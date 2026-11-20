"""Score a retrieval function against the labelled dataset.

``evaluate`` takes any ``retrieve(query, k) -> list[RetrievedChunk]`` callable, so dense retrieval, hybrid
retrieval and reranked pipelines (later milestones) are all measured by exactly the same code.

Per answerable question we compute Hit@k, Recall@k, nDCG@k for each cut-off k, and the reciprocal rank
(RR; its mean over questions is MRR). Unanswerable questions have no gold evidence, so those metrics are
undefined for them; we record their top-1 score instead (useful later for abstention thresholds).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from app.evaluation.dataset import EvalDataset, Question
from app.evaluation.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from app.evaluation.relevance import chunk_grade, contains_passage, covered_evidence
from app.retrieval.types import RetrievedChunk

Retrieve = Callable[[str, int], list[RetrievedChunk]]
_ND = 6  # decimals kept in stored numbers (keeps result files stable across platforms)


@dataclass(frozen=True)
class IndexedChunk:
    """A chunk that exists in the index under test (needed to judge every chunk for nDCG's ideal)."""

    chunk_id: str
    doc_id: str
    text: str


@dataclass(frozen=True)
class RankedHit:
    rank: int
    chunk_id: str
    doc_id: str
    score: float
    grade: int | None  # evidence grade contained in the chunk (0-2); None when the question has no gold evidence
    distractor: bool  # contains a distractor passage and no gold evidence


@dataclass(frozen=True)
class QuestionResult:
    id: str
    category: str
    answerable: bool
    hits: tuple[RankedHit, ...]
    metrics: dict[str, float] | None  # None for questions that must be refused
    top1_score: float | None
    reachable: float | None  # fraction of required quotes fully contained in at least one indexed chunk
    distractor_above_gold: bool | None  # None when the question has no distractors
    tie_with_gold: bool | None  # first gold chunk has the SAME score as a non-gold chunk: rank decided by tie-break
    no_signal: bool  # top-1 score is 0: the retriever found nothing in common, so the ranking is arbitrary


@dataclass(frozen=True)
class Evaluation:
    ks: tuple[int, ...]
    questions: tuple[QuestionResult, ...]
    latency_ms: dict[str, float]  # question id -> retrieval time (kept apart: it is not deterministic)


def _score_question(
    q: Question, chunks: Sequence[IndexedChunk], retrieved: list[RetrievedChunk], ks: tuple[int, ...]
) -> QuestionResult:
    hits: list[RankedHit] = []
    for r in retrieved:
        grade = chunk_grade(r.doc_id, r.text, q.evidence) if q.answerable else None
        distractor = grade in (0, None) and contains_passage(r.doc_id, r.text, q.distractors)
        hits.append(RankedHit(r.rank, r.chunk_id, r.doc_id, round(r.score, _ND), grade, distractor))
    top1 = hits[0].score if hits else None
    no_signal = top1 is not None and top1 <= 0.0

    first_distractor = next((h.rank for h in hits if h.distractor), None)
    first_gold = next((h.rank for h in hits if h.grade == 2), None)
    above: bool | None = None
    if q.answerable and q.distractors:
        above = first_distractor is not None and (first_gold is None or first_distractor < first_gold)

    if not q.answerable:
        return QuestionResult(q.id, str(q.category), False, tuple(hits), None, top1, None, above, None, no_signal)

    required = [e for e in q.evidence if e.grade == 2]
    grades = [h.grade or 0 for h in hits]
    relevant = [g == 2 for g in grades]
    ideal = [chunk_grade(c.doc_id, c.text, q.evidence) for c in chunks]  # every chunk is judged by content
    metrics: dict[str, float] = {"rr": reciprocal_rank(relevant)}
    for k in ks:
        covered = covered_evidence([(r.doc_id, r.text) for r in retrieved[:k]], required)
        metrics[f"hit@{k}"] = hit_at_k(relevant, k)
        metrics[f"recall@{k}"] = recall_at_k(len(covered), len(required))
        metrics[f"ndcg@{k}"] = ndcg_at_k(grades, ideal, k)
    reachable = len(covered_evidence([(c.doc_id, c.text) for c in chunks], required)) / len(required)
    gold = next((h for h in hits if h.grade == 2), None)
    tie_with_gold = gold is not None and any(h.grade != 2 and h.score == gold.score for h in hits)
    return QuestionResult(
        q.id, str(q.category), True, tuple(hits), {k: round(v, _ND) for k, v in metrics.items()},
        top1, round(reachable, _ND), above, tie_with_gold, no_signal,
    )


def evaluate(
    dataset: EvalDataset, chunks: Sequence[IndexedChunk], retrieve: Retrieve, ks: Sequence[int]
) -> Evaluation:
    """Run every question through ``retrieve`` and score the ranked results."""
    cutoffs = tuple(sorted(set(ks)))
    if not cutoffs or cutoffs[0] < 1:
        raise ValueError("ks must be a non-empty collection of integers >= 1")
    results: list[QuestionResult] = []
    latency: dict[str, float] = {}
    for q in dataset.questions:
        start = time.perf_counter()
        retrieved = retrieve(q.question, cutoffs[-1])
        latency[q.id] = round((time.perf_counter() - start) * 1000, 3)
        results.append(_score_question(q, chunks, retrieved, cutoffs))
    return Evaluation(cutoffs, tuple(results), latency)


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), _ND) if values else None


def _aggregate(results: Sequence[QuestionResult]) -> dict[str, float | int | None]:
    names = list(results[0].metrics) if results and results[0].metrics else []
    out: dict[str, float | int | None] = {"n": len(results)}
    for name in names:
        out["mrr" if name == "rr" else name] = _mean([r.metrics[name] for r in results if r.metrics])
    return out


def summarize(evaluation: Evaluation) -> dict:
    """Aggregate per-question results into overall, per-category and diagnostic summaries."""
    answerable = [r for r in evaluation.questions if r.answerable]
    refused = [r for r in evaluation.questions if not r.answerable]
    by_category: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in answerable:
        by_category[r.category].append(r)

    def top1(rs: Sequence[QuestionResult]) -> dict[str, float | int | None]:
        scores = [r.top1_score for r in rs if r.top1_score is not None]
        return {"n": len(scores), "mean": _mean(scores), "min": min(scores, default=None), "max": max(scores, default=None)}

    with_distractors = [r for r in answerable if r.distractor_above_gold is not None]
    depth = evaluation.ks[-1]
    return {
        "definitions": {
            "mrr": f"mean reciprocal rank over the retrieved top-{depth} (MRR@{depth}); a first relevant chunk "
                   f"below rank {depth} counts as 0",
            "hit@k": "1 if any of the top-k chunks contains required (grade 2) evidence",
            "recall@k": "fraction of required evidence quotes fully contained in the top-k chunks",
            "ndcg@k": "graded nDCG; every chunk in the index is judged by content, so the ideal ranking is exact",
        },
        "overall": _aggregate(answerable),
        "by_category": {c: _aggregate(rs) for c, rs in sorted(by_category.items())},
        "reachability": {
            "answerable_questions": len(answerable),
            "mean_required_quotes_reachable": _mean([r.reachable for r in answerable if r.reachable is not None]),
        },
        "distractors": {
            "questions_with_distractors": len(with_distractors),
            "distractor_above_gold_rate": _mean([float(r.distractor_above_gold) for r in with_distractors]),
        },
        "top1_score": {"answerable": top1(answerable), "must_refuse": top1(refused)},
        "ties": {
            "answerable_questions": len(answerable),
            "gold_tied_with_non_gold": sum(bool(r.tie_with_gold) for r in answerable),
            "no_signal": sum(r.no_signal for r in answerable),
        },
    }


def random_baseline(
    dataset: EvalDataset, chunks: Sequence[IndexedChunk], ks: Sequence[int], *, trials: int = 200, seed: int = 0
) -> dict[str, float | int | None]:
    """Expected overall metrics if the index were ranked uniformly at random (seeded Monte-Carlo).

    A control, not a competitor: with a tiny index even a random ranking scores well at larger k, so a
    result only means something relative to this. Runs through the same ``evaluate`` code as real pipelines.
    """
    rng = np.random.default_rng(seed)
    pool = list(chunks)
    totals: dict[str, float] = defaultdict(float)
    n = 0
    for _ in range(trials):
        def retrieve(query: str, k: int) -> list[RetrievedChunk]:
            order = rng.permutation(len(pool))[:k]
            return [RetrievedChunk(i, 0.0, pool[j].chunk_id, pool[j].doc_id, pool[j].doc_id, 0, None, None, pool[j].text)
                    for i, j in enumerate(order, start=1)]

        overall = summarize(evaluate(dataset, pool, retrieve, ks))["overall"]
        n = overall["n"]
        for name, value in overall.items():
            if name != "n" and value is not None:
                totals[name] += value
    return {"n": n, **{name: round(total / trials, _ND) for name, total in totals.items()}}
