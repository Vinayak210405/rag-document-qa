"""The evaluator, exercised with a tiny hand-built index and scripted retrievers."""

import pytest

from app.evaluation.dataset import EvalDataset
from app.evaluation.evaluator import IndexedChunk, evaluate, summarize
from app.retrieval.types import RetrievedChunk

CHUNKS = {
    "c1": IndexedChunk("c1", "a.txt", "Limit is 100 USD. Receipts are kept."),
    "c2": IndexedChunk("c2", "a.txt", "Unrelated text."),
    "c3": IndexedChunk("c3", "b.txt", "Old limit is 50 USD."),
    "c4": IndexedChunk("c4", "b.txt", "Another unrelated chunk."),
    "c5": IndexedChunk("c5", "c.txt", "Fee is 5 USD."),
}


def q(id_: str, category: str, answerable: bool, evidence=(), distractors=()) -> dict:
    return {"id": id_, "category": category, "question": f"question {id_}", "answerable": answerable,
            "reference_answer": "x", "evidence": list(evidence), "distractors": list(distractors),
            "label_source": "claude_authored"}


LIMIT = {"doc": "a.txt", "quote": "Limit is 100 USD.", "grade": 2}
RECEIPTS = {"doc": "a.txt", "quote": "Receipts are kept.", "grade": 1}
FEE = {"doc": "c.txt", "quote": "Fee is 5 USD.", "grade": 2}
OLD = {"doc": "b.txt", "quote": "Old limit is 50 USD."}

DATASET = EvalDataset.model_validate({
    "name": "t", "version": "0", "created": "2026-01-01", "description": "synthetic", "corpus_sha256": "0" * 64,
    "questions": [
        q("qa", "conflicting_outdated", True, [LIMIT, RECEIPTS], [OLD]),
        q("qm", "multi_document", True, [LIMIT, FEE]),
        q("qu", "unanswerable", False),
    ],
})


def retriever(order: list[str]):
    """A scripted retriever that always returns the given chunk ids, best first."""
    def retrieve(query: str, k: int) -> list[RetrievedChunk]:
        return [RetrievedChunk(i, 1.0 - 0.1 * i, c, CHUNKS[c].doc_id, CHUNKS[c].doc_id, 0, None, None, CHUNKS[c].text)
                for i, c in enumerate(order[:k], start=1)]
    return retrieve


def run(order: list[str], ks=(1, 3)):
    return {r.id: r for r in evaluate(DATASET, list(CHUNKS.values()), retriever(order), ks).questions}


def test_a_perfect_ranking_scores_one_everywhere() -> None:
    r = run(["c1", "c5", "c2", "c3", "c4"])
    assert r["qa"].metrics == {"rr": 1.0, "hit@1": 1.0, "recall@1": 1.0, "ndcg@1": 1.0,
                               "hit@3": 1.0, "recall@3": 1.0, "ndcg@3": 1.0}


def test_finding_the_gold_chunk_at_rank_two_halves_the_reciprocal_rank() -> None:
    m = run(["c2", "c1", "c4"])["qa"].metrics
    assert m["rr"] == 0.5 and m["hit@1"] == 0.0 and m["hit@3"] == 1.0
    assert m["recall@1"] == 0.0 and m["recall@3"] == 1.0
    assert 0 < m["ndcg@3"] < 1  # found, but discounted for being second


def test_recall_counts_how_many_required_pieces_were_found() -> None:
    r = run(["c1", "c2", "c5"])["qm"]  # needs LIMIT (in c1) and FEE (in c5)
    assert r.metrics["recall@1"] == 0.5 and r.metrics["recall@3"] == 1.0
    assert r.metrics["rr"] == 1.0  # a required chunk is already first


def test_finding_nothing_relevant_scores_zero() -> None:
    m = run(["c2", "c4"])["qa"].metrics
    assert all(v == 0.0 for v in m.values())


def test_outdated_distractor_ranked_above_the_gold_chunk_is_flagged() -> None:
    assert run(["c3", "c1"])["qa"].distractor_above_gold is True
    assert run(["c1", "c3"])["qa"].distractor_above_gold is False
    assert run(["c3", "c4"])["qa"].distractor_above_gold is True  # distractor found, gold never found
    assert run(["c3", "c1"])["qm"].distractor_above_gold is None  # question has no distractors


def test_must_refuse_questions_get_no_relevance_metrics_but_keep_their_top_score() -> None:
    r = run(["c1", "c2"])["qu"]
    assert r.metrics is None and r.reachable is None and r.top1_score == pytest.approx(0.9)
    assert all(h.grade is None for h in r.hits)


def test_reachability_exposes_evidence_destroyed_by_chunking() -> None:
    split = [IndexedChunk("s1", "a.txt", "Limit is"), IndexedChunk("s2", "a.txt", "100 USD."), CHUNKS["c5"]]
    results = evaluate(DATASET, split, lambda query, k: [], (1,)).questions
    by_id = {r.id: r for r in results}
    assert by_id["qa"].reachable == 0.0  # the only required quote was cut in half
    assert by_id["qm"].reachable == 0.5  # FEE survived, LIMIT did not


def test_summary_aggregates_by_category_and_excludes_refusals() -> None:
    summary = summarize(evaluate(DATASET, list(CHUNKS.values()), retriever(["c1", "c5", "c2"]), (1, 3)))
    assert summary["overall"]["n"] == 2 and summary["overall"]["mrr"] == 1.0
    assert set(summary["by_category"]) == {"conflicting_outdated", "multi_document"}  # no refusal categories
    assert summary["overall"]["recall@1"] == pytest.approx(0.75)  # qa 1.0, qm 0.5
    assert summary["top1_score"]["must_refuse"]["n"] == 1 and summary["top1_score"]["answerable"]["n"] == 2
    assert summary["distractors"] == {"questions_with_distractors": 1, "distractor_above_gold_rate": 0.0}
    assert summary["reachability"]["mean_required_quotes_reachable"] == 1.0


def test_latency_is_recorded_per_question_and_invalid_cutoffs_are_rejected() -> None:
    ev = evaluate(DATASET, list(CHUNKS.values()), retriever(["c1"]), (1,))
    assert set(ev.latency_ms) == {"qa", "qm", "qu"} and all(v >= 0 for v in ev.latency_ms.values())
    for bad in ((), (0,), (-1, 3)):
        with pytest.raises(ValueError):
            evaluate(DATASET, list(CHUNKS.values()), retriever(["c1"]), bad)


# ------------------------------------------------------------------ tie / no-signal diagnostics
def scored(scores: dict[str, float]):
    """A retriever that returns chunks with explicit scores (best first), to create ties."""
    def retrieve(query: str, k: int) -> list[RetrievedChunk]:
        order = sorted(scores, key=lambda c: -scores[c])[:k]
        return [RetrievedChunk(i, scores[c], c, CHUNKS[c].doc_id, CHUNKS[c].doc_id, 0, None, None, CHUNKS[c].text)
                for i, c in enumerate(order, start=1)]
    return retrieve


def test_a_gold_chunk_tied_with_a_non_gold_chunk_is_flagged() -> None:
    tied = {r.id: r for r in evaluate(DATASET, list(CHUNKS.values()), scored({"c1": 0.5, "c3": 0.5, "c2": 0.1}), (3,)).questions}
    assert tied["qa"].tie_with_gold is True  # c1 (gold) shares 0.5 with c3 (not gold)
    clear = {r.id: r for r in evaluate(DATASET, list(CHUNKS.values()), scored({"c1": 0.9, "c3": 0.5, "c2": 0.1}), (3,)).questions}
    assert clear["qa"].tie_with_gold is False
    assert clear["qu"].tie_with_gold is None  # undefined without gold evidence


def test_zero_top_score_means_the_ranking_carries_no_signal() -> None:
    ev = evaluate(DATASET, list(CHUNKS.values()), scored({"c2": 0.0, "c1": 0.0}), (2,))
    assert all(r.no_signal for r in ev.questions)
    # all scores are 0, so every gold chunk is also tied with a non-gold chunk: the two flags overlap
    assert summarize(ev)["ties"] == {"answerable_questions": 2, "gold_tied_with_non_gold": 2, "no_signal": 2}


# ------------------------------------------------------------------ random-ranking control
def _single_question_dataset() -> EvalDataset:
    return EvalDataset.model_validate({
        "name": "t", "version": "0", "created": "2026-01-01", "description": "x", "corpus_sha256": "0" * 64,
        "questions": [q("only", "direct_factual", True, [LIMIT])],
    })


def test_random_baseline_matches_the_analytic_expectation() -> None:
    from app.evaluation.evaluator import random_baseline

    # 5 chunks, exactly one relevant (c1): E[hit@1]=1/5, E[hit@3]=1-C(4,3)/C(5,3)=0.6.
    # Retrieval depth is max(ks)=3, so reciprocal rank is truncated there: E[MRR@3]=(1+1/2+1/3+0+0)/5.
    base = random_baseline(_single_question_dataset(), list(CHUNKS.values()), (1, 3), trials=4000, seed=1)
    assert base["hit@1"] == pytest.approx(0.2, abs=0.03)
    assert base["hit@3"] == pytest.approx(0.6, abs=0.03)
    assert base["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 5, abs=0.03)


def test_random_baseline_is_deterministic_for_a_seed_and_beaten_by_a_perfect_ranking() -> None:
    from app.evaluation.evaluator import random_baseline

    args = (_single_question_dataset(), list(CHUNKS.values()), (1,))
    assert random_baseline(*args, trials=50, seed=7) == random_baseline(*args, trials=50, seed=7)
    perfect = summarize(evaluate(*args[:2], retriever(["c1", "c2"]), (1,)))["overall"]
    assert perfect["hit@1"] > random_baseline(*args, trials=200, seed=7)["hit@1"]


def test_the_summary_states_the_depth_that_mrr_is_truncated_at() -> None:
    summary = summarize(evaluate(DATASET, list(CHUNKS.values()), retriever(["c1", "c5", "c2"]), (1, 3)))
    assert "MRR@3" in summary["definitions"]["mrr"] and "below rank 3" in summary["definitions"]["mrr"]


def test_ndcg_penalises_relevant_chunks_that_the_retriever_failed_to_return() -> None:
    import math

    # qm needs c1 AND c5 (both grade 2). Retrieved [c1, c2, c4] finds only one of them.
    # DCG@3 = 3; the ideal ranking of the WHOLE index is [2, 2, 0, ...] so IDCG@3 = 3 + 3/log2(3).
    m = run(["c1", "c2", "c4"])["qm"].metrics
    assert m["ndcg@3"] == pytest.approx(3 / (3 + 3 / math.log2(3)), abs=1e-5)  # about 0.613, not 1.0
    assert m["recall@3"] == 0.5
