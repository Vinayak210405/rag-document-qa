"""End-to-end evaluation with an LLM: answers every dataset question, then scores them with Ragas.

Needs a running answer LLM (RAG_LLM_*) and a judge LLM (RAG_JUDGE_*, defaults to the answer LLM). It is NOT run by the
ordinary test suite (no network / keys); the orchestration is tested with fake scorers in tests/unit/test_ragas_runner.py.

What is measured
  Ragas (LLM-judged, on answerable questions):  faithfulness, answer_relevancy (non-abstained answers only),
                                                context_precision, context_recall (need the reference answer)
  Deterministic (no judge):                     abstention on must-refuse questions, false abstentions, citation status
                                                counts, latency, token usage
Honest limits: Ragas scores come from an LLM judge: they vary with the judge model and between runs, they are not ground
truth, and a small judge can be wrong in both directions. The reference answers are Claude-authored (dataset README).
Retrieval metrics (Recall/MRR/nDCG) stay in app.evaluation.benchmark and never depend on Ragas.

Usage: RAG_LLM_MODEL=... RAG_JUDGE_MODEL=... uv run python -m app.evaluation.ragas_runner --label hybrid-llama --retriever hybrid
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from app.config import Settings
from app.evaluation.benchmark import _corpus_files, _git_info, _pkg
from app.evaluation.dataset import DEFAULT_CORPUS_DIR, DEFAULT_DATASET_PATH, EvalDataset, load_dataset
from app.service import RagService, build_service

SCHEMA_VERSION = 1
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
_ABSTAIN_STATUSES = {"abstained", "abstained_no_evidence"}


class Scorer(Protocol):
    """Anything with Ragas's async ``ascore(**kwargs)`` returning an object with ``.value``."""

    async def ascore(self, **kwargs: Any) -> Any: ...


@dataclass
class QuestionEval:
    id: str
    category: str
    answerable: bool
    status: str
    answer: str
    abstained: bool
    scores: dict[str, float | None] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    retrieval_ms: float = 0.0
    generation_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    invalid_citations: list[int] = field(default_factory=list)


class RagasEvalError(RuntimeError):
    """Evaluation cannot produce trustworthy output (e.g. every judged call failed)."""


def _score_inputs(metric: str, question: str, answer: str, contexts: list[str], reference: str) -> dict[str, Any]:
    return {
        "faithfulness": dict(user_input=question, response=answer, retrieved_contexts=contexts),
        "answer_relevancy": dict(user_input=question, response=answer),
        "context_precision": dict(user_input=question, reference=reference, retrieved_contexts=contexts),
        "context_recall": dict(user_input=question, retrieved_contexts=contexts, reference=reference),
    }[metric]


def run_ragas_eval(
    service: RagService, dataset: EvalDataset, scorers: Mapping[str, Scorer], *, top_k: int = 4,
) -> list[QuestionEval]:
    """Answer each question with the real pipeline and score it. Per-metric failures are recorded, never hidden."""
    unknown = set(scorers) - set(RAGAS_METRICS)
    if unknown:
        raise ValueError(f"unknown metrics: {sorted(unknown)}")
    rows: list[QuestionEval] = []
    for q in dataset.questions:
        result = service.answer(q.question, top_k)  # LLMError propagates: no fabricated answer
        row = QuestionEval(
            id=q.id, category=str(q.category), answerable=q.answerable, status=result.status, answer=result.answer,
            abstained=result.abstained, retrieval_ms=result.timings.retrieval_ms,
            generation_ms=result.timings.generation_ms, prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens, invalid_citations=list(result.invalid_citations),
        )
        contexts = [s.text for s in result.sources]
        if q.answerable:  # must-refuse questions have no reference to judge against: scored deterministically instead
            for name, scorer in scorers.items():
                if name in ("faithfulness", "answer_relevancy") and result.abstained:
                    continue  # nothing to judge in a refusal; the false abstention is counted separately
                try:
                    kwargs = _score_inputs(name, q.question, result.answer, contexts, q.reference_answer)
                    value = asyncio.run(scorer.ascore(**kwargs)).value
                    row.scores[name] = None if value is None else float(value)
                except Exception as exc:  # recorded per question and counted in the summary
                    row.scores[name] = None
                    row.errors[name] = f"{type(exc).__name__}: {exc}"[:300]
        rows.append(row)
    return rows


def summarize(rows: Sequence[QuestionEval], metric_names: Sequence[str]) -> dict:
    answerable = [r for r in rows if r.answerable]
    refuse = [r for r in rows if not r.answerable]
    metrics: dict[str, dict] = {}
    for name in metric_names:
        vals = [r.scores[name] for r in answerable if r.scores.get(name) is not None]
        metrics[name] = {
            "mean": round(statistics.fmean(vals), 4) if vals else None, "n_scored": len(vals),
            "n_errors": sum(1 for r in answerable if name in r.errors),
        }
    if metric_names and all(m["n_scored"] == 0 for m in metrics.values()) and any(m["n_errors"] for m in metrics.values()):
        raise RagasEvalError("every judged call failed; first error: " + next(
            e for r in answerable for e in r.errors.values()))
    gen = [r.generation_ms for r in rows if r.generation_ms is not None]
    return {
        "ragas": metrics,
        "abstention": {
            "must_refuse": len(refuse), "correctly_abstained": sum(r.abstained for r in refuse),
            "answerable": len(answerable), "false_abstentions": sum(r.abstained for r in answerable),
        },
        "answer_status": dict(Counter(r.status for r in rows)),
        "invalid_citation_answers": sum(1 for r in rows if r.invalid_citations),
        "latency_ms": {
            "mean_retrieval": round(statistics.fmean(r.retrieval_ms for r in rows), 2) if rows else None,
            "mean_generation": round(statistics.fmean(gen), 2) if gen else None,
        },
        "tokens": {
            "prompt": sum(r.prompt_tokens or 0 for r in rows), "completion": sum(r.completion_tokens or 0 for r in rows),
            "note": "as reported by the answer LLM endpoint; judge-model tokens are not tracked; no cost estimate (price unknown)",
        },
    }


def build_ragas_scorers(
    judge_model: str, base_url: str, api_key: str | None, embedding_model: str | None = None,
    metrics: Sequence[str] = RAGAS_METRICS,
) -> dict[str, Scorer]:
    """Construct real Ragas metric objects (no network call happens here)."""
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    except ImportError as exc:
        raise RagasEvalError("Ragas is not installed. Run: uv sync --extra eval") from exc
    client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed")
    llm = llm_factory(judge_model, client=client)
    built: dict[str, Scorer] = {}
    for name in metrics:
        if name == "faithfulness":
            built[name] = Faithfulness(llm=llm)
        elif name == "context_precision":
            built[name] = ContextPrecision(llm=llm)
        elif name == "context_recall":
            built[name] = ContextRecall(llm=llm)
        elif name == "answer_relevancy":
            if not embedding_model:
                raise RagasEvalError("answer_relevancy needs an embedding model (--judge-embedding-model)")
            built[name] = AnswerRelevancy(llm=llm, embeddings=OpenAIEmbeddings(client=client, model=embedding_model))
        else:
            raise RagasEvalError(f"unknown metric {name!r}")
    return built


def main(argv: Sequence[str] | None = None) -> int:
    base = Settings()
    p = argparse.ArgumentParser(description="Answer the dataset with a real LLM and score it with Ragas.")
    p.add_argument("--label", required=True)
    p.add_argument("--retriever", choices=["dense", "hybrid", "reranked"], default=base.retriever)
    p.add_argument("--top-k", type=int, default=base.default_top_k)
    p.add_argument("--metrics", default=",".join(RAGAS_METRICS))
    p.add_argument("--judge-model", default=None, help="default: the answer model (RAG_LLM_MODEL)")
    p.add_argument("--judge-base-url", default=None)
    p.add_argument("--judge-api-key", default=None, help="prefer the RAG_JUDGE_API_KEY env var")
    p.add_argument("--judge-embedding-model", default=None, help="OpenAI-compatible embedding model for answer_relevancy")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    import os
    judge_model = args.judge_model or os.environ.get("RAG_JUDGE_MODEL") or base.llm_model
    judge_url = args.judge_base_url or os.environ.get("RAG_JUDGE_BASE_URL") or base.llm_base_url
    judge_key = args.judge_api_key or os.environ.get("RAG_JUDGE_API_KEY") or (
        base.llm_api_key.get_secret_value() if base.llm_api_key else None)
    names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    dataset = load_dataset(DEFAULT_DATASET_PATH)
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(storage_dir=Path(tmp), retriever=args.retriever, chroma_collection=f"ragas-{uuid.uuid4().hex[:8]}")  # type: ignore[arg-type]
        service = build_service(settings)
        for path in _corpus_files(DEFAULT_CORPUS_DIR):
            service.ingest(path.name, path.read_bytes())
        scorers = build_ragas_scorers(judge_model, judge_url, judge_key, args.judge_embedding_model, names)
        rows = run_ragas_eval(service, dataset, scorers, top_k=args.top_k)
        summary = summarize(rows, names)
    result = {
        "schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "label": args.label, "retriever": args.retriever, "top_k": args.top_k, "answer_model": base.llm_model,
            "answer_base_url": base.llm_base_url, "temperature": base.llm_temperature, "judge_model": judge_model,
            "judge_base_url": judge_url, "judge_embedding_model": args.judge_embedding_model,
            "embedding_model": service.embedder.model_name, "chunker": service.chunker.strategy_name,
            "ragas_version": _pkg("ragas"),
        },
        "dataset": {"name": dataset.name, "version": dataset.version, "n_questions": len(dataset.questions)},
        "git": _git_info(), "summary": summary, "questions": [asdict(r) for r in rows],
        "caveats": [
            "LLM-judged scores are not ground truth, vary with the judge model and between runs.",
            "24 Claude-authored questions: differences of a few points are noise.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"ragas-{args.label}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
