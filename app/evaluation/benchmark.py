"""Reproducible benchmark runner.

Each run builds a fresh in-memory index from the corpus (so it never depends on earlier runs), evaluates the
retriever on the labelled dataset, and writes a JSON file (full detail) and a CSV file (one row per question).

Reproducibility design
----------------------
* The JSON separates what must be reproducible from what cannot be: ``config``, ``dataset``, ``index``,
  ``metrics`` and ``questions`` are deterministic; timestamps, git info, environment and ALL latencies are not.
* ``results_sha256`` hashes only the deterministic part, so "same configuration + same dataset" can be
  verified by comparing one string. Numbers are rounded to 6 decimals first. On one machine the hash is
  identical run to run; across machines it should match, but tiny floating-point differences could in
  principle change it, so compare metrics, not just the hash, when moving between machines.

Usage:  uv run python -m app.evaluation.benchmark --label dense-hashing --embedder hashing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Sequence

import numpy as np

from app.chunking import SemanticChunker, SentenceChunker
from app.config import Settings
from app.embeddings import Embedder, create_embedder
from app.evaluation.dataset import (
    DEFAULT_CORPUS_DIR, DEFAULT_DATASET_PATH, EvalDataset, load_dataset, validate_dataset,
)
from app.evaluation.evaluator import IndexedChunk, Retrieve, evaluate, random_baseline, summarize
from app.indexing import ChromaVectorStore
from app.ingestion import SUPPORTED_TYPES, parse_document
from app.retrieval.dense import DenseRetriever
from app.retrieval.rerank import CrossEncoderReranker, LexicalOverlapReranker, RerankedRetriever
from app.retrieval.hybrid import BM25Retriever, HybridRetriever

SCHEMA_VERSION = 1
_ROOT = Path(__file__).resolve().parents[2]


class BenchmarkError(RuntimeError):
    """The benchmark cannot produce trustworthy results (e.g. the dataset no longer matches the corpus)."""


@dataclass(frozen=True)
class PipelineConfig:
    """Everything that defines the pipeline under test. Logged verbatim with every result."""

    label: str
    retriever: str = "dense"  # dense | bm25 | hybrid | reranked
    candidates: int = 20  # per-list candidate count for hybrid
    rrf_k: int = 60
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # or "lexical" (offline stand-in)
    embedding_backend: str = "sentence_transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = ""
    chunking: str = "sentence"  # sentence | semantic
    semantic_percentile: float = 90.0
    semantic_min_chars: int = 100
    chunk_max_chars: int = 500
    chunk_overlap_chars: int = 80
    ks: tuple[int, ...] = (1, 3, 5)


def _corpus_files(corpus_dir: Path) -> list[Path]:
    return sorted(
        p for p in corpus_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in SUPPORTED_TYPES
    )


def _build_retriever(config: PipelineConfig, store: ChromaVectorStore, embedder: Embedder) -> Retrieve:
    if config.retriever == "dense":
        return DenseRetriever(store, embedder).retrieve
    if config.retriever == "bm25":
        return BM25Retriever(store).retrieve
    if config.retriever == "hybrid":
        return HybridRetriever(store, embedder, candidates=config.candidates, rrf_k=config.rrf_k).retrieve
    if config.retriever == "reranked":
        base = HybridRetriever(store, embedder, candidates=config.candidates, rrf_k=config.rrf_k)
        reranker = LexicalOverlapReranker() if config.reranker == "lexical" else CrossEncoderReranker(config.reranker)
        return RerankedRetriever(base, reranker, candidates=config.candidates).retrieve
    raise BenchmarkError(f"unknown retriever {config.retriever!r}; available: dense, bm25, hybrid, reranked")


def _canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _git_info() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=_ROOT, capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": commit, "uncommitted_changes": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "uncommitted_changes": None, "note": "git not available"}


def _pkg(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _latency_summary(per_question: dict[str, float], indexing_seconds: float) -> dict:
    values = np.array(list(per_question.values()), dtype=float)
    return {
        "indexing_seconds": round(indexing_seconds, 3),
        "retrieval_ms": {
            "mean": round(float(values.mean()), 3), "p50": round(float(np.percentile(values, 50)), 3),
            "p95": round(float(np.percentile(values, 95)), 3), "max": round(float(values.max()), 3),
            "note": "includes query embedding; p95 over a few dozen questions is only indicative",
        },
        "per_question_ms": per_question,
        "tokens_and_cost": None,  # retrieval-only run: no LLM calls, so no tokens or cost
    }


def run_benchmark(
    config: PipelineConfig,
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    embedder: Embedder | None = None,
) -> dict:
    """Build an index from ``corpus_dir``, evaluate ``config`` on the dataset, and return the result dict."""
    dataset: EvalDataset = load_dataset(dataset_path)
    problems = validate_dataset(dataset, corpus_dir)
    if problems:
        raise BenchmarkError(
            "dataset does not match the corpus, so results would be meaningless:\n  - " + "\n  - ".join(problems)
        )
    if embedder is None:
        embedder = create_embedder(Settings(
            embedding_backend=config.embedding_backend,  # type: ignore[arg-type]
            embedding_model=config.embedding_model,
            embedding_query_prefix=config.embedding_query_prefix,
        ))

    if config.chunking == "semantic":
        chunker = SemanticChunker(embedder, config.chunk_max_chars, config.semantic_min_chars, config.semantic_percentile)
    elif config.chunking == "sentence":
        chunker = SentenceChunker(config.chunk_max_chars, config.chunk_overlap_chars)
    else:
        raise BenchmarkError(f"unknown chunking {config.chunking!r}; available: sentence, semantic")
    store = ChromaVectorStore(None, f"bench-{uuid.uuid4().hex}", embedder.model_name, embedder.dimension)
    try:
        start = time.perf_counter()
        for path in _corpus_files(corpus_dir):
            doc = parse_document(path.name, path.read_bytes())
            chunks = chunker.chunk(doc)
            store.replace_document(doc.doc_id, chunks, embedder.embed_documents([c.text for c in chunks]))
        indexing_seconds = time.perf_counter() - start

        snapshot = store.load_snapshot()
        indexed = [
            IndexedChunk(cid, str(meta["doc_id"]), text)
            for cid, meta, text in zip(snapshot.ids, snapshot.metadatas, snapshot.texts)
        ]
        evaluation = evaluate(dataset, indexed, _build_retriever(config, store, embedder), config.ks)
    finally:
        store.drop()

    metrics = summarize(evaluation)
    baseline_trials, baseline_seed = 200, 0
    baselines = {"random": {
        "description": "expected metrics of a uniformly random ranking of the index (seeded Monte-Carlo control)",
        "trials": baseline_trials, "seed": baseline_seed,
        "overall": random_baseline(dataset, indexed, evaluation.ks, trials=baseline_trials, seed=baseline_seed),
    }}
    questions = [asdict(r) for r in evaluation.questions]
    resolved_config = {
        **asdict(config),
        "ks": list(evaluation.ks),
        "embedding_model": embedder.model_name,
        "embedding_dim": embedder.dimension,
        "chunker": chunker.strategy_name,
    }
    refuse = sum(1 for q in dataset.questions if not q.answerable)
    dataset_info = {
        "name": dataset.name, "version": dataset.version, "corpus_sha256": dataset.corpus_sha256,
        "n_questions": len(dataset.questions), "n_answerable": len(dataset.questions) - refuse, "n_must_refuse": refuse,
        "labels_human_verified": sum(q.human_verified for q in dataset.questions),
        "label_sources": dict(Counter(q.label_source for q in dataset.questions)),
    }
    index_info = {
        "documents": len({c.doc_id for c in indexed}), "chunks": len(indexed),
        "mean_chunk_chars": round(sum(len(c.text) for c in indexed) / len(indexed), 1),
        # embedding work done at ingest: one vector per chunk, plus one per sentence if the chunker embeds
        "embedded_texts": len(indexed) + getattr(chunker, "units_embedded", 0),
    }
    deterministic = {"config": resolved_config, "dataset": dataset_info, "index": index_info,
                     "metrics": metrics, "questions": questions}
    results_sha256 = hashlib.sha256(_canonical(deterministic).encode()).hexdigest()
    created = datetime.now(timezone.utc)
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "run_id": f"{config.label}-{created:%Y%m%dT%H%M%SZ}-{results_sha256[:8]}",
            "label": config.label, "created_at": created.isoformat(timespec="seconds"),
            "dataset_path": str(dataset_path), "git": _git_info(),
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "numpy": _pkg("numpy"), "chromadb": _pkg("chromadb")},
        },
        **deterministic,
        "baselines": baselines,  # outside the hash: depends on numpy's random stream, not on the pipeline
        "latency": _latency_summary(evaluation.latency_ms, indexing_seconds),
        "results_sha256": results_sha256,
    }


def write_results(result: dict, output_dir: Path) -> tuple[Path, Path]:
    """Write ``<run_id>.json`` (full) and ``<run_id>.csv`` (one row per question)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = result["run"]["run_id"]
    json_path, csv_path = output_dir / f"{run_id}.json", output_dir / f"{run_id}.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    ks = result["config"]["ks"]
    metric_cols = ["rr"] + [f"{m}@{k}" for k in ks for m in ("hit", "recall", "ndcg")]
    latency = result["latency"]["per_question_ms"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "category", "answerable", *metric_cols, "top1_doc", "top1_score", "reachable",
                         "distractor_above_gold", "tie_with_gold", "no_signal", "latency_ms", "retrieved(rank:doc:grade)"])
        for q in result["questions"]:
            m = q["metrics"] or {}
            top = q["hits"][0] if q["hits"] else None
            writer.writerow([
                q["id"], q["category"], q["answerable"], *[m.get(c, "") for c in metric_cols],
                top["doc_id"] if top else "", q["top1_score"] if q["top1_score"] is not None else "",
                "" if q["reachable"] is None else q["reachable"],
                "" if q["distractor_above_gold"] is None else q["distractor_above_gold"],
                "" if q["tie_with_gold"] is None else q["tie_with_gold"], q["no_signal"],
                latency[q["id"]],
                " ".join(f"{h['rank']}:{h['doc_id']}:{'-' if h['grade'] is None else h['grade']}" for h in q["hits"]),
            ])
    return json_path, csv_path


def format_summary(result: dict) -> str:
    """Human-readable summary printed after a run."""
    cfg, ds, idx = result["config"], result["dataset"], result["index"]
    ks = cfg["ks"]
    cols = ["n", "mrr"] + [f"{m}@{k}" for k in ks for m in ("hit", "recall", "ndcg")]
    titles = {"mrr": f"mrr@{max(ks)}"}

    def row(name: str, agg: dict) -> str:
        cells = [f"{agg[c]:.3f}" if isinstance(agg.get(c), float) else str(agg.get(c, "-")) for c in cols]
        return f"{name:<26}" + "".join(f"{c:>10}" for c in cells)

    lines = [
        f"Run      : {result['run']['run_id']}",
        f"Pipeline : {cfg['retriever']} | embedder {cfg['embedding_model']} (dim {cfg['embedding_dim']}) | {cfg['chunker']}",
        f"Index    : {idx['documents']} documents, {idx['chunks']} chunks (mean {idx['mean_chunk_chars']} chars)",
        f"Dataset  : {ds['name']} v{ds['version']} | {ds['n_answerable']} answerable, {ds['n_must_refuse']} must-refuse",
        f"Labels   : {ds['labels_human_verified']}/{ds['n_questions']} human-verified",
        "",
        f"{'':<26}" + "".join(f"{titles.get(c, c):>10}" for c in cols),
        row("OVERALL", result["metrics"]["overall"]),
        row("random ranking (control)", result["baselines"]["random"]["overall"]),
        *[row(cat, agg) for cat, agg in result["metrics"]["by_category"].items()],
        "",
        f"Ties / no signal             : {result['metrics']['ties']}  (these ranks are decided by the tie-break rule, not by the retriever)",
        f"Evidence reachable in index  : {result['metrics']['reachability']['mean_required_quotes_reachable']}",
        f"Outdated source above gold   : {result['metrics']['distractors']}",
        f"Top-1 score (answerable)     : {result['metrics']['top1_score']['answerable']}",
        f"Top-1 score (must refuse)    : {result['metrics']['top1_score']['must_refuse']}",
        f"Retrieval latency (ms)       : { {k: v for k, v in result['latency']['retrieval_ms'].items() if k != 'note'} }",
        f"results_sha256               : {result['results_sha256']}",
    ]
    if ds["labels_human_verified"] < ds["n_questions"]:
        lines += ["", "WARNING: labels are not fully human-verified and the set is small; treat differences as indicative."]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    defaults = Settings()
    p = argparse.ArgumentParser(description="Run the retrieval benchmark and save JSON + CSV results.")
    p.add_argument("--label", default="dense", help="short name for this run (used in the file name)")
    p.add_argument("--retriever", choices=["dense", "bm25", "hybrid", "reranked"], default="dense")
    p.add_argument("--candidates", type=int, default=20, help="per-list candidates fed to RRF (hybrid)")
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help='model name, or "lexical" (offline stand-in)')
    p.add_argument("--embedder", choices=["hashing", "sentence_transformers"], default=defaults.embedding_backend)
    p.add_argument("--model", default=defaults.embedding_model)
    p.add_argument("--chunking", choices=["sentence", "semantic"], default="sentence")
    p.add_argument("--semantic-percentile", type=float, default=90.0)
    p.add_argument("--chunk-max", type=int, default=defaults.chunk_max_chars)
    p.add_argument("--chunk-overlap", type=int, default=defaults.chunk_overlap_chars)
    p.add_argument("--ks", default="1,3,5", help="comma-separated cut-offs, e.g. 1,3,5")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    config = PipelineConfig(
        label=args.label, retriever=args.retriever, candidates=args.candidates, rrf_k=args.rrf_k, reranker=args.reranker,
        embedding_backend=args.embedder, embedding_model=args.model,
        embedding_query_prefix=defaults.embedding_query_prefix if args.embedder == "sentence_transformers" else "",
        chunking=args.chunking, semantic_percentile=args.semantic_percentile,
        chunk_max_chars=args.chunk_max, chunk_overlap_chars=args.chunk_overlap,
        ks=tuple(int(k) for k in args.ks.split(",")),
    )
    try:
        result = run_benchmark(config, dataset_path=args.dataset, corpus_dir=args.corpus)
    except BenchmarkError as exc:
        print(f"BENCHMARK REFUSED: {exc}", file=sys.stderr)
        return 1
    json_path, csv_path = write_results(result, args.output_dir)
    print(format_summary(result))
    print(f"\nSaved: {json_path}\n       {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
