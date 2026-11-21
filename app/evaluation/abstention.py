"""Evaluate the pre-generation evidence gate on the labelled dataset (retrieval only, no LLM).

For each question we retrieve top-k, compute coverage (app/generation/evidence.py) and ask: would the gate abstain?
  - must-refuse questions (unanswerable / insufficient evidence): abstaining is CORRECT (true refusal)
  - answerable questions: abstaining is a FALSE refusal (the user loses a good answer)
This evaluates the cheap lexical gate only. Whether an LLM refuses or hallucinates on the questions that pass the gate
needs a real model and is a separate (LLM-dependent) measurement that is NOT done here.

Usage: uv run python -m app.evaluation.abstention --embedder hashing
"""

from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.chunking import SentenceChunker
from app.config import Settings
from app.embeddings import create_embedder
from app.evaluation.benchmark import _corpus_files
from app.evaluation.dataset import DEFAULT_CORPUS_DIR, DEFAULT_DATASET_PATH, load_dataset
from app.generation.evidence import assess_evidence
from app.indexing import ChromaVectorStore
from app.ingestion import parse_document
from app.retrieval.hybrid import HybridRetriever


@dataclass(frozen=True)
class GateRow:
    qid: str
    category: str
    answerable: bool
    coverage: float
    missing: tuple[str, ...]


def coverage_rows(embedder_backend: str = "hashing", model: str | None = None, k: int = 4) -> list[GateRow]:
    settings = Settings(embedding_backend=embedder_backend, **({"embedding_model": model} if model else {}))  # type: ignore[arg-type]
    embedder = create_embedder(settings)
    dataset = load_dataset(DEFAULT_DATASET_PATH)
    store = ChromaVectorStore(None, f"gate-{uuid.uuid4().hex}", embedder.model_name, embedder.dimension)
    try:
        chunker = SentenceChunker(settings.chunk_max_chars, settings.chunk_overlap_chars)
        for path in _corpus_files(DEFAULT_CORPUS_DIR):
            doc = parse_document(path.name, path.read_bytes())
            chunks = chunker.chunk(doc)
            store.replace_document(doc.doc_id, chunks, embedder.embed_documents([c.text for c in chunks]))
        retriever = HybridRetriever(store, embedder)
        rows = []
        for q in dataset.questions:
            d = assess_evidence(q.question, retriever.retrieve(q.question, k), 1.0)
            rows.append(GateRow(q.id, q.category, q.answerable, d.coverage, d.missing_terms))
        return rows
    finally:
        store.drop()


def summarize_gate(rows: list[GateRow], threshold: float) -> dict:
    refuse = [r for r in rows if not r.answerable]
    answer = [r for r in rows if r.answerable]
    true_refusals = sum(r.coverage < threshold for r in refuse)
    false_refusals = sum(r.coverage < threshold for r in answer)
    return {
        "threshold": threshold, "must_refuse": len(refuse), "answerable": len(answer),
        "true_refusals": true_refusals, "missed_refusals": len(refuse) - true_refusals,
        "false_refusals": false_refusals,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--embedder", choices=["hashing", "sentence_transformers"], default="hashing")
    p.add_argument("--model", default=None)
    p.add_argument("--k", type=int, default=4)
    args = p.parse_args(argv)
    rows = coverage_rows(args.embedder, args.model, args.k)
    print("qid   category                 answerable  coverage  missing terms")
    for r in rows:
        print(f"{r.qid}  {r.category:<24} {str(r.answerable):<10}  {r.coverage:7.2f}   {', '.join(r.missing)}")
    print("\nthreshold  true_refusals  missed_refusals  false_refusals")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        s = summarize_gate(rows, t)
        print(f"{t:>9.1f}  {s['true_refusals']:>6}/{s['must_refuse']:<7} {s['missed_refusals']:>8}        {s['false_refusals']:>6}/{s['answerable']}")
    print("\nNOTE: 24 Claude-authored questions; any threshold chosen here is tuned on the same data it is measured on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
