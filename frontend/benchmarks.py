"""Load saved benchmark runs for the dashboard (pure functions, no Streamlit; unit-tested)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METRIC_ORDER = ("mrr", "hit@1", "hit@3", "hit@5", "recall@1", "recall@3", "recall@5", "ndcg@1", "ndcg@3", "ndcg@5")


def load_runs(dirs: list[Path]) -> list[dict[str, Any]]:
    """Retrieval benchmark results (schema from app.evaluation.benchmark) found in ``dirs``, oldest first.

    Files that are not retrieval results (e.g. Ragas runs) or are unreadable are skipped and reported in ``skipped``.
    """
    runs: list[dict[str, Any]] = []
    for d in dirs:
        for path in sorted(d.glob("*.json")) if d.is_dir() else []:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                overall = data["metrics"]["overall"]
                cfg = data["config"]
            except (OSError, ValueError, KeyError, TypeError):
                continue
            runs.append({
                "file": path.name, "label": cfg.get("label", path.stem), "retriever": cfg.get("retriever"),
                "embedding": cfg.get("embedding_model"), "chunker": cfg.get("chunker"),
                "created": data.get("run", {}).get("created_utc", ""), "overall": overall,
                "random": data.get("baselines", {}).get("random", {}).get("overall", {}),
                "latency": data.get("latency", {}).get("retrieval_ms", {}),
                "indexing_seconds": data.get("latency", {}).get("indexing_seconds"),
                "n_human_verified": data.get("dataset", {}).get("labels_human_verified"),
                "n_questions": data.get("dataset", {}).get("n_questions"),
                "stand_in": str(cfg.get("embedding_model", "")).startswith("hashing")
                            or cfg.get("reranker") == "lexical",
                "sha": data.get("results_sha256", "")[:12],
            })
    return sorted(runs, key=lambda r: (r["created"], r["file"]))


def comparison_rows(runs: list[dict[str, Any]], metrics: tuple[str, ...] = METRIC_ORDER) -> list[dict[str, Any]]:
    """One table row per run; the first row's values are the baseline for the delta columns."""
    rows = []
    for r in runs:
        row: dict[str, Any] = {"run": r["label"], "retriever": r["retriever"], "chunker": r["chunker"],
                               "stand-in": "yes" if r["stand_in"] else "no"}
        for m in metrics:
            row[m] = r["overall"].get(m)
        row["retrieval ms (mean)"] = r["latency"].get("mean")
        rows.append(row)
    return rows


def deltas(base: dict[str, Any], other: dict[str, Any], metrics: tuple[str, ...] = METRIC_ORDER) -> dict[str, float]:
    return {m: round(other["overall"][m] - base["overall"][m], 4)
            for m in metrics if m in base["overall"] and m in other["overall"]}
