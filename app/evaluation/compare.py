"""Compare two benchmark results: metric deltas, per-question wins/losses, and configuration differences.

With a few dozen questions, an average can move because of one question. So alongside each delta we report
how many questions got better, worse, or stayed the same. A "+0.05" built from one win is not evidence.

Usage:  uv run python -m app.evaluation.compare results/run_a.json results/run_b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_EPS = 1e-9


class ComparisonError(RuntimeError):
    """The two runs are not comparable (different dataset version or corpus)."""


@dataclass(frozen=True)
class MetricRow:
    metric: str
    a: float
    b: float
    wins: int  # questions where B is better than A
    losses: int  # questions where B is worse than A
    ties: int

    @property
    def delta(self) -> float:
        return self.b - self.a

    @property
    def verdict(self) -> str:
        return "better" if self.delta > _EPS else "worse" if self.delta < -_EPS else "same"


@dataclass(frozen=True)
class Comparison:
    label_a: str
    label_b: str
    config_differences: list[tuple[str, object, object]]
    overall: list[MetricRow]
    by_category: dict[str, list[MetricRow]]
    n_questions: int


def _question_metrics(run: dict) -> dict[str, dict[str, float]]:
    return {q["id"]: q["metrics"] for q in run["questions"] if q["metrics"]}


def _rows(a: dict, b: dict, a_agg: dict, b_agg: dict, ids: set[str] | None) -> list[MetricRow]:
    qa, qb = _question_metrics(a), _question_metrics(b)
    rows = []
    for name, value_a in a_agg.items():
        if name == "n" or name not in b_agg or value_a is None or b_agg[name] is None:
            continue
        key = "rr" if name == "mrr" else name
        wins = losses = ties = 0
        for qid in qa.keys() & qb.keys():
            if ids is not None and qid not in ids:
                continue
            diff = qb[qid][key] - qa[qid][key]
            wins, losses, ties = wins + (diff > _EPS), losses + (diff < -_EPS), ties + (abs(diff) <= _EPS)
        rows.append(MetricRow(name, value_a, b_agg[name], wins, losses, ties))
    return rows


def compare_runs(a: dict, b: dict, *, force: bool = False) -> Comparison:
    """Compare run ``b`` against baseline ``a``. Refuses mismatched datasets unless ``force``."""
    da, db = a["dataset"], b["dataset"]
    if not force and (da["version"], da["corpus_sha256"]) != (db["version"], db["corpus_sha256"]):
        raise ComparisonError(
            f"runs used different datasets (v{da['version']}/{da['corpus_sha256'][:8]} vs "
            f"v{db['version']}/{db['corpus_sha256'][:8]}); metrics are not comparable. Re-run on the same dataset."
        )
    diffs = [(k, a["config"].get(k), b["config"].get(k)) for k in sorted(set(a["config"]) | set(b["config"]))
             if a["config"].get(k) != b["config"].get(k)]
    by_category: dict[str, list[MetricRow]] = {}
    for cat, agg_a in a["metrics"]["by_category"].items():
        if cat in b["metrics"]["by_category"]:
            ids = {q["id"] for q in a["questions"] if q["category"] == cat}
            by_category[cat] = _rows(a, b, agg_a, b["metrics"]["by_category"][cat], ids)
    return Comparison(
        a["run"]["label"], b["run"]["label"], diffs,
        _rows(a, b, a["metrics"]["overall"], b["metrics"]["overall"], None), by_category,
        a["metrics"]["overall"]["n"],
    )


def _fmt(rows: Sequence[MetricRow], indent: str = "") -> list[str]:
    head = f"{indent}{'metric':<12}{'A':>8}{'B':>8}{'delta':>9}  {'better/worse/same questions':<28}verdict"
    lines = [head]
    for r in rows:
        lines.append(f"{indent}{r.metric:<12}{r.a:>8.3f}{r.b:>8.3f}{r.delta:>+9.3f}  "
                     f"{f'{r.wins} / {r.losses} / {r.ties}':<28}{r.verdict}")
    return lines


def format_comparison(c: Comparison) -> str:
    lines = [f"A (baseline): {c.label_a}", f"B (candidate): {c.label_b}", f"Answerable questions: {c.n_questions}", ""]
    lines.append("Configuration differences:" if c.config_differences else "Configuration differences: none")
    lines += [f"  {k}: {va!r}  ->  {vb!r}" for k, va, vb in c.config_differences]
    lines += ["", "OVERALL", *_fmt(c.overall, "  ")]
    focus = {"mrr"} | {r.metric for r in c.overall if r.metric.startswith(("recall@", "ndcg@"))}
    for cat, rows in c.by_category.items():
        lines += ["", f"{cat}", *_fmt([r for r in rows if r.metric in focus], "  ")]
    lines += ["", f"Caution: only {c.n_questions} answerable questions, and categories have 2-4 each. Differences backed by "
                  "one or two questions are noise, not evidence."]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare two benchmark result files (A = baseline, B = candidate).")
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--force", action="store_true", help="compare even if the datasets differ")
    args = p.parse_args(argv)
    try:
        comparison = compare_runs(json.loads(args.baseline.read_text("utf-8")), json.loads(args.candidate.read_text("utf-8")),
                                  force=args.force)
    except ComparisonError as exc:
        print(f"COMPARISON REFUSED: {exc}", file=sys.stderr)
        return 1
    print(format_comparison(comparison))
    return 0


if __name__ == "__main__":
    sys.exit(main())
