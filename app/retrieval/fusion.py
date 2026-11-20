"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009).

    RRF(d) = sum over ranked lists L containing d of  1 / (k + rank_L(d))      (rank is 1-based)

Only ranks are used, never raw scores, so lists with incomparable scales (cosine in [-1, 1] vs
unbounded BM25) can be fused without calibration. ``k`` (default 60) damps the advantage of
the very top ranks. Ties are broken by the best rank in any list, then by first appearance, so
output is deterministic.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[T]], k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[T, float]]:
    """Fuse best-first rankings of item ids into one list of (item, fused score), best first."""
    if k < 0:
        raise ValueError("k must be >= 0")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    scores: dict[T, float] = {}
    best_rank: dict[T, int] = {}
    first_seen: dict[T, int] = {}
    for weight, ranking in zip(weights, rankings):
        seen: set[T] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:  # a duplicate inside one list counts once, at its best rank
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
            first_seen.setdefault(item, len(first_seen))
    order = sorted(scores, key=lambda i: (-scores[i], best_rank[i], first_seen[i]))
    return [(item, scores[item]) for item in order]
