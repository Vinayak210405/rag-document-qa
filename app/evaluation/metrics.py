"""Ranking metrics, implemented from first principles (no evaluation library).

Conventions: ``k`` is a cut-off (>= 1); ranks are 1-based; "relevant" lists are ordered best-first.
Graded relevance uses gain = 2^grade - 1 (grades 0, 1, 2 give gains 0, 1, 3), so a "required" chunk is
worth three times a merely "helpful" one.
"""

from __future__ import annotations

import math
from typing import Sequence


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")


def reciprocal_rank(relevant: Sequence[bool]) -> float:
    """1 / rank of the first relevant item; 0 if none. The mean over questions is MRR."""
    for rank, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def hit_at_k(relevant: Sequence[bool], k: int) -> float:
    """1.0 if any of the top-k items is relevant, else 0.0."""
    _check_k(k)
    return 1.0 if any(relevant[:k]) else 0.0


def recall_at_k(covered: int, total: int) -> float:
    """Fraction of the required evidence items covered by the top-k results."""
    if total <= 0:
        raise ValueError("recall is undefined when there is no required evidence")
    if not 0 <= covered <= total:
        raise ValueError("covered must be between 0 and total")
    return covered / total


def dcg_at_k(grades: Sequence[int], k: int) -> float:
    """Discounted cumulative gain: sum over ranks i of (2^grade_i - 1) / log2(i + 1)."""
    _check_k(k)
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def ndcg_at_k(grades: Sequence[int], ideal_grades: Sequence[int], k: int) -> float:
    """DCG@k divided by the best achievable DCG@k.

    ``ideal_grades`` are the grades of *every* judged item available (any order); the ideal ranking
    is those grades sorted best-first. Returns 0.0 when nothing relevant exists.
    """
    ideal = dcg_at_k(sorted(ideal_grades, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(grades, k) / ideal
