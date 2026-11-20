"""Metric maths, checked against values worked out by hand."""

import math

import pytest

from app.evaluation.metrics import dcg_at_k, hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank


def test_reciprocal_rank() -> None:
    assert reciprocal_rank([True, False]) == 1.0
    assert reciprocal_rank([False, False, True, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([False, False]) == 0.0
    assert reciprocal_rank([]) == 0.0


def test_hit_at_k_respects_the_cutoff() -> None:
    relevant = [False, False, True]
    assert hit_at_k(relevant, 2) == 0.0 and hit_at_k(relevant, 3) == 1.0 and hit_at_k(relevant, 99) == 1.0
    with pytest.raises(ValueError):
        hit_at_k(relevant, 0)


def test_recall_is_the_covered_fraction_and_rejects_nonsense() -> None:
    assert recall_at_k(1, 2) == 0.5 and recall_at_k(2, 2) == 1.0 and recall_at_k(0, 3) == 0.0
    for covered, total in [(0, 0), (3, 2), (-1, 2)]:
        with pytest.raises(ValueError):
            recall_at_k(covered, total)


def test_dcg_matches_a_hand_calculation() -> None:
    # grades [2, 0, 1]: (2^2-1)/log2(2) + 0 + (2^1-1)/log2(4) = 3 + 0 + 0.5
    assert dcg_at_k([2, 0, 1], 3) == pytest.approx(3.5)
    assert dcg_at_k([2, 0, 1], 1) == pytest.approx(3.0)  # cut-off truncates


def test_ndcg_matches_the_worked_example() -> None:
    ideal_dcg = 3 + 1 / math.log2(3)  # ideal order [2, 1, 0]
    assert ndcg_at_k([2, 0, 1], [2, 1, 0], 3) == pytest.approx(3.5 / ideal_dcg)
    assert ndcg_at_k([2, 0, 1], [2, 1, 0], 3) == pytest.approx(0.9639, abs=1e-4)


def test_ndcg_is_one_for_a_perfect_ranking_and_zero_when_nothing_is_found() -> None:
    assert ndcg_at_k([2, 1, 0], [0, 1, 2, 0], 3) == pytest.approx(1.0)  # ideal order is derived by sorting
    assert ndcg_at_k([0, 0, 0], [2, 1], 3) == 0.0
    assert ndcg_at_k([0, 0], [0, 0], 2) == 0.0  # no relevant item exists at all


def test_ndcg_rewards_putting_the_best_chunk_first() -> None:
    ideal = [2, 1, 0, 0]
    assert ndcg_at_k([2, 1, 0], ideal, 3) > ndcg_at_k([1, 2, 0], ideal, 3) > ndcg_at_k([0, 1, 2], ideal, 3)


def test_ndcg_is_never_above_one_when_ranking_a_subset_of_the_judged_items() -> None:
    ideal = [2, 2, 1, 0, 0]
    for ranking in ([2, 2, 1], [1, 2, 2], [0, 2, 1], [2, 0, 0]):
        assert 0.0 <= ndcg_at_k(ranking, ideal, 3) <= 1.0 + 1e-12
