import numpy as np
import pytest

from app.retrieval.similarity import cosine_similarity, top_k


def test_cosine_known_values() -> None:
    q = np.array([1.0, 0.0])
    m = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]])
    assert cosine_similarity(q, m) == pytest.approx([1.0, 0.0, -1.0, 1 / np.sqrt(2)])


def test_cosine_ignores_vector_length() -> None:
    rng = np.random.default_rng(0)
    q, m = rng.normal(size=8), rng.normal(size=(5, 8))
    assert cosine_similarity(q * 7.5, m * 0.01) == pytest.approx(cosine_similarity(q, m))


def test_zero_vectors_score_zero_not_nan() -> None:
    m = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert cosine_similarity(np.array([1.0, 0.0]), m) == pytest.approx([0.0, 1.0])
    assert cosine_similarity(np.zeros(2), m) == pytest.approx([0.0, 0.0])


def test_cosine_shape_errors_and_empty_matrix() -> None:
    with pytest.raises(ValueError):
        cosine_similarity(np.ones(3), np.ones((2, 4)))
    assert cosine_similarity(np.ones(3), np.zeros((0, 3))).shape == (0,)


def test_top_k_orders_by_score_and_breaks_ties_by_lower_index() -> None:
    assert top_k(np.array([0.1, 0.9, 0.5, 0.9]), 3) == [(1, 0.9), (3, 0.9), (2, 0.5)]


def test_top_k_larger_than_n_returns_everything_sorted() -> None:
    assert [i for i, _ in top_k(np.array([0.2, 0.8, 0.5]), 10)] == [1, 2, 0]


def test_top_k_edge_cases() -> None:
    assert top_k(np.array([]), 3) == []
    with pytest.raises(ValueError):
        top_k(np.array([1.0]), 0)


def test_top_k_matches_a_full_sort_reference() -> None:
    scores = np.random.default_rng(42).normal(size=200)
    expected = sorted(range(200), key=lambda i: (-scores[i], i))[:7]
    assert [i for i, _ in top_k(scores, 7)] == expected
