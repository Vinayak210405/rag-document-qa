"""Cosine similarity and top-k ranking, implemented from first principles with NumPy.

    cos(q, d) = (q . d) / (||q|| * ||d||)

We compute the full formula (not just a dot product) so the result is correct even if an
embedder forgets to normalise. A zero vector has no direction, so its similarity is defined as 0.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query (d,) and every row of ``matrix`` (n, d) -> (n,)."""
    query = np.asarray(query, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if query.ndim != 1 or matrix.ndim != 2 or matrix.shape[1] != query.shape[0]:
        raise ValueError(f"shape mismatch: query {query.shape} vs matrix {matrix.shape}")
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    denom = np.linalg.norm(query) * np.linalg.norm(matrix, axis=1)
    dots = matrix @ query
    return np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)


def top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Indices and scores of the ``k`` highest scores, best first.

    Ties are broken by lower index, so results are deterministic. ``k`` larger than the number
    of scores simply returns everything.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    candidates = np.argpartition(-scores, k - 1)[:k] if k < scores.size else np.arange(scores.size)
    order = np.lexsort((candidates, -scores[candidates]))  # primary: score desc; secondary: index asc
    return [(int(candidates[i]), float(scores[candidates[i]])) for i in order]
