"""Embedding interface. Vectors are float32 and L2-normalised, so cosine == dot product."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class EmbeddingBackendError(RuntimeError):
    """The embedding backend is unavailable or misconfigured (message says how to fix it)."""


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dimension) float32 array."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return a (dimension,) float32 array."""
        ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; all-zero rows stay zero (no NaNs)."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return (matrix / safe).astype(np.float32)
