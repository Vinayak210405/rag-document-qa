"""A deterministic, dependency-free *lexical* embedder (feature hashing / bag of words).

NOT semantic: it only knows which words overlap, so paraphrases will not match. Its purpose is
(1) fully offline, reproducible tests and (2) letting you try the whole app without PyTorch.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

import numpy as np

from app.embeddings.base import l2_normalize

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an the of is are was were to for in on and or what how does do can i my be by with at it "
    "that this which who when where there their you your".split()
)


class HashingEmbedder:
    def __init__(self, dim: int = 1024) -> None:
        if dim < 16:
            raise ValueError("dim must be >= 16")
        self._dim = dim

    @property
    def model_name(self) -> str:
        return f"hashing-{self._dim}"

    @property
    def dimension(self) -> int:
        return self._dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        counts: dict[int, int] = {}
        for token in _TOKEN.findall(text.lower()):
            if token in _STOPWORDS:
                continue
            index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big") % self._dim
            counts[index] = counts.get(index, 0) + 1
        for index, count in counts.items():
            vec[index] = 1.0 + math.log(count)  # sublinear term frequency
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._vector(text)[None, :])[0]
