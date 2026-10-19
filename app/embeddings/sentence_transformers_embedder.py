"""Real semantic embeddings via sentence-transformers (optional extra: ``uv sync --extra embeddings``)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from app.embeddings.base import EmbeddingBackendError, l2_normalize


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model.

    Queries get ``query_prefix`` (bge-style instruction); documents are embedded as-is.
    The model is downloaded from Hugging Face on first use (needs network once).
    """

    def __init__(
        self, model_name: str = "BAAI/bge-small-en-v1.5", query_prefix: str = "", batch_size: int = 32
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingBackendError(
                "sentence-transformers is not installed. Install it with: uv sync --extra embeddings "
                "(or set RAG_EMBEDDING_BACKEND=hashing for the offline, non-semantic embedder)."
            ) from exc
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # network, bad model name, corrupt cache ...
            raise EmbeddingBackendError(
                f"could not load embedding model {model_name!r}: {exc}. Check the model name and your "
                "network connection (the model is downloaded on first use)."
            ) from exc
        # sentence-transformers 6.x renamed this method; support both spellings (we allow >=3,<7).
        dim_fn = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self._dim = int(dim_fn())
        self._name = model_name
        self._prefix = query_prefix
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dim

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return l2_normalize(np.asarray(out, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return self._encode(list(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self._prefix + text])[0]
