"""Build the configured embedder."""

from __future__ import annotations

from app.config import Settings
from app.embeddings.base import Embedder
from app.embeddings.hashing import HashingEmbedder
from app.embeddings.sentence_transformers_embedder import SentenceTransformerEmbedder


def create_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(
        model_name=settings.embedding_model, query_prefix=settings.embedding_query_prefix
    )
