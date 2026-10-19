"""Embedding providers behind one interface."""

from app.embeddings.base import Embedder, EmbeddingBackendError, l2_normalize
from app.embeddings.factory import create_embedder
from app.embeddings.hashing import HashingEmbedder

__all__ = ["Embedder", "EmbeddingBackendError", "HashingEmbedder", "create_embedder", "l2_normalize"]
