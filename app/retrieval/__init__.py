"""Retrieval: similarity maths, query handling, dense retrieval (BM25/RRF/reranking arrive in M3-M4)."""

from app.retrieval.query import InvalidQueryError, normalize_query
from app.retrieval.similarity import cosine_similarity, top_k
from app.retrieval.types import RetrievedChunk

__all__ = ["InvalidQueryError", "RetrievedChunk", "cosine_similarity", "normalize_query", "top_k"]
