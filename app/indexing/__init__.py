"""Vector store (ChromaDB). Lexical index and lifecycle management arrive in M3 and M6."""

from app.indexing.chroma_store import (
    ChromaVectorStore, DocumentInfo, IndexMismatchError, IndexSnapshot, StorageError,
)

__all__ = ["ChromaVectorStore", "DocumentInfo", "IndexMismatchError", "IndexSnapshot", "StorageError"]
