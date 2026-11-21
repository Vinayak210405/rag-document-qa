"""Vector store (ChromaDB). Lexical index and lifecycle management arrive in M3 and M6."""

from app.indexing.chroma_store import (
    AuditReport, ChromaVectorStore, DocumentInfo, DocumentState, IndexMismatchError, IndexSnapshot, StorageError,
)

__all__ = ["AuditReport", "ChromaVectorStore", "DocumentInfo", "DocumentState", "IndexMismatchError", "IndexSnapshot", "StorageError"]
