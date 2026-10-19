"""Vector store (ChromaDB). Lexical index and lifecycle management arrive in M3 and M6."""

from app.indexing.chroma_store import ChromaVectorStore, DocumentInfo, IndexMismatchError, IndexSnapshot

__all__ = ["ChromaVectorStore", "DocumentInfo", "IndexMismatchError", "IndexSnapshot"]
