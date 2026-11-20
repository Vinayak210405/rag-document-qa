import uuid

import numpy as np

from app.chunking import SentenceChunker
from app.embeddings.hashing import HashingEmbedder
from app.indexing import ChromaVectorStore
from app.ingestion import parse_document
from app.retrieval.hybrid import BM25Retriever, HybridRetriever


def _store(docs: dict[str, str]):
    emb = HashingEmbedder()
    store = ChromaVectorStore(None, f"t-{uuid.uuid4().hex}", emb.model_name, emb.dimension)
    chunker = SentenceChunker(200, 20)
    for name, text in docs.items():
        chunks = chunker.chunk(parse_document(name, text.encode()))
        store.replace_document(chunks[0].doc_id, chunks, np.asarray(emb.embed_documents([c.text for c in chunks])))
    return store, emb


DOCS = {
    "a.txt": "The refund window is thirty days. Contact support for refunds.",
    "b.txt": "Ticket ZX-4471 was escalated to tier two. Nothing else happened.",
    "c.txt": "Employees receive twenty vacation days each year.",
}


def test_bm25_finds_exact_identifier_and_skips_non_matches():
    store, _ = _store(DOCS)
    hits = BM25Retriever(store).retrieve("ZX-4471", 3)
    assert [h.doc_name for h in hits] == ["b.txt"]
    assert BM25Retriever(store).retrieve("zebra", 3) == []


def test_hybrid_returns_ranked_unique_chunks_with_rrf_scores():
    store, emb = _store(DOCS)
    hits = HybridRetriever(store, emb, candidates=5).retrieve("ZX-4471 escalation", 3)
    assert hits[0].doc_name == "b.txt"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert len({h.chunk_id for h in hits}) == len(hits)
    assert all(0 < h.score <= 2 / 61 for h in hits)


def test_hybrid_sees_new_documents_after_ingest_and_empty_index():
    store, emb = _store({})
    r = HybridRetriever(store, emb)
    assert r.retrieve("anything", 3) == []
    chunks = SentenceChunker(200, 20).chunk(parse_document("n.txt", b"Quarterly report ZQ-1."))
    store.replace_document(chunks[0].doc_id, chunks, np.asarray(emb.embed_documents([c.text for c in chunks])))
    assert r.retrieve("ZQ-1", 3)[0].doc_name == "n.txt"
