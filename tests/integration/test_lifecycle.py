"""Document lifecycle: incremental ingest, versions, update, delete, dense+lexical sync, audit."""
import numpy as np
import pytest

from app.chunking import SentenceChunker
from app.retrieval.hybrid import BM25Retriever, HybridRetriever
from app.service import DocumentNotFoundError

DOC_A = b"Refund policy. Customers may request a refund within thirty days of purchase. Contact support for ZX-4471."
DOC_A2 = b"Refund policy. Customers may request a refund within sixty days of purchase. Contact support for QQ-9001."


def hybrid(svc):
    return HybridRetriever(svc.store, svc.embedder)


def test_identical_reupload_is_skipped_without_embedding(make_service):
    svc = make_service()
    first = svc.ingest("a.txt", DOC_A)
    calls = []
    original = svc.embedder.embed_documents
    svc.embedder.embed_documents = lambda t: calls.append(t) or original(t)
    again = svc.ingest("a.txt", DOC_A)
    assert first.status == "new" and first.ingest_version == 1
    assert again.status == "unchanged" and calls == [] and again.chunks == first.chunks


def test_change_creates_new_version_and_leaves_no_stale_chunks(make_service):
    svc = make_service()
    svc.ingest("a.txt", DOC_A)
    v2 = svc.ingest("a.txt", DOC_A2)
    assert v2.status == "updated" and v2.ingest_version == 2 and v2.replaced_existing
    docs = svc.list_documents()
    assert len(docs) == 1 and docs[0].ingest_version == 2 and docs[0].content_hash == v2.content_hash
    assert svc.audit().clean
    texts = " ".join(svc.store.load_snapshot().texts)
    assert "sixty" in texts and "thirty" not in texts


def test_lexical_index_follows_update_and_delete(make_service):
    svc = make_service()
    svc.ingest("a.txt", DOC_A)
    bm25 = BM25Retriever(svc.store)
    assert bm25.retrieve("ZX-4471", 3)[0].doc_name == "a.txt"
    svc.ingest("a.txt", DOC_A2)
    assert bm25.retrieve("ZX-4471", 3) == [] and bm25.retrieve("QQ-9001", 3)  # old identifier gone, new one found
    svc.delete_document("a.txt")
    assert bm25.retrieve("QQ-9001", 3) == [] and hybrid(svc).retrieve("refund", 3) == []
    assert svc.store.count() == 0


def test_delete_unknown_document_raises_and_other_docs_survive(make_service):
    svc = make_service()
    svc.ingest("a.txt", DOC_A)
    svc.ingest("b.txt", b"Vacation policy grants twenty days of leave each year to employees.")
    with pytest.raises(DocumentNotFoundError):
        svc.delete_document("missing.txt")
    assert svc.delete_document("a.txt") > 0
    assert [d.doc_id for d in svc.list_documents()] == ["b.txt"]


def test_settings_change_forces_reindex_of_identical_bytes(make_service):
    svc = make_service()
    svc.ingest("a.txt", DOC_A)
    svc.chunker = SentenceChunker(120, 20)  # different chunking => different stored chunks
    r = svc.ingest("a.txt", DOC_A)
    assert r.status == "updated" and svc.audit().clean


def test_audit_detects_stale_mixed_and_duplicates(make_service):
    svc = make_service()
    long_doc = (DOC_A.decode() + " ") + " ".join(f"Clause {i} states that rule number {i} applies to all orders." for i in range(30))
    svc.ingest("a.txt", long_doc.encode())
    svc.ingest("copy.txt", long_doc.encode())  # same text under another name: duplicate content across documents
    report = svc.audit()
    assert report.clean and report.duplicate_texts  # duplicates are informational, not corruption
    # corrupt one chunk's text behind the store's back -> its ID no longer matches sha256(doc, ordinal, text)
    cid = svc.store.load_snapshot().ids[0]
    emb = svc.store._collection.get(ids=[cid], include=["embeddings"])["embeddings"]
    svc.store._collection.update(ids=[cid], documents=["tampered text"], embeddings=[list(map(float, emb[0]))])
    bad = svc.audit()
    assert bad.stale_chunks == (cid,) and not bad.clean
    # simulate an interrupted update: one chunk of a.txt carries a different content hash
    snapshot = svc.store.load_snapshot()
    other = next(i for i, m in zip(snapshot.ids, snapshot.metadatas) if m["doc_id"] == "a.txt" and i != cid)
    meta = svc.store._collection.get(ids=[other], include=["metadatas"])["metadatas"][0]
    svc.store._collection.update(ids=[other], metadatas=[{**meta, "content_hash": "deadbeef"}])
    assert svc.audit().mixed_versions


def test_replace_writes_new_chunks_before_deleting_old(make_service, monkeypatch):
    """Crash safety: if the delete step fails, the new version is already fully present (never missing content)."""
    svc = make_service()
    svc.ingest("a.txt", DOC_A)
    original_delete = svc.store._collection.delete

    def boom(*a, **k):
        raise RuntimeError("disk error")

    monkeypatch.setattr(svc.store._collection, "delete", boom)
    with pytest.raises(RuntimeError):
        svc.ingest("a.txt", DOC_A2)
    monkeypatch.undo()
    assert "sixty" in " ".join(svc.store.load_snapshot().texts)  # new content present
    assert not svc.audit().clean  # and the leftover old chunks are detectable
    assert original_delete  # (silence linters)
