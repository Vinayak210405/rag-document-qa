"""End-to-end service behaviour: real Chroma on disk, offline hashing embedder, fake LLM."""

import pytest

from app.chunking import SentenceChunker
from app.embeddings import HashingEmbedder
from app.generation import LLMError
from app.indexing import ChromaVectorStore, IndexMismatchError
from app.retrieval import InvalidQueryError
from app.service import EmptyIndexError
from tests.fixtures.fakes import FakeLLM


@pytest.fixture
def loaded(make_service, corpus_files):
    svc = make_service()
    reports = {name: svc.ingest(name, data) for name, data in corpus_files.items()}
    return svc, reports


def test_ingesting_the_corpus_stores_every_chunk(loaded) -> None:
    svc, reports = loaded
    assert len(reports) == 7
    assert svc.store.count() == sum(r.chunks for r in reports.values())
    docs = {d.name: d.chunks for d in svc.list_documents()}
    assert docs == {name: r.chunks for name, r in reports.items()}
    assert all(not r.replaced_existing and r.duration_ms >= 0 for r in reports.values())


def test_reingesting_the_same_file_does_not_duplicate_chunks(loaded, corpus_files) -> None:
    svc, reports = loaded
    before = svc.store.count()
    again = svc.ingest("product_faq.txt", corpus_files["product_faq.txt"])
    assert again.replaced_existing is True
    assert svc.store.count() == before and again.chunks == reports["product_faq.txt"].chunks


def test_updating_a_document_removes_its_stale_chunks(make_service) -> None:
    svc = make_service()
    svc.ingest("policy.txt", b"Policy\n\nThe limit is 100 USD. Approval is required. Receipts are kept.")
    old_ids = set(svc.store.load_snapshot().ids)
    report = svc.ingest("policy.txt", b"Policy\n\nThe limit is 500 USD.")
    new_ids = set(svc.store.load_snapshot().ids)
    assert report.replaced_existing and old_ids.isdisjoint(new_ids)
    texts = " ".join(svc.store.load_snapshot().texts)
    assert "500 USD" in texts and "100 USD" not in texts


def test_index_survives_a_restart(make_service, corpus_files, tmp_path) -> None:
    svc = make_service()
    svc.ingest("org_directory.txt", corpus_files["org_directory.txt"])
    count = svc.store.count()
    reopened = make_service()  # a brand-new store/service over the same directory
    assert reopened.store.count() == count > 0
    assert reopened.retrieve("Who leads the Berlin office?").sources[0].doc_name == "org_directory.txt"


def test_opening_an_index_with_a_different_embedding_model_fails_loudly(make_service, corpus_files, tmp_path) -> None:
    svc = make_service()
    svc.ingest("org_directory.txt", corpus_files["org_directory.txt"])
    with pytest.raises(IndexMismatchError, match="not comparable"):
        ChromaVectorStore(tmp_path / "chroma", "documents", "some-other-model", 384)
    assert HashingEmbedder().model_name in str(
        pytest.raises(IndexMismatchError, ChromaVectorStore, tmp_path / "chroma", "documents", "x", 1024).value)


# ------------------------------------------------------------------ retrieval
def test_exact_identifier_query_finds_its_document(loaded) -> None:
    svc, _ = loaded
    top = svc.retrieve("What does error code ERR-5043 mean?").sources[0]
    assert top.doc_name == "it_security_handbook.txt" and "ERR-5043" in top.text


def test_a_keyword_question_retrieves_the_right_document_in_the_top_results(loaded) -> None:
    svc, _ = loaded
    sources = svc.retrieve("What is the daily meal allowance for international travel?", top_k=3).sources
    assert "expense_policy.txt" in [s.doc_name for s in sources]


def test_results_are_ranked_numbered_and_respect_top_k(loaded) -> None:
    svc, _ = loaded
    result = svc.retrieve("annual leave days carry over", top_k=5)
    scores = [s.score for s in result.sources]
    assert len(result.sources) == 5 and scores == sorted(scores, reverse=True)
    assert [s.rank for s in result.sources] == [1, 2, 3, 4, 5]
    assert [s.source_number for s in result.sources] == [1, 2, 3, 4, 5]
    assert result.timings.generation_ms is None and result.embedding_model == "hashing-1024"


def test_retrieval_preserves_provenance_metadata(make_service, corpus_files) -> None:
    # Small chunks so each section gets its own chunk. (A chunk spanning several sections records
    # only its FIRST heading: a documented limitation of single-heading metadata.)
    svc = make_service(chunker=SentenceChunker(300, 60))
    svc.ingest("it_security_handbook.txt", corpus_files["it_security_handbook.txt"])
    hit = svc.retrieve("hardware security key SMS second factor", top_k=1).sources[0]
    assert hit.doc_name == "it_security_handbook.txt" and hit.heading == "2. Multi-factor authentication"
    assert "hardware security key" in hit.text
    assert hit.page is None and len(hit.chunk_id) == 16


def test_invalid_requests_are_rejected_not_answered(make_service, loaded) -> None:
    svc, _ = loaded
    for bad in ("", "   "):
        with pytest.raises(InvalidQueryError):
            svc.retrieve(bad)
    for k in (0, 21, -3):
        with pytest.raises(InvalidQueryError, match="top_k"):
            svc.retrieve("leave", top_k=k)
    with pytest.raises(EmptyIndexError):
        make_service(subdir="empty").retrieve("anything")


# ------------------------------------------------------------------ answering
def test_answer_validates_citations_against_the_retrieved_sources(make_service, corpus_files) -> None:
    llm = FakeLLM("The code means encryption is off [1]. Also see [99].")
    svc = make_service(llm=llm)
    svc.ingest("it_security_handbook.txt", corpus_files["it_security_handbook.txt"])
    result = svc.answer("What does error code ERR-5043 mean?", top_k=3)
    assert result.valid_citations == [1] and result.invalid_citations == [99]
    assert [s.cited for s in result.sources] == [True, False, False][: len(result.sources)]
    assert result.usage.prompt_tokens == 11 and result.llm_model == "fake-llm"
    assert result.timings.generation_ms is not None and result.timings.total_ms >= result.timings.retrieval_ms


def test_the_prompt_sent_to_the_llm_contains_the_retrieved_text(make_service, corpus_files) -> None:
    llm = FakeLLM()
    svc = make_service(llm=llm)
    svc.ingest("it_security_handbook.txt", corpus_files["it_security_handbook.txt"])
    svc.answer("What does error code ERR-5043 mean?", top_k=2)
    (messages,) = llm.calls
    assert "ERR-5043 indicates that encryption has not been enabled" in messages[1]["content"]
    assert messages[1]["content"].rstrip().endswith("Question: What does error code ERR-5043 mean?")


def test_abstention_phrase_is_flagged(make_service, corpus_files) -> None:
    svc = make_service(llm=FakeLLM("I cannot answer this from the provided documents."))
    svc.ingest("product_faq.txt", corpus_files["product_faq.txt"])
    result = svc.answer("How much does the X200 cost?")
    assert result.abstained and result.valid_citations == []


def test_llm_failure_propagates_and_is_never_replaced_by_a_made_up_answer(make_service, corpus_files) -> None:
    svc = make_service(llm=FakeLLM(error=LLMError("endpoint down")))
    svc.ingest("product_faq.txt", corpus_files["product_faq.txt"])
    with pytest.raises(LLMError, match="endpoint down"):
        svc.answer("What is the warranty?")


def test_no_llm_call_is_made_when_the_index_is_empty(make_service) -> None:
    llm = FakeLLM()
    with pytest.raises(EmptyIndexError):
        make_service(llm=llm).answer("anything")
    assert llm.calls == []


# ------------------------------------------------------------------ storage failures
def test_a_storage_failure_becomes_an_actionable_message(monkeypatch, tmp_path) -> None:
    import chromadb
    from chromadb.errors import InternalError

    from app.indexing import StorageError

    def boom(*args, **kwargs):
        raise InternalError("error returned from database: (code: 8) attempt to write a readonly database")

    monkeypatch.setattr(chromadb, "PersistentClient", boom)
    with pytest.raises(StorageError) as info:
        ChromaVectorStore(tmp_path / "chroma", "documents", "hashing-1024", 1024)
    message = str(info.value)
    assert str(tmp_path / "chroma") in message and "RAG_STORAGE_DIR" in message and "readonly database" in message
    assert isinstance(info.value.__cause__, InternalError)  # the original error is preserved, not hidden


def test_in_memory_stores_with_different_collection_names_are_isolated() -> None:
    import numpy as np

    from app.chunking import Chunk

    def store(name: str) -> ChromaVectorStore:
        return ChromaVectorStore(None, name, "m", 2)

    chunk = Chunk("c1", "d.txt", "d.txt", 0, "hello", None, None)
    a, b = store("mem-a"), store("mem-b")
    a.replace_document("d.txt", [chunk], np.array([[1.0, 0.0]], dtype=np.float32))
    assert a.count() == 1 and b.count() == 0
    assert store("mem-a").count() == 1  # documented: same name in the same process SHARES state
