"""Headless Streamlit run of frontend/app.py with the API client stubbed (no server needed)."""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
APP = str(FRONTEND / "app.py")

SOURCE = {"source_number": 1, "rank": 1, "score": 0.8123, "chunk_id": "abc123def4567890", "doc_id": "it.txt",
          "doc_name": "it.txt", "ordinal": 2, "page": None, "heading": "4. Device encryption",
          "text": "Error code ERR-5043 indicates that encryption has not been enabled.", "cited": True}
ANSWER = {"question": "q", "answer": "Encryption is off [1].", "abstained": False, "valid_citations": [1],
          "invalid_citations": [], "sources": [SOURCE], "timings": {"retrieval_ms": 4.0, "generation_ms": 900.0, "total_ms": 905.0},
          "usage": {"prompt_tokens": 120, "completion_tokens": 9}, "llm_model": "m", "embedding_model": "e"}


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.syspath_prepend(str(FRONTEND))  # same import path `streamlit run frontend/app.py` uses
    import api_client as module

    monkeypatch.setattr(module.ApiClient, "health", lambda self: {
        "status": "ok", "documents": 1, "indexed_chunks": 2, "embedding_model": "e", "default_top_k": 4})
    monkeypatch.setattr(module.ApiClient, "list_documents", lambda self: [{"doc_id": "it.txt", "name": "it.txt", "chunks": 2}])
    return module


def ask(at: AppTest, question: str) -> AppTest:
    at.text_input(key="question").set_value(question)
    next(b for b in at.button if b.label == "Ask").click()
    return at.run()


def test_page_renders_with_index_status(api_client) -> None:
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert any("1** documents" in m.value or "1 documents" in m.value.replace("**", "") for m in at.sidebar.markdown)


def test_answer_shows_citations_metrics_and_inspectable_sources(api_client, monkeypatch) -> None:
    received: list[tuple] = []
    monkeypatch.setattr(api_client.ApiClient, "ask", lambda self, q, k=None: received.append((q, k)) or ANSWER)
    at = ask(AppTest.from_file(APP, default_timeout=15).run(), "What does ERR-5043 mean?")
    assert received == [("What does ERR-5043 mean?", 4)]  # the typed question and the slider default reached the API client
    assert not at.exception
    assert any("Encryption is off [1]." in w.value for w in [*at.markdown, *at.text])
    assert not at.warning and not at.error
    assert any("it.txt" in e.label and "0.812" in e.label and "cited" in e.label for e in at.expander)


def test_invalid_citations_and_missing_citations_are_called_out(api_client, monkeypatch) -> None:
    bad = {**ANSWER, "valid_citations": [], "invalid_citations": [9], "sources": [{**SOURCE, "cited": False}]}
    monkeypatch.setattr(api_client.ApiClient, "ask", lambda self, q, k=None: bad)
    at = ask(AppTest.from_file(APP, default_timeout=15).run(), "q")
    assert any("do not exist" in e.value for e in at.error)

    uncited = {**ANSWER, "valid_citations": [], "sources": [{**SOURCE, "cited": False}]}
    monkeypatch.setattr(api_client.ApiClient, "ask", lambda self, q, k=None: uncited)
    at = ask(AppTest.from_file(APP, default_timeout=15).run(), "q")
    assert any("no citations" in w.value for w in at.warning)


def test_abstention_is_shown_as_a_clear_notice(api_client, monkeypatch) -> None:
    abstain = {**ANSWER, "abstained": True, "valid_citations": [], "answer": "I cannot answer this from the provided documents."}
    monkeypatch.setattr(api_client.ApiClient, "ask", lambda self, q, k=None: abstain)
    at = ask(AppTest.from_file(APP, default_timeout=15).run(), "What is the CEO salary?")
    assert any("do not contain enough information" in w.value for w in at.warning)


def test_api_errors_are_displayed_not_swallowed(api_client, monkeypatch) -> None:
    def boom(self, q, k=None):
        raise api_client.ApiError("no documents have been ingested yet (HTTP 409)")

    monkeypatch.setattr(api_client.ApiClient, "ask", boom)
    at = ask(AppTest.from_file(APP, default_timeout=15).run(), "hello")
    assert any("no documents have been ingested" in e.value for e in at.error)


def test_unreachable_api_shows_how_to_start_it(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(FRONTEND))
    import api_client as module

    def down(self):
        raise module.ApiError("Cannot reach the API at http://127.0.0.1:8000. Start it with: uv run uvicorn app.api.main:app")

    monkeypatch.setattr(module.ApiClient, "health", down)
    monkeypatch.setattr(module.ApiClient, "list_documents", lambda self: (_ for _ in ()).throw(module.ApiError("down")))
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception and any("uvicorn" in e.value for e in at.sidebar.error)
