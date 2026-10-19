"""HTTP behaviour via FastAPI's TestClient, over a real Chroma store and the offline embedder."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from app.generation import LLMError
from tests.fixtures.builders import make_pdf
from tests.fixtures.fakes import FakeLLM


@pytest.fixture
def api(make_service):
    def _build(llm: FakeLLM | None = None, max_upload: int = 1_000_000) -> TestClient:
        settings = Settings(_env_file=None, max_upload_bytes=max_upload)
        return TestClient(create_app(service=make_service(llm=llm), settings=settings))

    return _build


def upload(client: TestClient, name: str, data: bytes):
    return client.post("/documents", files={"file": (name, data)})


def test_health_on_an_empty_index(api) -> None:
    body = api().get("/health").json()
    assert body["status"] == "ok" and body["indexed_chunks"] == 0 and body["embedding_model"] == "hashing-1024"


def test_upload_list_retrieve_and_ask_round_trip(api, corpus_files) -> None:
    client = api(FakeLLM("It means encryption is off [1]."))
    r = upload(client, "it_security_handbook.txt", corpus_files["it_security_handbook.txt"])
    assert r.status_code == 200 and r.json()["chunks"] > 0 and r.json()["replaced_existing"] is False
    assert client.get("/documents").json() == [
        {"doc_id": "it_security_handbook.txt", "name": "it_security_handbook.txt", "chunks": r.json()["chunks"]}]
    assert client.get("/health").json()["documents"] == 1

    hits = client.post("/retrieve", json={"question": "ERR-5043", "top_k": 2}).json()
    assert len(hits["sources"]) == 2 and hits["sources"][0]["doc_name"] == "it_security_handbook.txt"

    answer = client.post("/ask", json={"question": "What does ERR-5043 mean?"}).json()
    assert answer["answer"] == "It means encryption is off [1]."
    assert answer["valid_citations"] == [1] and answer["invalid_citations"] == []
    assert answer["sources"][0]["cited"] is True and answer["timings"]["generation_ms"] is not None


def test_pdf_upload_reports_pages(api) -> None:
    r = upload(api(), "r.pdf", make_pdf(["The vault code is 4821.", "Rotate it quarterly."]))
    assert r.status_code == 200 and r.json()["pages"] == 2 and r.json()["file_type"] == "pdf"


def test_reuploading_reports_replacement(api, corpus_files) -> None:
    client = api()
    upload(client, "product_faq.txt", corpus_files["product_faq.txt"])
    assert upload(client, "product_faq.txt", corpus_files["product_faq.txt"]).json()["replaced_existing"] is True


def test_asking_before_ingesting_is_a_409_with_a_clear_message(api) -> None:
    r = api().post("/ask", json={"question": "anything"})
    assert r.status_code == 409 and "no documents" in r.json()["detail"]


@pytest.mark.parametrize(("payload", "status"), [
    ({"question": ""}, 422), ({"question": "   "}, 422), ({"question": "ok", "top_k": 0}, 422),
    ({"question": "ok", "top_k": 99}, 422), ({}, 422), ({"question": "x" * 1001}, 422),
])
def test_invalid_questions_are_422(api, corpus_files, payload, status) -> None:
    client = api()
    upload(client, "product_faq.txt", corpus_files["product_faq.txt"])
    assert client.post("/ask", json=payload).status_code == status


@pytest.mark.parametrize(("name", "data", "status"), [
    ("malware.exe", b"MZ", 415),
    ("empty.txt", b"", 422),
    ("blank.pdf", make_pdf([""]), 422),
    ("corrupt.pdf", b"nope", 422),
    ("corrupt.docx", b"nope", 422),
])
def test_bad_uploads_get_precise_errors(api, name, data, status) -> None:
    r = upload(api(), name, data)
    assert r.status_code == status and r.json()["detail"]


def test_oversized_upload_is_413(api) -> None:
    assert upload(api(max_upload=100), "big.txt", b"x" * 101).status_code == 413
    assert upload(api(max_upload=100), "ok.txt", b"Title\n\nsmall.").status_code == 200


def test_missing_file_field_is_422(api) -> None:
    assert api().post("/documents").status_code == 422


def test_llm_outage_is_a_502_not_a_fabricated_answer(api, corpus_files) -> None:
    client = api(FakeLLM(error=LLMError("endpoint down")))
    upload(client, "product_faq.txt", corpus_files["product_faq.txt"])
    r = client.post("/ask", json={"question": "What is the warranty?"})
    assert r.status_code == 502 and r.json()["detail"] == "endpoint down" and "answer" not in r.json()
    assert client.post("/retrieve", json={"question": "warranty"}).status_code == 200  # retrieval still works
