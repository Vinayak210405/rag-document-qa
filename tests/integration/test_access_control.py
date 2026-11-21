"""End-to-end access control: restricted text must never be retrievable, listed, or sent to the LLM."""
import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import BM25Retriever, HybridRetriever
from tests.fixtures.fakes import FakeLLM

ADMIN, HR, IT = "admin-token-0123456789", "hr-token-012345678901", "it-token-012345678901"
SECRET = b"Confidential salary bands. The HR director salary band is code ZETA-9917 for all grades."
PUBLIC = b"Public holiday policy. Employees receive ten public holidays each year."


@pytest.fixture
def secured(make_service):
    tokens = {ADMIN: {"name": "ops", "admin": True}, HR: {"name": "hr", "groups": ["hr"]}, IT: {"name": "it", "groups": ["it"]}}
    llm = FakeLLM("Holidays are ten [1].")
    svc = make_service(llm)
    client = TestClient(create_app(service=svc, settings=Settings(_env_file=None, auth_tokens=json.dumps(tokens))))
    h = {"Authorization": f"Bearer {ADMIN}"}
    assert client.post("/documents", files={"file": ("salary.txt", SECRET)}, data={"groups": "hr"}, headers=h).status_code == 200
    assert client.post("/documents", files={"file": ("holidays.txt", PUBLIC)}, headers=h).status_code == 200
    return client, svc, llm


def bearer(t):
    return {"Authorization": f"Bearer {t}"}


def test_unauthenticated_and_bad_tokens_are_401_and_only_admin_can_write(secured):
    client, _, _ = secured
    assert client.post("/ask", json={"question": "holidays"}).status_code == 401
    assert client.post("/ask", json={"question": "holidays"}, headers=bearer("nope")).status_code == 401
    assert client.get("/documents").status_code == 401
    for t in (HR, IT):
        assert client.post("/documents", files={"file": ("x.txt", PUBLIC)}, headers=bearer(t)).status_code == 403
        assert client.delete("/documents/holidays.txt", headers=bearer(t)).status_code == 403
        assert client.get("/audit", headers=bearer(t)).status_code == 403
    assert client.get("/health").status_code == 200  # liveness stays open and reveals only counts


def test_restricted_document_is_hidden_from_other_groups_everywhere(secured):
    client, svc, llm = secured
    names = lambda t: {d["name"] for d in client.get("/documents", headers=bearer(t)).json()}
    assert names(ADMIN) == {"salary.txt", "holidays.txt"} and names(HR) == {"salary.txt", "holidays.txt"}
    assert names(IT) == {"holidays.txt"}
    r = client.post("/retrieve", json={"question": "ZETA-9917 salary band", "top_k": 5}, headers=bearer(IT)).json()
    assert all(s["doc_name"] == "holidays.txt" for s in r["sources"])
    r = client.post("/retrieve", json={"question": "ZETA-9917 salary band", "top_k": 5}, headers=bearer(HR)).json()
    assert r["sources"][0]["doc_name"] == "salary.txt"
    client.post("/ask", json={"question": "What is the salary band ZETA-9917?"}, headers=bearer(IT))
    sent = json.dumps(llm.calls)  # the question itself contains the identifier; the document body must not appear
    assert "Confidential salary bands" not in sent and "HR director" not in sent and "salary.txt" not in sent


@pytest.mark.parametrize("make", [
    lambda s: DenseRetriever(s.store, s.embedder),
    lambda s: HybridRetriever(s.store, s.embedder),
    lambda s: BM25Retriever(s.store),
])
def test_every_retriever_filters_before_ranking(secured, make):
    _, svc, _ = secured
    r = make(svc)
    assert "salary.txt" not in {c.doc_name for c in r.retrieve("ZETA-9917 salary", 5, groups={"it"})}
    assert "salary.txt" not in {c.doc_name for c in r.retrieve("ZETA-9917 salary", 5, groups=set())}
    assert "salary.txt" in {c.doc_name for c in r.retrieve("ZETA-9917 salary", 5, groups={"hr"})}
    assert "salary.txt" in {c.doc_name for c in r.retrieve("ZETA-9917 salary", 5)}  # no ACL context: everything


def test_user_with_no_visible_documents_gets_a_refusal_not_a_leak(make_service):
    svc = make_service(FakeLLM("should not be called [1]"))
    svc.ingest("salary.txt", SECRET, ["hr"])
    r = svc.answer("What is the salary band ZETA-9917?", 3, groups={"it"})
    assert r.status == "abstained_no_evidence" and r.sources == [] and svc.llm.calls == []


def test_changing_groups_reindexes_and_updates_visibility(make_service):
    svc = make_service()
    assert svc.ingest("salary.txt", SECRET, ["hr"]).status == "new"
    assert svc.ingest("salary.txt", SECRET, ["hr"]).status == "unchanged"
    assert svc.ingest("salary.txt", SECRET, ["it"]).status == "updated"
    assert svc.retriever.retrieve("salary", 3, groups={"hr"}) == []
    assert svc.retriever.retrieve("salary", 3, groups={"it"})


def test_unexpected_errors_return_a_generic_500_without_internals(make_service):
    svc = make_service()
    svc.ingest("holidays.txt", PUBLIC)

    def boom(*a, **k):
        raise RuntimeError("secret path /srv/key=hunter2")

    svc.answer = boom
    client = TestClient(create_app(service=svc, settings=Settings(_env_file=None)), raise_server_exceptions=False)
    r = client.post("/ask", json={"question": "holidays"})
    assert r.status_code == 500 and "hunter2" not in r.text and "/srv" not in r.text
    assert r.json()["error"] == "InternalError"
