from tests.fixtures.fakes import FakeLLM


def _ingest_all(svc, corpus_files):
    for name, data in corpus_files.items():
        svc.ingest(name, data)


def test_answered_status_and_conflict_reporting(make_service, corpus_files):
    llm = FakeLLM("Full-time employees receive 24 days [1].")
    svc = make_service(llm)
    _ingest_all(svc, corpus_files)
    r = svc.answer("How many days of annual leave do full-time employees receive?", top_k=4)
    assert r.status == "answered" and r.valid_citations == [1]
    assert any("HR-POL-004" in c for c in r.version_conflicts)  # v1 and v2 both retrieved
    assert "Version warning" in llm.calls[0][1]["content"]


def test_invalid_citation_is_flagged_unverified_and_strict_withholds(make_service, corpus_files):
    svc = make_service(FakeLLM("It is 24 days [9]."))
    _ingest_all(svc, corpus_files)
    r = svc.answer("annual leave days", top_k=2)
    assert r.status == "unverified" and r.invalid_citations == [9] and "24 days" in r.answer
    svc.strict_citations = True
    w = svc.answer("annual leave days", top_k=2)
    assert w.status == "withheld" and "24 days" not in w.answer


def test_model_abstention_status(make_service, corpus_files):
    from app.generation import ABSTAIN_PHRASE
    svc = make_service(FakeLLM(ABSTAIN_PHRASE))
    _ingest_all(svc, corpus_files)
    r = svc.answer("annual leave days", top_k=2)
    assert r.status == "abstained" and r.abstained


def test_evidence_gate_abstains_without_calling_llm(make_service, corpus_files):
    llm = FakeLLM("should not be used [1]")
    svc = make_service(llm)
    svc.min_evidence_coverage = 0.9
    _ingest_all(svc, corpus_files)
    r = svc.answer("Which country won the 2022 FIFA World Cup?", top_k=3)
    assert r.status == "abstained_no_evidence" and r.abstained and llm.calls == []
    assert r.llm_model == "(not called)" and r.evidence_coverage < 0.9
