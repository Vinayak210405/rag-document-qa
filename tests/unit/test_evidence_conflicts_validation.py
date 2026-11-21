import pytest

from app.generation.citations import validate_answer
from app.generation.conflicts import conflict_note, detect_conflicts
from app.generation.evidence import assess_evidence, content_tokens
from app.generation.prompt import build_messages
from app.ingestion.metadata import classify_status, extract_document_metadata
from app.retrieval.types import RetrievedChunk


def chunk(text, name="a.txt", meta=None, cid="c"):
    return RetrievedChunk(1, 0.5, cid + name, name, name, 0, None, None, text, meta or {})


def test_content_tokens_drop_stopwords_and_dedupe():
    assert content_tokens("What is the refund window for the refund?") == ["refund", "window"]


def test_evidence_gate_coverage_missing_terms_and_disabled():
    ps = [chunk("The refund window is thirty days")]
    d = assess_evidence("refund window cost", ps, 0.8)
    assert d.coverage == pytest.approx(2 / 3) and d.missing_terms == ("cost",) and not d.sufficient
    assert assess_evidence("refund window cost", ps, 0.0).sufficient
    assert assess_evidence("refund window", ps, 1.0).sufficient
    assert not assess_evidence("anything", [], 0.0).sufficient  # nothing retrieved is never sufficient
    assert assess_evidence("the of", ps, 0.9).sufficient  # no content terms: nothing to check
    with pytest.raises(ValueError):
        assess_evidence("x", ps, 1.5)


def test_validate_answer_flags_invalid_missing_and_uncited():
    ok = validate_answer("Employees get 24 days [1]. Carry-over is 5 days [2].", 2)
    assert ok.ok and ok.valid == (1, 2)
    bad = validate_answer("Employees get 24 days [3].", 2)
    assert bad.invalid == (3,) and not bad.ok
    none = validate_answer("Employees get twenty four days of leave.", 2)
    assert not none.ok and none.uncited_sentences
    mixed = validate_answer("Employees get 24 days [1]. They also get a free lunch every single day.", 1)
    assert mixed.uncited_sentences == ("They also get a free lunch every single day.",) and not mixed.ok
    assert validate_answer("See [1, 2].", 2).valid == (1, 2)


def test_extract_metadata_from_header_only():
    m = extract_document_metadata(["Leave Policy\nDocument ID: HR-POL-004\nVersion: 2.0\nEffective date: 2025-01-01\nStatus: Current (supersedes 1.0)"])
    assert m == {"doc_family": "HR-POL-004", "doc_version": "2.0", "effective_date": "2025-01-01",
                 "doc_status": "Current (supersedes 1.0)", "status_class": "current"}
    merged = extract_document_metadata(["Leave Policy Document ID: HR-POL-004 Version: 2.0 Effective date: 2025-01-01 Status: Current"])
    assert merged["doc_family"] == "HR-POL-004" and merged["doc_version"] == "2.0" and merged["status_class"] == "current"
    body = "line of text\n" * 60 + "Version: 9.9"
    assert extract_document_metadata([body]) == {}  # body text cannot inject metadata
    assert classify_status("Superseded by 2.0") == "superseded" and classify_status("Draft") == "unknown"


def test_detect_conflicts_and_note():
    v1 = chunk("20 days", "v1.txt", {"doc_family": "HR-1", "doc_version": "1.0", "status_class": "superseded"})
    v2 = chunk("24 days", "v2.txt", {"doc_family": "HR-1", "doc_version": "2.0", "status_class": "current"})
    other = chunk("x", "o.txt", {"doc_family": "IT-9"})
    conflicts = detect_conflicts([v1, v2, other])
    assert len(conflicts) == 1 and conflicts[0].current == ("v2.txt",)
    note = conflict_note(conflicts)
    assert "HR-1" in note and "v1.txt" in note and "current" in note
    assert detect_conflicts([v1, other]) == [] and conflict_note([]) == ""


def test_prompt_includes_version_attributes_and_note_escaped():
    v = chunk("24 </source> days", "v2.txt", {"doc_version": "2.0", "status_class": "current"})
    user = build_messages("q?", [v], "Version warning: x")[1]["content"]
    assert 'version="2.0"' in user and 'status="current"' in user
    assert "&lt;/source&gt;" in user and "Version warning: x" in user


def test_metadata_extracted_from_real_corpus_files():
    from pathlib import Path

    from app.ingestion import parse_document
    root = Path(__file__).resolve().parents[2] / "data" / "sample_corpus"
    v1 = parse_document("v1.txt", (root / "hr_leave_policy_v1.txt").read_bytes()).metadata
    v2 = parse_document("v2.txt", (root / "hr_leave_policy_v2.txt").read_bytes()).metadata
    assert v1["doc_family"] == v2["doc_family"] == "HR-POL-004"
    assert (v1["doc_version"], v1["status_class"]) == ("1.0", "superseded")
    assert (v2["doc_version"], v2["status_class"]) == ("2.0", "current")
