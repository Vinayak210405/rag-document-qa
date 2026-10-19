import pytest

from app.ingestion import (
    EmptyDocumentError, ExtractionError, UnsupportedFileTypeError, parse_document,
)
from tests.fixtures.builders import make_docx, make_pdf
from tests.fixtures.fakes import FakeOcr


def test_txt_document_has_provenance_and_structure(corpus_files) -> None:
    doc = parse_document("expense_policy.txt", corpus_files["expense_policy.txt"])
    assert (doc.doc_id, doc.file_type, doc.page_count) == ("expense_policy.txt", "txt", None)
    assert len(doc.content_hash) == 64
    assert [b.text for b in doc.blocks if b.kind == "heading"][:2] == [
        "Velmora Systems - Travel and Expense Policy", "1. Meals"]


def test_content_hash_tracks_content() -> None:
    a, b = parse_document("a.txt", b"Hello there."), parse_document("a.txt", b"Hello there!")
    assert a.content_hash != b.content_hash


def test_utf8_bom_is_stripped() -> None:
    doc = parse_document("a.txt", b"\xef\xbb\xbfTitle line\n\nBody text here.")
    assert doc.blocks[0].text == "Title line"


def test_windows_1252_fallback_warns() -> None:
    doc = parse_document("a.txt", "Caf\u00e9 menu\n\nOpen daily.".encode("cp1252"))
    assert "Caf\u00e9" in doc.blocks[0].text and any("Windows-1252" in w for w in doc.warnings)


@pytest.mark.parametrize("name", ["malware.exe", "notes", "data.csv"])
def test_unsupported_types_are_rejected(name: str) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        parse_document(name, b"x")


@pytest.mark.parametrize("content", [b"", b"   \n\n  "])
def test_empty_files_are_rejected(content: bytes) -> None:
    with pytest.raises(EmptyDocumentError):
        parse_document("a.txt", content)


def test_path_traversal_in_the_name_is_neutralised() -> None:
    assert parse_document("../../etc/passwd.txt", b"Title\n\nBody.").doc_id == "passwd.txt"


def test_docx_headings_come_from_styles_and_tables_become_rows() -> None:
    doc = parse_document("p.docx", make_docx("Travel Policy", ("Economy only.",), (("Item", "Limit"), ("Meals", "60 USD"))))
    kinds = [(b.kind, b.text) for b in doc.blocks]
    assert kinds[0] == ("heading", "Travel Policy") and ("body", "Economy only.") in kinds
    assert ("body", "Item | Limit") in kinds and ("body", "Meals | 60 USD") in kinds
    assert doc.page_count is None


def test_pdf_pages_are_tracked() -> None:
    doc = parse_document("r.pdf", make_pdf(["Alpha page text.", "Beta page text."]))
    assert doc.page_count == 2 and doc.file_type == "pdf"
    pages = {b.page for b in doc.blocks}
    assert pages == {1, 2}
    assert any("Alpha" in b.text and b.page == 1 for b in doc.blocks)
    assert any("Beta" in b.text and b.page == 2 for b in doc.blocks)


def test_scanned_pdf_is_reported_not_silently_indexed() -> None:
    with pytest.raises(EmptyDocumentError, match="OCR"):
        parse_document("scan.pdf", make_pdf(["", ""]))


def test_partially_empty_pdf_warns_about_the_missing_pages() -> None:
    doc = parse_document("p.pdf", make_pdf(["Real text on page one.", ""]))
    assert any("no extractable text" in w and "[2]" in w for w in doc.warnings)


def test_ocr_extension_point_recovers_blank_pages() -> None:
    ocr = FakeOcr("Recovered text from the scan.")
    doc = parse_document("scan.pdf", make_pdf([""]), ocr=ocr)
    assert ocr.pages == [1]
    assert doc.blocks and any("OCR" in w for w in doc.warnings)


@pytest.mark.parametrize("name", ["bad.pdf", "bad.docx"])
def test_corrupt_files_raise_extraction_errors(name: str) -> None:
    with pytest.raises(ExtractionError):
        parse_document(name, b"this is not really a document")
