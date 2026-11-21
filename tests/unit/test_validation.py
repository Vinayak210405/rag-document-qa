import io
import zipfile

import pytest

from app.ingestion import InvalidFileError, parse_document
from app.ingestion.models import ExtractionError


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, d in files.items():
            z.writestr(n, d)
    return buf.getvalue()


def test_fake_pdf_docx_and_binary_txt_rejected_before_parsing():
    with pytest.raises(InvalidFileError, match="PDF"):
        parse_document("x.pdf", b"just text pretending to be a pdf")
    with pytest.raises(InvalidFileError, match="ZIP"):
        parse_document("x.docx", b"not a zip")
    with pytest.raises(InvalidFileError, match="Word"):
        parse_document("x.docx", zip_bytes({"hello.txt": b"hi"}))
    with pytest.raises(InvalidFileError, match="binary"):
        parse_document("x.txt", b"abc\x00\x01\x02def")


def test_zip_bomb_and_entry_limits():
    bomb = zip_bytes({"word/document.xml": b"0" * (210 * 1024 * 1024)})
    with pytest.raises(InvalidFileError, match="zip bomb|expands"):
        parse_document("b.docx", bomb)
    many = zip_bytes({"word/document.xml": b"<x/>", **{f"f{i}.xml": b"x" for i in range(2100)}})
    with pytest.raises(InvalidFileError, match="too many"):
        parse_document("m.docx", many)


def test_valid_files_still_parse_and_max_pages_enforced():
    from tests.fixtures.builders import make_pdf
    doc = parse_document("ok.txt", b"Hello world. This is fine.")
    assert doc.file_type == "txt"
    pdf = make_pdf(["one page of text here."] * 3)
    assert parse_document("p.pdf", pdf).page_count == 3
    with pytest.raises(ExtractionError, match="limit"):
        parse_document("p.pdf", pdf, max_pages=2)
