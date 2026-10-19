import pytest

from app.ingestion.models import IngestionError
from app.ingestion.normalize import normalize_text, safe_filename


def test_nfkc_folds_ligatures_and_nbsp() -> None:
    assert normalize_text("\ufb01nance\u00a0report") == "finance report"


def test_whitespace_and_line_endings_are_normalised() -> None:
    assert normalize_text("a  \t b\r\nc \r d") == "a b\nc\nd"


def test_blank_lines_are_limited_to_one() -> None:
    assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"


def test_control_chars_and_soft_hyphens_are_removed() -> None:
    assert normalize_text("a\x00b\x0cc\u00addef") == "abcdef"


def test_result_is_stripped() -> None:
    assert normalize_text("  \n hello \n ") == "hello"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd.txt", "passwd.txt"),
        ("C:\\Users\\me\\report.pdf", "report.pdf"),
        ("my file (1).docx", "my file (1).docx"),
        ("a<b>:c.txt", "a_b__c.txt"),
    ],
)
def test_safe_filename(raw: str, expected: str) -> None:
    assert safe_filename(raw) == expected


@pytest.mark.parametrize("bad", ["", "..", ".", "...", "/", "a/"])
def test_safe_filename_rejects_unusable_names(bad: str) -> None:
    with pytest.raises(IngestionError):
        safe_filename(bad)
