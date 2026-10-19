import pytest

from app.chunking import split_sentences
from app.ingestion.structure import looks_like_heading, text_to_blocks


def test_basic_split() -> None:
    assert split_sentences("One is first. Two is second! Is three third? Yes.") == [
        "One is first.", "Two is second!", "Is three third?", "Yes.",
    ]


def test_decimals_and_versions_do_not_split() -> None:
    assert split_sentences("Version 2.0 is current. See section 3.1 for details.") == [
        "Version 2.0 is current.", "See section 3.1 for details.",
    ]


def test_abbreviations_and_initials_do_not_split() -> None:
    assert split_sentences("Use tools, e.g. Git. Then commit.") == ["Use tools, e.g. Git.", "Then commit."]
    assert split_sentences("J. Smith wrote it. Good.") == ["J. Smith wrote it.", "Good."]


def test_empty_and_unpunctuated_input() -> None:
    assert split_sentences("") == []
    assert split_sentences("no terminal punctuation here") == ["no terminal punctuation here"]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("1. Annual leave", True),
        ("2.1 Carry-over rules", True),
        ("SECURITY POLICY", True),
        ("Passwords must be rotated every 180 days.", False),
        ("Note:", False),
        ("x" * 120, False),
        ("", False),
        ("Document ID: HR-POL-004", False),  # not the document start, not numbered
    ],
)
def test_looks_like_heading(line: str, expected: bool) -> None:
    assert looks_like_heading(line) is expected


def test_first_line_of_document_counts_as_title() -> None:
    assert looks_like_heading("Leave Policy", is_document_start=True) is True
    assert looks_like_heading("Leave Policy", is_document_start=False) is False


def test_text_to_blocks_splits_heading_and_unwraps_body() -> None:
    text = "Policy Title\nVersion: 2.0\nStatus: Current\n\n1. Leave\nEmployees get leave.\nIt accrues monthly."
    blocks = text_to_blocks(text, page=None, is_document_start=True)
    assert [(b.kind, b.text) for b in blocks] == [
        ("heading", "Policy Title"),
        ("body", "Version: 2.0 Status: Current"),
        ("heading", "1. Leave"),
        ("body", "Employees get leave. It accrues monthly."),
    ]


def test_pages_are_recorded_on_blocks() -> None:
    blocks = text_to_blocks("Some body text.", page=3, is_document_start=False)
    assert [b.page for b in blocks] == [3]
