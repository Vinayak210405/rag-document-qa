"""Heading detection and paragraph handling shared by the TXT and PDF parsers.

Heuristics are deliberately simple and documented; they are NOT a layout analyser.
DOCX uses real paragraph styles instead (see parsers.py).
"""

from __future__ import annotations

import re

from app.ingestion.models import Block

_NUMBERED = re.compile(r"^\d+(\.\d+)*[.)]?\s+\S")
_MAX_HEADING_CHARS = 80


def looks_like_heading(line: str, *, is_document_start: bool = False) -> bool:
    """Heuristic: short, no sentence punctuation, and numbered / ALL-CAPS / the document title."""
    line = line.strip()
    if not line or len(line) > _MAX_HEADING_CHARS or line[-1] in ".!?;,:":
        return False
    if _NUMBERED.match(line):
        return True
    letters = [c for c in line if c.isalpha()]
    if len(letters) >= 3 and all(c.isupper() for c in letters):
        return True
    return is_document_start


def text_to_blocks(text: str, page: int | None, *, is_document_start: bool) -> list[Block]:
    """Split normalized text into heading/body blocks.

    Paragraphs are separated by blank lines. If a paragraph's first line looks like a
    heading it becomes a heading block; the remaining lines are unwrapped (joined by
    spaces) into one body block.
    """
    blocks: list[Block] = []
    for i, paragraph in enumerate(re.split(r"\n\s*\n", text)):
        lines = [ln.strip() for ln in paragraph.split("\n") if ln.strip()]
        if not lines:
            continue
        first_ever = is_document_start and i == 0 and not blocks
        if looks_like_heading(lines[0], is_document_start=first_ever):
            blocks.append(Block(lines[0], "heading", page))
            lines = lines[1:]
        if lines:
            blocks.append(Block(" ".join(lines), "body", page))
    return blocks
