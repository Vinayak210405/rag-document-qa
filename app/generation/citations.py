"""Basic citation-reference validation (Milestone 1).

Checks only that every cited number refers to a source that was actually supplied. It does NOT
check that the cited source supports the claim; that needs entailment checks (Milestone 5/8).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class CitationReport:
    valid: tuple[int, ...]  # distinct source numbers that exist, in order of first appearance
    invalid: tuple[int, ...]  # distinct numbers that do not match any supplied source

    @property
    def has_citations(self) -> bool:
        return bool(self.valid or self.invalid)


def check_citations(answer: str, n_sources: int) -> CitationReport:
    """Extract ``[n]`` / ``[n, m]`` references and split them into valid and invalid."""
    seen: set[int] = set()
    valid: list[int] = []
    invalid: list[int] = []
    for match in _CITATION.finditer(answer):
        for raw in match.group(1).split(","):
            number = int(raw)
            if number in seen:
                continue
            seen.add(number)
            (valid if 1 <= number <= n_sources else invalid).append(number)
    return CitationReport(tuple(valid), tuple(invalid))


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class AnswerValidation:
    """Structural checks on an answer. Says nothing about whether cited sources truly SUPPORT the claims."""

    valid: tuple[int, ...]
    invalid: tuple[int, ...]
    uncited_sentences: tuple[str, ...]  # sentences with no [n] at all (ignoring the abstain phrase)

    @property
    def ok(self) -> bool:
        return not self.invalid and bool(self.valid) and not self.uncited_sentences


def validate_answer(answer: str, n_sources: int) -> AnswerValidation:
    report = check_citations(answer, n_sources)
    uncited = tuple(
        s.strip() for s in _SENTENCE_END.split(answer.strip())
        if s.strip() and not _CITATION.search(s) and len(s.split()) >= 4
    )
    return AnswerValidation(report.valid, report.invalid, uncited)
