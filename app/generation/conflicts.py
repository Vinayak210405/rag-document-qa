"""Detect retrieved passages that come from different versions of the same logical document.

Documents share a ``doc_family`` (the header's "Document ID"). If the retrieved passages include more than one
file of a family, the answer may mix superseded and current facts, so the prompt gets an explicit note and the
API reports the conflict. Detection is metadata-only: it does not read the passages' meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.retrieval.types import RetrievedChunk


@dataclass(frozen=True)
class VersionConflict:
    family: str
    versions: tuple[tuple[str, str, str], ...]  # (doc_name, version, status_class) sorted by doc_name

    @property
    def current(self) -> tuple[str, ...]:
        return tuple(n for n, _, s in self.versions if s == "current")


def detect_conflicts(passages: Sequence[RetrievedChunk]) -> list[VersionConflict]:
    families: dict[str, dict[str, tuple[str, str]]] = {}
    for p in passages:
        family = p.metadata.get("doc_family")
        if not family:
            continue
        families.setdefault(str(family), {})[p.doc_name] = (
            str(p.metadata.get("doc_version", "?")), str(p.metadata.get("status_class", "unknown")),
        )
    return [
        VersionConflict(f, tuple((n, v, s) for n, (v, s) in sorted(docs.items())))
        for f, docs in sorted(families.items()) if len(docs) > 1
    ]


def conflict_note(conflicts: Sequence[VersionConflict]) -> str:
    """Text appended to the prompt when versions conflict ('' if none)."""
    lines = []
    for c in conflicts:
        parts = ", ".join(f"{n} (version {v}, {s})" for n, v, s in c.versions)
        lines.append(f"- Document family {c.family} appears in several versions: {parts}.")
    if not lines:
        return ""
    return (
        "Version warning:\n" + "\n".join(lines) +
        "\nState which version each fact comes from. Prefer the version marked current, and say that older "
        "versions are superseded."
    )
