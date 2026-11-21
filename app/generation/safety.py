"""Prompt-injection safeguards for retrieved (untrusted) text.

Layers, none of which is a guarantee:
1. ``sanitize``: remove control / zero-width / bidi-override characters that can hide or reorder text.
2. ``scan_for_injection``: flag passages containing instruction-like phrases aimed at the model. Flagged passages are marked
   in the prompt (``flagged="possible-injection"``) and reported in the API; optionally dropped (``RAG_DROP_FLAGGED_PASSAGES``).
3. Prompt design (generation/prompt.py): sources are delimited, angle brackets escaped, and the system prompt says sources are
   data, not instructions.
These heuristics are bypassable (paraphrase, other languages, encodings). Treat them as defence-in-depth; the real mitigations
are least privilege (the model has no tools and cannot act) and showing sources so a person can notice manipulation.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Sequence

from app.retrieval.types import RetrievedChunk

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(rx, re.IGNORECASE))
    for name, rx in (
        ("override-instructions", r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|any|the)\b[^.\n]{0,30}\b(instructions?|rules?|prompts?|guidelines?)\b"),
        ("reveal-prompt", r"\b(reveal|print|show|repeat|output|leak)\b[^.\n]{0,40}\b(system|hidden|initial)\s+(prompt|message|instructions?)\b"),
        ("role-reassignment", r"\byou\s+are\s+now\b|\bact\s+as\s+(an?\s+)?(unrestricted|jailbroken|dan)\b|\bnew\s+instructions?\s*:"),
        ("addressing-the-assistant", r"\b(assistant|ai|chatbot|language model|llm)\s*[,:]\s*(you must|you should|please|always|never|do not)\b"),
        ("answer-hijack", r"\b(always|only)\s+(answer|respond|reply)\s+(with|that)\b|\bdo not (cite|mention|tell)\b"),
        ("fake-delimiters", r"</?\s*(source|system|assistant|user)\s*>|\[/?INST\]|<\|im_(start|end)\|>"),
    )
)
_STRIP = re.compile("[​-‏‪-‮⁠-⁤⁦-⁩﻿]")


def sanitize(text: str) -> str:
    """Drop invisible/bidi characters and non-printing control characters (keeps newlines and tabs)."""
    text = _STRIP.sub("", text)
    return "".join(c for c in text if c in "\n\t" or not unicodedata.category(c).startswith("C"))


def scan_for_injection(text: str) -> list[str]:
    """Names of the heuristics that matched (empty = nothing suspicious found; NOT proof of safety)."""
    return [name for name, rx in _PATTERNS if rx.search(text)]


@dataclass(frozen=True)
class SafetyReport:
    passages: list[RetrievedChunk]
    flagged: dict[str, list[str]]  # chunk_id -> matched heuristics
    dropped: list[str]  # chunk_ids removed because they were flagged and dropping is enabled


def apply_safeguards(passages: Sequence[RetrievedChunk], *, drop_flagged: bool = False) -> SafetyReport:
    clean: list[RetrievedChunk] = []
    flagged: dict[str, list[str]] = {}
    dropped: list[str] = []
    for p in passages:
        text = sanitize(p.text)
        hits = scan_for_injection(text)
        if hits:
            flagged[p.chunk_id] = hits
            if drop_flagged:
                dropped.append(p.chunk_id)
                continue
            p = replace(p, metadata={**p.metadata, "flagged": "possible-injection"})
        clean.append(replace(p, text=text))
    return SafetyReport(clean, flagged, dropped)
