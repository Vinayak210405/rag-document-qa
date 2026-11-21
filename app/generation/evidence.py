"""Pre-generation evidence check: do the retrieved passages even mention what the question asks about?

Coverage = fraction of the question's distinct CONTENT tokens (stop-words removed) that appear somewhere in
the retrieved passages. Low coverage means the passages share almost no vocabulary with the question, a
cheap, deterministic signal for "retrieval found nothing relevant", so we can abstain WITHOUT calling the LLM.

Limits (measured, see docs/decisions.md D23): purely lexical, so paraphrases ("login password" vs "password")
lower coverage and can cause false abstentions, while a question that reuses corpus words but asks for a missing
fact (e.g. "How much does the X200 cost?") still passes. It is a safety net, not a verdict on answerability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.retrieval.bm25 import tokenize
from app.retrieval.types import RetrievedChunk

STOPWORDS = frozenset(
    "a an the of to in on at for from by with and or but not is are was were be been being do does did have has had "
    "what which who whom whose when where why how can could should would will may might must shall i me my we our "
    "you your it its this that these those there their they them he she his her as if than then so such any all "
    "per each about into over under up down out".split()
)


@dataclass(frozen=True)
class EvidenceDecision:
    sufficient: bool
    coverage: float
    threshold: float
    missing_terms: tuple[str, ...]


def content_tokens(query: str) -> list[str]:
    seen: list[str] = []
    for t in tokenize(query):
        if t not in STOPWORDS and t not in seen:
            seen.append(t)
    return seen


def assess_evidence(query: str, passages: Sequence[RetrievedChunk], min_coverage: float = 0.0) -> EvidenceDecision:
    """``min_coverage`` 0 disables the gate (always sufficient unless there are no passages)."""
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be in [0, 1]")
    terms = content_tokens(query)
    if not passages:
        return EvidenceDecision(False, 0.0, min_coverage, tuple(terms))
    if not terms:
        return EvidenceDecision(True, 1.0, min_coverage, ())
    vocab: set[str] = set()
    for p in passages:
        vocab.update(tokenize(p.text))
    missing = tuple(t for t in terms if t not in vocab)
    coverage = (len(terms) - len(missing)) / len(terms)
    return EvidenceDecision(coverage >= min_coverage, coverage, min_coverage, missing)
