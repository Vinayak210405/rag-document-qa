"""Context assembly: choose which retrieved passages go into the prompt.

Steps, in order: (1) drop near-duplicates (token-Jaccard >= threshold) keeping the higher-ranked one;
(2) enforce a per-document cap for source diversity; (3) stop when the token budget is spent.
Token counts are ESTIMATES (characters / 4, a rough English average), not the LLM's tokenizer.
The best passage is always kept even if it alone exceeds the budget, so the context is never empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.retrieval.bm25 import tokenize
from app.retrieval.types import RetrievedChunk


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class ContextResult:
    passages: list[RetrievedChunk]
    dropped: dict[str, str]  # chunk_id -> reason ("duplicate" | "doc_cap" | "budget")
    estimated_tokens: int


def assemble_context(
    passages: Sequence[RetrievedChunk], *, token_budget: int = 1500, max_per_doc: int | None = None,
    dedupe_threshold: float = 0.85,
) -> ContextResult:
    if token_budget < 1 or (max_per_doc is not None and max_per_doc < 1) or not 0 < dedupe_threshold <= 1:
        raise ValueError("invalid context settings")
    kept: list[RetrievedChunk] = []
    kept_tokens: list[set[str]] = []
    per_doc: dict[str, int] = {}
    dropped: dict[str, str] = {}
    used = 0
    for p in passages:
        toks = set(tokenize(p.text))
        if any(jaccard(toks, k) >= dedupe_threshold for k in kept_tokens):
            dropped[p.chunk_id] = "duplicate"
            continue
        if max_per_doc is not None and per_doc.get(p.doc_id, 0) >= max_per_doc:
            dropped[p.chunk_id] = "doc_cap"
            continue
        cost = estimate_tokens(p.text)
        if kept and used + cost > token_budget:
            dropped[p.chunk_id] = "budget"
            continue
        kept.append(p)
        kept_tokens.append(toks)
        per_doc[p.doc_id] = per_doc.get(p.doc_id, 0) + 1
        used += cost
    return ContextResult([replace_rank(p, i) for i, p in enumerate(kept, start=1)], dropped, used)


def replace_rank(p: RetrievedChunk, rank: int) -> RetrievedChunk:
    from dataclasses import replace
    return replace(p, rank=rank)
