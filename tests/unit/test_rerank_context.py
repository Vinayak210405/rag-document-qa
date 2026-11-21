import pytest

from app.retrieval.context import assemble_context, estimate_tokens
from app.retrieval.rerank import LexicalOverlapReranker, RerankedRetriever, RerankerError
from app.retrieval.types import RetrievedChunk


def chunk(cid, text, doc="d1", rank=1):
    return RetrievedChunk(rank, 0.5, cid, doc, doc + ".txt", 0, None, None, text)


class FakeBase:
    def __init__(self, items):
        self.items, self.asked = items, None

    def retrieve(self, q, k, groups=None):
        self.asked = k
        return self.items[:k]


class FixedReranker:
    name = "fixed"

    def __init__(self, scores):
        self.scores = scores

    def score(self, q, passages):
        return self.scores


def test_reranker_reorders_and_requests_candidate_pool():
    base = FakeBase([chunk("a", "x"), chunk("b", "y"), chunk("c", "z")])
    out = RerankedRetriever(base, FixedReranker([0.1, 0.9, 0.5]), candidates=3).retrieve("q", 2)
    assert base.asked == 3
    assert [c.chunk_id for c in out] == ["b", "c"] and [c.rank for c in out] == [1, 2]
    assert out[0].score == 0.9


def test_reranker_ties_keep_first_stage_order_and_empty_and_mismatch():
    base = FakeBase([chunk("a", "x"), chunk("b", "y")])
    assert [c.chunk_id for c in RerankedRetriever(base, FixedReranker([1, 1]), 2).retrieve("q", 2)] == ["a", "b"]
    assert RerankedRetriever(FakeBase([]), FixedReranker([]), 2).retrieve("q", 2) == []
    with pytest.raises(RerankerError):
        RerankedRetriever(base, FixedReranker([1.0]), 2).retrieve("q", 2)
    with pytest.raises(ValueError):
        RerankedRetriever(base, FixedReranker([]), 0)


def test_lexical_standin_scores_coverage():
    s = LexicalOverlapReranker().score("refund window days", ["refund window is 30 days", "vacation"])
    assert s == [1.0, 0.0]
    assert LexicalOverlapReranker().score("", ["a"]) == [0.0]


def test_context_drops_duplicates_keeps_higher_ranked():
    a = chunk("a", "the refund window is thirty days for all orders", rank=1)
    b = chunk("b", "the refund window is thirty days for all orders", rank=2)
    c = chunk("c", "vacation policy grants twenty days", rank=3)
    r = assemble_context([a, b, c])
    assert [p.chunk_id for p in r.passages] == ["a", "c"] and r.dropped == {"b": "duplicate"}
    assert [p.rank for p in r.passages] == [1, 2]


def test_context_doc_cap_and_budget_and_always_keeps_first():
    ps = [chunk(f"c{i}", f"unique words number{i} " * 5, doc="d1" if i < 3 else "d2") for i in range(5)]
    capped = assemble_context(ps, max_per_doc=1)
    assert [p.chunk_id for p in capped.passages] == ["c0", "c3"]
    assert capped.dropped["c1"] == "doc_cap"
    one = estimate_tokens(ps[0].text)
    tight = assemble_context(ps, token_budget=one + 1)
    assert [p.chunk_id for p in tight.passages] == ["c0"] and tight.dropped["c1"] == "budget"
    huge = assemble_context([chunk("big", "w " * 1000)], token_budget=10)
    assert [p.chunk_id for p in huge.passages] == ["big"]


def test_context_validation():
    with pytest.raises(ValueError):
        assemble_context([], token_budget=0)
