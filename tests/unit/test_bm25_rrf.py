import math

import numpy as np
import pytest

from app.retrieval.bm25 import BM25Index, tokenize
from app.retrieval.fusion import reciprocal_rank_fusion


def test_tokenize_keeps_identifier_and_parts():
    assert tokenize("Invoice INV-2041 v1.2!") == ["invoice", "inv-2041", "inv", "2041", "v1.2", "v1", "2"]


def test_bm25_matches_hand_computation():
    # N=2, "cat" in 1 doc -> idf = ln(1 + 1.5/1.5) = ln 2. avgdl = 2. Doc0 len 2, tf 1, b=0.75 -> norm = k1.
    idx = BM25Index(["cat dog", "bird fish"], k1=1.5, b=0.75)
    expected = math.log(2) * 1 * 2.5 / (1 + 1.5)
    s = idx.scores("cat")
    assert s[0] == pytest.approx(expected) and s[1] == 0.0


def test_bm25_rare_term_outweighs_common_and_unknown_is_zero():
    idx = BM25Index(["the cat", "the dog", "the bird"])
    assert idx.scores("cat")[0] > idx.scores("the")[0]
    assert not idx.scores("zebra").any()


def test_bm25_repeated_query_term_not_double_counted_and_length_normalised():
    idx = BM25Index(["alpha beta", "alpha " + "pad " * 20])
    assert np.allclose(idx.scores("alpha alpha"), idx.scores("alpha"))
    assert idx.scores("alpha")[0] > idx.scores("alpha")[1]


def test_bm25_exact_identifier_match():
    idx = BM25Index(["order ZX-4471 shipped", "order ZX-9999 shipped"])
    s = idx.scores("ZX-4471")
    assert s[0] > 0
    # the shared parts "zx"/"order"/"shipped" still give the other doc some score, but less
    assert s[0] > s[1]


def test_bm25_empty_and_invalid():
    assert len(BM25Index([]).scores("x")) == 0
    with pytest.raises(ValueError):
        BM25Index(["a"], b=2)


def test_rrf_hand_computed():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)
    d = dict(fused)
    assert d["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert d["a"] == pytest.approx(1 / 61) and d["d"] == pytest.approx(1 / 63)
    assert [i for i, _ in fused] == ["b", "c", "a", "d"]


def test_rrf_agreement_beats_single_top_rank():
    assert reciprocal_rank_fusion([["x", "y"], ["z", "y"]])[0][0] == "y"


def test_rrf_deterministic_tie_break_and_dedup_and_weights():
    out = reciprocal_rank_fusion([["a"], ["b"]])
    assert [i for i, _ in out] == ["a", "b"]  # equal scores -> first seen
    assert dict(reciprocal_rank_fusion([["a", "a"]]))["a"] == pytest.approx(1 / 61)
    w = reciprocal_rank_fusion([["a"], ["b"]], weights=[1, 3])
    assert w[0][0] == "b"
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1, 2])
    assert reciprocal_rank_fusion([]) == []
