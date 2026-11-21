import numpy as np
import pytest

from app.chunking import SemanticChunker, SentenceChunker
from app.chunking.base import make_chunk_id
from app.ingestion.models import Block, ParsedDocument


class TopicEmbedder:
    """Deterministic fake: sentences mentioning 'cat' point one way, 'tax' another."""

    model_name, dimension = "topic-fake", 3

    def embed_documents(self, texts):
        return np.array([[1.0, 0.0, 0.0] if "cat" in t.lower() else [0.0, 1.0, 0.0] for t in texts])

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def doc(*blocks, meta=None):
    return ParsedDocument("d.txt", "d.txt", "h", "txt", tuple(blocks), None, (), meta or {})


BODY = Block("The cat sat on the mat. The cat likes fish. A cat sleeps all day. "
             "Tax law changed in 2020. Tax forms are due in April. Tax refunds take weeks.", "body", None)


def test_splits_at_topic_shift_not_by_size():
    chunker = SemanticChunker(TopicEmbedder(), max_chars=500, min_chars=20, percentile=60)
    chunks = chunker.chunk(doc(BODY))
    assert len(chunks) == 2
    assert "cat" in chunks[0].text.lower() and "tax" not in chunks[0].text.lower()
    assert chunks[1].text.startswith("Tax law") and "cat" not in chunks[1].text.lower()
    assert chunker.units_embedded == 6  # cost accounting


def test_ids_ordinals_metadata_and_determinism():
    d = doc(BODY, meta={"doc_family": "X"})
    c1 = SemanticChunker(TopicEmbedder(), min_chars=20, percentile=60).chunk(d)
    c2 = SemanticChunker(TopicEmbedder(), min_chars=20, percentile=60).chunk(d)
    assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]
    assert [c.ordinal for c in c1] == [0, 1]
    assert all(c.chunk_id == make_chunk_id("d.txt", c.ordinal, c.text) and c.metadata == {"doc_family": "X"} for c in c1)


def test_heading_forces_boundary_and_is_recorded():
    d = doc(Block("Cats", "heading", None), Block("The cat sat. The cat ran. The cat slept.", "body", None),
            Block("Taxes", "heading", None), Block("Tax is due. Tax is hard. Tax is yearly.", "body", None))
    chunks = SemanticChunker(TopicEmbedder(), min_chars=10, percentile=95).chunk(d)
    assert [c.heading for c in chunks] == ["Cats", "Taxes"]
    assert chunks[0].text.startswith("Cats") and chunks[1].text.startswith("Taxes")


def test_oversized_segment_is_packed_and_small_segments_merged():
    long_body = Block(" ".join(f"The cat did thing number {i} today." for i in range(40)), "body", None)
    chunks = SemanticChunker(TopicEmbedder(), max_chars=200, min_chars=20).chunk(doc(long_body))
    assert len(chunks) > 3 and all(len(c.text) <= 200 for c in chunks)
    tiny = doc(Block("Cat one. Tax two. Cat three. Tax four.", "body", None))
    merged = SemanticChunker(TopicEmbedder(), max_chars=500, min_chars=400, percentile=50).chunk(tiny)
    assert len(merged) == 1  # everything is below min_chars, so it ends up as one chunk


def test_single_sentence_and_validation():
    assert len(SemanticChunker(TopicEmbedder()).chunk(doc(Block("Only one sentence here.", "body", None)))) == 1
    with pytest.raises(ValueError):
        SemanticChunker(TopicEmbedder(), percentile=100)
    with pytest.raises(ValueError):
        SemanticChunker(TopicEmbedder(), max_chars=100, min_chars=100)


def test_strategy_name_distinguishes_settings_and_service_reindexes(make_service):
    from app.embeddings import HashingEmbedder
    a = SemanticChunker(HashingEmbedder(), percentile=80).strategy_name
    b = SemanticChunker(HashingEmbedder(), percentile=90).strategy_name
    assert a != b and a != SentenceChunker().strategy_name
    svc = make_service()
    svc.ingest("a.txt", b"Cats purr loudly. Cats nap often. Taxes are due in April. Taxes are annoying to file.")
    svc.chunker = SemanticChunker(HashingEmbedder(), min_chars=10, percentile=50)
    assert svc.ingest("a.txt", b"Cats purr loudly. Cats nap often. Taxes are due in April. Taxes are annoying to file.").status == "updated"
