import pytest

from app.chunking import SentenceChunker, make_chunk_id
from app.ingestion.loader import parse_document
from app.ingestion.models import Block, ParsedDocument
from tests.conftest import CORPUS_DIR


def doc(*blocks: Block, doc_id: str = "d.txt") -> ParsedDocument:
    return ParsedDocument(doc_id, doc_id, "hash", "txt", tuple(blocks), None)


def body(text: str, page: int | None = None) -> Block:
    return Block(text, "body", page)


def heading(text: str, page: int | None = None) -> Block:
    return Block(text, "heading", page)


def numbered_sentences(n: int) -> str:
    return " ".join(f"Sentence number {i:02d} is here." for i in range(n))  # 27 chars each


@pytest.mark.parametrize(("max_chars", "overlap"), [(100, 0), (150, 30), (300, 60), (500, 80)])
def test_no_chunk_exceeds_max_chars(max_chars: int, overlap: int) -> None:
    chunker = SentenceChunker(max_chars, overlap)
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        for chunk in chunker.chunk(parse_document(path.name, path.read_bytes())):
            assert 0 < chunk.char_count <= max_chars, (path.name, chunk.char_count)
    long_doc = doc(body(numbered_sentences(80)))
    assert all(c.char_count <= max_chars for c in chunker.chunk(long_doc))


def test_every_sentence_is_covered_by_some_chunk() -> None:
    chunks = SentenceChunker(200, 50).chunk(doc(body(numbered_sentences(40))))
    joined = "\n".join(c.text for c in chunks)
    assert all(f"Sentence number {i:02d} is here." in joined for i in range(40))


def test_consecutive_chunks_overlap_by_whole_sentences() -> None:
    chunks = SentenceChunker(200, 60).chunk(doc(body(numbered_sentences(40))))
    assert len(chunks) > 2
    for prev, nxt in zip(chunks, chunks[1:]):
        first_sentence_of_next = nxt.text.split(". ")[0] + "."
        assert first_sentence_of_next in prev.text  # carried over intact, never cut mid-sentence


def test_zero_overlap_means_no_shared_sentences() -> None:
    chunks = SentenceChunker(200, 0).chunk(doc(body(numbered_sentences(40))))
    seen: set[str] = set()
    for chunk in chunks:
        sentences = {s for s in chunk.text.split(". ")}
        assert not (sentences & seen)
        seen |= sentences


def test_chunking_is_deterministic_and_ordinals_are_sequential() -> None:
    d = doc(body(numbered_sentences(40)))
    a, b = SentenceChunker(200, 50).chunk(d), SentenceChunker(200, 50).chunk(d)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.ordinal for c in a] == list(range(len(a)))
    assert len({c.chunk_id for c in a}) == len(a)


def test_chunk_id_is_stable_and_sensitive_to_every_input() -> None:
    base = make_chunk_id("doc.txt", 0, "hello")
    assert base == make_chunk_id("doc.txt", 0, "hello")
    assert len(base) == 16 and int(base, 16) >= 0
    assert base != make_chunk_id("other.txt", 0, "hello")
    assert base != make_chunk_id("doc.txt", 1, "hello")
    assert base != make_chunk_id("doc.txt", 0, "hello!")


def test_heading_is_not_stranded_at_the_end_of_a_chunk() -> None:
    s1, s2, s3 = "A" * 88 + ".", "B" * 88 + ".", "C" * 88 + "."
    chunks = SentenceChunker(200, 0).chunk(
        doc(body(f"{s1} {s2}"), heading("2. Next section"), body(s3))
    )
    assert "Next section" not in chunks[0].text
    assert chunks[1].text.startswith("2. Next section\n")


def test_oversized_sentence_is_hard_split_without_losing_words() -> None:
    words = [f"w{i:03d}" for i in range(300)]
    chunks = SentenceChunker(300, 50).chunk(doc(body(" ".join(words))))
    assert len(chunks) > 3 and all(c.char_count <= 300 for c in chunks)
    covered = {w for c in chunks for w in c.text.split()}
    assert set(words) <= covered or len(set(words) - covered) <= len(chunks)  # only cut-edge words may be split


def test_chunks_never_cross_page_boundaries() -> None:
    d = doc(body("Alpha one. Alpha two.", page=1), body("Beta one. Beta two.", page=2))
    chunks = SentenceChunker(500, 80).chunk(d)
    assert [c.page for c in chunks] == [1, 2]
    assert "Beta" not in chunks[0].text and "Alpha" not in chunks[1].text


def test_heading_metadata_comes_from_the_first_new_sentence_not_the_overlap() -> None:
    path = CORPUS_DIR / "it_security_handbook.txt"
    chunks = SentenceChunker(300, 60).chunk(parse_document(path.name, path.read_bytes()))
    by_heading = {c.heading for c in chunks}
    assert "2. Multi-factor authentication" in by_heading
    mfa = next(c for c in chunks if "hardware security key" in c.text)
    assert mfa.heading == "2. Multi-factor authentication"


@pytest.mark.parametrize(("max_chars", "overlap"), [(49, 0), (200, 200), (200, -1)])
def test_invalid_configuration_is_rejected(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        SentenceChunker(max_chars, overlap)


def test_overlap_is_shrunk_when_a_large_next_sentence_would_overflow_the_chunk() -> None:
    s1, s2, s3 = "A" * 38 + ".", "B" * 38 + ".", "C" * 88 + "."  # 39, 39, 89 chars; max 100
    chunks = SentenceChunker(100, 60).chunk(doc(body(f"{s1} {s2} {s3}")))
    assert all(c.char_count <= 100 for c in chunks)
    assert chunks[-1].text == s3  # the s2 overlap had to be dropped: "s2 s3" would be 129 chars
