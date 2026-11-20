from app.evaluation.dataset import Evidence, Passage
from app.evaluation.relevance import chunk_grade, contains_passage, contains_quote, covered_evidence

EVIDENCE = [
    Evidence(doc="a.txt", quote="Limit is 100 USD.", grade=2),
    Evidence(doc="a.txt", quote="Receipts are kept.", grade=1),
    Evidence(doc="b.txt", quote="Limit is 100 USD.", grade=2),
]


def test_quote_matching_ignores_whitespace_differences() -> None:
    assert contains_quote("The   Limit is\n100 USD.  Next.", "Limit is 100 USD.")
    assert not contains_quote("Limit is 100", "Limit is 100 USD.")  # partial quotes do not count


def test_chunk_grade_is_the_best_contained_grade() -> None:
    assert chunk_grade("a.txt", "Limit is 100 USD. Receipts are kept.", EVIDENCE) == 2
    assert chunk_grade("a.txt", "Only: Receipts are kept.", EVIDENCE) == 1
    assert chunk_grade("a.txt", "Nothing relevant.", EVIDENCE) == 0


def test_the_same_sentence_in_another_document_is_not_relevant() -> None:
    assert chunk_grade("c.txt", "Limit is 100 USD.", EVIDENCE) == 0


def test_a_quote_split_across_two_chunks_is_covered_by_neither() -> None:
    halves = [("a.txt", "Limit is"), ("a.txt", "100 USD.")]
    assert covered_evidence(halves, EVIDENCE) == set()


def test_covered_evidence_reports_indices() -> None:
    chunks = [("a.txt", "x Limit is 100 USD."), ("b.txt", "Limit is 100 USD."), ("c.txt", "Receipts are kept.")]
    assert covered_evidence(chunks, EVIDENCE) == {0, 2}  # c.txt is the wrong document for item 1


def test_distractor_detection_is_document_specific() -> None:
    old = [Passage(doc="v1.txt", quote="20 days")]
    assert contains_passage("v1.txt", "you get 20 days", old)
    assert not contains_passage("v2.txt", "you get 20 days", old)
