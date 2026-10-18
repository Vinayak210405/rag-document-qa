"""Evaluation dataset: the real dataset is valid, and the validator catches bad ones."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.dataset import (
    Category,
    EvalDataset,
    Question,
    corpus_sha256,
    load_dataset,
    validate_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data/eval/dataset_v0.1.0.json"
CORPUS_DIR = ROOT / "data/sample_corpus"


# --------------------------------------------------------------------------
# The real, committed dataset
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dataset() -> EvalDataset:
    return load_dataset(DATASET_PATH)


def test_real_dataset_is_consistent_with_corpus(dataset: EvalDataset) -> None:
    assert validate_dataset(dataset, CORPUS_DIR) == []


def test_every_required_category_is_present(dataset: EvalDataset) -> None:
    assert {q.category for q in dataset.questions} == set(Category)


def test_dataset_mixes_answerable_and_abstain_questions(dataset: EvalDataset) -> None:
    answerable = [q for q in dataset.questions if q.answerable]
    abstain = [q for q in dataset.questions if not q.answerable]
    assert answerable and abstain


def test_labels_are_honestly_marked_unverified(dataset: EvalDataset) -> None:
    """Claude wrote these labels; none may claim human verification until a human flips it."""
    for q in dataset.questions:
        if q.label_source == "claude_authored":
            assert q.human_verified is False, q.id


def test_conflicting_questions_have_an_outdated_distractor(dataset: EvalDataset) -> None:
    for q in dataset.questions:
        if q.category is Category.CONFLICTING_OUTDATED:
            assert any(d.doc.endswith("_v1.txt") for d in q.distractors), q.id


# --------------------------------------------------------------------------
# The validator itself (tested on small synthetic data)
# --------------------------------------------------------------------------
def _corpus(tmp_path: Path) -> Path:
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "a.txt").write_text("Alpha is first. Beta is second. Beta is second.", encoding="utf-8")
    return d


def _question(**overrides: object) -> dict:
    base: dict = {
        "id": "t1",
        "category": "direct_factual",
        "question": "Which is first?",
        "answerable": True,
        "reference_answer": "Alpha.",
        "evidence": [{"doc": "a.txt", "quote": "Alpha is first.", "grade": 2}],
        "distractors": [],
        "label_source": "claude_authored",
    }
    base.update(overrides)
    return base


def _dataset(corpus: Path, *questions: dict) -> EvalDataset:
    return EvalDataset.model_validate(
        {
            "name": "t",
            "version": "0",
            "created": "2026-01-01",
            "description": "synthetic",
            "corpus_sha256": corpus_sha256(corpus),
            "questions": list(questions),
        }
    )


def test_validator_accepts_a_correct_dataset(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    assert validate_dataset(_dataset(corpus, _question()), corpus) == []


def test_validator_flags_a_quote_missing_from_the_document(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    bad = _question(evidence=[{"doc": "a.txt", "quote": "Gamma is third.", "grade": 2}])
    problems = validate_dataset(_dataset(corpus, bad), corpus)
    assert len(problems) == 1 and "quote not found" in problems[0]


def test_validator_flags_an_ambiguous_quote(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    bad = _question(evidence=[{"doc": "a.txt", "quote": "Beta is second.", "grade": 2}])
    problems = validate_dataset(_dataset(corpus, bad), corpus)
    assert len(problems) == 1 and "ambiguous" in problems[0]


def test_validator_flags_an_unknown_document(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    bad = _question(evidence=[{"doc": "nope.txt", "quote": "Alpha is first.", "grade": 2}])
    problems = validate_dataset(_dataset(corpus, bad), corpus)
    assert len(problems) == 1 and "document not found" in problems[0]


def test_validator_flags_a_bad_distractor_quote(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    bad = _question(distractors=[{"doc": "a.txt", "quote": "Not here."}])
    problems = validate_dataset(_dataset(corpus, bad), corpus)
    assert len(problems) == 1 and "distractor" in problems[0]


def test_validator_flags_an_edited_corpus(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    ds = _dataset(corpus, _question())
    (corpus / "a.txt").write_text("Alpha is first. Changed.", encoding="utf-8")
    problems = validate_dataset(ds, corpus)
    assert any("corpus hash mismatch" in p for p in problems)


def test_validator_flags_duplicate_ids(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    problems = validate_dataset(_dataset(corpus, _question(), _question()), corpus)
    assert any("duplicate question id" in p for p in problems)


# --------------------------------------------------------------------------
# Structural rules enforced by the schema
# --------------------------------------------------------------------------
def test_answerable_question_needs_required_evidence() -> None:
    weak = [{"doc": "a.txt", "quote": "x", "grade": 1}]  # only "helpful", none "required"
    with pytest.raises(ValidationError, match="grade-2"):
        Question.model_validate(_question(evidence=weak))


def test_abstain_question_must_not_have_gold_evidence() -> None:
    with pytest.raises(ValidationError, match="no gold evidence"):
        Question.model_validate(_question(category="unanswerable", answerable=False))


def test_category_and_answerable_flag_must_agree() -> None:
    with pytest.raises(ValidationError, match="requires answerable"):
        Question.model_validate(_question(category="unanswerable", answerable=True))


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(_question(typo_field="oops"))


def test_synthetic_labels_are_a_distinct_accepted_source() -> None:
    """M8 adds an LLM-generated set; it must be distinguishable from authored labels."""
    q = Question.model_validate(_question(label_source="synthetic_llm"))
    assert q.label_source == "synthetic_llm" and q.human_verified is False
    with pytest.raises(ValidationError):
        Question.model_validate(_question(label_source="made_up"))
