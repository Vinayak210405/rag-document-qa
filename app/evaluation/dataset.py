"""Evaluation dataset: schema, loading, and integrity validation.

Why evidence is labelled by quote, not chunk ID
-----------------------------------------------
Chunk IDs depend on the chunking strategy. A label such as "chunk 3f9a..." is
meaningless once we change the chunker (Milestone 7 compares chunkers). So each
question stores *verbatim evidence quotes* (document + exact sentence). At
benchmark time (Milestone 2) a retrieved chunk counts as relevant if it
contains a gold quote. The same labels then work for every chunker.

Two layers of validation
------------------------
1. Structural rules (pydantic, no filesystem): enforced when a dataset is loaded.
2. Corpus rules (``validate_dataset``, needs the corpus on disk): every quote
   must appear exactly once in its document, and the corpus hash must match.
   This catches "label rot" when someone edits a document but not the labels.

Run ``python -m app.evaluation.dataset --show`` to inspect the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_DATASET_PATH = Path("data/eval/dataset_v0.1.0.json")
DEFAULT_CORPUS_DIR = Path("data/sample_corpus")


class Category(StrEnum):
    """Question categories required by the project specification."""

    DIRECT_FACTUAL = "direct_factual"
    PARAPHRASE = "paraphrase"
    EXACT_IDENTIFIER = "exact_identifier"
    MULTI_DOCUMENT = "multi_document"
    MULTI_HOP = "multi_hop"
    CONFLICTING_OUTDATED = "conflicting_outdated"
    UNANSWERABLE = "unanswerable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# Categories whose correct behaviour is to abstain.
ABSTAIN_CATEGORIES = frozenset({Category.UNANSWERABLE, Category.INSUFFICIENT_EVIDENCE})


class Passage(BaseModel):
    """A verbatim quote from one corpus document."""

    model_config = ConfigDict(extra="forbid")

    doc: str = Field(min_length=1, description="File name inside the corpus directory.")
    quote: str = Field(min_length=1, description="Exact text; must occur once in the document.")


class Evidence(Passage):
    """A gold passage with a graded relevance (used for nDCG in Milestone 2)."""

    grade: int = Field(
        ge=1,
        le=2,
        description="2 = required to answer; 1 = helpful supporting context.",
    )


class Question(BaseModel):
    """One labelled evaluation question."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    category: Category
    question: str = Field(min_length=1)
    answerable: bool
    reference_answer: str = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)
    # Plausible-but-wrong or non-answering passages (e.g. a superseded policy).
    # Retrieval may surface them; the answer must not rely on them.
    distractors: list[Passage] = Field(default_factory=list)
    label_source: Literal["claude_authored", "human_authored", "synthetic_llm"]
    human_verified: bool = False

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        must_abstain = self.category in ABSTAIN_CATEGORIES
        if self.answerable == must_abstain:
            raise ValueError(
                f"{self.id}: category '{self.category}' requires "
                f"answerable={not must_abstain}, got answerable={self.answerable}"
            )
        if self.answerable and not any(e.grade == 2 for e in self.evidence):
            raise ValueError(f"{self.id}: answerable question needs >= 1 grade-2 evidence")
        if not self.answerable and self.evidence:
            raise ValueError(f"{self.id}: unanswerable question must have no gold evidence")
        return self


class EvalDataset(BaseModel):
    """A versioned set of questions tied to one specific corpus snapshot."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    created: str
    description: str
    corpus_sha256: str = Field(min_length=64, max_length=64)
    questions: list[Question] = Field(min_length=1)

    def category_counts(self) -> Counter[str]:
        return Counter(str(q.category) for q in self.questions)


def corpus_sha256(corpus_dir: Path) -> str:
    """Hash a corpus directory deterministically.

    Hashes (file name, bytes) for every non-hidden file in sorted name order, so
    the result is independent of directory listing order and ignores files such
    as ``.gitkeep``. Renaming or editing any document changes the hash.
    """
    digest = hashlib.sha256()
    files = sorted(
        (p for p in corpus_dir.iterdir() if p.is_file() and not p.name.startswith(".")),
        key=lambda p: p.name,
    )
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_dataset(path: Path) -> EvalDataset:
    """Load and structurally validate a dataset file (raises on invalid data)."""
    return EvalDataset.model_validate_json(path.read_text(encoding="utf-8"))


def validate_dataset(dataset: EvalDataset, corpus_dir: Path) -> list[str]:
    """Check a dataset against the corpus on disk; return a list of problems.

    An empty list means the dataset is consistent with the corpus.
    """
    problems: list[str] = []

    ids = [q.id for q in dataset.questions]
    for dup, n in Counter(ids).items():
        if n > 1:
            problems.append(f"duplicate question id: {dup} (x{n})")

    actual_hash = corpus_sha256(corpus_dir)
    if actual_hash != dataset.corpus_sha256:
        problems.append(
            "corpus hash mismatch: dataset records "
            f"{dataset.corpus_sha256[:12]}..., corpus is {actual_hash[:12]}... "
            "(corpus changed: re-verify labels, bump the dataset version, update the hash)"
        )

    texts: dict[str, str | None] = {}

    def text_of(doc: str) -> str | None:
        if doc not in texts:
            p = corpus_dir / doc
            texts[doc] = p.read_text(encoding="utf-8") if p.is_file() else None
        return texts[doc]

    for q in dataset.questions:
        passages: list[tuple[str, Passage]] = [("evidence", e) for e in q.evidence]
        passages += [("distractor", d) for d in q.distractors]
        for kind, p in passages:
            text = text_of(p.doc)
            if text is None:
                problems.append(f"{q.id} {kind}: document not found: {p.doc}")
                continue
            n = text.count(p.quote)
            if n == 0:
                problems.append(f"{q.id} {kind}: quote not found in {p.doc}: {p.quote!r}")
            elif n > 1:
                problems.append(
                    f"{q.id} {kind}: quote is ambiguous (occurs {n}x) in {p.doc}: {p.quote!r}"
                )
    return problems


def _print_summary(dataset: EvalDataset, corpus_dir: Path, show: bool) -> None:
    n = len(dataset.questions)
    answerable = sum(q.answerable for q in dataset.questions)
    verified = sum(q.human_verified for q in dataset.questions)
    print(f"Dataset  : {dataset.name} v{dataset.version} (created {dataset.created})")
    print(f"Questions: {n} ({answerable} answerable, {n - answerable} must-abstain)")
    print(f"Human-verified labels: {verified}/{n}")
    print(f"Recorded corpus sha256: {dataset.corpus_sha256}")
    print(f"Actual   corpus sha256: {corpus_sha256(corpus_dir)}")
    print("\nBy category:")
    for cat, count in sorted(dataset.category_counts().items()):
        print(f"  {cat:<22}{count}")
    if show:
        print()
        for q in dataset.questions:
            flag = "verified" if q.human_verified else "UNVERIFIED"
            print(f"[{q.id}] ({q.category}, {flag}, {q.label_source})")
            print(f"  Q: {q.question}")
            print(f"  A: {q.reference_answer}")
            for e in q.evidence:
                print(f"  + grade {e.grade}  {e.doc}: {e.quote}")
            for d in q.distractors:
                print(f"  - distractor  {d.doc}: {d.quote}")
            print()


def main(argv: list[str] | None = None) -> int:
    """CLI: summarise and validate the dataset. Exit code 1 if problems exist."""
    parser = argparse.ArgumentParser(description="Inspect and validate the evaluation dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--show", action="store_true", help="print every question and its evidence")
    args = parser.parse_args(argv)

    dataset = load_dataset(args.dataset)
    _print_summary(dataset, args.corpus, args.show)
    problems = validate_dataset(dataset, args.corpus)
    if problems:
        print("VALIDATION PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Validation: OK (all quotes found exactly once; corpus hash matches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
