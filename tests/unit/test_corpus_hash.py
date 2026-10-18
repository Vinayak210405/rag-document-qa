"""corpus_sha256: deterministic, content-sensitive, ignores hidden files."""

from __future__ import annotations

from pathlib import Path

from app.evaluation.dataset import corpus_sha256


def _write(d: Path, name: str, text: str) -> None:
    (d / name).write_text(text, encoding="utf-8")


def test_hash_is_deterministic(tmp_path: Path) -> None:
    _write(tmp_path, "a.txt", "alpha")
    _write(tmp_path, "b.txt", "beta")
    assert corpus_sha256(tmp_path) == corpus_sha256(tmp_path)


def test_hash_is_independent_of_creation_order(tmp_path: Path) -> None:
    d1, d2 = tmp_path / "one", tmp_path / "two"
    d1.mkdir()
    d2.mkdir()
    _write(d1, "a.txt", "alpha")
    _write(d1, "b.txt", "beta")
    _write(d2, "b.txt", "beta")
    _write(d2, "a.txt", "alpha")
    assert corpus_sha256(d1) == corpus_sha256(d2)


def test_hash_changes_when_content_changes(tmp_path: Path) -> None:
    _write(tmp_path, "a.txt", "alpha")
    before = corpus_sha256(tmp_path)
    _write(tmp_path, "a.txt", "alpha!")
    assert corpus_sha256(tmp_path) != before


def test_hash_changes_when_a_file_is_renamed(tmp_path: Path) -> None:
    _write(tmp_path, "a.txt", "alpha")
    before = corpus_sha256(tmp_path)
    (tmp_path / "a.txt").rename(tmp_path / "z.txt")
    assert corpus_sha256(tmp_path) != before


def test_hash_ignores_hidden_files(tmp_path: Path) -> None:
    _write(tmp_path, "a.txt", "alpha")
    before = corpus_sha256(tmp_path)
    _write(tmp_path, ".gitkeep", "")
    assert corpus_sha256(tmp_path) == before
