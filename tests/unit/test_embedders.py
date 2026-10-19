import sys
import types

import numpy as np
import pytest

from app.embeddings import EmbeddingBackendError, HashingEmbedder
from app.embeddings.sentence_transformers_embedder import SentenceTransformerEmbedder


def test_hashing_vectors_are_unit_length_and_deterministic() -> None:
    texts = ["Passwords must be rotated every 180 days.", "Meals are capped at 60 USD."]
    a, b = HashingEmbedder().embed_documents(texts), HashingEmbedder().embed_documents(texts)
    assert a.shape == (2, 1024) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(a, b)


def test_hashing_query_matches_document_embedding_of_same_text() -> None:
    e = HashingEmbedder()
    assert np.allclose(e.embed_query("error code ERR-5043"), e.embed_documents(["error code ERR-5043"])[0])


def test_hashing_prefers_lexical_overlap() -> None:
    e = HashingEmbedder()
    q = e.embed_query("what does error code ERR-5043 mean")
    docs = e.embed_documents(["Error code ERR-5043 indicates encryption is off.", "Meals are capped at 60 USD."])
    assert (docs @ q)[0] > (docs @ q)[1]


def test_hashing_stopword_only_text_is_a_zero_vector_not_nan() -> None:
    v = HashingEmbedder().embed_query("what is the")
    assert not np.isnan(v).any() and np.linalg.norm(v) == 0


def test_hashing_empty_input() -> None:
    assert HashingEmbedder().embed_documents([]).shape == (0, 1024)


# --- the sentence-transformers wrapper, tested with a stub (no PyTorch, no download) -------------
class _NewApiModel:
    """Mimics sentence-transformers 6.x (get_embedding_dimension)."""

    def __init__(self, name: str) -> None:
        self.seen: list[list[str]] = []

    def get_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        self.seen.append(list(texts))
        return np.array([[3.0 + i, 4.0, 0.0, 0.0] for i, _ in enumerate(texts)])  # float64, un-normalised


class _OldApiModel(_NewApiModel):
    """Mimics sentence-transformers 3.x-5.x (only the old method name)."""

    get_embedding_dimension = None  # type: ignore[assignment]

    def get_sentence_embedding_dimension(self) -> int:
        return 4


def _install(monkeypatch: pytest.MonkeyPatch, model_cls: type) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=model_cls))


def test_st_wrapper_prefixes_queries_only_and_returns_unit_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _NewApiModel)
    e = SentenceTransformerEmbedder("any-model", query_prefix="QUERY: ")
    q = e.embed_query("hello")
    docs = e.embed_documents(["doc one", "doc two"])
    assert e._model.seen == [["QUERY: hello"], ["doc one", "doc two"]]
    assert q.dtype == np.float32 and docs.dtype == np.float32
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-5)
    assert e.dimension == 4 and e.model_name == "any-model"


def test_st_wrapper_supports_the_old_dimension_method_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _OldApiModel)
    assert SentenceTransformerEmbedder("m").dimension == 4


def test_st_wrapper_gives_an_actionable_error_when_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # makes `import` raise ImportError
    with pytest.raises(EmbeddingBackendError, match="uv sync --extra embeddings"):
        SentenceTransformerEmbedder("m")
