"""Our dense retriever vs LlamaIndex reading the SAME Chroma collection (optional: needs --extra llamaindex)."""
import uuid

import numpy as np
import pytest

pytest.importorskip("llama_index.core")
pytest.importorskip("llama_index.vector_stores.chroma")

from app.chunking import SentenceChunker  # noqa: E402
from app.embeddings import HashingEmbedder  # noqa: E402
from app.evaluation.benchmark import _corpus_files  # noqa: E402
from app.evaluation.dataset import DEFAULT_CORPUS_DIR, DEFAULT_DATASET_PATH, load_dataset  # noqa: E402
from app.indexing import ChromaVectorStore  # noqa: E402
from app.ingestion import parse_document  # noqa: E402
from app.integrations.llamaindex_retriever import LlamaIndexRetriever  # noqa: E402
from app.retrieval.dense import DenseRetriever  # noqa: E402


@pytest.fixture(scope="module")
def stores():
    emb = HashingEmbedder()
    store = ChromaVectorStore(None, f"li-{uuid.uuid4().hex}", emb.model_name, emb.dimension)
    chunker = SentenceChunker(500, 80)
    for path in _corpus_files(DEFAULT_CORPUS_DIR):
        doc = parse_document(path.name, path.read_bytes())
        chunks = chunker.chunk(doc)
        store.replace_document(doc.doc_id, chunks, emb.embed_documents([c.text for c in chunks]))
    return store, emb


def test_llamaindex_ranks_equal_ours_up_to_ties(stores):
    """Same cosine scores at every rank. The offline hashing embedder produces many exact ties, and tie order is
    implementation-defined (our rule: lower index; Chroma HNSW: arbitrary), so identical ID lists are NOT expected."""
    from app.retrieval.similarity import cosine_similarity

    store, emb = stores
    snap = store.load_snapshot()
    ours, theirs = DenseRetriever(store, emb), LlamaIndexRetriever(store, emb)
    id_to_row = {cid: i for i, cid in enumerate(snap.ids)}
    exact = 0
    questions = [q.question for q in load_dataset(DEFAULT_DATASET_PATH).questions]
    for q in questions:
        scores = cosine_similarity(emb.embed_query(q), snap.embeddings)
        their_ids = [c.chunk_id for c in theirs.retrieve(q, 5)]
        our_ids = [c.chunk_id for c in ours.retrieve(q, 5)]
        their_scores = [float(scores[id_to_row[i]]) for i in their_ids]
        our_scores = [float(scores[id_to_row[i]]) for i in our_ids]
        assert np.allclose(their_scores, our_scores, atol=1e-5), (q, our_scores, their_scores)
        exact += their_ids == our_ids
    assert exact >= len(questions) // 2  # most questions have no ties at all and match exactly


def test_results_carry_provenance_and_metadata_and_refuse_acl(stores):
    store, emb = stores
    hit = LlamaIndexRetriever(store, emb).retrieve("What is the daily meal allowance?", 3)[0]
    assert hit.doc_name.endswith(".txt") and hit.chunk_id and hit.text and hit.metadata.get("doc_family")
    with pytest.raises(NotImplementedError):
        LlamaIndexRetriever(store, emb).retrieve("x", 3, groups={"hr"})
