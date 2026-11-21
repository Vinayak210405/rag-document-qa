# Architecture

## Data flow

**Ingestion** (`RagService.ingest`): `validate_signature` (bytes must match the extension; zip-bomb and page limits) -> parser
(`app/ingestion/parsers.py`: TXT/PDF/DOCX -> blocks with kind heading/body and page) -> header metadata
(`Document ID`, `Version`, `Effective date`, `Status`) -> content hash. If the same file name already has this hash, chunker/embedder
signature and ACL groups, the upload is skipped (`unchanged`). Otherwise: chunker (sentence-aware or semantic) -> embeddings ->
`ChromaVectorStore.replace_document` (upsert new chunks, then delete stale ones). Chunk IDs are `sha256(doc_id, ordinal, text)[:16]`.

**Query** (`RagService.answer`): normalise query -> retriever(`groups`) -> context assembly -> safeguards -> evidence gate -> prompt
(+ version warning) -> LLM -> `validate_answer` -> status. Retrievers share one interface `retrieve(query, k, groups=None)`:
`DenseRetriever`, `BM25Retriever`, `HybridRetriever` (RRF), `RerankedRetriever(base, reranker)`.

## Module responsibilities

| Module | Responsibility | Why separate |
|---|---|---|
| `retrieval/similarity.py` | cosine + deterministic top-k | transparent baseline, unit-testable maths |
| `retrieval/bm25.py`, `fusion.py` | lexical scoring, RRF | each hand-verified in tests |
| `retrieval/rerank.py`, `context.py` | second stage, selection | swap rerankers; budget logic isolated |
| `retrieval/access.py` | ACL mask applied before ranking | one place to audit the boundary |
| `generation/*` | prompt, LLM client, citations, evidence, conflicts, safety | each checkable without an LLM |
| `indexing/chroma_store.py` | persistence, lifecycle, audit | Chroma is a store only; ranking is ours |
| `evaluation/*` | metrics, dataset, benchmark, Ragas, abstention | deterministic metrics independent of Ragas |
| `service.py` | orchestration | API/UI contain no business logic |

## Invariants (tested)

- Dense and lexical retrieval read the same store snapshot (keyed by write counter): an update or delete is visible to both.
- A document update never leaves a document with missing content (new chunks written before old ones are removed).
- Restricted chunks never enter ranking, results, prompts or document listings for callers outside their groups.
- Retrieval metrics are reproducible: same config + dataset => same `results_sha256`.
- Ordinary tests need no network, API key or model download.

## Not built (by design)

Agents/orchestration frameworks, async endpoints (blocking model calls run in FastAPI's thread pool), approximate-NN tuning, a
persistent incremental BM25 index, old-version retention, OCR engine, user management.
