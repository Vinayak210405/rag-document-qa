# Evaluated Document QA Bot with RAG

An evaluation-driven retrieval-augmented generation system for question answering over private documents,
with verifiable citations and measured retrieval/generation quality. Built incrementally; each milestone has a
verification path. The retrieval core is implemented in this repo, not hidden behind an orchestration framework
(see [docs/decisions.md](docs/decisions.md)).

## Status

| Milestone | Scope | Status |
|---|---|---|
| 0 | Foundation: structure, config, sample corpus, labelled eval set, offline tests | Done |
| 1 | Baseline RAG: ingestion, chunking, embeddings, ChromaDB, own cosine top-k, grounded answers, API + UI | **Implemented, verified offline** (real embedding model and Ollama not yet run; see "Verification status") |
| 2 | Evaluation: Recall@k, MRR, nDCG, benchmark runner, run comparison | **Implemented, verified offline** (numbers so far come from the non-semantic stand-in embedder only) |
| 3 | Hybrid retrieval: BM25 + RRF | Not started |
| 4 | Cross-encoder reranking, context assembly | Not started |
| 5 | Citation validation, abstention | Not started |
| 6 | Document lifecycle: incremental ingest, update, delete | Not started |
| 7 | Semantic chunking | Not started |
| 8 | Ragas metrics | Not started |
| 9 | Security, benchmark dashboard, demo docs | Not started |
| 10 | LlamaIndex pipeline over the same Chroma index, benchmarked against ours | Not started |

No benchmark results exist yet; none are claimed. See [docs/claims.md](docs/claims.md) for what each milestone earns.

## Architecture (Milestone 1)

```
OFFLINE   file --> parse + normalize --> blocks (heading/body, page) --> sentence-aware chunks --> embed --> ChromaDB
ONLINE    question --> validate --> embed --> our cosine + top-k over stored vectors --> grounded prompt --> LLM --> citation check
UI        Streamlit  --HTTP-->  FastAPI  -->  RagService  -->  (modules above)
```

Code map: `app/ingestion` (parsers) | `app/chunking` | `app/embeddings` | `app/indexing` (Chroma) | `app/retrieval`
(similarity maths, dense retriever) | `app/generation` (prompt, LLM client, citations) | `app/service.py` (orchestration) |
`app/api` | `frontend/`. Design reasoning: [docs/decisions.md](docs/decisions.md).

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync                  # core dependencies + pytest (no PyTorch)
uv run pytest            # 233 tests, fully offline, ~15 s
```

**Windows tip:** if your system Python is broken or missing, make uv use its own, permanently:
`[Environment]::SetEnvironmentVariable("UV_PYTHON_PREFERENCE","only-managed","User")` then reopen PowerShell.

### Run it offline in two minutes (no downloads; retrieval only needs no LLM)

```powershell
$env:RAG_EMBEDDING_BACKEND = "hashing"      # non-semantic stand-in embedder
uv run uvicorn app.api.main:app             # terminal 1: API on http://127.0.0.1:8000 (docs at /docs)
uv run streamlit run frontend/app.py        # terminal 2: UI on http://localhost:8501
```
Upload files from `data/sample_corpus/` in the sidebar and tick "Retrieve only" to inspect evidence without an LLM.

### Run it for real (semantic embeddings + a local LLM)

1. `uv sync --extra embeddings` (installs PyTorch; the model, ~130 MB, downloads on first use).
2. Install [Ollama](https://ollama.com), then `ollama pull llama3.2:3b`.
3. Copy `.env.example` to `.env` (defaults already point at Ollama), then start the API and UI as above, without the
   `RAG_EMBEDDING_BACKEND` override.

**An index is tied to the embedding model that built it.** If you switch models (e.g. hashing to bge) the API refuses to
start with `IndexMismatchError`. Use a new `RAG_STORAGE_DIR` or delete the `storage/` folder.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status, document/chunk counts, embedding model |
| `POST /documents` | Upload PDF/DOCX/TXT (multipart `file`); re-uploading a name replaces it |
| `GET /documents` | List indexed documents |
| `POST /retrieve` | `{"question", "top_k"}`: ranked passages only, no LLM |
| `POST /ask` | Same input: answer + citations + sources + timings + token usage |

```powershell
curl.exe -F "file=@data/sample_corpus/expense_policy.txt" http://127.0.0.1:8000/documents
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json" `
  -Body '{"question":"What is the daily meal allowance for international travel?"}'
```
Errors: 415 unsupported type, 422 bad/empty/scanned file or invalid question, 409 nothing indexed (or model mismatch),
413 file too large, 502 LLM unreachable (never a made-up answer), 503 embedding backend unavailable.

## Evaluation (Milestone 2)

```powershell
# Retrieval benchmark: builds a fresh in-memory index, scores every question, saves JSON + CSV under results\
uv run python -m app.evaluation.benchmark --label dense-hashing --embedder hashing
# With the real model (after: uv sync --extra embeddings)
uv run python -m app.evaluation.benchmark --label dense-bge --embedder sentence_transformers
# Compare two runs: deltas plus how many questions got better / worse / stayed the same
uv run python -m app.evaluation.compare results\run_a.json results\run_b.json
```

Metrics (details in [docs/decisions.md](docs/decisions.md), D15-D20): Hit@k, Recall@k (share of *required* evidence found),
reciprocal rank (MRR is its mean, truncated at the retrieval depth), and graded nDCG@k where every chunk in the index is
judged by content. Every run also records a **random-ranking control**, tie/no-signal flags, and the outdated-source-above-gold
rate, because averages alone can mislead on a tiny index. Same configuration + same dataset reproduces the same
`results_sha256`; the runner refuses to run if the dataset no longer matches the corpus.

### Reference run (stand-in embedder, NOT semantic) - `docs/results/m2-dense-hashing-20261105T120543Z-19560425`

| Metric (18 answerable questions) | Pipeline | Random ranking |
|---|---|---|
| MRR@5 | 0.778 | 0.219 |
| Hit@1 | 0.667 | 0.103 |
| Hit@3 | 0.889 | 0.285 |
| Hit@5 | 0.889 | 0.462 |
| Recall@5 (required evidence) | 0.861 | 0.418 |
| nDCG@5 | 0.785 | 0.260 |

How to read this honestly:
- These numbers describe the offline **hashing** embedder (bag of words), not a semantic model. They exist to prove the
  harness works. The meaningful baseline is the real `bge-small` run, which has not been done yet.
- The index has only 12 chunks, so even random ranking reaches Hit@5 of 0.46. Compare against the control, not against 1.0.
- Two questions have their gold rank decided by a score tie and one by a zero score; the conflicting-documents questions
  fail partly because `v1` sorts before `v2` when scores tie (D20).
- The questions are Claude-authored and not human-verified; categories have 2-4 questions each. Differences are indicative only.

## Configuration

All settings use the `RAG_` prefix; see `.env.example`. Never commit `.env`. Chunk sizes are in characters.

## Verification status (Milestone 1)

| Verified here | Not verified here |
|---|---|
| 221 automated tests, incl. real ChromaDB on disk, PDF (hand-built), DOCX, TXT parsing | The real `bge-small` model (no Hugging Face access in the build sandbox): only the wrapper is tested, with a stub; its API names were checked against the sentence-transformers 6.1.0 wheel source. Run `uv run pytest -m requires_model` |
| Real uvicorn server over HTTP: upload, retrieve, ask, error statuses, restart persistence | A real LLM (Ollama). Only an OpenAI-compatible stand-in server was used. Run `uv run pytest -m requires_api` |
| Real Streamlit server boots; UI driven headlessly against the live API | A browser session, and anything on Windows |
| 7 injected bugs in core logic were all caught by the tests | Retrieval *quality* (no metrics until Milestone 2) |

## Evaluation dataset (v0.1.0)

24 questions over the Velmora Systems corpus, 8 categories: direct factual (4), paraphrase (3), exact identifier (4),
multi-document (2), multi-hop (2), conflicting/outdated (3), unanswerable (3), insufficient evidence (3).
18 are answerable and 6 require abstention.

- **Labels are quote-based**, so they stay valid across chunking strategies (decision D1).
- **Integrity is machine-checked:** every quote must occur exactly once in its document, and the dataset records
  a SHA-256 of the corpus. Edit a document without updating the labels and the tests fail.
- **Provenance:** all labels are `claude_authored` and `human_verified: false`. They have **not** been reviewed by a
  human. Review them with `--show`, fix any you disagree with, and set `human_verified` to `true` for those you accept.

### Limitations

- 24 questions is a **smoke-test set**, not a benchmark: per-category metrics rest on 2-4 questions, so
  differences between pipeline variants will often be within noise. Do not draw strong conclusions from it.
- The corpus is tiny (788 words, 7 documents), so retrieval is easy; results will not predict performance on a
  large real corpus. A larger set is planned before drawing ablation conclusions.
- Corpus and questions were written by the same author (an LLM), so lexical overlap between question and source
  may be unrealistically high or low in places.

## Hybrid retrieval (Milestone 3)

`RAG_RETRIEVER=hybrid` fuses dense cosine and our own BM25 with Reciprocal Rank Fusion. Benchmark and compare:

```
uv run python -m app.evaluation.benchmark --label dense  --retriever dense  --embedder hashing
uv run python -m app.evaluation.benchmark --label hybrid --retriever hybrid --embedder hashing
uv run python -m app.evaluation.compare results/<dense>.json results/<hybrid>.json
```

Reference runs (stand-in hashing embedder, 24 Claude-authored questions) are in `docs/results/m3-*`. The difference is
about one question and is not a claim that hybrid beats dense; see decision D21. Tests: 272 passing offline (M8 Ragas runner: D26; M7 semantic chunking: D25; M4 reranking + context: D22; M5 validation, abstention, version conflicts: D23; M6 lifecycle: D24). New endpoints: `DELETE /documents/{doc_id}`, `GET /audit`.

## Ragas evaluation (Milestone 8)

```
uv sync --extra eval
# answer model + judge (any OpenAI-compatible endpoint; default is local Ollama)
RAG_LLM_MODEL=llama3.2:3b RAG_JUDGE_MODEL=llama3.1:8b \
  uv run python -m app.evaluation.ragas_runner --label hybrid-llama --retriever hybrid \
  --judge-embedding-model nomic-embed-text
```
No Ragas scores are committed: producing them needs an LLM that this repo's CI/sandbox does not have. See decision D26 for
what is and is not verified. Deterministic retrieval metrics never depend on Ragas.
