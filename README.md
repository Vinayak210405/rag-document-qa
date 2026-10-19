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
| 2 | Evaluation: Recall@k, MRR, nDCG, benchmark runner | Not started |
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
uv run pytest            # 176 tests, fully offline, ~15 s
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

## Configuration

All settings use the `RAG_` prefix; see `.env.example`. Never commit `.env`. Chunk sizes are in characters.

## Verification status (Milestone 1)

| Verified here | Not verified here |
|---|---|
| 176 automated tests, incl. real ChromaDB on disk, PDF (hand-built), DOCX, TXT parsing | The real `bge-small` model (no Hugging Face access in the build sandbox): only the wrapper is tested, with a stub; its API names were checked against the sentence-transformers 6.1.0 wheel source. Run `uv run pytest -m requires_model` |
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
