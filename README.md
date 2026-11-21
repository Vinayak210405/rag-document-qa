# Evaluated Document QA Bot with RAG

An evaluation-driven retrieval-augmented generation system for question answering over private documents, with
verifiable citations and *measured* retrieval and answer quality. The retrieval core (chunking, cosine top-k, BM25, RRF,
rerank orchestration, context assembly, citation validation, metrics, benchmark runner, document lifecycle) is written in this
repo rather than hidden behind an orchestration framework. Design reasoning: [docs/decisions.md](docs/decisions.md) (D1-D26),
architecture: [docs/architecture.md](docs/architecture.md), threat model: [docs/security.md](docs/security.md).

> **Read this first: what has and has not been measured.** Every benchmark number in this repo comes from an *offline stand-in*
> (a bag-of-words hashing embedder, and for reranking a lexical stand-in) on a 24-question, Claude-authored, not human-verified
> set over 7 short documents. They prove the harness works and show how to compare configurations; they say nothing about
> how the system performs with real models. No cross-encoder, `bge`, LLM-judged (Ragas) or end-to-end LLM numbers exist yet,
> because the build sandbox has no model downloads or LLM. Commands to produce them are below.

## Status

| Milestone | Scope | Status |
|---|---|---|
| 0 | Foundation: structure, config, sample corpus, labelled eval set | Done |
| 1 | Baseline RAG: ingestion, sentence chunking, embeddings, ChromaDB, own cosine top-k, grounded answers, API + UI | Done, tested offline |
| 2 | Metrics (Hit/Recall/MRR/nDCG), benchmark runner, run comparison, random control | Done, tested offline |
| 3 | Own BM25 + Reciprocal Rank Fusion, hybrid retriever | Done, tested offline |
| 4 | Reranker orchestration + cross-encoder wrapper, context assembly (dedupe, per-doc cap, token budget) | Done; cross-encoder itself **not run** here |
| 5 | Citation validation, abstention, evidence gate, version-conflict handling | Done, tested offline; LLM behaviour **not run** here |
| 6 | Document lifecycle: hashes, incremental ingest, versions, safe update order, delete, audit | Done, tested offline |
| 7 | Semantic chunker + chunking ablation | Done, tested offline |
| 8 | Ragas runner (configurable judge) | Done with fake scorers; **no live run, no Ragas scores** |
| 9 | File validation, injection safeguards, access control, safe errors, benchmark dashboard, docs | Done, tested offline |
| 10 | LlamaIndex comparison over the same Chroma index | Done: retrieval over our Chroma index matches ours (D28) |

Tests: run `uv run pytest`. Everything runs offline in ~15 s; two opt-in tests (`-m requires_model`, `-m requires_api`) need a model
download / a running LLM.

## Architecture

```
OFFLINE  file -> validate bytes -> parse (txt/pdf/docx) -> blocks (heading/body, page) + header metadata
              -> chunker (sentence | semantic) -> embed -> ChromaDB (vectors + text + provenance + metadata + ACL groups)
ONLINE   question -> validate -> [ACL filter] dense cosine top-k (ours)  +  BM25 top-k (ours)  -> RRF
              -> optional cross-encoder rerank -> context assembly (dedupe, per-doc cap, token budget)
              -> safeguards (sanitise + injection scan) -> evidence gate -> grounded prompt (+ version warning) -> LLM
              -> citation validation -> status: answered | abstained | abstained_no_evidence | unverified | withheld
UI       Streamlit --HTTP--> FastAPI --> RagService --> modules above
EVAL     dataset (quote-based labels) -> benchmark runner -> JSON/CSV  |  Ragas runner (LLM judge) -> JSON
```

Code map: `app/ingestion` | `app/chunking` | `app/embeddings` | `app/indexing` | `app/retrieval` (similarity, BM25, fusion, hybrid,
rerank, context, access) | `app/generation` (prompt, LLM client, citations, evidence, conflicts, safety) | `app/evaluation` |
`app/security.py` | `app/service.py` | `app/api` | `frontend/`.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync                  # core dependencies + pytest (no PyTorch)
uv run pytest            # offline
```
Extras: `--extra embeddings` (PyTorch, sentence-transformers: real embeddings and the cross-encoder), `--extra eval` (Ragas),
`--extra llamaindex` (comparison).

**Windows tip:** if your system Python is broken or missing, make uv use its own:
`[Environment]::SetEnvironmentVariable("UV_PYTHON_PREFERENCE","only-managed","User")`, then reopen PowerShell.

### Offline demo (no downloads, no LLM)

```powershell
$env:RAG_EMBEDDING_BACKEND = "hashing"      # non-semantic stand-in embedder
uv run uvicorn app.api.main:app             # terminal 1: http://127.0.0.1:8000  (interactive docs at /docs)
uv run streamlit run frontend/app.py        # terminal 2: http://localhost:8501
```
Upload files from `data/sample_corpus/`, tick "Retrieve only" to inspect evidence, and open the **Benchmarks** tab.

### Real demo (semantic embeddings + local LLM)

1. `uv sync --extra embeddings` (the ~130 MB `bge-small` model downloads on first use).
2. Install [Ollama](https://ollama.com), then `ollama pull llama3.2:3b`.
3. Copy `.env.example` to `.env`; set e.g. `RAG_RETRIEVER=hybrid`. Start the API and UI as above, without the hashing override.

An index is tied to the embedding model and chunker that built it: switching the model raises `IndexMismatchError` (use a new
`RAG_STORAGE_DIR`); changing the chunker or ACL groups re-indexes identical files automatically.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, counts, embedding model (always open) |
| `POST /documents` | Upload PDF/DOCX/TXT (multipart `file`, optional `groups`). Identical re-upload: `status: unchanged`; changed: `updated` with `ingest_version` + 1. Admin |
| `GET /documents` | Indexed documents (restricted ones hidden from callers outside their groups) |
| `DELETE /documents/{doc_id}` | Remove a document; dense and lexical views update together. Admin |
| `GET /audit` | Stale chunks, interrupted updates, duplicate positions/texts. Admin |
| `POST /retrieve` | `{"question","top_k"}`: ranked passages only, no LLM |
| `POST /ask` | Answer + `status` + citations + sources + version conflicts + injection flags + timings + token usage |

Errors: 401/403 auth, 404 unknown document, 409 nothing indexed or model mismatch, 413 too large, 415 unsupported type, 422 invalid
or empty/scanned/fake file or question, 500 generic (details only in server logs), 502 LLM unreachable (never a made-up
answer), 503 backend unavailable. Authentication is off unless `RAG_AUTH_TOKENS` is set (see [docs/security.md](docs/security.md)).

## Evaluation

```powershell
uv run python -m app.evaluation.benchmark --label dense  --retriever dense  --embedder hashing
uv run python -m app.evaluation.benchmark --label hybrid --retriever hybrid --embedder hashing
uv run python -m app.evaluation.benchmark --label rr --retriever reranked --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 --embedder sentence_transformers
uv run python -m app.evaluation.benchmark --label sem --retriever hybrid --chunking semantic --embedder sentence_transformers
uv run python -m app.evaluation.compare results\a.json results\b.json      # deltas + better/worse/same question counts
uv run python -m app.evaluation.abstention --embedder hashing                 # evidence-gate measurement (no LLM)
```
Metrics: Hit@k, Recall@k over *required* evidence, MRR (truncated at retrieval depth), graded nDCG@k judged by content over the
whole index; every run records a random-ranking control, tie/no-signal flags, outdated-source-above-gold rate, latency and a
`results_sha256` (same config + dataset reproduces it). Retrieval metrics never depend on Ragas.

### Ablation A-D on the same 24 questions (stand-in embedder; NOT semantic; `docs/results/`)

| Configuration | MRR | Hit@1 | Hit@5 | Recall@5 | nDCG@5 | chunks |
|---|---|---|---|---|---|---|
| A. dense only | 0.778 | 0.667 | 0.889 | 0.861 | 0.785 | 12 |
| (BM25 only) | 0.819 | 0.722 | 0.944 | 0.944 | 0.848 | 12 |
| B. dense + BM25 + RRF | 0.796 | 0.667 | 0.944 | 0.917 | 0.810 | 12 |
| C. hybrid + lexical STAND-IN reranker | 0.769 | 0.611 | 0.944 | 0.917 | 0.800 | 12 |
| D. hybrid, semantic chunking | 0.741 | 0.611 | 0.889 | 0.861 | 0.756 | 31 |

LlamaIndex reading the same Chroma collection (`--retriever llamaindex`, `docs/results/m10-*`) matched our dense retriever on all 18
questions (same scores at every rank; ties may order differently), at 2.4 ms vs 0.19 ms mean latency on this tiny index (D28).

What this does and does not show:
- On this setup **BM25 alone beat the hybrid**, and the lexical stand-in reranker and semantic chunking both **regressed** versus
  hybrid. Reporting those regressions is the point: a more complex pipeline is not assumed to be better.
- The embedder is bag-of-words, so "dense" here is itself lexical; BM25 + RRF may help or hurt very differently with `bge`.
  The reranker is a stand-in, not a cross-encoder. Semantic chunking cost ~10x the embeddings at ingest (D25).
- 12-31 chunks, 18 answerable questions: one question moves a metric by ~0.05. These are smoke-test numbers.
- **Evidence gate** (`--embedder hashing`, D23): no lexical-coverage threshold separates answerable from unanswerable questions
  (0.6 refuses 4/6 must-refuse but also 4/18 answerable), so the gate ships **off** by default.

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

## Ragas and end-to-end evaluation (Milestone 8)

```powershell
uv sync --extra eval
$env:RAG_LLM_MODEL="llama3.2:3b"; $env:RAG_JUDGE_MODEL="llama3.1:8b"
uv run python -m app.evaluation.ragas_runner --label hybrid-llama --retriever hybrid --judge-embedding-model nomic-embed-text
```
Answers every question through the real pipeline, scores faithfulness, answer relevancy, context precision and context recall with
a configurable judge (any OpenAI-compatible endpoint; `RAG_JUDGE_*`), and scores abstention on must-refuse questions
deterministically. **Not yet run**: it needs an LLM. Judge scores are not ground truth and vary with the judge model.

## Security (Milestone 9)

Upload validation (magic bytes, zip-bomb limits, page cap, size cap), prompt-injection heuristics + delimiting + sanitising,
group-based access control enforced *before* ranking, optional bearer-token auth, safe 500s, no secrets in Git. These reduce risk;
they do not make processing untrusted documents safe. Details and limits: [docs/security.md](docs/security.md).

## Configuration

All settings use the `RAG_` prefix; see `.env.example`. Never commit `.env`. Sizes are in characters, token counts are estimates
(chars / 4). Key options: `RAG_RETRIEVER` (dense|hybrid|reranked), `RAG_CHUNKING` (sentence|semantic), `RAG_RERANKER_MODEL`,
`RAG_CONTEXT_TOKEN_BUDGET`, `RAG_MIN_EVIDENCE_COVERAGE`, `RAG_STRICT_CITATIONS`, `RAG_DROP_FLAGGED_PASSAGES`, `RAG_AUTH_TOKENS`.

## Known limitations

- Citation checks are structural (numbers exist, sentences cited); they do not prove a source *supports* a claim. Not claimed to
  eliminate hallucination.
- Brute-force exact search and an in-memory BM25 rebuilt per index change: fine to ~1e5 chunks, not beyond. Single-writer assumed.
- Scanned PDFs need OCR (extension point exists; no engine bundled). Token budgets are estimates.
- Access control and token auth are application-level and basic (static tokens, no expiry); see docs/security.md.
- No human-verified labels yet; the 24-question set is a smoke test (see Dataset limitations above).

## Interview prompts this repo supports

Why RRF instead of score fusion? Why filter by ACL before top-k? What does nDCG with a random control tell you on a 12-chunk
index? Why can an LLM judge not be treated as ground truth? What fails when the update path crashes between upsert and delete?
