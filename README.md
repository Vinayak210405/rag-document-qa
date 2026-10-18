# Evaluated Document QA Bot with RAG

An evaluation-driven retrieval-augmented generation system for question answering over private documents,
with verifiable citations and measured retrieval/generation quality. Built incrementally; each milestone has a
verification path. The retrieval core is implemented in this repo, not hidden behind an orchestration framework
(see [docs/decisions.md](docs/decisions.md)).

## Status

| Milestone | Scope | Status |
|---|---|---|
| 0 | Foundation: structure, config, sample corpus, labelled eval set, offline tests | **Done** |
| 1 | Baseline RAG: ingestion, chunking, embeddings, ChromaDB, own cosine top-k, grounded answers, API + UI | Not started |
| 2 | Evaluation: Recall@k, MRR, nDCG, benchmark runner | Not started |
| 3 | Hybrid retrieval: BM25 + RRF | Not started |
| 4 | Cross-encoder reranking, context assembly | Not started |
| 5 | Citation validation, abstention | Not started |
| 6 | Document lifecycle: incremental ingest, update, delete | Not started |
| 7 | Semantic chunking | Not started |
| 8 | Ragas metrics | Not started |
| 9 | Security, benchmark dashboard, demo docs | Not started |
| 10 | LlamaIndex pipeline over the same Chroma index, benchmarked against ours | Not started |

No benchmark results exist yet; none are claimed.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # core dependencies + pytest (no PyTorch)
cp .env.example .env         # optional for now; nothing in M0 needs it
uv run pytest                # 25 tests, fully offline, < 1 second
uv run python -m app.evaluation.dataset --show   # inspect the eval dataset
```

Heavy or optional dependencies are extras: `embeddings` (PyTorch, for real embedding models), `bm25`, `eval` (Ragas),
`llamaindex`. `uv.lock` resolves all extras together, so incompatibilities surface early.
See [docs/claims.md](docs/claims.md) for which claims each milestone earns.

## Repository layout

```
app/            config, ingestion, chunking, embeddings, indexing, retrieval, generation, evaluation, api, utils
frontend/       Streamlit UI (M1+)
tests/          unit/, integration/, fixtures/
data/
  sample_corpus/  7 fictional documents (committed, versioned by hash)
  eval/           labelled evaluation dataset (JSON)
  raw/            your own documents (git-ignored)
docs/           decisions.md (design log)
```

Generated state (`storage/`, `results/`) and secrets (`.env`) are git-ignored.

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

## Configuration

All settings use the `RAG_` prefix (see `.env.example`). Never commit `.env`.
