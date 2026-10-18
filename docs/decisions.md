# Design decisions

A running log. Each entry: what we chose, why, and what we gave up.

## D1. Evidence is labelled by verbatim quote, not chunk ID
**Chosen:** each question stores `(document, exact quote, grade)` evidence. At benchmark time (M2) a retrieved
chunk is relevant if it contains a gold quote.
**Why:** chunk IDs depend on the chunker. Labelling against one chunker's IDs makes every other chunker
(M7) unmeasurable. Quotes are chunker-independent.
**Cost:** slightly more work in the metric code (quote-in-chunk matching, and a chunk that straddles a
quote boundary needs a rule). Quotes must be exact and unique, enforced by `validate_dataset`.

## D2. The core is framework-independent
**Chosen:** chunking, cosine/top-k, BM25 wiring, RRF, reranking orchestration, context assembly, citation
validation and metrics are our own modules. Libraries supply parsing, embeddings, ChromaDB, the LLM, and Ragas.
**Why:** every algorithm in the retrieval path must be explainable and unit-testable in isolation.
**Cost:** more code than calling a framework. LlamaIndex may return later as an optional comparison.

## D3. Ordinary tests are offline
**Chosen:** the default `pytest` run needs no network, API key or model download. Tests that do are marked
`requires_model` / `requires_api` and are excluded by default (`addopts` in `pyproject.toml`).
**Why:** fast, deterministic CI, and a project anyone can clone and verify in seconds.

## D4. Sample corpus lives in `data/sample_corpus/`, beside the eval set
The suggested layout had only `data/eval/`. The corpus is the dataset's ground truth, so it is committed
(it is tiny and fictional) and **versioned by hash** alongside the labels. User documents go in the
git-ignored `data/raw/`.

## D5. Dependencies: extras + lockfile, resolved up front
Core dependencies are minimal; heavy packages (ChromaDB, PyTorch via sentence-transformers, Ragas, FastAPI,
Streamlit) are optional extras. `uv.lock` resolves **all** extras together, so incompatibilities surface now
instead of at the milestone that first needs them. Resolved set (M0): 181 packages, no conflicts.
Note: Ragas pulls in `openai` and `langchain-core` transitively.

## D6. Settings are added per milestone
Only settings that existing code needs are defined. Chunk size and top-k are deferred to M1, where we
decide their units (characters vs. tokens) deliberately.

## D7. Label provenance is explicit
All 24 questions are `label_source: claude_authored`, `human_verified: false`. A test enforces that
Claude-authored labels cannot claim human verification. Reviewing them and flipping the flag is a task for the
project owner (see README).
