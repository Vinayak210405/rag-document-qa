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

## Amendments (Milestone 0 revision)

### D2a. LlamaIndex is kept, as a measured comparison (M10), not as the core
The project description names LlamaIndex, so we make that claim true without letting a framework hide the
retrieval logic. The core stays ours; M10 builds a LlamaIndex pipeline over the **same Chroma index** and benchmarks
it on the **same evaluation set**. The interview story becomes "built it from scratch, then benchmarked it against the
framework". Compatibility verified: `llama-index-core 0.14.25` + `llama-index-vector-stores-chroma 0.6.0` +
`chromadb 1.5.9` resolve together (`uv lock`).

### D5a. Baseline dependencies moved into core
After M0, the packages Milestone 1 needs (pypdf, python-docx, chromadb, FastAPI, uvicorn, python-multipart, Streamlit,
httpx) are core dependencies so a plain `uv sync` yields a working system (important for Windows users who would
otherwise have to remember `--extra` flags on every command). Extras remain only for genuinely heavy or optional
pieces: `embeddings` (PyTorch), `bm25`, `eval` (Ragas), `llamaindex`. The default `pytest` run still needs no network.

### D8. Claims are earned, not asserted
`docs/claims.md` maps every phrase of the project description to the milestone and the measurement that earns it.
The dataset schema gained a `synthetic_llm` label source so the Ragas-generated "automated QA test set" (M8) can never
be confused with authored or human-verified labels.

## Milestone 1 decisions

### D9. Chunk sizes are in characters
Model-independent and needs no tokenizer; ~4 characters is about 1 English token. Defaults: 500 chars, 80 overlap
(`RAG_CHUNK_MAX_CHARS`, `RAG_CHUNK_OVERLAP_CHARS`). Token budgets arrive with context assembly (M4).

### D10. Chroma stores; our code ranks
ChromaDB is the durable store for (id, text, vector, metadata). Ranking is our own NumPy cosine similarity and top-k
over the full matrix (exact, brute-force, O(n*d) per query). Transparent and unit-tested; fine to ~1e5 chunks.
Beyond that, Chroma's approximate HNSW search becomes the thing to benchmark against this baseline.
Verified on chromadb 1.5.9: reopening a collection with different metadata silently keeps the old metadata, so the
embedding-model guard (`IndexMismatchError`) is our own explicit check.

### D11. Streamlit calls FastAPI over HTTP; endpoints are plain `def`
The UI is a real client of the API (as in deployment) and only the API process touches Chroma (two writers on one
persistent store is unsafe). Embedding and the LLM call are blocking, so endpoints are `def`: FastAPI runs them in a
thread pool and the event loop stays free. `async def` would have blocked it.

### D12. An offline, non-semantic `hashing` embedder
A deterministic bag-of-words (feature hashing) embedder: no download, no PyTorch. It makes tests reproducible and lets
the app run offline. It is NOT semantic and has no IDF weighting; do not judge retrieval quality with it.
Observed in the live demo: adding two tiny unrelated PDF chunks pushed the right chunk for `ERR-5043` from #1 to #2,
because the generic word "code" counted as much as the rare identifier and short chunks score high under cosine.
That is the motivation for BM25 (M3) and for measuring (M2) instead of eyeballing.

### D13. Re-ingesting a file replaces it
`replace_document` deletes the document's older chunks, then adds the new ones, so edits never leave stale chunks.
It is not atomic and does not skip unchanged files; hashing, versioning and atomicity are Milestone 6.

### D14. Grounding is requested, not guaranteed
The prompt asks for sources-only answers, citations, abstention, and distrust of source text (angle brackets inside
sources are escaped so a document cannot close the `<source>` tag). The code only checks that cited numbers refer to
sources that were actually supplied. Whether claims are supported, and whether abstention happens when it should, are
measured later (M5, M8), not assumed.

### Known limitations after M1
- A chunk spanning several sections records only its first heading.
- Very short chunks (e.g. a 30-character PDF page) are indexed as-is and can rank oddly; context selection (M4) may
  need a minimum size or merging.
- Top-k always returns k results, even with near-zero scores; no relevance threshold yet.
- Heading detection is heuristic for TXT/PDF (numbered, ALL CAPS, first line); DOCX uses real styles. PDF layout, tables
  and multi-column pages are not analysed. Scanned PDFs are rejected with a clear message (OCR hook only).
- No authentication or per-document permissions yet (M9). The upload size cap and file-name sanitising are minimal.
- `starlette.testclient` emits a deprecation warning when used with httpx; harmless today, may need `httpx2` later.

## Milestone 2 decisions

### D15. Relevance is defined by content, so every chunk can be judged
A chunk's grade is the highest grade among gold quotes it fully contains (and only quotes from its own document).
Because grading needs only the chunk text, we can grade **every chunk in the index**, so nDCG's ideal ranking is exact
instead of assumed. A quote cut in half by a chunk boundary is contained by neither chunk; the `reachable` diagnostic
reports that loss so a bad chunker cannot hide behind "retrieval failed".

### D16. Metric definitions (`docs/` and every result file repeat them)
Relevant = contains required (grade 2) evidence. Hit@k = any relevant chunk in the top k. Recall@k = share of required
quotes covered by the top k (matters for multi-hop). RR = 1 / rank of the first relevant chunk; MRR is its mean.
nDCG@k uses gains 2^grade - 1 (grade 2 is worth 3x grade 1). **MRR is computed over the retrieved depth, max(ks)**, so
it is really MRR@max-k (found out when a test expectation disagreed with the code); results label it that way.
Unanswerable questions have no gold evidence: they are excluded from these averages and reported separately.

### D17. Each run builds a fresh in-memory index with a unique collection name
Verified on chromadb 1.5.9: in-memory clients **share state within a process** when the collection name matches, so a
second run could silently inherit the first run's chunks. A uuid collection name per run plus `drop()` prevents it. An
in-memory store also avoids the Windows file-locking problems that temp-directory SQLite files cause.

### D18. Deterministic results are separated from non-deterministic ones
`config`, `dataset`, `index`, `metrics`, `questions` are deterministic; timestamps, git state, environment and all
latencies are not and live in other keys. `results_sha256` covers only the deterministic part, so "same configuration +
same dataset" is checkable by comparing one string. Numbers are rounded to 6 decimals first. Same machine: identical hash.
Across machines it should match, but tiny floating-point differences could in principle change it; compare metrics too.
The benchmark refuses to run if the dataset no longer matches the corpus (label rot).

### D19. Every result carries controls and diagnostics, because averages can mislead
- **Random-ranking control**: expected metrics of a uniformly random ranking (seeded Monte-Carlo, same evaluator code).
  With a 12-chunk index, random already gets hit@5 of about 0.46, so a result means something only relative to it.
- **Tie flags**: `tie_with_gold` (the first gold chunk has the same score as a non-gold chunk) and `no_signal` (top score
  is 0). In both cases the rank is decided by the tie-break rule, not by the retriever.
- **Outdated-above-gold rate** for questions with a superseded distractor (the conflicting-documents failure mode).
- **Comparison tool** reports, next to every metric delta, how many questions got better / worse / stayed the same.

### D20. Tie-break rule and its side effect
Exact score ties resolve to the lower index; the index is ordered by (document id, chunk position). Since `v1` sorts
before `v2`, a tie between the two leave-policy versions silently favours the **outdated** file. This is deterministic
and documented, not hidden. A lexical embedder cannot tell "20 days" from "24 days", so such ties are real; fixing them
(recency/status metadata, reranking) is Milestone 4-5 work, and the tie flags measure how often it matters.

### Known dataset issues found while benchmarking (not yet fixed, so v0.1.0 stays valid)
- Paraphrase questions q006 (names "AeroLink X200") and q007 (shares the word "second") overlap with their source in
  distinctive keywords, so they are not pure tests of semantic matching. Suggest rewriting during label review.
- 24 questions over 12 chunks: per-category numbers rest on 2-4 questions each. Treat them as indicative only.

### D21. BM25 and RRF are our own code (Milestone 3)
- **BM25**: Okapi with the Lucene idf `ln(1 + (N-n+0.5)/(n+0.5))`, k1=1.5, b=0.75. Unit-tested against a hand computation.
  Tokeniser keeps identifiers whole *and* splits them (`ZX-4471` -> `zx-4471`, `zx`, `4471`). No stemming/stop words:
  untested language-specific choices. The index is rebuilt from the same store snapshot as the dense retriever (keyed on
  store version), so the two views cannot disagree about which chunks exist; fine at this scale, a persistent
  incremental lexical index is Milestone 6.
- **RRF**: `sum 1/(k+rank)`, k=60, rank-only so incomparable score scales fuse safely. Deterministic tie-break
  (best rank, then first seen). BM25 chunks with score 0 are excluded from the lexical list so they cannot vote.
- **Result**: on the 24-question set with the *stand-in hashing embedder*, hybrid vs dense: MRR 0.778 -> 0.796,
  Recall@5 0.861 -> 0.917, Hit@1 unchanged, Recall@3 and nDCG@3 each got worse on one question. That is a 1-question
  effect on a small, Claude-authored set; it is **not** evidence that hybrid wins with a real embedder. Re-run with
  `--embedder sentence_transformers` (needs the model download) before drawing conclusions.

### D22. Reranking and context assembly (Milestone 4)
- `RerankedRetriever(base, reranker, candidates)` re-scores the first-stage pool; a failing/unavailable reranker raises
  `RerankerError` (no silent fallback to the first-stage order). `CrossEncoderReranker` wraps a pretrained
  sentence-transformers CrossEncoder (needs a download; **not run in this sandbox**, so no cross-encoder numbers exist yet).
- `LexicalOverlapReranker` is an offline stand-in used only to test orchestration. Measured result with the hashing
  embedder: it *hurt* the top ranks (MRR 0.796 -> 0.769, Hit@1 0.667 -> 0.611, one question worse, none better at rank 1).
  This is a stand-in result, not evidence about cross-encoders; re-run with `--reranker cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Context assembly: near-duplicate removal (token Jaccard >= 0.85), per-document cap, token budget. Token counts are
  chars/4 estimates, not the LLM tokenizer. The best passage is always kept.
- Latency/cost of reranking on real models is unmeasured here.

### D23. Citation validation, evidence gate, version conflicts (Milestone 5)
- **Answer status** (`answered | abstained | abstained_no_evidence | unverified | withheld`) is computed from deterministic
  checks: citation numbers must exist in the supplied sources; every sentence of 4+ words needs a `[n]`. `unverified`
  answers are still shown but flagged; `RAG_STRICT_CITATIONS=true` withholds them. These checks are STRUCTURAL: they
  do not show that a cited source supports the claim (that needs an entailment/LLM judge; Ragas faithfulness, M8).
- **Evidence gate** (`RAG_MIN_EVIDENCE_COVERAGE`, default 0 = off): lexical coverage of question terms by the retrieved
  passages; below threshold the service refuses without calling the LLM (fixed abstain phrase, `llm_model="(not called)"`).
  Measured on the 24 questions (`docs/results/m5-evidence-gate-hashing.txt`): no threshold separates the classes.
  Threshold 0.6 refuses 4/6 must-refuse questions but also wrongly refuses 4/18 answerable ones; 0.9 catches 6/6 but
  wrongly refuses 16/18. The cheap lexical gate is therefore left OFF by default. A paraphrase-robust signal (reranker
  score / dense similarity with a real embedder) is the next thing to try and is unmeasured here. Thresholds picked on
  this set would be tuned on their own test data.
- **Version conflicts**: header metadata (`Document ID`, `Version`, `Effective date`, `Status`) is extracted at ingest,
  stored on every chunk, and shown to the model; if retrieved passages span several files of one `Document ID`, the
  prompt gets an explicit version warning and the API lists the conflict. Whether the LLM then picks the current version
  is NOT verified without a real model.
- **Bug caught by the corpus test**: text normalisation merges header lines into one paragraph, so line-based header
  parsing silently found one field; the extractor now matches `Key:` markers, with a regression test on the real files.

### D24. Document lifecycle (Milestone 6)
- **Identity**: `doc_id` = sanitised file name (stable across uploads); `content_hash` = sha256 of the bytes;
  `ingest_version` = 1 on first ingest, +1 whenever the content or the index settings change. Stored on every chunk.
- **Incremental ingestion**: identical bytes AND identical settings (`chunker|embedding model` signature, ACL groups) are
  skipped: the file is parsed and hashed but nothing is re-chunked or re-embedded (`status: unchanged`). A settings change
  (e.g. a different chunker) forces a rebuild of identical bytes.
- **Update safety**: new chunks are upserted first (IDs are content-derived, so unchanged chunks are overwritten in place),
  then chunks no longer in the document are deleted. An interruption can leave stale extras (found by `GET /audit`) but
  never missing content. This is ordering, not a transaction: Chroma has none.
- **Lexical/dense sync**: both views are built from the same store snapshot, keyed on a write counter, so updates and
  deletes reach BM25 and dense retrieval together (tested). Limitation: the counter is per process; a second process
  writing to the same on-disk store would not invalidate this process's cache. Single-writer deployment assumed.
- **Audit**: reports stale/corrupt chunks (ID != sha256(doc, ordinal, text)), documents with mixed content hashes
  (interrupted update), duplicate positions, and identical text shared by different documents (informational).
- Not implemented: keeping old versions queryable, soft deletes, or a persistent incremental BM25 index.

### D25. Semantic chunking and the chunking ablation (Milestone 7)
- `SemanticChunker` (same `Chunker` interface): embeds each sentence, cuts where adjacent-sentence cosine distance exceeds the
  page's p-th percentile or a heading starts, merges fragments under `min_chars`, packs oversized groups. Costs one embedding per
  sentence at ingest, counted as `embedded_texts` in benchmark output.
- **Ablation D** (hybrid retrieval, hashing stand-in embedder, 24 questions, `docs/results/m7-*`): sentence chunking 12 chunks,
  MRR 0.796, Hit@5 0.944, Recall@5 0.917; semantic (p=70 and p=90 gave identical splits on this tiny corpus) 31 chunks (mean 159
  vs 424 chars), MRR 0.741, Hit@5 0.889, Recall@5 0.861. Embedded texts 119 vs 12 (~10x), ingest time ~2x. So on this setup
  semantic chunking was **worse and costlier**. That does not generalise: the corpus is 7 short documents (a handful of
  sentences each, so there is little topic structure to find), the embedder is lexical not semantic, and the labels are small and
  Claude-authored. It does show why "semantic is better" must be measured. Re-run with `--embedder sentence_transformers`
  on a larger corpus before drawing any conclusion.
