# Claims map: what the project description may say, and when

Rule: a claim goes on a CV only when the milestone behind it is **implemented, tested, and (for any performance
claim) measured**. Numbers in brackets are filled in only from saved benchmark output in `results/`.

| Phrase in the description | Built in | Status | Evidence required before claiming |
|---|---|---|---|
| "RAG QA system" over "private document repositories" | M1 | Not started | Ingest PDF/DOCX/TXT, answer with sources, tests passing |
| "Chroma vector database" | M1 | Not started | Persistent store, restart-safe (integration test) |
| "Inline source citation" | M1 basic, M5 validated | Not started | Citations checked against retrieved chunks; citation-correctness metric |
| "Hybrid search" | M3 | Not started | BM25 + RRF **measured against dense-only** on the same set |
| "Re-ranking" | M4 | Not started | Cross-encoder **measured against M3**, incl. latency cost |
| "Semantic chunking" | M7 | Not started | Compared with sentence-aware chunking; trade-offs reported |
| "Optimize retrieval precision" | M2-M4 | Not started | Recall@k / MRR / nDCG before-and-after. If no gain, do not claim it |
| "Minimize hallucinations" | M5, M8 | **Do not claim as worded** | Reword to what is measured: faithfulness score, abstention rate on unanswerable questions |
| "Ragas: faithfulness, answer relevance, context recall" | M8 | Not started | Runner saves Ragas scores with the judge model recorded |
| "Automated QA test sets" | M8 | **Partly false today** | Today's set is hand-structured (24 Qs). M8 adds a Ragas-generated set, labelled `synthetic_llm` and reported separately |
| "Using LlamaIndex" | M10 | Not started | A LlamaIndex pipeline over the same Chroma index, benchmarked on the same questions as our own pipeline |
| "Enterprise" | n/a | Avoid | A 7-document corpus does not support the word; prefer "private document collections" |

## Wording template (fill `[X]` from measured results only)

- Built a document QA system over PDF/DOCX/TXT collections using ChromaDB, with custom hybrid retrieval (dense + BM25 fused by
  Reciprocal Rank Fusion), cross-encoder reranking, and citations validated against retrieved chunks.
- Compared sentence-aware and semantic chunking and measured each pipeline stage on a labelled [N]-question set (including
  unanswerable and conflicting-source questions): Recall@5 [X to Y], MRR [X to Y]; benchmarked against a LlamaIndex implementation.
- Built a reproducible benchmark runner (retrieval metrics, citation correctness, latency, Ragas faithfulness / answer relevancy /
  context recall) with abstention measured at [X]% on unanswerable questions.
