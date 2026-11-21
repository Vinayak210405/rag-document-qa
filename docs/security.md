# Security notes and threat model

Scope: a single-service RAG app over documents that may be **untrusted** (uploaded files, text inside them) and queried by users
who may have **different permissions**. These are safeguards, not guarantees.

| Threat | Mitigation | Tested | Limit |
|---|---|---|---|
| File extension lies (renamed exe/zip) | magic-byte / structure check before parsing (`ingestion/validation.py`) | yes | does not make the PDF/DOCX parsers safe against parser exploits; sandbox ingestion for hostile input |
| Zip bomb / huge PDF | entry count, uncompressed size and ratio limits, `RAG_MAX_PDF_PAGES`, `RAG_MAX_UPLOAD_BYTES` | yes | limits are heuristics |
| Prompt injection inside documents | sanitise invisible/bidi chars, delimiter + escape of `<`/`>`, "sources are data" system rule, heuristic scan with flagging (or dropping), UI warning | yes (heuristics) | paraphrases, other languages and encodings bypass heuristics; the model can still be swayed. Least privilege helps most: the model has no tools |
| Reading documents you should not | `groups` per document, filtered **before** ranking; listings hide restricted names; restricted text never reaches the prompt | yes (all retrievers, API, LLM prompt) | only as strong as authentication; no row-level audit log |
| Unauthenticated access | optional bearer tokens (`RAG_AUTH_TOKENS`), constant-time compare, admin-only ingest/delete/audit | yes | static shared secrets: no expiry/rotation/TLS; use a reverse proxy + real IdP for production. **Auth is off by default** (localhost) |
| Information leakage in errors | generic 500 body, details only in logs; secrets are `SecretStr` | yes | logs may contain user questions |
| Secrets in Git | `.gitignore` for `.env`, storage, results; `.env.example` only | manual | rotate any key that was ever committed |
| Hallucinated / unsupported answers | structural citation validation, optional withholding, abstention prompt | yes (structure) | support of claims by sources is **not** verified |

Not covered: rate limiting, malware scanning, per-user quotas, encryption at rest, multi-tenant isolation beyond group filters.
