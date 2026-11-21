"""Extract document-level metadata from a plain header block (Document ID / Version / Effective date / Status).

Only the first ~600 characters are inspected and only the four known keys are used, so text further down the
document cannot inject metadata. Missing fields are simply absent. Values are strings.
"""

from __future__ import annotations

import re

_HEADER_CHARS = 600  # only the top of the document is inspected
_FIELDS = {
    "document id": "doc_family",
    "version": "doc_version",
    "effective date": "effective_date",
    "status": "doc_status",
}
# Header fields may sit on separate lines or be merged into one paragraph by text normalisation, so match
# "Key:" markers and take the value up to the next marker / line break.
_KEY = re.compile(r"(?i)\b(document id|version|effective date|status)\s*:\s*")


def classify_status(status: str) -> str:
    s = status.strip().lower()
    if s.startswith(("superseded", "obsolete", "withdrawn")):
        return "superseded"
    if s.startswith(("current", "active", "in force")):
        return "current"
    return "unknown"


def extract_document_metadata(texts: list[str]) -> dict[str, str]:
    """``texts`` are the document's block texts in order."""
    head = "\n".join(texts)[:_HEADER_CHARS]
    matches = list(_KEY.finditer(head))
    meta: dict[str, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(head)
        value = head[m.end():end].split("\n")[0].strip()[:120]
        if value:
            meta.setdefault(_FIELDS[m.group(1).lower()], value)
    if "doc_status" in meta:
        meta["status_class"] = classify_status(meta["doc_status"])
    return meta
