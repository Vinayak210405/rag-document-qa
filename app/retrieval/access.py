"""Document-level access control applied BEFORE ranking.

Each chunk carries ``groups`` metadata (comma-separated; empty = public). A caller with a set of groups may see a chunk if it
is public or they share a group. ``groups=None`` means "no access-control context" (authentication disabled): everything is
visible. Filtering happens before top-k so restricted text can neither appear in results nor consume result slots, and
nothing restricted reaches the prompt. This is an application-level boundary: it is only as trustworthy as the
authentication that produces the caller's groups (see docs/security.md).
"""

from __future__ import annotations

from typing import Any, Collection, Mapping

import numpy as np


def chunk_visible(meta: Mapping[str, Any], groups: Collection[str] | None) -> bool:
    if groups is None:
        return True
    allowed = {g for g in str(meta.get("groups", "")).split(",") if g}
    return not allowed or bool(allowed & set(groups))


def visible_mask(metadatas: Collection[Mapping[str, Any]], groups: Collection[str] | None) -> np.ndarray:
    return np.array([chunk_visible(m, groups) for m in metadatas], dtype=bool)
