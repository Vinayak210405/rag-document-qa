"""Bearer-token authentication and group-based authorisation (optional, off by default).

Set ``RAG_AUTH_TOKENS`` to a JSON object to turn it on:
    {"<token, >=16 chars>": {"name": "alice", "groups": ["hr"], "admin": false},
     "<another token>":      {"name": "ops", "admin": true}}
Admins may ingest, delete and audit, and see every document. Other callers can only query, and only chunks that are public or
share one of their groups (enforced before ranking, app/retrieval/access.py). With no tokens configured the API is open: that
is intended for a single user on localhost, NOT for a shared deployment. Tokens are compared in constant time; they are
static shared secrets (no expiry, no rotation, no TLS handling): put a real identity provider and HTTPS in front for production.
"""

from __future__ import annotations

import hmac
import json
from dataclasses import dataclass

MIN_TOKEN_CHARS = 16


class AuthConfigError(ValueError):
    """RAG_AUTH_TOKENS is malformed."""


@dataclass(frozen=True)
class Principal:
    name: str
    groups: frozenset[str] | None  # None = unrestricted (admin or auth disabled)
    admin: bool


ANONYMOUS = Principal("anonymous", None, True)  # used only when authentication is disabled


@dataclass(frozen=True)
class AuthConfig:
    tokens: tuple[tuple[str, Principal], ...] = ()

    @property
    def enabled(self) -> bool:
        return bool(self.tokens)

    def authenticate(self, authorization: str | None) -> Principal | None:
        """Principal for an ``Authorization: Bearer <token>`` header value, or None if missing/invalid."""
        if not authorization:
            return None
        scheme, _, presented = authorization.partition(" ")
        if scheme.lower() != "bearer" or not presented.strip():
            return None
        found: Principal | None = None
        for token, principal in self.tokens:  # check every token: no early exit on the first match
            if hmac.compare_digest(token.encode(), presented.strip().encode()):
                found = principal
        return found


def parse_auth_tokens(raw: str | None) -> AuthConfig:
    if not raw or not raw.strip():
        return AuthConfig()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AuthConfigError(f"RAG_AUTH_TOKENS is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise AuthConfigError("RAG_AUTH_TOKENS must be a non-empty JSON object mapping token -> {name, groups, admin}")
    tokens: list[tuple[str, Principal]] = []
    for token, spec in data.items():
        if len(token) < MIN_TOKEN_CHARS:
            raise AuthConfigError(f"every token must be at least {MIN_TOKEN_CHARS} characters")
        if not isinstance(spec, dict):
            raise AuthConfigError("each token needs an object value like {\"name\": ..., \"groups\": [...], \"admin\": false}")
        admin = bool(spec.get("admin", False))
        groups = spec.get("groups", [])
        if not isinstance(groups, list) or not all(isinstance(g, str) and g and "," not in g for g in groups):
            raise AuthConfigError("groups must be a list of non-empty strings without commas")
        tokens.append((token, Principal(str(spec.get("name", "user")), None if admin else frozenset(groups), admin)))
    return AuthConfig(tuple(tokens))
