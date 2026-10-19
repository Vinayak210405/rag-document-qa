"""Thin HTTP client for the backend API (kept separate from Streamlit so it is unit-testable)."""

from __future__ import annotations

from typing import Any

import httpx


class ApiError(RuntimeError):
    """The API was unreachable or returned an error; the message is safe to show to the user."""


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 300.0, transport: httpx.BaseTransport | None = None) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, self._base + path, **kwargs)
        except httpx.ConnectError as exc:
            raise ApiError(
                f"Cannot reach the API at {self._base}. Start it with: uv run uvicorn app.api.main:app"
            ) from exc
        except httpx.HTTPError as exc:
            raise ApiError(f"Request to the API failed: {exc}") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(f"{detail} (HTTP {response.status_code})")
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def list_documents(self) -> list[dict[str, Any]]:
        return self._request("GET", "/documents")

    def upload(self, filename: str, content: bytes) -> dict[str, Any]:
        return self._request("POST", "/documents", files={"file": (filename, content)})

    def retrieve(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        return self._request("POST", "/retrieve", json={"question": question, "top_k": top_k})

    def ask(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        return self._request("POST", "/ask", json={"question": question, "top_k": top_k})
