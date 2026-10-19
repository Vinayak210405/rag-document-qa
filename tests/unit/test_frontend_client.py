import httpx
import pytest

from frontend.api_client import ApiClient, ApiError


def client(handler) -> ApiClient:
    return ApiClient("http://api.test/", transport=httpx.MockTransport(handler))


def test_successful_calls_return_parsed_json() -> None:
    c = client(lambda r: httpx.Response(200, json={"status": "ok", "path": r.url.path}))
    assert c.health() == {"status": "ok", "path": "/health"}


def test_api_error_detail_and_status_are_surfaced() -> None:
    c = client(lambda r: httpx.Response(409, json={"detail": "no documents have been ingested yet"}))
    with pytest.raises(ApiError, match=r"no documents.*409"):
        c.ask("hello")


def test_unreachable_api_tells_the_user_how_to_start_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ApiError, match="uvicorn"):
        client(handler).health()


def test_upload_sends_multipart_with_the_filename() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ctype"], seen["body"] = request.headers["content-type"], request.content
        return httpx.Response(200, json={})

    client(handler).upload("a.txt", b"hello")
    assert seen["ctype"].startswith("multipart/form-data") and b'filename="a.txt"' in seen["body"]
