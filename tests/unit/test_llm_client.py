import json

import httpx
import pytest

from app.generation import LLMError, OpenAICompatibleClient

MSGS = [{"role": "user", "content": "hi"}]
GOOD = {"model": "m1", "choices": [{"message": {"content": " Hello [1] "}}], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}


def client(handler, api_key=None) -> OpenAICompatibleClient:
    return OpenAICompatibleClient("http://llm.test/v1/", "m1", api_key=api_key, transport=httpx.MockTransport(handler))


def test_success_parses_text_model_and_usage_and_sends_expected_request() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers.get("authorization"), json.loads(request.content)
        return httpx.Response(200, json=GOOD)

    r = client(handler, api_key="sk-x").complete(MSGS)
    assert (r.text, r.model, r.prompt_tokens, r.completion_tokens) == ("Hello [1]", "m1", 9, 2)
    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-x"
    assert seen["body"]["temperature"] == 0.0 and seen["body"]["stream"] is False and seen["body"]["messages"] == MSGS


def test_no_authorization_header_without_a_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json=GOOD)

    client(handler).complete(MSGS)


def test_http_errors_become_llm_errors_with_status_and_body() -> None:
    with pytest.raises(LLMError, match="HTTP 404.*model not found"):
        client(lambda r: httpx.Response(404, text="model not found")).complete(MSGS)


def test_connection_failure_gives_an_actionable_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="Is the server running"):
        client(handler).complete(MSGS)


def test_timeout_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(LLMError, match="timed out"):
        client(handler).complete(MSGS)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"no": "choices"}),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]}),
    ],
)
def test_malformed_or_empty_responses_are_errors_never_silent(response: httpx.Response) -> None:
    with pytest.raises(LLMError):
        client(lambda r: response).complete(MSGS)
