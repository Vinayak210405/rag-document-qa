"""Opt-in smoke tests against REAL backends. Skipped by default; run with:

    uv sync --extra embeddings
    uv run pytest -m requires_model      # downloads bge-small (~130 MB) on first run
    uv run pytest -m requires_api        # needs your LLM endpoint running (see .env)
"""

import pytest

from app.config import Settings


@pytest.mark.requires_model
def test_real_embedding_model_matches_a_paraphrase_with_no_shared_keywords() -> None:
    from app.embeddings import create_embedder

    embedder = create_embedder(Settings(_env_file=None, embedding_backend="sentence_transformers"))
    docs = ["The AeroLink X200 is covered by a 24-month limited warranty.", "Passwords must be rotated every 180 days."]
    sims = embedder.embed_documents(docs) @ embedder.embed_query("How long does the guarantee on the device last?")
    assert embedder.dimension == 384 and sims[0] > sims[1]


@pytest.mark.requires_api
def test_real_llm_endpoint_returns_text() -> None:
    from app.generation import OpenAICompatibleClient

    s = Settings()  # reads your .env
    client = OpenAICompatibleClient(
        s.llm_base_url, s.llm_model, s.llm_api_key.get_secret_value() if s.llm_api_key else None, s.llm_timeout_seconds)
    reply = client.complete([{"role": "user", "content": "Reply with the single word: pong"}])
    assert reply.text.strip()
