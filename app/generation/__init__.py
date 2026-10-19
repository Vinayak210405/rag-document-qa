"""Generation: LLM client, grounded prompts, citation checks."""

from app.generation.citations import CitationReport, check_citations
from app.generation.llm import LLMClient, LLMError, LLMResponse, OpenAICompatibleClient
from app.generation.prompt import ABSTAIN_PHRASE, build_messages

__all__ = [
    "ABSTAIN_PHRASE", "CitationReport", "LLMClient", "LLMError", "LLMResponse",
    "OpenAICompatibleClient", "build_messages", "check_citations",
]
