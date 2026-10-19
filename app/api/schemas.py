"""Request/response models for the HTTP API (these produce the OpenAPI docs at /docs)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(description="The user's question (1-1000 characters).")
    top_k: int | None = Field(default=None, ge=1, le=20, description="Passages to retrieve; default from settings.")


class PassageOut(BaseModel):
    source_number: int
    rank: int
    score: float
    chunk_id: str
    doc_id: str
    doc_name: str
    ordinal: int
    page: int | None
    heading: str | None
    text: str
    cited: bool


class TimingsOut(BaseModel):
    retrieval_ms: float
    generation_ms: float | None
    total_ms: float


class UsageOut(BaseModel):
    prompt_tokens: int | None
    completion_tokens: int | None


class RetrieveOut(BaseModel):
    question: str
    sources: list[PassageOut]
    timings: TimingsOut
    embedding_model: str


class AnswerOut(BaseModel):
    question: str
    answer: str
    abstained: bool
    valid_citations: list[int]
    invalid_citations: list[int]
    sources: list[PassageOut]
    timings: TimingsOut
    usage: UsageOut
    llm_model: str
    embedding_model: str


class IngestOut(BaseModel):
    doc_id: str
    name: str
    file_type: str
    content_hash: str
    pages: int | None
    chunks: int
    chars: int
    replaced_existing: bool
    warnings: list[str]
    duration_ms: float
    chunker: str
    embedding_model: str


class DocumentOut(BaseModel):
    doc_id: str
    name: str
    chunks: int


class HealthOut(BaseModel):
    status: str
    documents: int
    indexed_chunks: int
    embedding_model: str
    default_top_k: int
