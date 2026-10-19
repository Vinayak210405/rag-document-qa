"""Application service: orchestrates ingestion and question answering.

This is the only place that wires the modules together. API and UI layers call it and contain
no business logic. Retrieval, prompting, generation and citation checking remain separate modules.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.chunking import Chunker, SentenceChunker
from app.config import Settings
from app.embeddings import Embedder, create_embedder
from app.generation import ABSTAIN_PHRASE, LLMClient, OpenAICompatibleClient, build_messages, check_citations
from app.indexing import ChromaVectorStore, DocumentInfo
from app.ingestion import EmptyDocumentError, parse_document
from app.retrieval import InvalidQueryError, RetrievedChunk, normalize_query
from app.retrieval.dense import DenseRetriever
from app.utils.logging import log_event

logger = logging.getLogger(__name__)

MAX_TOP_K = 20


class EmptyIndexError(RuntimeError):
    """A question was asked before any document was ingested."""


@dataclass(frozen=True)
class IngestReport:
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


@dataclass(frozen=True)
class PassageResult:
    """A retrieved passage as shown to the user; ``source_number`` is the [n] used in citations."""

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


@dataclass(frozen=True)
class Timings:
    retrieval_ms: float
    generation_ms: float | None
    total_ms: float


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(frozen=True)
class RetrievalResult:
    question: str  # normalized
    sources: list[PassageResult]
    timings: Timings
    embedding_model: str


@dataclass(frozen=True)
class AnswerResult:
    question: str  # normalized
    answer: str
    abstained: bool  # the model used the exact abstain phrase (a prompt-compliance signal, not a guarantee)
    valid_citations: list[int]
    invalid_citations: list[int]
    sources: list[PassageResult]
    timings: Timings
    usage: Usage
    llm_model: str
    embedding_model: str


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class RagService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        store: ChromaVectorStore,
        retriever: DenseRetriever,
        chunker: Chunker,
        llm: LLMClient,
        default_top_k: int = 4,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.retriever = retriever
        self.chunker = chunker
        self.llm = llm
        self.default_top_k = default_top_k

    # ------------------------------------------------------------ ingestion
    def ingest(self, filename: str, content: bytes) -> IngestReport:
        start = time.perf_counter()
        doc = parse_document(filename, content)
        chunks = self.chunker.chunk(doc)
        if not chunks:
            raise EmptyDocumentError(f"{doc.name}: produced no chunks")
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        replaced = self.store.replace_document(doc.doc_id, chunks, embeddings)
        report = IngestReport(
            doc_id=doc.doc_id, name=doc.name, file_type=doc.file_type, content_hash=doc.content_hash,
            pages=doc.page_count, chunks=len(chunks), chars=doc.char_count, replaced_existing=replaced,
            warnings=list(doc.warnings), duration_ms=_ms(start), chunker=self.chunker.strategy_name,
            embedding_model=self.embedder.model_name,
        )
        log_event(logger, "ingest", doc_id=doc.doc_id, chunks=report.chunks, chars=report.chars,
                  replaced=replaced, warnings=len(report.warnings), ms=report.duration_ms)
        return report

    def list_documents(self) -> list[DocumentInfo]:
        return self.store.list_documents()

    # ------------------------------------------------------------ retrieval / QA
    def _retrieve(self, question: str, top_k: int | None) -> tuple[str, list[RetrievedChunk], float]:
        query = normalize_query(question)
        k = top_k if top_k is not None else self.default_top_k
        if not 1 <= k <= MAX_TOP_K:
            raise InvalidQueryError(f"top_k must be between 1 and {MAX_TOP_K}")
        if self.store.count() == 0:
            raise EmptyIndexError("no documents have been ingested yet")
        start = time.perf_counter()
        passages = self.retriever.retrieve(query, k)
        return query, passages, _ms(start)

    @staticmethod
    def _to_results(passages: list[RetrievedChunk], cited: set[int]) -> list[PassageResult]:
        return [
            PassageResult(
                source_number=i, rank=p.rank, score=round(p.score, 6), chunk_id=p.chunk_id, doc_id=p.doc_id,
                doc_name=p.doc_name, ordinal=p.ordinal, page=p.page, heading=p.heading, text=p.text,
                cited=i in cited,
            )
            for i, p in enumerate(passages, start=1)
        ]

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResult:
        """Retrieval only (no LLM): lets you inspect evidence without a model running."""
        total = time.perf_counter()
        query, passages, retrieval_ms = self._retrieve(question, top_k)
        timings = Timings(retrieval_ms=retrieval_ms, generation_ms=None, total_ms=_ms(total))
        log_event(logger, "retrieve", question_chars=len(query), hits=len(passages), ms=retrieval_ms)
        return RetrievalResult(query, self._to_results(passages, set()), timings, self.embedder.model_name)

    def answer(self, question: str, top_k: int | None = None) -> AnswerResult:
        total = time.perf_counter()
        query, passages, retrieval_ms = self._retrieve(question, top_k)
        messages = build_messages(query, passages)
        gen_start = time.perf_counter()
        response = self.llm.complete(messages)  # raises LLMError; never replaced by a made-up answer
        generation_ms = _ms(gen_start)
        report = check_citations(response.text, len(passages))
        timings = Timings(retrieval_ms, generation_ms, _ms(total))
        result = AnswerResult(
            question=query,
            answer=response.text,
            abstained=ABSTAIN_PHRASE in response.text,
            valid_citations=list(report.valid),
            invalid_citations=list(report.invalid),
            sources=self._to_results(passages, set(report.valid)),
            timings=timings,
            usage=Usage(response.prompt_tokens, response.completion_tokens),
            llm_model=response.model,
            embedding_model=self.embedder.model_name,
        )
        log_event(
            logger, "answer", hits=len(passages), valid_citations=len(report.valid),
            invalid_citations=len(report.invalid), abstained=result.abstained,
            retrieval_ms=retrieval_ms, generation_ms=generation_ms, total_ms=timings.total_ms,
            prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens,
        )
        return result


def build_service(settings: Settings) -> RagService:
    """Construct the production service from settings (may download an embedding model on first use)."""
    embedder = create_embedder(settings)
    store = ChromaVectorStore(
        settings.chroma_dir, settings.chroma_collection, embedder.model_name, embedder.dimension
    )
    llm = OpenAICompatibleClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
        timeout=settings.llm_timeout_seconds,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    return RagService(
        embedder=embedder,
        store=store,
        retriever=DenseRetriever(store, embedder),
        chunker=SentenceChunker(settings.chunk_max_chars, settings.chunk_overlap_chars),
        llm=llm,
        default_top_k=settings.default_top_k,
    )
