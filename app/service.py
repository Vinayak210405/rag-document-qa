"""Application service: orchestrates ingestion and question answering.

This is the only place that wires the modules together. API and UI layers call it and contain
no business logic. Retrieval, prompting, generation and citation checking remain separate modules.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Protocol, Sequence

from app.chunking import Chunker, SemanticChunker, SentenceChunker
from app.config import Settings
from app.embeddings import Embedder, create_embedder
from app.generation import ABSTAIN_PHRASE, LLMClient, OpenAICompatibleClient, build_messages
from app.generation.citations import validate_answer
from app.indexing import ChromaVectorStore, DocumentInfo
from app.ingestion import EmptyDocumentError, parse_document
from app.retrieval import InvalidQueryError, normalize_query
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.generation.conflicts import conflict_note, detect_conflicts
from app.generation.evidence import assess_evidence
from app.retrieval.context import assemble_context
from app.retrieval.rerank import CrossEncoderReranker, LexicalOverlapReranker, RerankedRetriever
from app.retrieval.types import RetrievedChunk
from app.utils.logging import log_event

logger = logging.getLogger(__name__)

MAX_TOP_K = 20


class DocumentNotFoundError(LookupError):
    """The requested document is not in the index."""


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
    status: str = "new"  # new | updated | unchanged (identical bytes and settings: nothing was re-embedded)
    ingest_version: int = 1


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
    status: str = "answered"  # answered | abstained | abstained_no_evidence | unverified | withheld
    evidence_coverage: float | None = None
    uncited_sentences: list[str] = field(default_factory=list)
    version_conflicts: list[str] = field(default_factory=list)  # human-readable, e.g. "HR-POL-004: v1 (superseded), v2 (current)"
    context_dropped: dict[str, str] = field(default_factory=dict)  # chunk_id -> reason


class Retriever(Protocol):
    def retrieve(self, query: str, top_k_: int) -> list[RetrievedChunk]: ...


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class RagService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        store: ChromaVectorStore,
        retriever: Retriever,
        chunker: Chunker,
        llm: LLMClient,
        default_top_k: int = 4,
        context_token_budget: int = 1500,
        max_chunks_per_doc: int | None = None,
        min_evidence_coverage: float = 0.0,
        strict_citations: bool = False,
        max_pages: int | None = None,
    ) -> None:
        self.max_pages = max_pages
        self.min_evidence_coverage = min_evidence_coverage
        self.strict_citations = strict_citations
        self.context_token_budget = context_token_budget
        self.max_chunks_per_doc = max_chunks_per_doc
        self.embedder = embedder
        self.store = store
        self.retriever = retriever
        self.chunker = chunker
        self.llm = llm
        self.default_top_k = default_top_k

    # ------------------------------------------------------------ ingestion
    @property
    def index_signature(self) -> str:
        """Settings that determine what the stored chunks look like; a change forces re-indexing."""
        return f"{self.chunker.strategy_name}|{self.embedder.model_name}"

    def ingest(self, filename: str, content: bytes, allowed_groups: Sequence[str] | None = None) -> IngestReport:
        """Index a document. Identical bytes + identical settings are skipped (no parsing cost beyond hashing)."""
        start = time.perf_counter()
        doc = parse_document(filename, content, max_pages=self.max_pages)
        groups = ",".join(sorted(set(allowed_groups))) if allowed_groups else ""
        state = self.store.document_state(doc.doc_id)
        if (
            state is not None and state.content_hash == doc.content_hash
            and state.index_signature == self.index_signature and state.groups == groups
        ):
            log_event(logger, "ingest", doc_id=doc.doc_id, status="unchanged", ms=_ms(start))
            return IngestReport(
                doc_id=doc.doc_id, name=doc.name, file_type=doc.file_type, content_hash=doc.content_hash,
                pages=doc.page_count, chunks=len(state.chunk_ids), chars=doc.char_count, replaced_existing=False,
                warnings=list(doc.warnings), duration_ms=_ms(start), chunker=self.chunker.strategy_name,
                embedding_model=self.embedder.model_name, status="unchanged", ingest_version=state.ingest_version,
            )
        chunks = self.chunker.chunk(doc)
        if not chunks:
            raise EmptyDocumentError(f"{doc.name}: produced no chunks")
        version = 1 if state is None else state.ingest_version + 1
        stamp: dict[str, str | int | bool] = {
            "content_hash": doc.content_hash, "index_signature": self.index_signature, "ingest_version": version,
            "groups": groups,
        }
        chunks = [replace(c, metadata={**c.metadata, **stamp}) for c in chunks]
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        replaced = self.store.replace_document(doc.doc_id, chunks, embeddings)
        report = IngestReport(
            doc_id=doc.doc_id, name=doc.name, file_type=doc.file_type, content_hash=doc.content_hash,
            pages=doc.page_count, chunks=len(chunks), chars=doc.char_count, replaced_existing=replaced,
            warnings=list(doc.warnings), duration_ms=_ms(start), chunker=self.chunker.strategy_name,
            embedding_model=self.embedder.model_name, status="updated" if replaced else "new", ingest_version=version,
        )
        log_event(logger, "ingest", doc_id=doc.doc_id, chunks=report.chunks, chars=report.chars,
                  status=report.status, version=version, warnings=len(report.warnings), ms=report.duration_ms)
        return report

    def delete_document(self, doc_id: str) -> int:
        """Remove a document from the index; raises DocumentNotFoundError if it is not indexed."""
        removed = self.store.delete_document(doc_id)
        if removed == 0:
            raise DocumentNotFoundError(f"no indexed document with id {doc_id!r}")
        return removed

    def audit(self):
        return self.store.audit()

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
        context = assemble_context(
            passages, token_budget=self.context_token_budget, max_per_doc=self.max_chunks_per_doc
        )
        passages = context.passages  # source numbers in the prompt follow the assembled context
        decision = assess_evidence(query, passages, self.min_evidence_coverage)
        conflicts = detect_conflicts(passages)
        conflict_lines = [
            f"{c.family}: " + ", ".join(f"{n} v{v} ({st})" for n, v, st in c.versions) for c in conflicts
        ]
        common = dict(
            question=query, embedding_model=self.embedder.model_name, evidence_coverage=round(decision.coverage, 4),
            version_conflicts=conflict_lines, context_dropped=dict(context.dropped),
        )
        if not decision.sufficient:
            # Deterministic refusal: no LLM call, and the text is the fixed abstain phrase, not generated content.
            timings = Timings(retrieval_ms, None, _ms(total))
            log_event(logger, "answer", status="abstained_no_evidence", coverage=decision.coverage,
                      missing=len(decision.missing_terms), retrieval_ms=retrieval_ms)
            return AnswerResult(
                answer=ABSTAIN_PHRASE, abstained=True, valid_citations=[], invalid_citations=[],
                sources=self._to_results(passages, set()), timings=timings, usage=Usage(None, None),
                llm_model="(not called)", status="abstained_no_evidence", **common,
            )
        messages = build_messages(query, passages, conflict_note(conflicts))
        gen_start = time.perf_counter()
        response = self.llm.complete(messages)  # raises LLMError; never replaced by a made-up answer
        generation_ms = _ms(gen_start)
        validation = validate_answer(response.text, len(passages))
        abstained = ABSTAIN_PHRASE in response.text
        if abstained:
            status = "abstained"
        elif validation.ok:
            status = "answered"
        else:
            status = "unverified"  # invalid / missing citations or uncited sentences: shown, but flagged
        text = response.text
        if status == "unverified" and self.strict_citations:
            status, text = "withheld", "Answer withheld: it failed citation validation. See the source passages."
        timings = Timings(retrieval_ms, generation_ms, _ms(total))
        result = AnswerResult(
            answer=text, abstained=abstained, valid_citations=list(validation.valid),
            invalid_citations=list(validation.invalid), sources=self._to_results(passages, set(validation.valid)),
            timings=timings, usage=Usage(response.prompt_tokens, response.completion_tokens),
            llm_model=response.model, status=status, uncited_sentences=list(validation.uncited_sentences), **common,
        )
        log_event(
            logger, "answer", status=status, hits=len(passages), valid_citations=len(validation.valid),
            invalid_citations=len(validation.invalid), uncited=len(validation.uncited_sentences),
            conflicts=len(conflicts), retrieval_ms=retrieval_ms, generation_ms=generation_ms,
            total_ms=timings.total_ms, prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        return result


def make_chunker(settings: Settings, embedder: Embedder) -> Chunker:
    if settings.chunking == "semantic":
        return SemanticChunker(
            embedder, settings.chunk_max_chars, settings.semantic_min_chars, settings.semantic_percentile
        )
    return SentenceChunker(settings.chunk_max_chars, settings.chunk_overlap_chars)


def _build_retriever(settings: Settings, store: ChromaVectorStore, embedder: Embedder) -> Retriever:
    if settings.retriever == "dense":
        return DenseRetriever(store, embedder)
    hybrid = HybridRetriever(store, embedder, candidates=settings.hybrid_candidates, rrf_k=settings.rrf_k)
    if settings.retriever == "hybrid":
        return hybrid
    reranker = (
        LexicalOverlapReranker() if settings.reranker_model == "lexical"
        else CrossEncoderReranker(settings.reranker_model)
    )
    return RerankedRetriever(hybrid, reranker, candidates=settings.hybrid_candidates)


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
        retriever=_build_retriever(settings, store, embedder),
        context_token_budget=settings.context_token_budget,
        max_chunks_per_doc=settings.max_chunks_per_doc,
        min_evidence_coverage=settings.min_evidence_coverage,
        strict_citations=settings.strict_citations,
        max_pages=settings.max_pdf_pages,
        chunker=make_chunker(settings, embedder),
        llm=llm,
        default_top_k=settings.default_top_k,
    )
