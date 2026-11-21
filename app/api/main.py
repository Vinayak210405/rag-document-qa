"""FastAPI application.

Endpoints are plain ``def`` (not ``async def``) on purpose: embedding and the LLM call are blocking,
and FastAPI runs ``def`` endpoints in a thread pool, so they do not stall the event loop.
Domain errors are mapped to precise HTTP statuses; nothing is swallowed.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.api.schemas import (
    AnswerOut, AskRequest, AuditOut, DeleteOut, DocumentOut, HealthOut, IngestOut, RetrieveOut,
)
from app.config import Settings, get_settings
from app.embeddings import EmbeddingBackendError
from app.generation import LLMError
from app.indexing import IndexMismatchError, StorageError
from app.ingestion import (
    EmptyDocumentError, ExtractionError, IngestionError, UnsupportedFileTypeError,
)
from app.retrieval import InvalidQueryError
from app.service import DocumentNotFoundError, EmptyIndexError, RagService, build_service
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)

# Most specific classes first is not required: Starlette resolves handlers by walking the MRO.
_ERROR_STATUS: list[tuple[type[Exception], int]] = [
    (InvalidQueryError, 422),
    (UnsupportedFileTypeError, 415),
    (EmptyDocumentError, 422),
    (ExtractionError, 422),
    (IngestionError, 400),
    (EmptyIndexError, 409),
    (DocumentNotFoundError, 404),
    (IndexMismatchError, 409),
    (StorageError, 503),
    (LLMError, 502),
    (EmbeddingBackendError, 503),
]


def create_app(service: RagService | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Pass ``service`` to inject fakes in tests; otherwise it is built on startup."""
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(cfg.log_level)
        if app.state.service is None:
            app.state.service = build_service(cfg)
        yield

    app = FastAPI(title="Evaluated Document QA Bot", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    app.state.max_upload_bytes = cfg.max_upload_bytes

    def make_handler(status: int):
        def handler(_: Request, exc: Exception) -> JSONResponse:
            logger.warning("request failed: %s: %s", type(exc).__name__, exc)
            return JSONResponse(status_code=status, content={"error": type(exc).__name__, "detail": str(exc)})
        return handler

    for exc_type, status in _ERROR_STATUS:
        app.add_exception_handler(exc_type, make_handler(status))

    def get_service(request: Request) -> RagService:
        svc = request.app.state.service
        if svc is None:
            raise HTTPException(status_code=503, detail="service is not initialised")
        return svc

    @app.get("/health", response_model=HealthOut)
    def health(svc: RagService = Depends(get_service)) -> HealthOut:
        docs = svc.list_documents()
        return HealthOut(
            status="ok", documents=len(docs), indexed_chunks=svc.store.count(),
            embedding_model=svc.embedder.model_name, default_top_k=svc.default_top_k,
        )

    @app.post("/documents", response_model=IngestOut)
    def upload_document(request: Request, file: UploadFile = File(...), svc: RagService = Depends(get_service)):
        if not file.filename:
            raise HTTPException(status_code=400, detail="uploaded file has no name")
        limit = request.app.state.max_upload_bytes
        content = file.file.read(limit + 1)
        if len(content) > limit:
            raise HTTPException(status_code=413, detail=f"file exceeds the {limit} byte upload limit")
        return svc.ingest(file.filename, content)

    @app.get("/documents", response_model=list[DocumentOut])
    def list_documents(svc: RagService = Depends(get_service)):
        return svc.list_documents()

    @app.delete("/documents/{doc_id}", response_model=DeleteOut)
    def delete_document(doc_id: str, svc: RagService = Depends(get_service)):
        """Remove a document from the index (dense and lexical views share one store, so both are updated)."""
        return DeleteOut(doc_id=doc_id, chunks_removed=svc.delete_document(doc_id))

    @app.get("/audit", response_model=AuditOut)
    def audit(svc: RagService = Depends(get_service)):
        """Index consistency report: stale chunks, interrupted updates, duplicate positions and texts."""
        r = svc.audit()
        return AuditOut(
            documents=r.documents, chunks=r.chunks, clean=r.clean, stale_chunks=list(r.stale_chunks),
            mixed_versions=list(r.mixed_versions), duplicate_ordinals=list(r.duplicate_ordinals),
            duplicate_texts=[list(g) for g in r.duplicate_texts],
        )

    @app.post("/retrieve", response_model=RetrieveOut)
    def retrieve(body: AskRequest, svc: RagService = Depends(get_service)):
        """Retrieval only: inspect the evidence without calling the LLM."""
        return svc.retrieve(body.question, body.top_k)

    @app.post("/ask", response_model=AnswerOut)
    def ask(body: AskRequest, svc: RagService = Depends(get_service)):
        return svc.answer(body.question, body.top_k)

    return app


app = create_app()
