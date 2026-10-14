"""FastAPI application entrypoint for Conversational Document RAG platform.

Configures lifespan context with server-boot BM25 index hydration from ChromaDB,
CORS middleware, standardized error envelopes, and aggregated API v1 routing.
"""

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryFile
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import AppException
from app.services.retrieval.bm25_store import get_bm25_store
from app.services.retrieval.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class ReadinessResponse(BaseModel):
    """Public initialization status without storage or exception details."""

    status: Literal["ready", "not_ready"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    On server startup:
    1. Ensures persistent host storage directories exist.
    2. Re-hydrates in-memory BM25Store from persistent ChromaDB via get_all_chunks(),
       keeping dense and sparse indices perfectly synchronized across restarts.
    """
    app.state.ready = False
    settings = get_settings()

    try:
        for directory in {
            settings.resolved_storage_dir,
            settings.resolved_upload_dir,
            settings.resolved_chroma_dir,
            settings.resolve_path(Path("storage")),
        }:
            directory.mkdir(parents=True, exist_ok=True)
            with TemporaryFile(dir=directory, prefix=".ready-", mode="w+b") as probe:
                probe.write(b"ready")
                probe.flush()
                probe.seek(0)
                if probe.read() != b"ready":
                    raise OSError("Storage probe read failed")
        vector_store = get_vector_store()
        bm25_store = get_bm25_store()
        existing_chunks = vector_store.get_all_chunks()
        bm25_store.clear()
        if existing_chunks:
            bm25_store.add_chunks(existing_chunks)
            logger.info(
                "BM25 startup hydration complete: hydrated %d chunks from ChromaDB",
                len(existing_chunks),
            )
        else:
            logger.info("ChromaDB vector store is empty; BM25 initialized with 0 chunks")
        app.state.ready = True
    except Exception as err:
        logger.warning("BM25 startup hydration warning: %s", err)

    try:
        yield
    finally:
        app.state.ready = False


def create_application() -> FastAPI:
    """Instantiate and configure the FastAPI application."""
    if os.environ.get("RAG_COST_DEBUG") == "1":
        from app.api.v1.cost_debug import create_debug_application

        return create_debug_application()

    settings = get_settings()

    application = FastAPI(
        title=settings.PROJECT_NAME,
        version="1.0.0",
        description="Production-grade Conversational Document RAG API",
        lifespan=lifespan,
    )
    application.state.ready = False

    # Cross-Origin Resource Sharing (CORS)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Standardized Error Envelopes (docs/ARCHITECTURE.md Section 4.1)
    @application.exception_handler(AppException)
    async def app_exception_handler(_: Request, exc: AppException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        from fastapi.encoders import jsonable_encoder

        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed.",
                    "details": jsonable_encoder(exc.errors()),
                }
            },
        )

    @application.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "HTTP_ERROR",
                    "message": str(exc.detail),
                    "details": None,
                }
            },
        )

    # Mount API v1 router
    application.include_router(api_router, prefix=settings.API_V1_STR)

    @application.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        """Liveness only; initialization is reported separately by /ready."""
        return {"status": "ok", "project": settings.PROJECT_NAME}

    @application.get(
        "/ready",
        tags=["system"],
        response_model=ReadinessResponse,
        responses={503: {"model": ReadinessResponse}},
    )
    async def readiness_check(response: Response) -> ReadinessResponse:
        """Report completed local initialization without invoking any provider."""
        if application.state.ready:
            return ReadinessResponse(status="ready")
        response.status_code = 503
        return ReadinessResponse(status="not_ready")

    return application


app = create_application()
