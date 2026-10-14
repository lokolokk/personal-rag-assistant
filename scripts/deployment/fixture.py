"""Disposable container acceptance fixture; never copied into production images."""

import asyncio
import hashlib
import json
import sys
from collections.abc import AsyncIterator
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import fitz
import uvicorn
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.main import create_application
from app.schemas.document import ChunkMetadata, DocumentChunk, DocumentItem
from app.services.evaluation.offline import OfflineNetworkError, offline_network
from app.services.ingestion.registry import get_document_registry
from app.services.retrieval.bm25_store import get_bm25_store
from app.services.retrieval.conversation_registry import get_conversation_registry
from app.services.retrieval.vector_store import get_vector_store

TEXT = "Synthetic quartz persistence evidence. No private source material."


def seed() -> None:
    """Seed only an empty disposable volume using real deterministic embeddings."""
    settings = get_settings()
    settings.ensure_directories()
    assert not get_vector_store().get_all_chunks(), "Refusing to seed a populated collection"
    assert not get_document_registry().get_all()
    assert not get_conversation_registry().get_all()
    with fitz.open() as pdf:
        pdf.new_page().insert_text((72, 72), TEXT)
        pdf.save(settings.resolved_upload_dir / "synthetic_synthetic.pdf")
    chunk = DocumentChunk(
        chunk_id="synthetic-chunk",
        content=TEXT,
        metadata=ChunkMetadata(
            chunk_id="synthetic-chunk",
            document_id="synthetic",
            filename="synthetic.pdf",
            page_number=1,
            section_title="Fixture",
            token_count=10,
            char_start=0,
            char_end=len(TEXT),
        ),
    )
    get_vector_store().add_chunks([chunk])
    get_document_registry().add_or_update(
        DocumentItem(
            id="synthetic",
            filename="synthetic.pdf",
            page_count=1,
            chunk_count=1,
        )
    )
    get_conversation_registry().record_turn("synthetic-conversation", "user", TEXT)


def state() -> dict[str, object]:
    """Read persisted content and the actual server's startup-hydrated BM25."""
    settings = get_settings()
    vector = get_vector_store()
    assert vector.count() == 1
    assert vector.query("quartz")[0].chunk_id == "synthetic-chunk"
    assert get_bm25_store().query("quartz")[0].chunk_id == "synthetic-chunk"
    assert get_document_registry().get_all()[0].filename == "synthetic.pdf"
    conversation = get_conversation_registry().get("synthetic-conversation")
    assert conversation is not None and conversation.messages[0].content == TEXT
    pdf = settings.resolved_upload_dir / "synthetic_synthetic.pdf"
    with fitz.open(pdf) as document:
        assert "quartz" in document[0].get_text()
    return {
        "chunks": vector.count(),
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "documents_sha256": hashlib.sha256(
            (settings.BASE_DIR / "storage/documents.json").read_bytes()
        ).hexdigest(),
        "conversations_sha256": hashlib.sha256(
            (settings.BASE_DIR / "storage/conversations.json").read_bytes()
        ).hexdigest(),
    }


def serve() -> None:
    application = create_application()

    @application.get("/api/v1/_verification/state")
    async def read_state() -> dict[str, object]:
        return state()

    @application.get("/api/v1/_verification/events")
    async def events() -> StreamingResponse:
        async def delayed() -> AsyncIterator[str]:
            yield 'event: token\ndata: {"content":"synthetic"}\n\n'
            await asyncio.sleep(2)
            yield 'event: done\ndata: {"finishReason":"stop"}\n\n'

        return StreamingResponse(delayed(), media_type="text/event-stream")

    uvicorn.run(application, host="0.0.0.0", port=8000, workers=1)


if __name__ == "__main__":
    assert get_settings().BASE_DIR == Path("/app")
    assert not get_settings().OPENAI_API_KEY
    with ExitStack() as stack:
        stack.enter_context(offline_network())
        for constructor in ("openai.OpenAI", "openai.AsyncOpenAI"):
            stack.enter_context(
                patch(constructor, side_effect=OfflineNetworkError("Provider call"))
            )
        if sys.argv[1:] == ["seed"]:
            seed()
            print(json.dumps({"seeded": True}))
        elif sys.argv[1:] == ["serve"]:
            serve()
        else:
            raise SystemExit("Expected seed or serve")
