"""Document management endpoints (Multi-Document Lifecycle).

Handles PDF upload, layout-aware parsing, structural chunking, dual indexing (ChromaDB + BM25),
atomic replace on re-upload, document listing from registry, and clean deletion.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status

from app.core.config import get_settings
from app.core.exceptions import (
    DocumentNotFoundError,
    FileSizeLimitExceededError,
    InvalidFileFormatError,
)
from app.schemas.document import DocumentDeleteResponse, DocumentItem
from app.services.ingestion.chunker import StructuralChunker
from app.services.ingestion.parser import PDFParser
from app.services.ingestion.registry import get_document_registry
from app.services.retrieval.bm25_store import get_bm25_store
from app.services.retrieval.vector_store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_FILE_SIZE = 30 * 1024 * 1024  # 30MB limit


@router.post(
    "/upload",
    response_model=DocumentItem,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a PDF document",
)
async def upload_document(
    file: Annotated[UploadFile, File()],
) -> DocumentItem:
    """Upload and ingest a PDF document into the knowledge library.

    Validates file format and size, coordinates layout parsing, structural chunking,
    dual indexing (ChromaDB + BM25), enforces atomic replace on re-upload, and
    records metadata in the lightweight DocumentRegistry.
    """
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise InvalidFileFormatError("Only PDF files are supported.")

    content = await file.read()
    if not content:
        raise InvalidFileFormatError("Uploaded PDF file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise FileSizeLimitExceededError("File exceeds maximum size limit of 30MB.")

    settings = get_settings()
    registry = get_document_registry()
    vector_store = get_vector_store()
    bm25_store = get_bm25_store()

    # Check for existing document with same filename to enforce Atomic Replace
    existing_doc = registry.get_by_filename(filename)
    if existing_doc is not None:
        doc_id = existing_doc.id
        logger.info(
            "Atomic replace: Purging prior chunks for existing document %s (%s)",
            filename,
            doc_id,
        )
        vector_store.delete_document(doc_id)
        bm25_store.delete_document(doc_id)
    else:
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"

    # Parse PDF with layout awareness and Markdown table serialization
    parser = PDFParser()
    parsed_doc = parser.parse_bytes(content=content, filename=filename)

    # Chunk with structural boundaries and table isolation
    chunker = StructuralChunker()
    chunks = chunker.chunk_document(parsed_doc=parsed_doc, document_id=doc_id)

    # Index into ChromaDB (dense) and BM25 (sparse)
    vector_store.add_chunks(chunks)
    bm25_store.add_chunks(chunks)

    # Persist uploaded raw PDF to host storage
    settings.ensure_directories()
    raw_pdf_path = settings.resolved_upload_dir / f"{doc_id}_{filename}"
    raw_pdf_path.write_bytes(content)

    # Record in persistent document registry
    now_iso = datetime.now(UTC).isoformat()
    doc_item = DocumentItem(
        id=doc_id,
        filename=filename,
        page_count=parsed_doc.page_count,
        chunk_count=len(chunks),
        uploaded_at=now_iso,
        status="ready",
    )
    registry.add_or_update(doc_item)
    logger.info(
        "Successfully ingested document %s (%s) with %d chunks",
        filename,
        doc_id,
        len(chunks),
    )

    return doc_item


@router.get(
    "",
    response_model=list[DocumentItem],
    status_code=status.HTTP_200_OK,
    summary="List all indexed documents",
)
async def list_documents() -> list[DocumentItem]:
    """Retrieve all indexed documents from the persistent DocumentRegistry."""
    registry = get_document_registry()
    return registry.get_all()


@router.get(
    "/{document_id}",
    response_model=DocumentItem,
    status_code=status.HTTP_200_OK,
    summary="Get document metadata by ID",
)
async def get_document(document_id: str) -> DocumentItem:
    """Retrieve metadata summary for an individual document by its ID."""
    registry = get_document_registry()
    doc = registry.get(document_id)
    if doc is None:
        raise DocumentNotFoundError(document_id)
    return doc


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a document and purge all associated chunks",
)
async def delete_document(document_id: str) -> DocumentDeleteResponse:
    """Atomically remove document chunks from ChromaDB, BM25, and DocumentRegistry."""
    registry = get_document_registry()
    existing_doc = registry.get(document_id)
    if existing_doc is None:
        raise DocumentNotFoundError(document_id)

    vector_store = get_vector_store()
    bm25_store = get_bm25_store()
    settings = get_settings()

    # Purge vectors and BM25 token index
    vector_store.delete_document(document_id)
    bm25_store.delete_document(document_id)

    # Remove from DocumentRegistry
    registry.remove(document_id)

    # Clean up raw PDF file if present
    raw_pdf_path = settings.resolved_upload_dir / f"{document_id}_{existing_doc.filename}"
    if raw_pdf_path.exists():
        try:
            raw_pdf_path.unlink()
        except OSError as err:
            logger.debug("Failed to remove raw file %s: %s", raw_pdf_path, err)

    logger.info("Successfully deleted document %s (%s)", existing_doc.filename, document_id)
    return DocumentDeleteResponse(
        message="Document deleted successfully",
        id=document_id,
    )
