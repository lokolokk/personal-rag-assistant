"""Bounded PDF parsing process and recoverable workspace mutations."""

import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import fitz

from app.core.exceptions import AppException
from app.schemas.document import DocumentItem, ParsedDocument
from app.services.ingestion.chunker import StructuralChunker
from app.services.ingestion.parser import PDFParser
from app.services.live_provider import BoundProvider
from app.services.workspaces import Workspace, WorkspaceManager, durable_bytes

# Directory that contains the `app` package, so the parser child process can import it
# no matter which working directory the server was started from.
BACKEND_ROOT = Path(__file__).resolve().parents[3]


def parse_pdf(content: bytes, filename: str, root: Path) -> ParsedDocument:
    pdf = root / "staging/input.pdf"
    result = root / "staging/parsed.json"
    pdf.write_bytes(content)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(BACKEND_ROOT), env.get("PYTHONPATH")) if path
    )
    try:
        process = subprocess.run(
            [sys.executable, "-m", __name__, str(pdf), str(result), filename],
            capture_output=True,
            timeout=30,
            check=False,
            env=env,
        )
        if (
            process.returncode != 0
            or not result.exists()
            or result.stat().st_size > 16 * 1024 * 1024
        ):
            raise AppException("PDF could not be processed within limits.", "INVALID_PDF", 422)
        return ParsedDocument.model_validate_json(result.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired:
        raise AppException("PDF parsing timed out.", "PARSE_TIMEOUT", 422) from None
    finally:
        pdf.unlink(missing_ok=True)
        result.unlink(missing_ok=True)


def ingest(
    bundle: Workspace,
    manager: WorkspaceManager,
    provider: BoundProvider,
    content: bytes,
    filename: str,
) -> DocumentItem:
    provider.require()
    if not content.startswith(b"%PDF-") or not filename.lower().endswith(".pdf"):
        raise AppException("A valid PDF is required.", "INVALID_PDF", 422)
    if len(content) > 10 * 1024 * 1024:
        raise AppException("PDF exceeds 10 MiB.", "FILE_TOO_LARGE", 413)
    if (
        len(filename) > 200
        or any(c in filename for c in ("/", "\\", "\x00"))
        or any(ord(c) < 32 for c in filename)
    ):
        raise AppException("Invalid filename.", "INVALID_FILENAME", 422)
    old = bundle.documents.get_by_filename(filename)
    docs = bundle.documents.get_all()
    if old is None and len(docs) >= 10:
        raise AppException("Document limit reached.", "QUOTA_EXCEEDED", 429)
    pdf_bytes = sum(p.stat().st_size for p in (bundle.root / "uploads").glob("*.pdf"))
    if pdf_bytes + len(content) > 100 * 1024 * 1024:
        raise AppException("PDF storage limit reached.", "QUOTA_EXCEEDED", 429)
    manager.storage_admission(bundle, len(content) * 3)
    parsed = parse_pdf(content, filename, bundle.root)
    doc_id = old.id if old else "doc_" + uuid.uuid4().hex
    chunks = StructuralChunker().chunk_document(parsed, doc_id)
    if not chunks or bundle.vectors.count() - (old.chunk_count if old else 0) + len(chunks) > 10000:
        raise AppException("Document chunk limit reached or no text found.", "QUOTA_EXCEEDED", 422)
    manager.storage_admission(bundle, len(content) * 3 + len(chunks) * 16384)
    total_chunks = sum(manager.get(w).vectors.count() for w in manager.access.workspace_ids())
    if total_chunks - (old.chunk_count if old else 0) + len(chunks) > 50000:
        raise AppException("Global chunk limit reached.", "QUOTA_EXCEEDED", 429)
    vectors: list[list[float]] = []
    provider.prepare_embeddings([chunk.content for chunk in chunks])
    for start in range(0, len(chunks), 16):
        vectors.extend(provider.embeddings([c.content for c in chunks[start : start + 16]]))
    provider.check()
    doc = DocumentItem(
        id=doc_id,
        filename=filename,
        page_count=parsed.page_count,
        chunk_count=len(chunks),
        uploaded_at=datetime.now(UTC).isoformat(),
    )
    with bundle.lock:
        bundle.begin_mutation(doc_id)
        try:
            bundle.vectors.delete_document(doc_id)
            bundle.vectors.add_chunks(chunks, embeddings=vectors)
            durable_bytes(bundle.root / "uploads" / f"{doc_id}.pdf", content)
            bundle.documents.add_or_update(doc)
            bundle.commit_mutation()
            bundle.bm25.clear()
            bundle.bm25.add_chunks(bundle.vectors.get_all_chunks())
        except BaseException:
            bundle.recover()
            bundle.hydrate()
            raise
    return doc


def delete_document(bundle: Workspace, document_id: str) -> None:
    with bundle.lock:
        bundle.validate_documents([document_id])
        bundle.begin_mutation(document_id)
        try:
            bundle.vectors.delete_document(document_id)
            bundle.documents.remove(document_id)
            (bundle.root / "uploads" / f"{document_id}.pdf").unlink()
            bundle.commit_mutation()
            bundle.bm25.delete_document(document_id)
        except BaseException:
            bundle.recover()
            bundle.hydrate()
            raise


if __name__ == "__main__":
    source, destination, original_name = sys.argv[1:]
    with fitz.open(source) as document:
        if len(document) > 100 or document.needs_pass:
            raise ValueError("Page limit or encrypted document")
    parsed_document = PDFParser().parse_bytes(Path(source).read_bytes(), original_name)
    Path(destination).write_text(json.dumps(parsed_document.model_dump()), encoding="utf-8")
