"""Explicit isolated services; no live request changes any process singleton."""

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from app.core.config import get_settings
from app.core.exceptions import AppException
from app.schemas.chat import ChatMessage
from app.schemas.document import DocumentChunk, FullPassage, ScoredChunk
from app.services.access import AccessStore
from app.services.ingestion.registry import DocumentRegistry
from app.services.live_provider import BoundProvider, Provider
from app.services.retrieval.bm25_store import BM25Store
from app.services.retrieval.conversation_registry import ConversationRegistry
from app.services.retrieval.memory import ConversationMemoryManager
from app.services.retrieval.vector_store import VectorStore


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def durable_bytes(path: Path, value: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


class ExplicitEmbeddings(EmbeddingFunction[Documents]):
    def __call__(self, input: Documents) -> Embeddings:
        _ = input
        raise AppException("Explicit live embeddings required.", "PROVIDER_UNAVAILABLE", 503)


@dataclass
class Workspace:
    root: Path
    documents: DocumentRegistry
    conversations: ConversationRegistry
    vectors: VectorStore
    bm25: BM25Store
    memory: ConversationMemoryManager
    lock: threading.RLock

    def full_passage(self, chunk_id: str) -> FullPassage:
        """Caller holds the workspace lock; never consult another workspace."""
        chunk = self.vectors.get_chunk(chunk_id)
        document = self.documents.get(chunk.metadata.document_id) if chunk else None
        if (
            chunk is None
            or document is None
            or chunk.metadata.chunk_id != chunk.chunk_id
            or chunk.metadata.filename != document.filename
        ):
            raise AppException("Source is unavailable.", "CHUNK_NOT_FOUND", 404)
        return FullPassage(
            chunkId=chunk.chunk_id,
            text=chunk.content,
            filename=chunk.metadata.filename,
            pageNumber=chunk.metadata.page_number,
        )

    def citation_history(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Annotate old previews from stored text without changing saved evidence."""
        result: list[ChatMessage] = []
        for message in messages:
            citations: list[dict[str, Any] | int] = []
            for citation in message.citations:
                annotated = dict(citation) if isinstance(citation, dict) else citation
                if isinstance(annotated, dict) and "isTruncated" not in annotated:
                    chunk_id = annotated.get("chunkId", annotated.get("chunk_id"))
                    if isinstance(chunk_id, str) and chunk_id:
                        try:
                            original = self.full_passage(chunk_id).text.strip()
                        except AppException as error:
                            if error.code != "CHUNK_NOT_FOUND":
                                raise
                        else:
                            preview = annotated.get("sourceText", annotated.get("quote", ""))
                            annotated["isTruncated"] = (
                                len(original) > 200
                                and preview != original
                                and preview == original[:200] + "..."
                            )
                citations.append(annotated)
            result.append(message.model_copy(update={"citations": citations}))
        return result

    def validate_documents(self, ids: list[str]) -> list[str]:
        if len(ids) > 10 or len(set(ids)) != len(ids):
            raise AppException("Invalid document scope.", "INVALID_SCOPE", 422)
        if any(self.documents.get(doc_id) is None for doc_id in ids):
            raise AppException("Document does not exist.", "DOCUMENT_NOT_FOUND", 404)
        return ids

    def check_candidates(self, chunks: list[ScoredChunk], ids: list[str]) -> None:
        allowed = set(ids) if ids else {doc.id for doc in self.documents.get_all()}
        known = {chunk.chunk_id: chunk for chunk in self.vectors.get_all_chunks()}
        for chunk in chunks:
            original = known.get(chunk.chunk_id)
            if (
                chunk.metadata.document_id not in allowed
                or original is None
                or (original.content != chunk.content or original.metadata != chunk.metadata)
            ):
                raise AppException("Retrieval scope validation failed.", "ISOLATION_FAILURE", 503)

    def hydrate(self) -> None:
        docs = {doc.id: doc for doc in self.documents.get_all()}
        chunks = self.vectors.get_all_chunks()
        if any(c.metadata.document_id not in docs for c in chunks):
            raise ValueError("Orphaned live chunks")
        for doc_id, doc in docs.items():
            owned = [chunk for chunk in chunks if chunk.metadata.document_id == doc_id]
            if len(owned) != doc.chunk_count or any(
                chunk.metadata.filename != doc.filename for chunk in owned
            ):
                raise ValueError("Live document/vector integrity mismatch")
            if doc.chunk_count and not (self.root / "uploads" / f"{doc_id}.pdf").is_file():
                raise ValueError("Missing live PDF")
        self.bm25.clear()
        self.bm25.add_chunks(chunks)
        self.memory.reset()
        for conversation in self.conversations.get_all():
            self.memory.get_or_create_session(conversation.id, conversation.document_ids)
            for message in conversation.messages:
                self.memory.add_message(
                    conversation.id, message.role, message.content, message.citations
                )

    def recover(self) -> None:
        journal_path = self.root / "staging/mutation.json"
        if not journal_path.exists():
            return
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        document_id = str(journal["document_id"])
        if (
            not document_id.startswith("doc_")
            or len(document_id) != 36
            or not all(c in "0123456789abcdef" for c in document_id[4:])
        ):
            raise ValueError("Invalid journal document ID")
        self.vectors.delete_document(document_id)
        chunks = [DocumentChunk.model_validate(item) for item in journal["chunks"]]
        self.vectors.add_chunks(chunks, embeddings=journal["vectors"])
        atomic_json(self.root / "documents.json", journal["documents"])
        pdf = self.root / "uploads" / f"{document_id}.pdf"
        old_pdf = self.root / "staging/old.pdf"
        if journal["had_pdf"]:
            durable_bytes(pdf, old_pdf.read_bytes())
        else:
            pdf.unlink(missing_ok=True)
        journal_path.unlink()
        old_pdf.unlink(missing_ok=True)

    def begin_mutation(self, document_id: str) -> None:
        chunks, vectors = self.vectors.document_snapshot(document_id)
        pdf = self.root / "uploads" / f"{document_id}.pdf"
        if pdf.exists():
            durable_bytes(self.root / "staging/old.pdf", pdf.read_bytes())
        atomic_json(
            self.root / "staging/mutation.json",
            {
                "document_id": document_id,
                "chunks": [c.model_dump() for c in chunks],
                "vectors": vectors,
                "documents": [d.model_dump() for d in self.documents.get_all()],
                "had_pdf": pdf.exists(),
            },
        )

    def commit_mutation(self) -> None:
        (self.root / "staging/mutation.json").unlink()
        (self.root / "staging/old.pdf").unlink(missing_ok=True)


class WorkspaceManager:
    def __init__(self, root: Path, access: AccessStore, provider: Provider | None) -> None:
        self.root = root
        self.access = access
        self.provider = provider
        self._bundles: dict[str, Workspace] = {}
        self._lock = threading.RLock()
        with access.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS initialized(workspace TEXT PRIMARY KEY)")

    def get(self, workspace_id: str) -> Workspace:
        if workspace_id not in self.access.workspace_ids():
            raise AppException("Workspace unavailable.", "ACCESS_INVALID", 401)
        with self._lock:
            if workspace_id in self._bundles:
                return self._bundles[workspace_id]
            root = self.root / "workspaces" / workspace_id
            settings = get_settings()
            expected = {
                "version": 1,
                "workspace": workspace_id,
                "embedding_model": settings.EMBEDDING_MODEL,
                "dimensions": 1024,
            }
            with self.access.transaction() as db:
                initialized = db.execute(
                    "SELECT workspace FROM initialized WHERE workspace=?", (workspace_id,)
                ).fetchone()
                if not root.exists() and initialized is None:
                    root.mkdir(parents=True)
                    for directory in ("uploads", "chroma", "staging"):
                        (root / directory).mkdir()
                    atomic_json(root / "manifest.json", expected)
                    atomic_json(root / "documents.json", [])
                    atomic_json(root / "conversations.json", [])
                    db.execute("INSERT INTO initialized VALUES(?)", (workspace_id,))
                elif initialized is None:
                    raise ValueError("Unregistered workspace directory")
            if root.is_symlink() or any(
                (root / name).is_symlink()
                for name in (
                    "uploads",
                    "chroma",
                    "staging",
                    "documents.json",
                    "conversations.json",
                    "manifest.json",
                )
            ):
                raise ValueError("Workspace symlinks are prohibited")
            if json.loads((root / "manifest.json").read_text(encoding="utf-8")) != expected:
                raise ValueError("Workspace manifest mismatch")
            if not all((root / name).is_dir() for name in ("uploads", "chroma", "staging")):
                raise OSError("Missing workspace storage directory")
            bundle = Workspace(
                root,
                DocumentRegistry(root / "documents.json", strict=True),
                ConversationRegistry(root / "conversations.json", strict=True),
                VectorStore(root / "chroma", embedding_function=ExplicitEmbeddings()),
                BM25Store(),
                ConversationMemoryManager(),
                threading.RLock(),
            )
            bundle.recover()
            bundle.hydrate()
            self._bundles[workspace_id] = bundle
            return bundle

    def storage_admission(self, bundle: Workspace, extra: int) -> None:
        def size(root: Path) -> int:
            return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())

        if size(bundle.root) + extra > 1024**3 or size(self.root) + extra > 5 * 1024**3:
            raise AppException("Storage limit reached.", "QUOTA_EXCEEDED", 429)


class RequestVectorStore(VectorStore):
    """Adapter binds the provider to this request while retaining native dense math."""

    def __init__(self, bundle: Workspace, provider: BoundProvider) -> None:
        self.bundle = bundle
        self.provider = provider

    def get_all_chunks(self) -> list[DocumentChunk]:
        """Read continuation evidence from this workspace's existing vectors only."""
        return self.bundle.vectors.get_all_chunks()

    def query(
        self,
        query_text: str,
        document_ids: list[str] | None = None,
        top_n: int = 10,
        query_embedding: list[float] | None = None,
    ) -> list[ScoredChunk]:
        _ = query_embedding
        self.bundle.validate_documents(document_ids or [])
        if self.bundle.vectors.count() == 0:
            return []
        vector = self.provider.embeddings([query_text])[0]
        result = self.bundle.vectors.query(query_text, document_ids, top_n, query_embedding=vector)
        self.bundle.check_candidates(result, document_ids or [])
        return result


class RequestBM25Store(BM25Store):
    def __init__(self, bundle: Workspace) -> None:
        self.bundle = bundle

    def query(
        self, query_text: str, document_ids: list[str] | None = None, top_n: int = 10
    ) -> list[ScoredChunk]:
        result = self.bundle.bm25.query(query_text, document_ids, top_n)
        self.bundle.check_candidates(result, document_ids or [])
        return result

    def get_chunks_for_pages(self, pages: set[tuple[str, int]]) -> list[DocumentChunk]:
        """Read caption and continuation chunks from this request's workspace only."""
        return self.bundle.bm25.get_chunks_for_pages(pages)
