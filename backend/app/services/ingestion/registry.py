"""Lightweight document registry for instantaneous document listing and metadata tracking."""

import json
import logging
import threading
from pathlib import Path

from app.core.config import get_settings
from app.schemas.document import DocumentItem

logger = logging.getLogger(__name__)


class DocumentRegistry:
    """Manages persistent document metadata in a lightweight JSON registry.

    Stores summaries (id, filename, page_count, chunk_count, uploaded_at, status)
    in storage/documents.json for instantaneous UI listing without scanning the vector database.
    """

    def __init__(self, registry_path: Path | str | None = None) -> None:
        settings = get_settings()
        if registry_path is not None:
            self._path = Path(registry_path)
        else:
            self._path = settings.resolve_path(Path("storage/documents.json"))

        self._lock = threading.RLock()
        self._ensure_parent_dir()

    def _ensure_parent_dir(self) -> None:
        """Ensure parent directory of the registry file exists."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> list[DocumentItem]:
        """Read and parse document records from disk under lock."""
        with self._lock:
            if not self._path.exists():
                return []
            try:
                content = self._path.read_text(encoding="utf-8")
                if not content.strip():
                    return []
                raw_items = json.loads(content)
                if not isinstance(raw_items, list):
                    return []
                return [DocumentItem.model_validate(item) for item in raw_items]
            except Exception as err:
                logger.warning("Failed to load document registry from %s: %s", self._path, err)
                return []

    def _save_data(self, items: list[DocumentItem]) -> None:
        """Save document records atomically to disk under lock."""
        with self._lock:
            self._ensure_parent_dir()
            data = [item.model_dump() for item in items]
            tmp_path = self._path.with_suffix(".tmp")
            try:
                tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                tmp_path.replace(self._path)
            except Exception as err:
                logger.error("Failed to persist document registry to %s: %s", self._path, err)
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError as unlink_err:
                        logger.debug("Failed to clean up temp file: %s", unlink_err)
                raise

    def get_all(self) -> list[DocumentItem]:
        """Return all registered documents."""
        return self._load_data()

    def get(self, document_id: str) -> DocumentItem | None:
        """Retrieve a document by its ID (or document_id alias)."""
        items = self._load_data()
        for item in items:
            if item.id == document_id or item.document_id == document_id:
                return item
        return None

    def get_by_filename(self, filename: str) -> DocumentItem | None:
        """Retrieve a document by its original filename."""
        items = self._load_data()
        for item in items:
            if item.filename == filename:
                return item
        return None

    def add_or_update(self, doc: DocumentItem) -> None:
        """Insert or atomically replace a document record in the registry."""
        with self._lock:
            items = self._load_data()
            updated = False
            for idx, existing in enumerate(items):
                if existing.id == doc.id or existing.filename == doc.filename:
                    items[idx] = doc
                    updated = True
                    break
            if not updated:
                items.append(doc)
            self._save_data(items)

    def remove(self, document_id: str) -> bool:
        """Remove a document record from the registry by ID.

        Returns True if the document was found and removed, False otherwise.
        """
        with self._lock:
            items = self._load_data()
            initial_count = len(items)
            items = [d for d in items if d.id != document_id and d.document_id != document_id]
            if len(items) < initial_count:
                self._save_data(items)
                return True
            return False

    def clear(self) -> None:
        """Clear all records from the registry."""
        with self._lock:
            self._save_data([])


_document_registry_instance: DocumentRegistry | None = None


def get_document_registry() -> DocumentRegistry:
    """Return the global singleton instance of DocumentRegistry."""
    global _document_registry_instance
    if _document_registry_instance is None:
        _document_registry_instance = DocumentRegistry()
    return _document_registry_instance


def set_document_registry(registry: DocumentRegistry | None) -> None:
    """Set or reset the global singleton instance of DocumentRegistry (for testing/DI)."""
    global _document_registry_instance
    _document_registry_instance = registry
