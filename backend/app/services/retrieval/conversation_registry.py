"""Lightweight persistent conversation registry for session lifecycle tracking."""

import json
import logging
import re
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from app.core.config import get_settings
from app.schemas.chat import ChatMessage, ConversationSummaryItem

logger = logging.getLogger(__name__)


def generate_auto_title(query: str, max_chars: int = 40) -> str:
    """Generate a clean, readable conversation title from the first user query.

    Args:
        query: Raw user query string.
        max_chars: Maximum character length for the title (excluding ellipsis).

    Returns:
        Formatted title string.
    """
    cleaned = query.strip()
    if not cleaned:
        return "New Conversation"

    # Strip markdown headers, bullet points, and code markers
    cleaned = re.sub(r"^[#*`>\-\d\.\s]+", "", cleaned)
    # Collapse multiple whitespace characters and newlines into a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return "New Conversation"

    if len(cleaned) <= max_chars:
        return cleaned

    # Truncate at word boundary if possible
    truncated = cleaned[:max_chars].rstrip()
    last_space = truncated.rfind(" ")
    if last_space > int(max_chars * 0.6):
        truncated = truncated[:last_space]

    return f"{truncated}..."


class ConversationRegistry:
    """Manages persistent conversation session summaries and message history.

    Persists sessions in storage/conversations.json to support listing past sessions,
    restoring messages across server restarts, and deleting conversations.
    """

    def __init__(self, registry_path: Path | str | None = None) -> None:
        settings = get_settings()
        if registry_path is not None:
            self._path = Path(registry_path)
        else:
            self._path = settings.resolve_path(Path("storage/conversations.json"))

        self._lock = threading.RLock()
        self._ensure_parent_dir()

    def _ensure_parent_dir(self) -> None:
        """Ensure parent directory of the registry file exists."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> list[ConversationSummaryItem]:
        """Read and parse conversation records from disk under lock."""
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
                return [ConversationSummaryItem.model_validate(item) for item in raw_items]
            except Exception as err:
                logger.warning("Failed to load conversation registry from %s: %s", self._path, err)
                return []

    def _save_data(self, items: list[ConversationSummaryItem]) -> None:
        """Save conversation records atomically to disk under lock."""
        with self._lock:
            self._ensure_parent_dir()
            data = [item.model_dump() for item in items]
            tmp_path = self._path.with_suffix(".tmp")
            try:
                tmp_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
                tmp_path.replace(self._path)
            except Exception as err:
                logger.error("Failed to persist conversation registry to %s: %s", self._path, err)
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError as unlink_err:
                        logger.debug("Failed to clean up temp file: %s", unlink_err)
                raise

    def get_all(self) -> list[ConversationSummaryItem]:
        """Return all registered conversations ordered by updated_at descending."""
        items = self._load_data()
        return sorted(items, key=lambda x: x.updated_at, reverse=True)

    def get(self, conversation_id: str) -> ConversationSummaryItem | None:
        """Retrieve a conversation summary by its ID."""
        items = self._load_data()
        for item in items:
            if item.id == conversation_id or item.conversation_id == conversation_id:
                return item
        return None

    def add_or_update(self, conv: ConversationSummaryItem) -> None:
        """Insert or atomically update a conversation record in the registry."""
        with self._lock:
            items = self._load_data()
            updated = False
            for idx, existing in enumerate(items):
                if existing.id == conv.id or existing.conversation_id == conv.id:
                    items[idx] = conv
                    updated = True
                    break
            if not updated:
                items.append(conv)
            self._save_data(items)

    def record_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: Sequence[dict[str, Any] | int] | None = None,
        auto_title: bool = True,
        document_ids: list[str] | None = None,
    ) -> ConversationSummaryItem:
        """Record a single dialogue turn into persistent storage.

        Auto-titles the conversation if it currently has a default title and role is user.
        """
        now = datetime.now(UTC)
        literal_role: Literal["user", "assistant", "system"]
        if role == "assistant":
            literal_role = "assistant"
        elif role == "user":
            literal_role = "user"
        else:
            literal_role = "system"

        message = ChatMessage(
            role=literal_role,
            content=content,
            timestamp=now,
            citations=list(citations) if citations is not None else [],
        )

        with self._lock:
            items = self._load_data()
            target_conv: ConversationSummaryItem | None = None

            for conv in items:
                if conv.id == conversation_id or conv.conversation_id == conversation_id:
                    target_conv = conv
                    break

            if target_conv is None:
                title = (
                    generate_auto_title(content) if (role == "user" and auto_title) else "New Chat"
                )
                target_conv = ConversationSummaryItem(
                    id=conversation_id,
                    title=title,
                    document_ids=list(document_ids or []),
                    created_at=now,
                    updated_at=now,
                    message_count=1,
                    messages=[message],
                )
                items.append(target_conv)
            else:
                target_conv.messages.append(message)
                target_conv.message_count = len(target_conv.messages)
                target_conv.updated_at = now
                if document_ids is not None:
                    target_conv.document_ids = list(document_ids)
                    target_conv.documentIds = list(document_ids)

                # Auto-title if user message and title is default
                is_default_title = (
                    target_conv.title in ("New Chat", "New Conversation", "")
                    or not target_conv.title
                )
                if role == "user" and auto_title and is_default_title:
                    target_conv.title = generate_auto_title(content)

            self._save_data(items)
            return target_conv

    def update_title(self, conversation_id: str, title: str) -> ConversationSummaryItem | None:
        """Update the title of an existing conversation in persistent storage.

        Args:
            conversation_id: Unique identifier of the conversation.
            title: New non-empty title string.

        Returns:
            Updated ConversationSummaryItem if found, else None.
        """
        with self._lock:
            items = self._load_data()
            for item in items:
                if item.id == conversation_id or item.conversation_id == conversation_id:
                    item.title = title
                    now = datetime.now(UTC)
                    item.updated_at = now
                    item.updatedAt = now.isoformat()
                    self._save_data(items)
                    return item
            return None

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation from the registry."""
        with self._lock:
            items = self._load_data()
            initial_count = len(items)
            remaining = [
                item
                for item in items
                if item.id != conversation_id and item.conversation_id != conversation_id
            ]
            if len(remaining) < initial_count:
                self._save_data(remaining)
                return True
            return False

    def clear(self) -> None:
        """Clear all conversation records (testing utility)."""
        with self._lock:
            self._save_data([])


_GLOBAL_CONVERSATION_REGISTRY: ConversationRegistry | None = None
_CONV_REGISTRY_LOCK = threading.Lock()


def get_conversation_registry() -> ConversationRegistry:
    """Return the cached singleton ConversationRegistry instance."""
    global _GLOBAL_CONVERSATION_REGISTRY
    if _GLOBAL_CONVERSATION_REGISTRY is None:
        with _CONV_REGISTRY_LOCK:
            if _GLOBAL_CONVERSATION_REGISTRY is None:
                _GLOBAL_CONVERSATION_REGISTRY = ConversationRegistry()
    return _GLOBAL_CONVERSATION_REGISTRY


def set_conversation_registry(registry: ConversationRegistry | None) -> None:
    """Set or reset the cached singleton ConversationRegistry instance."""
    global _GLOBAL_CONVERSATION_REGISTRY
    with _CONV_REGISTRY_LOCK:
        _GLOBAL_CONVERSATION_REGISTRY = registry
