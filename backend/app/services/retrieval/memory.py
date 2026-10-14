"""Thread-safe conversation memory manager and sliding window session store."""

import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from app.schemas.chat import (
    ChatMessage,
    ConversationalRAGState,
    ConversationSession,
)


class ConversationMemoryManager:
    """Manages conversational session state, message history, and dynamic document scoping.

    Maintains a sliding window of the last N dialogue turns (by default 6 turns = 12 messages)
    to enforce context limits during downstream query reformulation and generation.
    Thread-safe for concurrent read/write access across multiple sessions.
    """

    def __init__(self, max_turns: int = 6) -> None:
        """Initialize memory manager with turn limit.

        Args:
            max_turns: Number of dialogue turns (user + assistant pairs) to preserve.
        """
        self._max_turns = max_turns
        self._max_messages = max_turns * 2
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = threading.Lock()

    def get_or_create_session(
        self,
        conversation_id: str,
        document_ids: list[str] | None = None,
    ) -> ConversationSession:
        """Retrieve an existing conversation session or initialize a new one.

        Args:
            conversation_id: Unique conversation identifier.
            document_ids: Optional document scoping IDs to initialize/update.

        Returns:
            The ConversationSession instance.
        """
        with self._lock:
            if conversation_id not in self._sessions:
                self._sessions[conversation_id] = ConversationSession(
                    conversation_id=conversation_id,
                    active_document_ids=list(document_ids) if document_ids is not None else [],
                )
            elif document_ids is not None:
                self._sessions[conversation_id].active_document_ids = list(document_ids)
                self._sessions[conversation_id].updated_at = datetime.now(UTC)

            return self._sessions[conversation_id]

    def add_message(
        self,
        conversation_id: str,
        role: Literal["user", "assistant", "system"],
        content: str,
        citations: Sequence[dict[str, Any] | int] | None = None,
    ) -> ChatMessage:
        """Append a message turn to the session and enforce the sliding window.

        Args:
            conversation_id: Unique conversation identifier.
            role: The participant role ('user', 'assistant', 'system').
            content: Text message payload.
            citations: Optional sequence of chunk citation references (rich dicts or indices).

        Returns:
            The newly created ChatMessage instance.
        """
        now = datetime.now(UTC)
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=now,
            citations=list(citations) if citations is not None else [],
        )

        with self._lock:
            if conversation_id not in self._sessions:
                session = ConversationSession(
                    conversation_id=conversation_id,
                    created_at=now,
                    updated_at=now,
                )
                self._sessions[conversation_id] = session
            else:
                session = self._sessions[conversation_id]

            session.messages.append(message)

            # Trim older messages exceeding the sliding window
            if len(session.messages) > self._max_messages:
                session.messages = session.messages[-self._max_messages :]

            session.updated_at = now

        return message

    def get_messages(self, conversation_id: str) -> list[ChatMessage]:
        """Return the current message history for a conversation in chronological order.

        Args:
            conversation_id: Unique conversation identifier.

        Returns:
            List of ChatMessage objects (or empty list if session not found).
        """
        with self._lock:
            session = self._sessions.get(conversation_id)
            if session is None:
                return []
            return list(session.messages)

    def get_formatted_history(self, conversation_id: str) -> str:
        """Format the dialogue history into a plain text transcript for prompt injection.

        Args:
            conversation_id: Unique conversation identifier.

        Returns:
            String with lines formatted as "Role: Content".
        """
        messages = self.get_messages(conversation_id)
        if not messages:
            return ""

        lines: list[str] = []
        for msg in messages:
            role_display = msg.role.capitalize()
            lines.append(f"{role_display}: {msg.content}")

        return "\n".join(lines)

    def set_active_documents(self, conversation_id: str, document_ids: list[str]) -> None:
        """Update active document scope for a conversation.

        Args:
            conversation_id: Unique conversation identifier.
            document_ids: List of document IDs (empty list = global search).
        """
        with self._lock:
            if conversation_id not in self._sessions:
                self._sessions[conversation_id] = ConversationSession(
                    conversation_id=conversation_id,
                    active_document_ids=list(document_ids),
                )
            else:
                self._sessions[conversation_id].active_document_ids = list(document_ids)
                self._sessions[conversation_id].updated_at = datetime.now(UTC)

    def get_active_documents(self, conversation_id: str) -> list[str]:
        """Get active document scoping IDs for a conversation.

        Args:
            conversation_id: Unique conversation identifier.

        Returns:
            List of active document IDs (empty list = global search).
        """
        with self._lock:
            session = self._sessions.get(conversation_id)
            if session is None:
                return []
            return list(session.active_document_ids)

    def build_initial_state(
        self,
        conversation_id: str,
        user_query: str,
        active_document_titles: str = "",
    ) -> ConversationalRAGState:
        """Construct the initial LangGraph ConversationalRAGState dictionary for a query.

        Args:
            conversation_id: Unique conversation identifier.
            user_query: Raw user query input.
            active_document_titles: Formatted string of active document filenames.

        Returns:
            ConversationalRAGState populated with session history and document scope.
        """
        with self._lock:
            session = self._sessions.get(conversation_id)
            if session is None:
                active_doc_ids: list[str] = []
                messages: list[ChatMessage] = []
            else:
                active_doc_ids = list(session.active_document_ids)
                messages = list(session.messages)

        return {
            "conversation_id": conversation_id,
            "active_document_ids": active_doc_ids,
            "active_document_titles": active_document_titles,
            "messages": messages,
            "user_query": user_query,
            "standalone_query": "",
            "retrieved_chunks": [],
            "retrieval_grade": "insufficient",
            "response_text": "",
            "citations": [],
        }

    def clear_session(self, conversation_id: str) -> None:
        """Clear message history of a session while preserving metadata and active documents.

        Args:
            conversation_id: Unique conversation identifier.
        """
        with self._lock:
            session = self._sessions.get(conversation_id)
            if session is not None:
                session.messages.clear()
                session.updated_at = datetime.now(UTC)

    def delete_session(self, conversation_id: str) -> bool:
        """Delete an entire conversation session.

        Args:
            conversation_id: Unique conversation identifier.

        Returns:
            True if session existed and was deleted, False otherwise.
        """
        with self._lock:
            if conversation_id in self._sessions:
                del self._sessions[conversation_id]
                return True
            return False

    def list_sessions(self) -> list[str]:
        """Return list of all active session conversation IDs."""
        with self._lock:
            return list(self._sessions.keys())

    def reset(self) -> None:
        """Clear all active sessions (primarily for testing and resets)."""
        with self._lock:
            self._sessions.clear()


_GLOBAL_MEMORY_MANAGER: ConversationMemoryManager | None = None
_INIT_LOCK = threading.Lock()


def get_memory_manager() -> ConversationMemoryManager:
    """Return the cached singleton ConversationMemoryManager instance."""
    global _GLOBAL_MEMORY_MANAGER
    if _GLOBAL_MEMORY_MANAGER is None:
        with _INIT_LOCK:
            if _GLOBAL_MEMORY_MANAGER is None:
                _GLOBAL_MEMORY_MANAGER = ConversationMemoryManager()
    return _GLOBAL_MEMORY_MANAGER


def set_memory_manager(manager: ConversationMemoryManager | None) -> None:
    """Set or reset the cached singleton ConversationMemoryManager instance."""
    global _GLOBAL_MEMORY_MANAGER
    with _INIT_LOCK:
        _GLOBAL_MEMORY_MANAGER = manager
