"""Pydantic schemas and LangGraph state types for conversational retrieval."""

from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    """Represents a single message turn in a multi-turn conversation."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    citations: list[dict[str, Any] | int] = Field(default_factory=list)


class ConversationSession(BaseModel):
    """Tracks session-level message history and active document scoping."""

    conversation_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
    active_document_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationSummaryItem(BaseModel):
    """Metadata summary of a conversation session for listing and persistence."""

    id: str
    conversation_id: str = ""
    conversationId: str = ""
    title: str = "New Chat"
    document_ids: list[str] = Field(default_factory=list)
    documentIds: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    createdAt: str = ""
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updatedAt: str = ""
    message_count: int = 0
    messageCount: int = 0
    messages: list[ChatMessage] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.conversation_id:
            self.conversation_id = self.id
        if not self.conversationId:
            self.conversationId = self.id
        if not self.documentIds and self.document_ids:
            self.documentIds = list(self.document_ids)
        elif not self.document_ids and self.documentIds:
            self.document_ids = list(self.documentIds)
        if not self.createdAt:
            self.createdAt = self.created_at.isoformat()
        if not self.updatedAt:
            self.updatedAt = self.updated_at.isoformat()
        if self.messageCount == 0 and self.message_count != 0:
            self.messageCount = self.message_count
        elif self.message_count == 0 and self.messageCount != 0:
            self.message_count = self.messageCount


class ConversationDeleteResponse(BaseModel):
    """Response payload confirming conversation deletion."""

    status: str = "deleted"
    id: str
    conversation_id: str = ""
    conversationId: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.conversation_id:
            self.conversation_id = self.id
        if not self.conversationId:
            self.conversationId = self.id


class ChatMessageRequest(BaseModel):
    """Payload for submitting a user message to an active conversation."""

    content: str = Field(default="", description="User message content / query string")
    message: str | None = Field(default=None, description="Alias for content")
    document_ids: list[str] = Field(
        default_factory=list,
        description="Optional document IDs to scope retrieval to",
    )
    documentIds: list[str] | None = Field(
        default=None,
        description="CamelCase alias for document_ids",
    )

    @model_validator(mode="after")
    def validate_query_content(self) -> "ChatMessageRequest":
        """Ensure either content or message provides a non-empty query string."""
        raw_query = self.content.strip() if self.content else ""
        if not raw_query and self.message:
            raw_query = self.message.strip()

        if not raw_query:
            raise ValueError("Query content cannot be empty or whitespace.")

        if not self.content and self.message:
            self.content = self.message

        return self

    def get_query(self) -> str:
        """Return the trimmed user query string."""
        if self.content and self.content.strip():
            return self.content.strip()
        if self.message and self.message.strip():
            return self.message.strip()
        return ""

    def get_document_ids(self) -> list[str]:
        """Return the effective document IDs for retrieval scoping."""
        if self.documentIds is not None:
            return list(self.documentIds)
        return list(self.document_ids)


class ConversationCreateRequest(BaseModel):
    """Payload for initializing a new conversational session."""

    document_ids: list[str] = Field(default_factory=list)
    documentIds: list[str] | None = Field(default=None)
    title: str | None = Field(default=None)

    def get_document_ids(self) -> list[str]:
        """Return the effective document IDs for retrieval scoping."""
        if self.documentIds is not None:
            return list(self.documentIds)
        return list(self.document_ids)


class ConversationCreateResponse(BaseModel):
    """Response returned upon creating a new conversation."""

    conversation_id: str
    conversationId: str
    created_at: datetime
    createdAt: str


class ConversationUpdateRequest(BaseModel):
    """Payload for updating conversation metadata (e.g., title)."""

    title: str = Field(description="New title for the conversation")

    @model_validator(mode="after")
    def validate_title(self) -> "ConversationUpdateRequest":
        """Ensure title is non-empty after trimming."""
        cleaned = self.title.strip()
        if not cleaned:
            raise ValueError("Conversation title cannot be empty or whitespace.")
        self.title = cleaned
        return self


class CitationPayload(BaseModel):
    """Structured citation metadata payload emitted in SSE stream."""

    id: int
    citationIndex: int
    chunk_id: str
    chunkId: str
    document_id: str
    documentId: str
    filename: str
    page_number: int
    pageNumber: int
    section_title: str
    sectionTitle: str
    quote: str
    sourceText: str


class ConversationalRAGState(TypedDict):
    """LangGraph state schema for conversational document retrieval.

    Strictly conforms to docs/DESIGN.md Section 3.1 specification.
    """

    conversation_id: str
    active_document_ids: list[str]  # Empty = all docs, or subset for scoped chat
    active_document_titles: str  # Formatted string of filenames for prompt injection
    messages: list[ChatMessage]  # Recent dialogue history (sliding window)
    user_query: str  # Raw user input
    standalone_query: str  # Reformulated standalone search query
    retrieved_chunks: list[Any]  # Top-K fused DocumentChunk/ScoredChunk objects from Layer 1
    retrieval_grade: Literal["grounded", "insufficient"]
    response_text: str  # Generated response content
    citations: list[dict[str, Any]]  # Resolved citations mapped to chunk metadata
