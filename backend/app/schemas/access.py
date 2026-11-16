"""Minimal capability exchange schemas; secrets never enter error details."""

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.chat import (
    ChatMessageRequest,
    ConversationCreateRequest,
    ConversationUpdateRequest,
)


class RedemptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    invitation: str = Field(min_length=43, max_length=43)


class SessionResponse(BaseModel):
    workspaceId: str
    expiresAt: str
    limits: dict[str, int]


class RedemptionResponse(SessionResponse):
    sessionToken: str


class LiveChatRequest(ChatMessageRequest):
    model_config = ConfigDict(extra="forbid")


class LiveConversationCreate(ConversationCreateRequest):
    model_config = ConfigDict(extra="forbid")


class LiveConversationUpdate(ConversationUpdateRequest):
    model_config = ConfigDict(extra="forbid")
