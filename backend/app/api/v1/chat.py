"""Conversational Chat and SSE Streaming Endpoints (Task 4.2).

Mounts:
- POST /api/v1/conversations: Initialize a conversation session.
- GET /api/v1/conversations/{conversation_id}/messages: Retrieve session message history.
- POST /api/v1/conversations/{conversation_id}/messages: SSE streaming conversational endpoint
  executing the compiled LangGraph StateGraph (reformulation, scoped hybrid search, grading,
  grounded generation / fallback) and yielding lifecycle status, token chunks, citations,
  and completion markers.
"""

import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import StreamingResponse

from app.core.exceptions import ConversationNotFoundError
from app.schemas.chat import (
    ChatMessage,
    ChatMessageRequest,
    ConversationCreateRequest,
    ConversationCreateResponse,
    ConversationDeleteResponse,
    ConversationSummaryItem,
    ConversationUpdateRequest,
)
from app.services.generation.graph import get_conversational_rag_graph
from app.services.ingestion.registry import get_document_registry
from app.services.retrieval.conversation_registry import get_conversation_registry
from app.services.retrieval.memory import get_memory_manager

logger = logging.getLogger(__name__)

router = APIRouter()


def _format_sse(event: str, data: Any) -> str:
    """Format an event and payload into the standard Server-Sent Events text wire format."""
    if isinstance(data, (dict, list)):
        payload = json.dumps(data)
    else:
        payload = str(data)
    return f"event: {event}\ndata: {payload}\n\n"


@router.get(
    "",
    response_model=list[ConversationSummaryItem],
    summary="List recent conversations",
)
async def list_conversations() -> list[ConversationSummaryItem]:
    """List all saved conversations ordered by updated_at descending."""
    registry = get_conversation_registry()
    return registry.get_all()


@router.post(
    "",
    response_model=ConversationCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Initialize conversation session",
)
async def create_conversation(
    payload: ConversationCreateRequest | None = None,
) -> ConversationCreateResponse:
    """Initialize a new conversational session with optional scoping and title."""
    req = payload or ConversationCreateRequest()
    conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
    memory = get_memory_manager()
    doc_ids = req.get_document_ids()
    session = memory.get_or_create_session(conversation_id, document_ids=doc_ids)

    # Persist in conversation registry
    registry = get_conversation_registry()
    registry.add_or_update(
        ConversationSummaryItem(
            id=conversation_id,
            title=req.title or "New Chat",
            document_ids=doc_ids,
            created_at=session.created_at,
            updated_at=session.updated_at,
            message_count=0,
        )
    )

    return ConversationCreateResponse(
        conversation_id=conversation_id,
        conversationId=conversation_id,
        created_at=session.created_at,
        createdAt=session.created_at.isoformat(),
    )


@router.delete(
    "/{conversation_id}",
    response_model=ConversationDeleteResponse,
    summary="Delete conversation session",
)
async def delete_conversation(conversation_id: str) -> ConversationDeleteResponse:
    """Delete a conversation session and its persisted history."""
    registry = get_conversation_registry()
    memory = get_memory_manager()

    deleted_from_reg = registry.delete(conversation_id)
    deleted_from_mem = memory.delete_session(conversation_id)

    if not deleted_from_reg and not deleted_from_mem:
        raise ConversationNotFoundError(conversation_id)

    return ConversationDeleteResponse(
        status="deleted",
        id=conversation_id,
        conversation_id=conversation_id,
        conversationId=conversation_id,
    )


@router.patch(
    "/{conversation_id}",
    response_model=ConversationSummaryItem,
    summary="Update conversation metadata",
)
async def update_conversation(
    conversation_id: str,
    payload: ConversationUpdateRequest,
) -> ConversationSummaryItem:
    """Update conversation session metadata (e.g. title)."""
    registry = get_conversation_registry()
    item = registry.update_title(conversation_id, payload.title)
    if item is None:
        raise ConversationNotFoundError(conversation_id)
    return item


@router.get(
    "/{conversation_id}/messages",
    response_model=list[ChatMessage],
    summary="Get conversation message history",
)
async def get_conversation_messages(conversation_id: str) -> list[ChatMessage]:
    """Retrieve message history for a conversation session."""
    memory = get_memory_manager()
    messages = memory.get_messages(conversation_id)
    if not messages:
        registry = get_conversation_registry()
        conv = registry.get(conversation_id)
        if conv and conv.messages:
            for msg in conv.messages:
                memory.add_message(
                    conversation_id=conversation_id,
                    role=msg.role,
                    content=msg.content,
                    citations=msg.citations,
                )
            return memory.get_messages(conversation_id)
    return messages


@router.post(
    "/{conversation_id}/messages",
    summary="Stream conversational RAG response via Server-Sent Events",
)
async def stream_conversation_message(
    conversation_id: str,
    payload: ChatMessageRequest,
    request: Request,
) -> StreamingResponse:
    """Execute conversational retrieval and stream response tokens and citations via SSE.

    Lifecycle:
    1. Extracts query and dynamic document scoping filter.
    2. Builds initial state with sliding window dialogue context.
    3. Streams lifecycle status changes ('reformulating', 'searching', 'grading').
    4. Streams incremental response tokens ('token').
    5. Concludes with structured citation payload ('citations') and completion marker ('done').
    6. Gracefully aborts without resource leakage on client disconnection.
    """
    user_query = payload.get_query()
    memory = get_memory_manager()

    # Retrieve or create session, updating active documents if specified in request
    target_doc_ids = payload.get_document_ids()
    if payload.documentIds is not None or "document_ids" in payload.model_fields_set:
        memory.set_active_documents(conversation_id, target_doc_ids)
        active_doc_ids = target_doc_ids
    else:
        session = memory.get_or_create_session(conversation_id)
        active_doc_ids = list(session.active_document_ids)

    # Resolve document filenames for reformulator context injection
    registry = get_document_registry()
    if active_doc_ids:
        titles = [doc.filename for doc_id in active_doc_ids if (doc := registry.get(doc_id))]
    else:
        titles = [doc.filename for doc in registry.get_all()]
    active_document_titles = ", ".join(titles)

    # Construct initial state with previous turns prior to this query
    initial_state = memory.build_initial_state(
        conversation_id=conversation_id,
        user_query=user_query,
        active_document_titles=active_document_titles,
    )

    # Append user turn to memory history and persistent registry
    memory.add_message(conversation_id=conversation_id, role="user", content=user_query)
    conv_registry = get_conversation_registry()
    conv_registry.record_turn(
        conversation_id=conversation_id,
        role="user",
        content=user_query,
        auto_title=True,
        document_ids=active_doc_ids,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        # 1. Emit initial status (reformulating for multi-turn history, searching for Turn 1)
        has_history = len(initial_state.get("messages", [])) > 0
        initial_status = "reformulating" if has_history else "searching"
        yield _format_sse("status", {"status": initial_status, "state": initial_status})

        graph = get_conversational_rag_graph()
        collected_tokens: list[str] = []
        final_citations: list[dict[str, Any]] = []

        try:
            async for mode, event_data in graph.astream(
                initial_state,
                stream_mode=["updates", "custom"],
            ):
                if await request.is_disconnected():
                    logger.debug(
                        "Client disconnected during SSE stream for conversation %s", conversation_id
                    )
                    return

                if mode == "custom":
                    if isinstance(event_data, dict):
                        token = str(event_data.get("token", ""))
                        if token:
                            collected_tokens.append(token)
                            yield _format_sse("token", {"token": token, "content": token})
                    elif isinstance(event_data, str):
                        collected_tokens.append(event_data)
                        yield _format_sse("token", {"token": event_data, "content": event_data})
                elif mode == "updates":
                    if isinstance(event_data, dict):
                        if "reformulate_query" in event_data:
                            yield _format_sse(
                                "status", {"status": "searching", "state": "searching"}
                            )
                        elif "scoped_hybrid_retriever" in event_data:
                            yield _format_sse(
                                "status", {"status": "streaming", "state": "streaming"}
                            )
                        elif "grade_retrieval" in event_data:
                            yield _format_sse(
                                "status", {"status": "streaming", "state": "streaming"}
                            )
                        elif "generate_grounded_response" in event_data:
                            node_dict = event_data["generate_grounded_response"]
                            if isinstance(node_dict, dict):
                                raw_cits = node_dict.get("citations", [])
                                if isinstance(raw_cits, list):
                                    final_citations = raw_cits
                        elif "inability_fallback" in event_data:
                            node_dict = event_data["inability_fallback"]
                            if isinstance(node_dict, dict):
                                raw_cits = node_dict.get("citations", [])
                                if isinstance(raw_cits, list):
                                    final_citations = raw_cits

            # 2. Emit structured citation payload
            yield _format_sse("citations", final_citations)

            # 3. Emit completion marker
            yield _format_sse("done", {"finishReason": "stop"})

            # 4. Record assistant response turn into memory and persistent registry
            full_assistant_reply = "".join(collected_tokens)
            if full_assistant_reply:
                memory.add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=full_assistant_reply,
                    citations=final_citations,
                )
                conv_registry.record_turn(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=full_assistant_reply,
                    citations=final_citations,
                    auto_title=False,
                )

        except Exception as err:
            logger.error("Error during conversational stream: %s", err)
            yield _format_sse(
                "error",
                {"error": {"code": "STREAM_ERROR", "message": str(err)}},
            )
            yield _format_sse("done", {"finishReason": "error"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
