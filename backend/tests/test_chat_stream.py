"""Comprehensive unit and integration tests for Conversational SSE Streaming Endpoint (Task 4.2).

Tests:
1. SSE Stream Protocol: Valid sequence of status, token, citations, and done events.
2. Scoped Document Filtering: documentIds=["doc_1"] strictly scopes retrieval with 0% leakage.
3. Global Retrieval: Empty documentIds searches across all stored documents.
4. Multi-Turn Memory Continuity: Resolves multi-turn references using conversation history.
5. Inability Fallback Stream: Out-of-domain context emits standardized fallback and citations.
6. Validation Error: Whitespace or empty queries return HTTP 422 before starting the stream.
7. Client Disconnect Handling: Premature disconnect handled cleanly without server errors.
8. Conversation Lifecycle: POST /conversations and GET /conversations/{id}/messages work.
"""

import json
from collections.abc import AsyncGenerator, Iterator
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from app.core.config import get_settings
from app.core.exceptions import LLMConfigurationError
from app.main import create_application
from app.schemas.document import ChunkMetadata, ScoredChunk
from app.services.generation.grounding import (
    FALLBACK_RESPONSE,
    GroundedGenerator,
    RetrievalGrader,
    set_grounded_generator,
    set_retrieval_grader,
)
from app.services.generation.llm_client import (
    ChatLLMClient,
    get_llm_client,
    set_llm_client,
)
from app.services.ingestion.registry import DocumentRegistry, set_document_registry
from app.services.retrieval.hybrid import HybridRetriever, set_hybrid_retriever
from app.services.retrieval.memory import ConversationMemoryManager, set_memory_manager
from app.services.retrieval.reformulator import QueryReformulator, set_query_reformulator


def parse_sse_events(raw_text: str) -> list[tuple[str, Any]]:
    """Parse raw Server-Sent Events text wire payload into a list of (event, data) tuples."""
    events: list[tuple[str, Any]] = []
    blocks = raw_text.strip().split("\n\n")
    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().split("\n")
        event_name = ""
        data_str = ""
        for line in lines:
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:") :].strip()
        if event_name:
            try:
                parsed_data = json.loads(data_str)
            except Exception:
                parsed_data = data_str
            events.append((event_name, parsed_data))
    return events


def _create_sample_chunk(
    chunk_id: str,
    document_id: str,
    filename: str,
    content: str,
    page_number: int = 1,
    section_title: str = "Introduction",
) -> ScoredChunk:
    """Helper to construct a typed ScoredChunk."""
    return ScoredChunk(
        chunk_id=chunk_id,
        content=content,
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document_id,
            filename=filename,
            page_number=page_number,
            section_title=section_title,
            token_count=len(content.split()),
            char_start=0,
            char_end=len(content),
        ),
        score=0.95,
    )


@pytest.fixture(autouse=True)
def reset_singletons(tmp_path: Any) -> Iterator[None]:
    """Ensure a clean, isolated environment for each test."""
    temp_registry_path = tmp_path / "documents.json"
    registry = DocumentRegistry(registry_path=temp_registry_path)
    set_document_registry(registry)

    memory_manager = ConversationMemoryManager(max_turns=6)
    set_memory_manager(memory_manager)

    yield

    set_document_registry(None)
    set_memory_manager(None)
    set_query_reformulator(None)
    set_hybrid_retriever(None)
    set_retrieval_grader(None)
    set_grounded_generator(None)


@pytest.fixture
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Fixture providing an async httpx client configured with test application."""
    app = create_application()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.mark.asyncio
async def test_chat_stream_protocol_and_events(async_client: httpx.AsyncClient) -> None:
    """Verify complete SSE event sequence: status -> token -> citations -> done."""
    chunk = _create_sample_chunk(
        chunk_id="chk_1",
        document_id="doc_attn",
        filename="attention.pdf",
        content="Multi-head attention maps queries, keys, and values through 8 heads.",
        page_number=3,
        section_title="Multi-Head Attention",
    )

    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_retriever.retrieve.return_value = [chunk]
    set_hybrid_retriever(mock_retriever)

    mock_grader = RetrievalGrader(grading_fn=lambda _q, _c: "grounded")
    set_retrieval_grader(mock_grader)

    mock_generator = GroundedGenerator(
        generating_fn=lambda _p, _c: "The model employs multi-head attention with 8 heads [1]."
    )
    set_grounded_generator(mock_generator)

    response = await async_client.post(
        "/api/v1/conversations/conv_protocol/messages",
        json={"content": "How many heads are used in multi-head attention?"},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    events = parse_sse_events(response.text)
    event_names = [e[0] for e in events]

    assert "status" in event_names
    assert "token" in event_names
    assert "citations" in event_names
    assert "done" in event_names

    # Check status transitions (Turn 1: starts searching, transitions to streaming)
    statuses = [e[1].get("status") for e in events if e[0] == "status"]
    assert "searching" in statuses
    assert "streaming" in statuses

    # Check streamed tokens reconstruct full response
    tokens = [e[1].get("token") for e in events if e[0] == "token"]
    full_text = "".join(tokens)
    assert "multi-head attention" in full_text
    assert "[1]" in full_text

    # Check structured citations payload
    citations_events = [e[1] for e in events if e[0] == "citations"]
    assert len(citations_events) == 1
    citations = citations_events[0]
    assert len(citations) == 1
    assert citations[0]["id"] == 1
    assert citations[0]["documentId"] == "doc_attn"
    assert citations[0]["filename"] == "attention.pdf"
    assert citations[0]["pageNumber"] == 3
    assert "Multi-Head Attention" in citations[0]["sectionTitle"]

    # Check completion event
    done_events = [e[1] for e in events if e[0] == "done"]
    assert len(done_events) == 1
    assert done_events[0].get("finishReason") == "stop"


@pytest.mark.asyncio
async def test_chat_stream_scoped_retrieval_isolation(
    async_client: httpx.AsyncClient,
) -> None:
    """Verify that specifying documentIds=['doc_1'] strictly scopes retrieval with 0% leakage."""
    chunk_1 = _create_sample_chunk("c1", "doc_1", "doc1.pdf", "Content from Doc 1.")

    captured_doc_ids: list[Any] = []

    def mock_retrieve_scoped(
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        _ = (query, top_k)
        captured_doc_ids.append(document_ids)
        if document_ids == ["doc_1"]:
            return [chunk_1]
        return []

    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_retriever.retrieve.side_effect = mock_retrieve_scoped
    set_hybrid_retriever(mock_retriever)

    set_retrieval_grader(RetrievalGrader(grading_fn=lambda _q, _c: "grounded"))
    set_grounded_generator(GroundedGenerator(generating_fn=lambda _p, _c: "Answer from Doc 1 [1]."))

    response = await async_client.post(
        "/api/v1/conversations/conv_scoped/messages",
        json={
            "content": "Explain Doc 1 concept",
            "documentIds": ["doc_1"],
        },
    )

    assert response.status_code == 200
    assert len(captured_doc_ids) == 1
    assert captured_doc_ids[0] == ["doc_1"]

    events = parse_sse_events(response.text)
    citations = next(e[1] for e in events if e[0] == "citations")
    assert len(citations) == 1
    assert citations[0]["documentId"] == "doc_1"


@pytest.mark.asyncio
async def test_chat_stream_global_retrieval(async_client: httpx.AsyncClient) -> None:
    """Verify that omitting documentIds passes empty/None scoping for global search."""
    captured_doc_ids: list[Any] = []

    def mock_retrieve_global(
        query: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        _ = (query, top_k)
        captured_doc_ids.append(document_ids)
        return []

    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_retriever.retrieve.side_effect = mock_retrieve_global
    set_hybrid_retriever(mock_retriever)

    response = await async_client.post(
        "/api/v1/conversations/conv_global/messages",
        json={"content": "Search across all documents", "documentIds": []},
    )

    assert response.status_code == 200
    assert len(captured_doc_ids) == 1
    assert captured_doc_ids[0] == [] or captured_doc_ids[0] is None


@pytest.mark.parametrize("scope_field", ["documentIds", "document_ids"])
@pytest.mark.asyncio
async def test_scope_transitions_and_persistence(
    async_client: httpx.AsyncClient, scope_field: str
) -> None:
    from app.services.retrieval.conversation_registry import get_conversation_registry
    from app.services.retrieval.memory import get_memory_manager

    captured_scopes: list[list[str]] = []
    retriever = MagicMock(spec=HybridRetriever)

    def retrieve(query: str, document_ids: list[str], top_k: int) -> list[ScoredChunk]:
        _ = (query, top_k)
        captured_scopes.append(list(document_ids))
        return []

    retriever.retrieve.side_effect = retrieve
    set_hybrid_retriever(retriever)
    conv_id = "conv_scope_transition"
    for scope in [["doc_A"], [], ["doc_B"]]:
        response = await async_client.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"content": "Question", scope_field: scope},
        )
        assert "event: error" not in response.text
        assert captured_scopes[-1] == scope
        assert get_memory_manager().get_active_documents(conv_id) == scope
        saved = get_conversation_registry().get(conv_id)
        assert saved is not None
        assert saved.document_ids == scope
        assert saved.documentIds == scope
    await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages", json={"content": "Follow-up"}
    )
    assert captured_scopes == [["doc_A"], [], ["doc_B"], ["doc_B"]]


@pytest.mark.asyncio
async def test_chat_stream_multi_turn_coreference_continuity(
    async_client: httpx.AsyncClient,
) -> None:
    """Verify Turn 2 resolves pronouns and relative references using multi-turn history."""
    chunk = _create_sample_chunk("c1", "doc_attn", "attention.pdf", "Self-attention mechanism.")

    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_retriever.retrieve.return_value = [chunk]
    set_hybrid_retriever(mock_retriever)

    set_retrieval_grader(RetrievalGrader(grading_fn=lambda _q, _c: "grounded"))
    set_grounded_generator(
        GroundedGenerator(generating_fn=lambda _p, _c: "Self-attention connects all positions [1].")
    )

    # Mock reformulator to capture history on Turn 2
    reformulated_queries: list[str] = []

    def mock_reformulate_node(state: dict[str, Any]) -> dict[str, str]:
        history = state.get("messages", [])
        query = state.get("user_query", "")
        if len(history) > 0:
            standalone = f"Reformulated: {query} with prior context"
        else:
            standalone = query
        reformulated_queries.append(standalone)
        return {"standalone_query": standalone}

    mock_reformulator = MagicMock(spec=QueryReformulator)
    mock_reformulator.reformulate_node.side_effect = mock_reformulate_node
    set_query_reformulator(mock_reformulator)

    # Turn 1: Reformulation is bypassed (0 calls to reformulator node)
    resp_turn1 = await async_client.post(
        "/api/v1/conversations/conv_multiturn/messages",
        json={"content": "What is self-attention?"},
    )
    assert resp_turn1.status_code == 200
    assert len(reformulated_queries) == 0

    # Turn 2: Follow-up referencing Turn 1 (reformulation node is executed)
    resp_turn2 = await async_client.post(
        "/api/v1/conversations/conv_multiturn/messages",
        json={"content": "How does it compare to recurrent layers?"},
    )
    assert resp_turn2.status_code == 200
    assert len(reformulated_queries) == 1
    assert "Reformulated: How does it compare to recurrent layers?" in reformulated_queries[0]


@pytest.mark.asyncio
async def test_chat_stream_inability_fallback(async_client: httpx.AsyncClient) -> None:
    """Verify that ungrounded context triggers standardized fallback stream with empty citations."""
    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_retriever.retrieve.return_value = []
    set_hybrid_retriever(mock_retriever)

    set_retrieval_grader(RetrievalGrader(grading_fn=lambda _q, _c: "insufficient"))

    response = await async_client.post(
        "/api/v1/conversations/conv_fallback/messages",
        json={"content": "What is the capital of Mars?"},
    )

    assert response.status_code == 200
    events = parse_sse_events(response.text)

    tokens = [e[1].get("token") for e in events if e[0] == "token"]
    full_text = "".join(tokens)
    assert FALLBACK_RESPONSE in full_text

    citations = next(e[1] for e in events if e[0] == "citations")
    assert citations == []

    done = next(e[1] for e in events if e[0] == "done")
    assert done.get("finishReason") == "stop"


@pytest.mark.asyncio
async def test_chat_stream_empty_query_validation(async_client: httpx.AsyncClient) -> None:
    """Verify that an empty or whitespace query is rejected with HTTP 422 before streaming."""
    response = await async_client.post(
        "/api/v1/conversations/conv_val/messages",
        json={"content": "   \n  "},
    )
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_conversation_management_endpoints(async_client: httpx.AsyncClient) -> None:
    """Verify POST /conversations and GET /conversations/{id}/messages endpoints."""
    # 1. Create a conversation
    create_resp = await async_client.post(
        "/api/v1/conversations",
        json={"documentIds": ["doc_abc"], "title": "Test Chat"},
    )
    assert create_resp.status_code == 201
    created_data = create_resp.json()
    conv_id = created_data.get("conversationId") or created_data.get("conversation_id")
    assert conv_id is not None

    # 2. Add message to memory and verify GET /messages
    memory = ConversationMemoryManager(max_turns=6)
    set_memory_manager(memory)
    memory.add_message(conv_id, role="user", content="Hello document")
    memory.add_message(conv_id, role="assistant", content="Hello user", citations=[1])

    get_resp = await async_client.get(f"/api/v1/conversations/{conv_id}/messages")
    assert get_resp.status_code == 200
    messages = get_resp.json()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello document"
    assert messages[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_chat_stream_client_disconnect_handling(async_client: httpx.AsyncClient) -> None:
    """Verify that a client disconnecting prematurely does not raise unhandled server exceptions."""
    chunk = _create_sample_chunk("c1", "doc_attn", "attention.pdf", "Attention is all you need.")
    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_retriever.retrieve.return_value = [chunk]
    set_hybrid_retriever(mock_retriever)
    set_retrieval_grader(RetrievalGrader(grading_fn=lambda _q, _c: "grounded"))
    set_grounded_generator(
        GroundedGenerator(generating_fn=lambda _p, _c: "Response chunk 1. Response chunk 2.")
    )

    async with async_client.stream(
        "POST",
        "/api/v1/conversations/conv_disconnect/messages",
        json={"content": "Stream and disconnect"},
    ) as stream_response:
        assert stream_response.status_code == 200
        async for line in stream_response.aiter_lines():
            if line.startswith("event:"):
                break


def test_chat_llm_client_operations() -> None:
    """Verify ChatLLMClient completions, streaming, error handling, and singleton access."""
    # 1. Mock client with chat completions
    mock_choice = MagicMock()
    mock_choice.message.content = "Complete answer"
    mock_response = MagicMock(choices=[mock_choice])

    mock_delta_1 = MagicMock()
    mock_delta_1.choices = [MagicMock(delta=MagicMock(content="Token 1 "))]
    mock_delta_2 = MagicMock()
    mock_delta_2.choices = [MagicMock(delta=MagicMock(content="Token 2"))]

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        mock_response,
        [mock_delta_1, mock_delta_2],
    ]

    llm = ChatLLMClient(client=mock_client)
    assert llm.client is mock_client
    assert llm.model == get_settings().chat_model

    # Complete chat
    complete_result = llm.complete_chat([{"role": "user", "content": "Hi"}])
    assert complete_result == "Complete answer"

    # Stream chat
    stream_tokens = list(llm.stream_chat([{"role": "user", "content": "Hi"}]))
    assert stream_tokens == ["Token 1 ", "Token 2"]

    # 2. Callable client
    callable_llm = ChatLLMClient(client=lambda p: f"Echo: {p}")
    assert callable_llm.complete_chat([{"role": "user", "content": "Test"}]) == "Echo: Test"
    callable_tokens = list(callable_llm.stream_chat([{"role": "user", "content": "A B"}]))
    assert len(callable_tokens) > 0

    # 3. Missing API key and client raises LLMConfigurationError
    unconfigured_llm = ChatLLMClient(api_key=None, client=None)
    with pytest.raises(LLMConfigurationError):
        unconfigured_llm.complete_chat([{"role": "user", "content": "Hi"}])

    with pytest.raises(LLMConfigurationError):
        list(unconfigured_llm.stream_chat([{"role": "user", "content": "Hi"}]))

    # 4. Singleton getter and setter
    set_llm_client(None)
    client_instance = get_llm_client()
    assert isinstance(client_instance, ChatLLMClient)
    set_llm_client(llm)
    assert get_llm_client() is llm
    set_llm_client(None)
