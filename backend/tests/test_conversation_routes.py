"""Unit and integration tests for Conversation Session Management & History (Task 5.4).

Verifies:
1. ConversationRegistry persistence, atomic disk writes, and update ordering.
2. Auto-titling generation from user queries.
3. GET /api/v1/conversations listing recent chats ordered by updated_at desc.
4. DELETE /api/v1/conversations/{id} removing chat from registry and memory.
5. Message history persistence across memory resets.
"""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from app.core.config import Settings
from app.main import create_application
from app.schemas.chat import ConversationSummaryItem
from app.schemas.document import ChunkMetadata, ScoredChunk
from app.services.generation.grounding import (
    GroundedGenerator,
    RetrievalGrader,
    set_grounded_generator,
    set_retrieval_grader,
)
from app.services.generation.llm_client import ChatLLMClient, set_llm_client
from app.services.ingestion.registry import DocumentRegistry, set_document_registry
from app.services.retrieval.conversation_registry import (
    ConversationRegistry,
    generate_auto_title,
    get_conversation_registry,
    set_conversation_registry,
)
from app.services.retrieval.hybrid import HybridRetriever, set_hybrid_retriever
from app.services.retrieval.memory import ConversationMemoryManager, set_memory_manager
from app.services.retrieval.reformulator import QueryReformulator, set_query_reformulator


@pytest.fixture
def isolated_conversation_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Configure an isolated filesystem environment with a temporary conversation registry."""
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    conv_file = storage_dir / "conversations.json"
    doc_file = storage_dir / "documents.json"

    test_settings = Settings(
        BASE_DIR=tmp_path,
        STORAGE_DIR=storage_dir,
    )
    monkeypatch.setattr("app.core.config.get_settings", lambda: test_settings)
    monkeypatch.setattr("app.main.get_settings", lambda: test_settings)

    conv_registry = ConversationRegistry(registry_path=conv_file)
    set_conversation_registry(conv_registry)

    doc_registry = DocumentRegistry(registry_path=doc_file)
    set_document_registry(doc_registry)

    memory_manager = ConversationMemoryManager()
    set_memory_manager(memory_manager)

    mock_llm = MagicMock(spec=ChatLLMClient)

    async def mock_stream_chat(*_args: Any, **_kwargs: Any) -> AsyncGenerator[str, None]:
        yield "Test response."

    async def mock_complete_chat(*_args: Any, **_kwargs: Any) -> str:
        return "Test response."

    mock_llm.stream_chat = mock_stream_chat
    mock_llm.complete_chat = mock_complete_chat
    set_llm_client(mock_llm)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="reformulated test query"))
    ]
    reformulator = QueryReformulator(llm_client=mock_client)
    set_query_reformulator(reformulator)

    dummy_chunk = ScoredChunk(
        chunk_id="chunk_1",
        score=0.95,
        content="Multi-Head Attention projects queries, keys, and values linearly.",
        metadata=ChunkMetadata(
            chunk_id="chunk_1",
            document_id="doc_test",
            filename="attention.pdf",
            page_number=4,
            section_title="Multi-Head Attention",
            token_count=10,
            char_start=0,
            char_end=60,
        ),
    )

    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_retriever.retrieve.return_value = [dummy_chunk]
    set_hybrid_retriever(mock_retriever)

    grader = RetrievalGrader(grading_fn=lambda _q, _c: "grounded")
    set_retrieval_grader(grader)

    generator = GroundedGenerator(
        generating_fn=lambda _p, _c: "Multi-Head Attention explanation [1]."
    )
    set_grounded_generator(generator)

    return {
        "conv_registry": conv_registry,
        "doc_registry": doc_registry,
        "memory_manager": memory_manager,
        "conv_file": conv_file,
    }


def test_auto_titling_helper() -> None:
    """Verify generate_auto_title produces clean, truncated titles from queries."""
    assert generate_auto_title("What is Multi-Head Attention?") == "What is Multi-Head Attention?"
    long_q = "How does Segment-Level Recurrence improve over vanilla Transformer in Transformer-XL?"
    title = generate_auto_title(long_q, max_chars=40)
    assert len(title) <= 43  # 40 chars + optional ellipsis
    assert title.endswith("...")
    assert generate_auto_title("   ") == "New Conversation"
    assert generate_auto_title("### Query with markdown: test") == "Query with markdown: test"


def test_conversation_registry_lifecycle(tmp_path: Path) -> None:
    """Verify ConversationRegistry add, get, list ordering, and delete functionality."""
    reg_path = tmp_path / "conversations.json"
    registry = ConversationRegistry(registry_path=reg_path)

    now = datetime.now(UTC)
    item1 = ConversationSummaryItem(
        id="conv_1",
        title="Attention Query",
        document_ids=["doc_1"],
        created_at=now - timedelta(minutes=10),
        updated_at=now - timedelta(minutes=10),
        message_count=2,
    )
    item2 = ConversationSummaryItem(
        id="conv_2",
        title="Llama 2 Comparison",
        document_ids=["doc_1", "doc_2"],
        created_at=now,
        updated_at=now,
        message_count=4,
    )

    registry.add_or_update(item1)
    registry.add_or_update(item2)

    # Verify order: item2 was updated most recently
    all_items = registry.get_all()
    assert len(all_items) == 2
    assert all_items[0].id == "conv_2"
    assert all_items[1].id == "conv_1"

    # Reload from disk to verify atomic persistence
    reloaded_reg = ConversationRegistry(registry_path=reg_path)
    reloaded_items = reloaded_reg.get_all()
    assert len(reloaded_items) == 2
    assert reloaded_items[0].id == "conv_2"
    assert reloaded_items[0].document_ids == ["doc_1", "doc_2"]

    # Delete conv_1
    assert reloaded_reg.delete("conv_1") is True
    assert len(reloaded_reg.get_all()) == 1
    assert reloaded_reg.get("conv_1") is None
    assert reloaded_reg.delete("non_existent") is False


@pytest.mark.asyncio
async def test_api_conversation_lifecycle(isolated_conversation_env: dict[str, Any]) -> None:
    """Test full API conversation flow: create -> list -> auto-title on message -> delete."""
    _ = isolated_conversation_env
    test_app = create_application()
    transport = httpx.ASGITransport(app=test_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Create conversation
        create_res = await client.post(
            "/api/v1/conversations",
            json={"document_ids": ["doc_1"], "title": "New Chat"},
        )
        assert create_res.status_code == 201
        data = create_res.json()
        conv_id = data["conversationId"]
        assert conv_id.startswith("conv_")

        # 2. List conversations: should include newly created conversation
        list_res = await client.get("/api/v1/conversations")
        assert list_res.status_code == 200
        convs = list_res.json()
        assert len(convs) == 1
        assert convs[0]["id"] == conv_id
        assert convs[0]["title"] == "New Chat"
        assert convs[0]["document_ids"] == ["doc_1"]

        # 3. Stream a message to trigger auto-titling and turn count update
        msg_res = await client.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "Explain how multi-head attention calculates keys and values"},
        )
        assert msg_res.status_code == 200

        # 4. Verify conversation is auto-titled and message count is updated
        list_after_msg = await client.get("/api/v1/conversations")
        assert list_after_msg.status_code == 200
        updated_convs = list_after_msg.json()
        assert len(updated_convs) == 1
        assert updated_convs[0]["id"] == conv_id
        title = updated_convs[0]["title"]
        assert "Multi-Head Attention" in title or "Explain how" in title
        assert updated_convs[0]["message_count"] >= 2  # user + assistant

        # 5. Fetch message history
        messages_res = await client.get(f"/api/v1/conversations/{conv_id}/messages")
        assert messages_res.status_code == 200
        messages_data = messages_res.json()
        assert len(messages_data) >= 2
        assert messages_data[0]["role"] == "user"
        assert messages_data[1]["role"] == "assistant"

        # 6. Delete conversation
        del_res = await client.delete(f"/api/v1/conversations/{conv_id}")
        assert del_res.status_code == 200
        del_data = del_res.json()
        assert del_data["status"] == "deleted"
        assert del_data["id"] == conv_id

        # 7. List should now be empty
        list_after_del = await client.get("/api/v1/conversations")
        assert list_after_del.status_code == 200
        assert len(list_after_del.json()) == 0

        # 8. Deleting non-existent conversation returns 404
        del_404 = await client.delete("/api/v1/conversations/non_existent_conv")
        assert del_404.status_code == 404


@pytest.mark.asyncio
async def test_conversation_persistence_across_memory_resets(
    isolated_conversation_env: dict[str, Any],
) -> None:
    """Verify that messages and metadata are preserved across memory manager resets."""
    _ = isolated_conversation_env
    test_app = create_application()

    transport = httpx.ASGITransport(app=test_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Create and send message
        create_res = await client.post("/api/v1/conversations", json={})
        conv_id = create_res.json()["conversationId"]

        await client.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "What is the primary loss function?"},
        )

        # Simulate backend reboot by clearing in-memory manager
        memory = ConversationMemoryManager()
        set_memory_manager(memory)

        # Calling GET /messages should hydrate from persistent registry
        messages_res = await client.get(f"/api/v1/conversations/{conv_id}/messages")
        assert messages_res.status_code == 200
        messages_data = messages_res.json()
        assert len(messages_data) >= 2
        assert messages_data[0]["content"] == "What is the primary loss function?"


@pytest.mark.asyncio
async def test_conversation_rich_citations_roundtrip(
    isolated_conversation_env: dict[str, Any],
) -> None:
    """Verify that rich citation metadata is persisted and returned without degradation."""
    _ = isolated_conversation_env
    test_app = create_application()
    transport = httpx.ASGITransport(app=test_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_res = await client.post("/api/v1/conversations", json={})
        assert create_res.status_code == 201
        conv_id = create_res.json()["conversationId"]

        # Post user message to chat stream
        msg_res = await client.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "Explain Multi-Head Attention"},
        )
        assert msg_res.status_code == 200

        # Retrieve messages for the conversation
        get_res = await client.get(f"/api/v1/conversations/{conv_id}/messages")
        assert get_res.status_code == 200
        messages = get_res.json()
        assert len(messages) >= 2

        assistant_msg = next(m for m in messages if m["role"] == "assistant")
        assert len(assistant_msg["citations"]) > 0
        first_cit = assistant_msg["citations"][0]
        # Rich citation must be a dictionary, not just an integer
        assert isinstance(first_cit, dict)
        assert first_cit["filename"] == "attention.pdf"
        assert first_cit.get("page_number") == 4 or first_cit.get("pageNumber") == 4
        assert first_cit.get("chunk_id") == "chunk_1" or first_cit.get("chunkId") == "chunk_1"
        cit_text = first_cit.get("quote") or first_cit.get("sourceText") or ""
        assert "Multi-Head Attention" in cit_text


@pytest.mark.asyncio
async def test_update_conversation_title_success(
    isolated_conversation_env: dict[str, Any],
) -> None:
    """Verify that updating a conversation title via PATCH persists to storage."""
    _ = isolated_conversation_env
    test_app = create_application()
    transport = httpx.ASGITransport(app=test_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_res = await client.post("/api/v1/conversations", json={})
        assert create_res.status_code == 201
        conv_id = create_res.json()["conversationId"]

        # Rename conversation
        patch_res = await client.patch(
            f"/api/v1/conversations/{conv_id}",
            json={"title": "Attention Mechanism Analysis"},
        )
        assert patch_res.status_code == 200
        patch_data = patch_res.json()
        assert patch_data["title"] == "Attention Mechanism Analysis"
        assert patch_data["id"] == conv_id

        # Verify listed conversations reflect updated title
        list_res = await client.get("/api/v1/conversations")
        assert list_res.status_code == 200
        conversations = list_res.json()
        matched = next(c for c in conversations if c["id"] == conv_id)
        assert matched["title"] == "Attention Mechanism Analysis"


@pytest.mark.asyncio
async def test_update_conversation_title_validation_error(
    isolated_conversation_env: dict[str, Any],
) -> None:
    """Verify that empty or whitespace titles are rejected with status 422."""
    _ = isolated_conversation_env
    test_app = create_application()
    transport = httpx.ASGITransport(app=test_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        create_res = await client.post("/api/v1/conversations", json={})
        assert create_res.status_code == 201
        conv_id = create_res.json()["conversationId"]

        # Attempt to patch with whitespace title
        patch_res = await client.patch(
            f"/api/v1/conversations/{conv_id}",
            json={"title": "   "},
        )
        assert patch_res.status_code == 422


@pytest.mark.asyncio
async def test_update_conversation_not_found(
    isolated_conversation_env: dict[str, Any],
) -> None:
    """Verify that updating a non-existent conversation returns 404."""
    _ = isolated_conversation_env
    test_app = create_application()
    transport = httpx.ASGITransport(app=test_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        patch_res = await client.patch(
            "/api/v1/conversations/conv_nonexistent",
            json={"title": "Valid Title"},
        )
        assert patch_res.status_code == 404


def test_conversation_registry_update_title(
    isolated_conversation_env: dict[str, Any],
) -> None:
    """Unit test ConversationRegistry.update_title on existing and non-existing records."""
    _ = isolated_conversation_env
    registry = get_conversation_registry()

    conv = registry.record_turn(
        conversation_id="conv_unit_test",
        role="user",
        content="Hello world",
        auto_title=False,
    )
    assert conv.title == "New Chat"

    updated = registry.update_title("conv_unit_test", "Custom Renamed Title")
    assert updated is not None
    assert updated.title == "Custom Renamed Title"

    # Fetch fresh from registry
    fetched = registry.get("conv_unit_test")
    assert fetched is not None
    assert fetched.title == "Custom Renamed Title"

    # Non-existent ID returns None
    missing = registry.update_title("conv_unknown", "Will not work")
    assert missing is None
