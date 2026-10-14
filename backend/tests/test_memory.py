"""Unit and integration tests for conversational memory manager and LangGraph state schema."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatMessage, ConversationalRAGState
from app.services.retrieval.memory import ConversationMemoryManager, get_memory_manager


def test_chat_message_schema_validation() -> None:
    """Assert ChatMessage validates roles, defaults timestamp and citations, and rejects invalid."""
    msg = ChatMessage(role="user", content="What are the main results?")
    assert msg.role == "user"
    assert msg.content == "What are the main results?"
    assert isinstance(msg.timestamp, datetime)
    assert msg.timestamp.tzinfo is not None
    assert msg.citations == []

    # Valid roles
    assistant_msg = ChatMessage(
        role="assistant",
        content="The results are in Table 1.",
        citations=[1, 2],
    )
    assert assistant_msg.role == "assistant"
    assert assistant_msg.citations == [1, 2]

    system_msg = ChatMessage(role="system", content="System instruction.")
    assert system_msg.role == "system"

    # Invalid role should raise ValidationError
    with pytest.raises(ValidationError):
        ChatMessage.model_validate({"role": "invalid_role", "content": "Hello"})


def test_conversational_rag_state_schema() -> None:
    """Assert ConversationalRAGState conforms to DESIGN.md Section 3.1 specification."""
    msg = ChatMessage(role="user", content="Compare the two papers")
    state: ConversationalRAGState = {
        "conversation_id": "conv_123",
        "active_document_ids": ["doc_1", "doc_2"],
        "active_document_titles": "paper1.pdf, paper2.pdf",
        "messages": [msg],
        "user_query": "Compare the two papers",
        "standalone_query": "Compare paper1.pdf and paper2.pdf findings",
        "retrieved_chunks": [],
        "retrieval_grade": "grounded",
        "response_text": "Both papers show significant gains.",
        "citations": [{"citation_id": 1, "filename": "paper1.pdf", "page": 3}],
    }
    assert state["conversation_id"] == "conv_123"
    assert len(state["active_document_ids"]) == 2
    assert state["retrieval_grade"] in ("grounded", "insufficient")


def test_session_lifecycle_and_message_appending() -> None:
    """Assert sessions can be created, retrieved, messages appended, and history generated."""
    memory = ConversationMemoryManager()
    session = memory.get_or_create_session("session_abc")
    assert session.conversation_id == "session_abc"
    assert session.messages == []
    assert session.active_document_ids == []

    # Append user message
    user_msg = memory.add_message(
        "session_abc",
        role="user",
        content="Hello, tell me about Section 1.",
    )
    assert user_msg.role == "user"
    assert user_msg.content == "Hello, tell me about Section 1."

    # Append assistant message
    asst_msg = memory.add_message(
        "session_abc",
        role="assistant",
        content="Section 1 introduces the framework.",
        citations=[1],
    )
    assert asst_msg.role == "assistant"
    assert asst_msg.citations == [1]

    # Verify history retrieval
    history = memory.get_messages("session_abc")
    assert len(history) == 2
    assert history[0].content == "Hello, tell me about Section 1."
    assert history[1].content == "Section 1 introduces the framework."

    # Verify formatted transcript
    formatted = memory.get_formatted_history("session_abc")
    expected = (
        "User: Hello, tell me about Section 1.\nAssistant: Section 1 introduces the framework."
    )
    assert formatted.strip() == expected


def test_sliding_window_pruning() -> None:
    """Assert dialogue history is trimmed to strictly preserve the last 6 turns (12 messages)."""
    memory = ConversationMemoryManager(max_turns=6)
    conv_id = "sliding_window_test"

    # Add 7 full dialogue turns (14 messages total: 7 user + 7 assistant)
    for turn_idx in range(1, 8):
        memory.add_message(conv_id, role="user", content=f"User question {turn_idx}")
        memory.add_message(conv_id, role="assistant", content=f"Assistant answer {turn_idx}")

    messages = memory.get_messages(conv_id)
    # Must preserve exactly 6 turns * 2 messages/turn = 12 messages
    assert len(messages) == 12

    # Turn 1 should have been dropped; the earliest message should now be Turn 2
    assert messages[0].content == "User question 2"
    assert messages[1].content == "Assistant answer 2"
    assert messages[-2].content == "User question 7"
    assert messages[-1].content == "Assistant answer 7"


def test_dynamic_document_scoping() -> None:
    """Assert setting, retrieving, and clearing document scopes."""
    memory = ConversationMemoryManager()
    conv_id = "scoping_test"

    # Default is empty list (global retrieval)
    assert memory.get_active_documents(conv_id) == []

    # Update to scoped documents
    memory.set_active_documents(conv_id, ["doc_paper_a", "doc_paper_b"])
    assert memory.get_active_documents(conv_id) == ["doc_paper_a", "doc_paper_b"]

    # Clear scope back to global
    memory.set_active_documents(conv_id, [])
    assert memory.get_active_documents(conv_id) == []


def test_build_initial_state() -> None:
    """Assert build_initial_state initializes a full ConversationalRAGState dictionary."""
    memory = ConversationMemoryManager()
    conv_id = "state_build_test"
    memory.set_active_documents(conv_id, ["doc_123"])
    memory.add_message(conv_id, role="user", content="Initial greeting")
    memory.add_message(conv_id, role="assistant", content="Hello! How can I help?")

    state = memory.build_initial_state(
        conversation_id=conv_id,
        user_query="What does Table 2 show?",
        active_document_titles="Report2026.pdf",
    )

    assert state["conversation_id"] == conv_id
    assert state["active_document_ids"] == ["doc_123"]
    assert state["active_document_titles"] == "Report2026.pdf"
    assert len(state["messages"]) == 2
    assert state["user_query"] == "What does Table 2 show?"
    assert state["standalone_query"] == ""
    assert state["retrieved_chunks"] == []
    assert state["retrieval_grade"] == "insufficient"
    assert state["response_text"] == ""
    assert state["citations"] == []


def test_multi_session_isolation() -> None:
    """Assert multiple sessions operate independently without crosstalk."""
    memory = ConversationMemoryManager()

    memory.add_message("session_1", role="user", content="Question for session 1")
    memory.set_active_documents("session_1", ["doc_1"])

    memory.add_message("session_2", role="user", content="Question for session 2")
    memory.set_active_documents("session_2", ["doc_2", "doc_3"])

    s1_msgs = memory.get_messages("session_1")
    s2_msgs = memory.get_messages("session_2")

    assert len(s1_msgs) == 1
    assert s1_msgs[0].content == "Question for session 1"
    assert memory.get_active_documents("session_1") == ["doc_1"]

    assert len(s2_msgs) == 1
    assert s2_msgs[0].content == "Question for session 2"
    assert memory.get_active_documents("session_2") == ["doc_2", "doc_3"]


def test_session_deletion_and_clear() -> None:
    """Assert clear_session empties message history and delete_session removes the session."""
    memory = ConversationMemoryManager()
    conv_id = "lifecycle_test"

    memory.add_message(conv_id, role="user", content="Test message")
    assert len(memory.get_messages(conv_id)) == 1

    memory.clear_session(conv_id)
    assert len(memory.get_messages(conv_id)) == 0
    # Session still exists, active documents retained
    assert memory.get_active_documents(conv_id) == []

    # Delete session
    assert memory.delete_session(conv_id) is True
    assert memory.delete_session("non_existent") is False
    assert conv_id not in memory.list_sessions()


def test_thread_safe_concurrent_access() -> None:
    """Assert thread safety during concurrent message appends across multiple threads."""
    memory = ConversationMemoryManager(max_turns=50)
    conv_id = "concurrency_test"

    def append_turn(idx: int) -> None:
        memory.add_message(conv_id, role="user", content=f"User {idx}")
        memory.add_message(conv_id, role="assistant", content=f"Assistant {idx}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append_turn, range(20)))

    messages = memory.get_messages(conv_id)
    # 20 turns * 2 messages = 40 messages total
    assert len(messages) == 40


def test_singleton_get_memory_manager() -> None:
    """Assert get_memory_manager returns the cached singleton instance."""
    memory = ConversationMemoryManager()
    memory.reset()
    mgr1 = get_memory_manager()
    mgr2 = get_memory_manager()
    assert mgr1 is mgr2
