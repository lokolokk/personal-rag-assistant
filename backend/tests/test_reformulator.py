"""Unit tests for multi-turn query reformulator LangGraph node (Task 3.2)."""

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import LLMConfigurationError
from app.schemas.chat import ChatMessage, ConversationalRAGState
from app.services.retrieval.memory import ConversationMemoryManager
from app.services.retrieval.reformulator import (
    REFORMULATION_PROMPT_TEMPLATE,
    QueryReformulator,
    build_reformulation_prompt,
    bypass_reformulation_node,
    clean_reformulated_query,
    format_chat_history,
    get_query_reformulator,
    reformulate_query_node,
    set_query_reformulator,
)


def test_format_chat_history() -> None:
    """Verify chat messages format into role-prefixed lines or fallback message."""
    # Empty history
    assert format_chat_history([]) == "(No previous conversation history)"

    # Formatted turns
    messages = [
        ChatMessage(role="user", content="What is Transformer?"),
        ChatMessage(role="assistant", content="Transformer is a sequence model."),
    ]
    expected = "User: What is Transformer?\nAssistant: Transformer is a sequence model."
    assert format_chat_history(messages) == expected


def test_build_reformulation_prompt() -> None:
    """Verify prompt formatting inserts document titles, history, and user query."""
    doc_titles = "Attention_Is_All_You_Need.pdf, Hybrid_RRF_Paper.pdf"
    messages = [
        ChatMessage(role="user", content="Tell me about self-attention."),
        ChatMessage(
            role="assistant",
            content="Self-attention relates different positions of a single sequence.",
        ),
    ]
    user_query = "How does the second paper improve on it?"

    prompt = build_reformulation_prompt(
        active_document_titles=doc_titles,
        chat_history=messages,
        user_query=user_query,
    )

    assert "You are a search query optimizer for a document retrieval system." in prompt
    assert prompt.startswith(REFORMULATION_PROMPT_TEMPLATE[:50])
    assert doc_titles in prompt
    assert "User: Tell me about self-attention." in prompt
    assert "Assistant: Self-attention relates different positions of a single sequence." in prompt
    assert f"User Follow-up:\n{user_query}" in prompt
    assert prompt.endswith("Standalone Search Query:")


def test_build_reformulation_prompt_empty_fallbacks() -> None:
    """Verify prompt uses fallback text when document titles or history are empty."""
    prompt = build_reformulation_prompt(
        active_document_titles="",
        chat_history=[],
        user_query="What are the key findings?",
    )
    assert "Active Documents:\n(None)" in prompt
    assert "Conversation History:\n(No previous conversation history)" in prompt
    assert "User Follow-up:\nWhat are the key findings?" in prompt


def test_clean_reformulated_query() -> None:
    """Verify cleaning raw model responses strips prefixes, quotes, and whitespace."""
    fallback = "What is the attention mechanism?"

    # Normal response
    assert (
        clean_reformulated_query("What is self-attention?", fallback) == "What is self-attention?"
    )

    # Strips redundant prefixes
    assert (
        clean_reformulated_query("Standalone Search Query: What is self-attention?", fallback)
        == "What is self-attention?"
    )
    assert (
        clean_reformulated_query("standalone search query:\nWhat is self-attention?", fallback)
        == "What is self-attention?"
    )
    assert (
        clean_reformulated_query("Search Query: What is self-attention?", fallback)
        == "What is self-attention?"
    )

    # Strips surrounding quotation marks
    assert (
        clean_reformulated_query('"What is self-attention?"', fallback) == "What is self-attention?"
    )
    assert (
        clean_reformulated_query("'What is self-attention?'", fallback) == "What is self-attention?"
    )

    # Strips surrounding whitespace
    assert (
        clean_reformulated_query("   What is self-attention?   \n", fallback)
        == "What is self-attention?"
    )

    # Fallback when response is empty or whitespace
    assert clean_reformulated_query("", fallback) == fallback
    assert clean_reformulated_query("   \n  ", fallback) == fallback


def test_standalone_first_turn_passthrough_with_mock_client() -> None:
    """Verify first-turn standalone question passes through verbatim without alterations."""
    standalone_input = "What is the multi-head attention mechanism?"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=standalone_input))]
    )

    reformulator = QueryReformulator(llm_client=mock_client)
    state: ConversationalRAGState = {
        "conversation_id": "conv_first_turn",
        "active_document_ids": ["doc_123"],
        "active_document_titles": "Attention_Is_All_You_Need.pdf",
        "messages": [],
        "user_query": standalone_input,
        "standalone_query": "",
        "retrieved_chunks": [],
        "retrieval_grade": "insufficient",
        "response_text": "",
        "citations": [],
    }

    result = reformulator.reformulate_node(state)
    assert result == {"standalone_query": standalone_input}

    # Verify LLM was invoked with the proper prompt
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    prompt_sent = call_kwargs["messages"][0]["content"]
    assert "Active Documents:\nAttention_Is_All_You_Need.pdf" in prompt_sent
    assert "Conversation History:\n(No previous conversation history)" in prompt_sent
    assert f"User Follow-up:\n{standalone_input}" in prompt_sent


def test_standalone_first_turn_passthrough_without_client_or_key() -> None:
    """Verify first-turn question without history passes through if no client or key exists."""
    reformulator = QueryReformulator(llm_client=None, api_key=None)
    # Ensure client is None
    assert reformulator.client is None

    standalone_input = "What is the Transformer architecture?"
    state: ConversationalRAGState = {
        "conversation_id": "conv_no_key_first_turn",
        "active_document_ids": [],
        "active_document_titles": "",
        "messages": [],
        "user_query": standalone_input,
        "standalone_query": "",
        "retrieved_chunks": [],
        "retrieval_grade": "insufficient",
        "response_text": "",
        "citations": [],
    }

    result = reformulator.reformulate_node(state)
    assert result == {"standalone_query": standalone_input}


def test_single_document_coreference_resolution() -> None:
    """Verify pronouns ('its', 'it') are resolved to concrete entities from dialogue context."""
    rewritten_query = (
        "What is the attention mechanism in Attention_Is_All_You_Need.pdf and "
        "how many heads does it use?"
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=rewritten_query))]
    )

    reformulator = QueryReformulator(llm_client=mock_client)
    state: ConversationalRAGState = {
        "conversation_id": "conv_single_doc",
        "active_document_ids": ["doc_attention"],
        "active_document_titles": "Attention_Is_All_You_Need.pdf",
        "messages": [
            ChatMessage(role="user", content="Can you summarize the Transformer model?"),
            ChatMessage(
                role="assistant",
                content=(
                    "The Transformer model uses an encoder-decoder architecture based on "
                    "attention mechanisms."
                ),
            ),
        ],
        "user_query": "What is its attention mechanism and how many heads does it use?",
        "standalone_query": "",
        "retrieved_chunks": [],
        "retrieval_grade": "insufficient",
        "response_text": "",
        "citations": [],
    }

    result = reformulator.reformulate_node(state)
    assert result == {"standalone_query": rewritten_query}

    # Verify prompt contents
    prompt_sent = mock_client.chat.completions.create.call_args[1]["messages"][0]["content"]
    assert "Active Documents:\nAttention_Is_All_You_Need.pdf" in prompt_sent
    assert "Can you summarize the Transformer model?" in prompt_sent
    assert (
        "User Follow-up:\nWhat is its attention mechanism and how many heads does it use?"
        in prompt_sent
    )


def test_multi_document_comparative_resolution() -> None:
    """Verify relative document references resolve to exact filenames."""
    rewritten_query = (
        "How does Hybrid_RRF_Paper.pdf improve retrieval accuracy compared to "
        "Attention_Is_All_You_Need.pdf?"
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=rewritten_query))]
    )

    reformulator = QueryReformulator(llm_client=mock_client)
    state: ConversationalRAGState = {
        "conversation_id": "conv_multi_doc",
        "active_document_ids": ["doc_1", "doc_2"],
        "active_document_titles": "Attention_Is_All_You_Need.pdf, Hybrid_RRF_Paper.pdf",
        "messages": [
            ChatMessage(
                role="user",
                content="Compare the retrieval techniques in the first and second papers.",
            ),
            ChatMessage(
                role="assistant",
                content=(
                    "The first paper uses dot-product attention while the second paper introduces "
                    "hybrid BM25 and dense RRF."
                ),
            ),
        ],
        "user_query": "How does the second paper improve on the first?",
        "standalone_query": "",
        "retrieved_chunks": [],
        "retrieval_grade": "insufficient",
        "response_text": "",
        "citations": [],
    }

    result = reformulator.reformulate_node(state)
    assert result == {"standalone_query": rewritten_query}

    # Verify prompt contents
    prompt_sent = mock_client.chat.completions.create.call_args[1]["messages"][0]["content"]
    assert "Attention_Is_All_You_Need.pdf, Hybrid_RRF_Paper.pdf" in prompt_sent
    assert "How does the second paper improve on the first?" in prompt_sent


def test_reformulate_query_node_conforms_to_langgraph_contract() -> None:
    """Verify pure reformulate_query_node function conforms to LangGraph node specification."""
    custom_rewritten = "What are the token limits in Attention_Is_All_You_Need.pdf?"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=custom_rewritten))]
    )

    custom_reformulator = QueryReformulator(llm_client=mock_client)
    set_query_reformulator(custom_reformulator)

    state: ConversationalRAGState = {
        "conversation_id": "conv_node_contract",
        "active_document_ids": ["doc_1"],
        "active_document_titles": "Attention_Is_All_You_Need.pdf",
        "messages": [ChatMessage(role="user", content="Tell me about the paper.")],
        "user_query": "What are its token limits?",
        "standalone_query": "",
        "retrieved_chunks": [],
        "retrieval_grade": "insufficient",
        "response_text": "",
        "citations": [],
    }

    node_result = reformulate_query_node(state)
    assert isinstance(node_result, dict)
    assert "standalone_query" in node_result
    assert node_result["standalone_query"] == custom_rewritten

    # Reset global reformulator
    set_query_reformulator(None)


def test_missing_api_key_and_client_with_history_raises_error() -> None:
    """Verify multi-turn queries raise LLMConfigurationError if no client or key exists."""
    reformulator = QueryReformulator(llm_client=None, api_key=None)

    state: ConversationalRAGState = {
        "conversation_id": "conv_missing_key",
        "active_document_ids": ["doc_1"],
        "active_document_titles": "sample.pdf",
        "messages": [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
        ],
        "user_query": "Tell me more about it",
        "standalone_query": "",
        "retrieved_chunks": [],
        "retrieval_grade": "insufficient",
        "response_text": "",
        "citations": [],
    }

    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY is not configured"):
        reformulator.reformulate_node(state)


def test_dependency_injection_custom_callable() -> None:
    """Verify QueryReformulator accepts custom callable function for testing."""
    prompts_captured: list[str] = []

    def mock_completion(prompt: str) -> str:
        prompts_captured.append(prompt)
        return "Standalone Search Query: Reformulated by callable"

    reformulator = QueryReformulator(llm_client=mock_completion)
    output = reformulator.reformulate(
        user_query="How does it work?",
        chat_history="User: Check table 1\nAssistant: Table 1 contains metrics",
        active_document_titles="report.pdf",
    )

    assert output == "Reformulated by callable"
    assert len(prompts_captured) == 1
    assert "report.pdf" in prompts_captured[0]


def test_empty_or_whitespace_user_query() -> None:
    """Verify empty or whitespace user queries return empty standalone query."""
    reformulator = QueryReformulator(llm_client=None, api_key=None)
    state: ConversationalRAGState = {
        "conversation_id": "conv_empty",
        "active_document_ids": [],
        "active_document_titles": "",
        "messages": [],
        "user_query": "   \n\t  ",
        "standalone_query": "",
        "retrieved_chunks": [],
        "retrieval_grade": "insufficient",
        "response_text": "",
        "citations": [],
    }

    result = reformulator.reformulate_node(state)
    assert result == {"standalone_query": ""}


def test_integration_with_memory_manager_initial_state() -> None:
    """Verify integration between ConversationMemoryManager initial state and reformulator."""
    memory_manager = ConversationMemoryManager()
    conv_id = "conv_integration_test"
    memory_manager.get_or_create_session(conv_id, document_ids=["doc_attention", "doc_rrf"])
    memory_manager.add_message(conv_id, "user", "What are the models discussed?")
    memory_manager.add_message(
        conv_id, "assistant", "Paper 1 discusses Transformer and Paper 2 discusses RRF."
    )

    initial_state = memory_manager.build_initial_state(
        conversation_id=conv_id,
        user_query="Which one has higher precision?",
        active_document_titles="Attention_Is_All_You_Need.pdf, Hybrid_RRF_Paper.pdf",
    )

    expected_query = (
        "Which model between Attention_Is_All_You_Need.pdf and Hybrid_RRF_Paper.pdf "
        "has higher precision?"
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=expected_query))]
    )

    reformulator = QueryReformulator(llm_client=mock_client)
    node_output = reformulator.reformulate_node(initial_state)

    assert node_output["standalone_query"] == expected_query


def test_get_query_reformulator_singleton() -> None:
    """Verify get_query_reformulator returns cached instance."""
    set_query_reformulator(None)
    ref1 = get_query_reformulator()
    ref2 = get_query_reformulator()
    assert ref1 is ref2
    set_query_reformulator(None)


def test_bypass_reformulation_node() -> None:
    """Verify bypass_reformulation_node copies user_query directly with zero LLM calls."""
    state: ConversationalRAGState = {
        "conversation_id": "conv_bypass",
        "active_document_ids": ["doc_123"],
        "active_document_titles": "Attention_Is_All_You_Need.pdf",
        "messages": [],
        "user_query": "What is multi-head attention?",
        "standalone_query": "",
        "retrieved_chunks": [],
        "retrieval_grade": "insufficient",
        "response_text": "",
        "citations": [],
    }
    result = bypass_reformulation_node(state)
    assert result == {"standalone_query": "What is multi-head attention?"}

    # Verify whitespace stripping
    state["user_query"] = "   \n\t  "
    assert bypass_reformulation_node(state) == {"standalone_query": ""}
