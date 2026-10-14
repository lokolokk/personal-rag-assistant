"""Unit and integration tests for grounding, citation resolution, and StateGraph (Task 3.4).

Tests:
1. Source-prefixed chunk framing format (--- CHUNK [N] --- Document: ...).
2. Grounding system prompt assembly and strict instructions.
3. Citation resolution (bracketed citations [1], [2] to chunk metadata, deduplication, bounds).
4. Retrieval quality grading logic and grade_retrieval_node.
5. Inability fallback node standard phrasing and empty citations.
6. Grounded response generation node with citation mapping.
7. Complete LangGraph StateGraph execution (Grounded path vs Insufficient path).
"""

from typing import Any
from unittest.mock import MagicMock

from app.schemas.chat import ChatMessage, ConversationalRAGState
from app.schemas.document import ChunkMetadata, DocumentChunk, ScoredChunk
from app.services.generation.graph import (
    build_conversational_rag_graph,
    compile_conversational_rag_graph,
    get_conversational_rag_graph,
    run_conversational_rag_graph,
)
from app.services.generation.grounding import (
    FALLBACK_RESPONSE,
    GroundedGenerator,
    RetrievalGrader,
    build_grounding_prompt,
    format_chunk_header,
    format_context_chunks,
    generate_grounded_response_node,
    grade_retrieval_node,
    inability_fallback_node,
    resolve_citations,
    set_grounded_generator,
    set_retrieval_grader,
)
from app.services.retrieval.hybrid import HybridRetriever, set_hybrid_retriever
from app.services.retrieval.reformulator import QueryReformulator, set_query_reformulator


def _create_sample_chunk(
    chunk_id: str = "chunk_1",
    document_id: str = "doc_1",
    filename: str = "attention_is_all_you_need.pdf",
    page_number: int = 3,
    section_title: str = "3.2 Multi-Head Attention",
    content: str = "Multi-head attention allows the model to jointly attend to information.",
    score: float = 0.85,
) -> ScoredChunk:
    """Helper to construct a realistic ScoredChunk for tests."""
    return ScoredChunk(
        chunk_id=chunk_id,
        content=content,
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document_id,
            filename=filename,
            page_number=page_number,
            section_title=section_title,
            is_table=False,
            token_count=len(content.split()),
            char_start=0,
            char_end=len(content),
        ),
        score=score,
    )


# ---------------------------------------------------------------------------
# 1. Source-Prefixed Chunk Framing & Prompt Assembly
# ---------------------------------------------------------------------------


def test_format_chunk_header_single_chunk() -> None:
    """Verify single chunk framing produces standardized format for Scored and DocumentChunk."""
    chunk = _create_sample_chunk(
        chunk_id="chunk_1",
        filename="llama2.pdf",
        page_number=5,
        section_title="Pretraining Architecture",
        content="We adopt the standard Transformer architecture with SwiGLU activations.",
    )
    formatted = format_chunk_header(chunk, index=1)
    expected_lines = [
        "--- CHUNK [1] ---",
        "Document: llama2.pdf",
        "Page: 5",
        "Section: Pretraining Architecture",
        "Content:",
        "We adopt the standard Transformer architecture with SwiGLU activations.",
    ]
    for line in expected_lines:
        assert line in formatted

    # Also test DocumentChunk
    doc_chunk = DocumentChunk(
        chunk_id="chunk_base",
        content="Base document chunk text.",
        metadata=ChunkMetadata(
            chunk_id="chunk_base",
            document_id="doc_base",
            filename="base.pdf",
            page_number=2,
            section_title="Introduction",
            is_table=False,
            token_count=4,
            char_start=0,
            char_end=24,
        ),
    )
    doc_formatted = format_chunk_header(doc_chunk, index=2)
    assert "--- CHUNK [2] ---" in doc_formatted
    assert "Document: base.pdf" in doc_formatted


def test_format_chunk_header_missing_section() -> None:
    """Verify missing section title falls back gracefully without breaking."""
    chunk = _create_sample_chunk(section_title="")
    formatted = format_chunk_header(chunk, index=2)
    assert "--- CHUNK [2] ---" in formatted
    assert "Section: (None)" in formatted or "Section: N/A" in formatted


def test_format_context_chunks_multiple() -> None:
    """Verify multiple chunks are indexed sequentially starting at 1."""
    chunk1 = _create_sample_chunk(
        chunk_id="c1",
        filename="doc_a.pdf",
        page_number=1,
        content="Content of document A.",
    )
    chunk2 = _create_sample_chunk(
        chunk_id="c2",
        filename="doc_b.pdf",
        page_number=4,
        content="Content of document B.",
    )
    framed = format_context_chunks([chunk1, chunk2])
    assert "--- CHUNK [1] ---" in framed
    assert "Document: doc_a.pdf" in framed
    assert "--- CHUNK [2] ---" in framed
    assert "Document: doc_b.pdf" in framed


def test_format_context_chunks_empty() -> None:
    """Verify empty chunk list returns clear placeholder without errors."""
    framed = format_context_chunks([])
    assert "(No relevant document chunks retrieved)" in framed


def test_build_grounding_prompt_contains_rules() -> None:
    """Verify grounding prompt template enforces all 4 core rules and injects query & context."""
    chunks = [_create_sample_chunk()]
    prompt = build_grounding_prompt(
        context_chunks=chunks,
        standalone_query="How does multi-head attention work?",
    )
    # Rule 1: Strict Grounding
    assert "Strict Grounding" in prompt
    # Rule 2: Multi-Document Attribution
    assert "Multi-Document Attribution" in prompt
    # Rule 3: Bracketed citations
    assert "[1]" in prompt
    # Rule 4: Standardized Inability Fallback
    assert FALLBACK_RESPONSE in prompt
    # Query & Content
    assert "How does multi-head attention work?" in prompt
    assert "attention_is_all_you_need.pdf" in prompt


# ---------------------------------------------------------------------------
# 2. Citation Resolution
# ---------------------------------------------------------------------------


def test_resolve_citations_single() -> None:
    """Verify resolving a single citation [1] to chunk metadata."""
    chunk1 = _create_sample_chunk(
        chunk_id="c1",
        document_id="doc_attn",
        filename="attention.pdf",
        page_number=3,
        section_title="Architecture",
        content="Attention mechanisms model dependencies.",
    )
    response_text = "The Transformer utilizes attention mechanisms [1]."
    citations = resolve_citations(response_text, [chunk1])

    assert len(citations) == 1
    cit = citations[0]
    assert cit["id"] == 1
    assert cit["citationIndex"] == 1
    assert cit["chunk_id"] == "c1"
    assert cit["document_id"] == "doc_attn"
    assert cit["filename"] == "attention.pdf"
    assert cit["page_number"] == 3
    assert cit["section_title"] == "Architecture"
    assert "Attention mechanisms model dependencies." in cit["quote"]


def test_resolve_citations_multiple_and_ordering() -> None:
    """Verify multiple citations like [2] and [1] are resolved with exact attribution."""
    chunk1 = _create_sample_chunk(chunk_id="c1", filename="doc1.pdf", page_number=1)
    chunk2 = _create_sample_chunk(chunk_id="c2", filename="doc2.pdf", page_number=2)
    response_text = (
        "According to doc2, performance is high [2], whereas doc1 states baseline results [1]."
    )
    citations = resolve_citations(response_text, [chunk1, chunk2])

    assert len(citations) == 2
    assert citations[0]["id"] == 2
    assert citations[0]["filename"] == "doc2.pdf"
    assert citations[1]["id"] == 1
    assert citations[1]["filename"] == "doc1.pdf"


def test_resolve_citations_deduplication() -> None:
    """Verify repeated citation markers [1] in text produce only one resolved citation entry."""
    chunk1 = _create_sample_chunk(chunk_id="c1", filename="doc1.pdf")
    response_text = "First claim [1]. Second claim supporting first [1]."
    citations = resolve_citations(response_text, [chunk1])

    assert len(citations) == 1
    assert citations[0]["id"] == 1


def test_resolve_citations_out_of_bounds_and_invalid() -> None:
    """Verify citations referencing non-existent chunk indices (e.g. [99], [0]) are ignored."""
    chunk1 = _create_sample_chunk(chunk_id="c1")
    response_text = "Speculative claim [99] and zero [0] and valid [1]."
    citations = resolve_citations(response_text, [chunk1])

    assert len(citations) == 1
    assert citations[0]["id"] == 1


def test_resolve_citations_no_citations() -> None:
    """Verify response without bracketed numbers returns an empty list."""
    chunk1 = _create_sample_chunk(chunk_id="c1")
    response_text = "General statement without any citations."
    citations = resolve_citations(response_text, [chunk1])
    assert citations == []


# ---------------------------------------------------------------------------
# 3. Retrieval Quality Grading Logic & Nodes
# ---------------------------------------------------------------------------


def test_retrieval_grader_empty_chunks() -> None:
    """Verify empty chunks immediately evaluate to 'insufficient' without calling LLM."""
    grader = RetrievalGrader(llm_client=MagicMock())
    grade = grader.grade_retrieval(query="Any question", chunks=[])
    assert grade == "insufficient"
    assert grader.client is not None
    grader.client.chat.completions.create.assert_not_called()


def test_retrieval_grader_mock_client_grounded() -> None:
    """Verify LLM grader returning 'grounded' is recognized."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="grounded"))]
    mock_client.chat.completions.create.return_value = mock_completion

    grader = RetrievalGrader(llm_client=mock_client)
    chunks = [_create_sample_chunk(content="The learning rate was set to 0.001.")]
    grade = grader.grade_retrieval(query="What was the learning rate?", chunks=chunks)

    assert grade == "grounded"


def test_retrieval_grader_mock_client_insufficient() -> None:
    """Verify LLM grader returning 'insufficient' is recognized."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="insufficient"))]
    mock_client.chat.completions.create.return_value = mock_completion

    grader = RetrievalGrader(llm_client=mock_client)
    chunks = [_create_sample_chunk(content="This text is completely unrelated to astrophysics.")]
    grade = grader.grade_retrieval(query="What is the speed of light in vacuum?", chunks=chunks)

    assert grade == "insufficient"


def test_grade_retrieval_node_delta() -> None:
    """Verify grade_retrieval_node conforms to LangGraph node delta contract."""
    grader = RetrievalGrader(grading_fn=lambda _q, _c: "grounded")
    set_retrieval_grader(grader)
    try:
        state: ConversationalRAGState = {
            "conversation_id": "test_conv",
            "active_document_ids": ["doc_1"],
            "active_document_titles": "test.pdf",
            "messages": [],
            "user_query": "Test query",
            "standalone_query": "Test query",
            "retrieved_chunks": [_create_sample_chunk()],
            "retrieval_grade": "insufficient",
            "response_text": "",
            "citations": [],
        }
        delta = grade_retrieval_node(state)
        assert delta == {"retrieval_grade": "grounded"}
    finally:
        set_retrieval_grader(None)


# ---------------------------------------------------------------------------
# 4. Grounded Generation & Inability Fallback Nodes
# ---------------------------------------------------------------------------


def test_inability_fallback_node() -> None:
    """Verify inability_fallback_node sets exact standardized message and empty citations."""
    state: ConversationalRAGState = {
        "conversation_id": "test_conv",
        "active_document_ids": [],
        "active_document_titles": "",
        "messages": [],
        "user_query": "Irrelevant query",
        "standalone_query": "Irrelevant query",
        "retrieved_chunks": [],
        "retrieval_grade": "insufficient",
        "response_text": "",
        "citations": [],
    }
    delta = inability_fallback_node(state)
    assert delta["response_text"] == FALLBACK_RESPONSE
    assert delta["citations"] == []


def test_generate_grounded_response_node() -> None:
    """Verify generate_grounded_response_node synthesizes answer and resolves citations."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content="The model uses multi-head attention [1]."))
    ]
    mock_client.chat.completions.create.return_value = mock_completion

    generator = GroundedGenerator(llm_client=mock_client)
    set_grounded_generator(generator)
    try:
        chunk = _create_sample_chunk(chunk_id="c1", filename="attention.pdf")
        state: ConversationalRAGState = {
            "conversation_id": "test_conv",
            "active_document_ids": ["doc_1"],
            "active_document_titles": "attention.pdf",
            "messages": [],
            "user_query": "How does attention work?",
            "standalone_query": "How does attention work?",
            "retrieved_chunks": [chunk],
            "retrieval_grade": "grounded",
            "response_text": "",
            "citations": [],
        }
        delta = generate_grounded_response_node(state)
        assert "multi-head attention [1]" in delta["response_text"]
        assert len(delta["citations"]) == 1
        assert delta["citations"][0]["id"] == 1
        assert delta["citations"][0]["filename"] == "attention.pdf"
    finally:
        set_grounded_generator(None)


def test_generate_grounded_response_node_empty_completion_falls_back_to_fallback() -> None:
    """Verify empty LLM completions fall back to FALLBACK_RESPONSE and insufficient grade."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content=""))]
    mock_client.chat.completions.create.return_value = mock_completion

    generator = GroundedGenerator(llm_client=mock_client)
    set_grounded_generator(generator)
    try:
        chunk = _create_sample_chunk(chunk_id="c1", filename="attention.pdf")
        state: ConversationalRAGState = {
            "conversation_id": "test_conv_empty",
            "active_document_ids": ["doc_1"],
            "active_document_titles": "attention.pdf",
            "messages": [],
            "user_query": "What is the exact score?",
            "standalone_query": "What is the exact score?",
            "retrieved_chunks": [chunk],
            "retrieval_grade": "insufficient",
            "response_text": "",
            "citations": [],
        }
        delta = generate_grounded_response_node(state)
        assert delta["response_text"] == FALLBACK_RESPONSE
        assert delta["citations"] == []
        assert delta["retrieval_grade"] == "insufficient"

        # Also verify streaming mode emits FALLBACK_RESPONSE token when completion is empty
        writer_events: list[dict[str, Any]] = []

        def custom_writer(event: dict[str, Any]) -> None:
            writer_events.append(event)

        resp, cits = generator.generate(
            standalone_query="What is the exact score?",
            chunks=[chunk],
            writer=custom_writer,
        )
        assert resp == FALLBACK_RESPONSE
        assert cits == []
        assert {"token": FALLBACK_RESPONSE} in writer_events
    finally:
        set_grounded_generator(None)


# ---------------------------------------------------------------------------
# 5. Full LangGraph StateGraph Execution (Grounded & Insufficient Paths)
# ---------------------------------------------------------------------------


def test_graph_builder_and_singleton() -> None:
    """Verify build_conversational_rag_graph and get_conversational_rag_graph instances."""
    uncompiled = build_conversational_rag_graph()
    assert uncompiled is not None
    compiled = get_conversational_rag_graph()
    assert compiled is not None


def test_stategraph_grounded_execution_path() -> None:
    """Verify complete StateGraph execution when retrieval is grounded."""
    # 1. Setup mock reformulator returning standalone query
    mock_reformulator = MagicMock(spec=QueryReformulator)
    mock_reformulator.reformulate_node.return_value = {
        "standalone_query": "What is multi-head attention in attention.pdf?"
    }
    set_query_reformulator(mock_reformulator)

    # 2. Setup mock hybrid retriever returning chunk
    chunk = _create_sample_chunk(
        chunk_id="chunk_42",
        filename="attention.pdf",
        content="Multi-head attention maps queries and keys.",
    )
    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_retriever.retrieve.return_value = [chunk]
    set_hybrid_retriever(mock_retriever)

    # 3. Setup mock grader returning grounded
    mock_grader = RetrievalGrader(grading_fn=lambda _q, _c: "grounded")
    set_retrieval_grader(mock_grader)

    # 4. Setup mock generator returning grounded text with citation [1]
    mock_generator = GroundedGenerator(
        generating_fn=lambda _prompt, _chunks: "Attention models dependencies across positions [1]."
    )
    set_grounded_generator(mock_generator)

    try:
        app = compile_conversational_rag_graph()
        initial_state: dict[str, Any] = {
            "conversation_id": "conv_1",
            "active_document_ids": ["doc_attn"],
            "active_document_titles": "attention.pdf",
            "messages": [
                ChatMessage(role="user", content="Tell me about Transformers."),
                ChatMessage(role="assistant", content="They are neural network models."),
            ],
            "user_query": "What is multi-head attention?",
            "standalone_query": "",
            "retrieved_chunks": [],
            "retrieval_grade": "insufficient",
            "response_text": "",
            "citations": [],
        }

        final_state = app.invoke(initial_state)

        assert final_state["standalone_query"] == "What is multi-head attention in attention.pdf?"
        assert len(final_state["retrieved_chunks"]) == 1
        assert final_state["retrieval_grade"] == "grounded"
        assert "Attention models dependencies across positions [1]." in final_state["response_text"]
        assert len(final_state["citations"]) == 1
        assert final_state["citations"][0]["filename"] == "attention.pdf"
        assert final_state["citations"][0]["chunk_id"] == "chunk_42"
    finally:
        set_query_reformulator(None)
        set_hybrid_retriever(None)
        set_retrieval_grader(None)
        set_grounded_generator(None)


def test_stategraph_insufficient_fallback_execution_path() -> None:
    """Verify complete StateGraph routing to inability fallback on insufficient retrieval."""
    # 1. Setup mock reformulator
    mock_reformulator = MagicMock(spec=QueryReformulator)
    mock_reformulator.reformulate_node.return_value = {
        "standalone_query": "What is the recipe for chocolate cake?"
    }
    set_query_reformulator(mock_reformulator)

    # 2. Setup mock hybrid retriever returning empty
    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_retriever.retrieve.return_value = []
    set_hybrid_retriever(mock_retriever)

    # 3. Setup mock grader returning insufficient
    mock_grader = RetrievalGrader(grading_fn=lambda _q, _c: "insufficient")
    set_retrieval_grader(mock_grader)

    # 4. Inability fallback node should be reached, NOT the generator
    mock_generator = GroundedGenerator(
        generating_fn=MagicMock(side_effect=RuntimeError("Generator should not be called!"))
    )
    set_grounded_generator(mock_generator)

    try:
        app = compile_conversational_rag_graph()
        initial_state: dict[str, Any] = {
            "conversation_id": "conv_2",
            "active_document_ids": ["doc_attn"],
            "active_document_titles": "attention.pdf",
            "messages": [],
            "user_query": "What is the recipe for chocolate cake?",
            "standalone_query": "",
            "retrieved_chunks": [],
            "retrieval_grade": "insufficient",
            "response_text": "",
            "citations": [],
        }

        final_state = app.invoke(initial_state)

        assert final_state["retrieval_grade"] == "insufficient"
        assert final_state["response_text"] == FALLBACK_RESPONSE
        assert final_state["citations"] == []
    finally:
        set_query_reformulator(None)
        set_hybrid_retriever(None)
        set_retrieval_grader(None)
        set_grounded_generator(None)


def test_stategraph_multi_document_attribution() -> None:
    """Verify multi-document comparative scenario maps citations across different files."""
    chunk_a = _create_sample_chunk(
        chunk_id="chunk_a",
        document_id="doc_a",
        filename="attention.pdf",
        content="Transformer uses 8 attention heads.",
    )
    chunk_b = _create_sample_chunk(
        chunk_id="chunk_b",
        document_id="doc_b",
        filename="llama2.pdf",
        content="Llama 2 uses 32 attention heads in 7B model.",
    )

    mock_reformulator = MagicMock(spec=QueryReformulator)
    mock_reformulator.reformulate_node.return_value = {
        "standalone_query": "Compare attention heads between attention.pdf and llama2.pdf"
    }
    set_query_reformulator(mock_reformulator)

    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_retriever.retrieve.return_value = [chunk_a, chunk_b]
    set_hybrid_retriever(mock_retriever)

    mock_grader = RetrievalGrader(grading_fn=lambda _q, _c: "grounded")
    set_retrieval_grader(mock_grader)

    mock_generator = GroundedGenerator(
        generating_fn=lambda _p, _c: (
            "Attention uses 8 heads [1], whereas Llama 2 uses 32 heads [2]."
        )
    )
    set_grounded_generator(mock_generator)

    try:
        result = run_conversational_rag_graph(
            {
                "conversation_id": "conv_multi",
                "active_document_ids": ["doc_a", "doc_b"],
                "active_document_titles": "attention.pdf, llama2.pdf",
                "messages": [],
                "user_query": "Compare their attention heads.",
                "standalone_query": "",
                "retrieved_chunks": [],
                "retrieval_grade": "insufficient",
                "response_text": "",
                "citations": [],
            }
        )

        assert len(result["citations"]) == 2
        assert result["citations"][0]["filename"] == "attention.pdf"
        assert result["citations"][0]["id"] == 1
        assert result["citations"][1]["filename"] == "llama2.pdf"
        assert result["citations"][1]["id"] == 2
    finally:
        set_query_reformulator(None)
        set_hybrid_retriever(None)
        set_retrieval_grader(None)
        set_grounded_generator(None)


def test_checkpoint_3_multi_turn_coreference_and_retrieval_flow() -> None:
    """Verify Checkpoint 3: Multi-turn flow where user follow-up references another document.

    Asserts:
    1. Turn history with initial single document (attention.pdf).
    2. User follow-up referencing another document (llama2.pdf).
    3. Query reformulator resolves document names into standalone query.
    4. Scoped hybrid retrieval returns top chunks from scoped documents.
    5. Prompt assembler frames chunks with filename headers.
    6. Citations contain valid filename and page mappings.
    """
    chunk_attn = _create_sample_chunk(
        chunk_id="chk_attn_1",
        document_id="doc_attn",
        filename="attention.pdf",
        page_number=3,
        section_title="Multi-Head Attention",
        content="Multi-head attention maps queries, keys, and values through 8 heads.",
    )
    chunk_llama = _create_sample_chunk(
        chunk_id="chk_llama_1",
        document_id="doc_llama",
        filename="llama2.pdf",
        page_number=7,
        section_title="Architecture Details",
        content="Llama 2 uses 32 heads for attention with grouped-query attention in 70B.",
    )

    # Mock reformulator to rewrite multi-turn follow-up with document awareness
    mock_reformulator = MagicMock(spec=QueryReformulator)
    mock_reformulator.reformulate_node.return_value = {
        "standalone_query": "Compare attention head counts between attention.pdf and llama2.pdf"
    }
    set_query_reformulator(mock_reformulator)

    # Mock hybrid retriever returning both chunks
    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_retriever.retrieve.return_value = [chunk_attn, chunk_llama]
    set_hybrid_retriever(mock_retriever)

    # Mock grader confirming grounded context
    mock_grader = RetrievalGrader(grading_fn=lambda _q, _c: "grounded")
    set_retrieval_grader(mock_grader)

    # Mock grounded generator emitting grounded answer with [1] and [2]
    mock_generator = GroundedGenerator(
        generating_fn=lambda _p, _c: (
            "In attention.pdf, 8 heads are used [1], whereas llama2.pdf uses 32 heads [2]."
        )
    )
    set_grounded_generator(mock_generator)

    try:
        dialogue_history = [
            ChatMessage(role="user", content="How many attention heads does the Transformer use?"),
            ChatMessage(
                role="assistant", content="The Transformer uses 8 parallel attention heads."
            ),
        ]
        state_input: dict[str, Any] = {
            "conversation_id": "checkpoint_3_session",
            "active_document_ids": ["doc_attn", "doc_llama"],
            "active_document_titles": "attention.pdf, llama2.pdf",
            "messages": dialogue_history,
            "user_query": "How does Llama 2 compare to this?",
            "standalone_query": "",
            "retrieved_chunks": [],
            "retrieval_grade": "insufficient",
            "response_text": "",
            "citations": [],
        }

        output = run_conversational_rag_graph(state_input)

        # 1. Reformulator resolved standalone query
        assert output["standalone_query"] == (
            "Compare attention head counts between attention.pdf and llama2.pdf"
        )
        # 2. Retriever retrieved scoped chunks
        assert len(output["retrieved_chunks"]) == 2
        # 3. Grader set grounded
        assert output["retrieval_grade"] == "grounded"
        # 4. Generator synthesized grounded response
        assert "In attention.pdf, 8 heads are used [1]" in output["response_text"]
        assert "llama2.pdf uses 32 heads [2]" in output["response_text"]
        # 5. Citations map with exact filename, page, and chunk IDs
        assert len(output["citations"]) == 2
        cit1 = output["citations"][0]
        assert cit1["id"] == 1
        assert cit1["filename"] == "attention.pdf"
        assert cit1["page_number"] == 3
        assert cit1["chunk_id"] == "chk_attn_1"

        cit2 = output["citations"][1]
        assert cit2["id"] == 2
        assert cit2["filename"] == "llama2.pdf"
        assert cit2["page_number"] == 7
        assert cit2["chunk_id"] == "chk_llama_1"
    finally:
        set_query_reformulator(None)
        set_hybrid_retriever(None)
        set_retrieval_grader(None)
        set_grounded_generator(None)
