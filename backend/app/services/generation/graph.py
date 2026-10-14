"""Conversational RAG LangGraph StateGraph assembly and orchestration (Task 3.4 & Task 6.3).

Assembles the LangGraph StateGraph in two modes:
1. Streamlined Mode (Default, Task 6.3):
   START -> [Conditional Edge: route_reformulation_entry]
            -> bypass_reformulation (Turn 1: pure-Python passthrough, 0 LLM tokens)
            -> reformulate_query (Multi-turn: LLM rewrite)
         -> scoped_hybrid_retriever
         -> [Conditional Edge: route_retrieval_candidate_quality]
            -> inability_fallback (0 chunks: immediate pure-Python fallback, 0 LLM tokens)
            -> generate_grounded_response (>=1 chunks: single-pass grounded generation)
         -> END

2. Legacy Evaluation Mode (streamlined=False, Task 6.1 harness):
   START -> reformulate_query -> scoped_hybrid_retriever -> grade_retrieval
         -> [Conditional Edge: route_retrieval_grade]
            -> generate_grounded_response -> END
            -> inability_fallback -> END

Strictly conforms to docs/RULES.md, docs/DESIGN.md Section 3, and docs/ARCHITECTURE.md.
"""

import threading
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.schemas.chat import ConversationalRAGState
from app.services.generation.grounding import (
    generate_grounded_response_node,
    grade_retrieval_node,
    inability_fallback_node,
)
from app.services.retrieval.hybrid import scoped_hybrid_retriever_node
from app.services.retrieval.reformulator import (
    bypass_reformulation_node,
    reformulate_query_node,
)


def route_reformulation_entry(state: ConversationalRAGState) -> str:
    """Conditional entry routing for query reformulation (Step 6.3.1).

    If conversation history is empty (Turn 1), bypasses the LLM rewrite pass entirely,
    saving ~500 prompt tokens and preventing prompt truncation on complex initial queries.
    If multi-turn history exists, routes to the LLM reformulator.

    Args:
        state: Current ConversationalRAGState dictionary.

    Returns:
        Next node name: 'bypass_reformulation' or 'reformulate_query'.
    """
    messages = state.get("messages", [])
    if len(messages) == 0:
        return "bypass_reformulation"
    return "reformulate_query"


def route_retrieval_candidate_quality(state: ConversationalRAGState) -> str:
    """Pure-Python candidate quality check before generation (Step 6.3.2).

    If zero chunks are retrieved, immediately short-circuits to inability_fallback
    with zero LLM token consumption.
    If retrieved chunks are present, routes directly to single-pass grounded generation.

    Args:
        state: Current ConversationalRAGState dictionary.

    Returns:
        Next node name: 'generate_grounded_response' or 'inability_fallback'.
    """
    chunks = state.get("retrieved_chunks", [])
    if not chunks or len(chunks) == 0:
        return "inability_fallback"
    return "generate_grounded_response"


def route_retrieval_grade(state: ConversationalRAGState) -> str:
    """Conditional routing function after retrieval grading (Legacy Mode).

    Args:
        state: Current ConversationalRAGState dictionary.

    Returns:
        Next node name: 'generate_grounded_response' or 'inability_fallback'.
    """
    grade = state.get("retrieval_grade", "insufficient")
    if grade == "grounded":
        return "generate_grounded_response"
    return "inability_fallback"


ConversationalRAGStateGraph = StateGraph[ConversationalRAGState, Any, Any, Any]
CompiledConversationalRAGGraph = CompiledStateGraph[ConversationalRAGState, Any, Any, Any]


def build_conversational_rag_graph(
    streamlined: bool = True,
) -> ConversationalRAGStateGraph:
    """Assemble the Conversational RAG StateGraph.

    Args:
        streamlined: When True (default), builds the streamlined single-pass architecture
                     with Turn-1 bypass and pure-Python gating. When False, builds the
                     legacy 4-node evaluation graph for Task 6.1 harness verification.

    Returns:
        Uncompiled StateGraph instance.
    """
    workflow = StateGraph(ConversationalRAGState)

    if streamlined:
        # Streamlined Architecture (Task 6.3):
        # 1. Register Nodes
        workflow.add_node("bypass_reformulation", bypass_reformulation_node)
        workflow.add_node("reformulate_query", reformulate_query_node)
        workflow.add_node("scoped_hybrid_retriever", scoped_hybrid_retriever_node)
        workflow.add_node("generate_grounded_response", generate_grounded_response_node)
        workflow.add_node("inability_fallback", inability_fallback_node)

        # 2. Entry Branching (Turn-1 bypass vs Multi-turn rewrite)
        workflow.add_conditional_edges(
            START,
            route_reformulation_entry,
            {
                "bypass_reformulation": "bypass_reformulation",
                "reformulate_query": "reformulate_query",
            },
        )
        workflow.add_edge("bypass_reformulation", "scoped_hybrid_retriever")
        workflow.add_edge("reformulate_query", "scoped_hybrid_retriever")

        # 3. Candidate Quality Gating (Pure-Python check before generation)
        workflow.add_conditional_edges(
            "scoped_hybrid_retriever",
            route_retrieval_candidate_quality,
            {
                "generate_grounded_response": "generate_grounded_response",
                "inability_fallback": "inability_fallback",
            },
        )

        # 4. Terminal Edges
        workflow.add_edge("generate_grounded_response", END)
        workflow.add_edge("inability_fallback", END)

    else:
        # Legacy Evaluation Mode (Task 6.1):
        workflow.add_node("reformulate_query", reformulate_query_node)
        workflow.add_node("scoped_hybrid_retriever", scoped_hybrid_retriever_node)
        workflow.add_node("grade_retrieval", grade_retrieval_node)
        workflow.add_node("generate_grounded_response", generate_grounded_response_node)
        workflow.add_node("inability_fallback", inability_fallback_node)

        workflow.add_edge(START, "reformulate_query")
        workflow.add_edge("reformulate_query", "scoped_hybrid_retriever")
        workflow.add_edge("scoped_hybrid_retriever", "grade_retrieval")

        workflow.add_conditional_edges(
            "grade_retrieval",
            route_retrieval_grade,
            {
                "generate_grounded_response": "generate_grounded_response",
                "inability_fallback": "inability_fallback",
            },
        )

        workflow.add_edge("generate_grounded_response", END)
        workflow.add_edge("inability_fallback", END)

    return workflow


def compile_conversational_rag_graph(
    streamlined: bool = True,
) -> CompiledConversationalRAGGraph:
    """Build and compile the conversational RAG StateGraph.

    Args:
        streamlined: Whether to compile the streamlined architecture (default True).

    Returns:
        CompiledStateGraph ready for invocation.
    """
    graph = build_conversational_rag_graph(streamlined=streamlined)
    return graph.compile()


_COMPILED_GRAPH: CompiledConversationalRAGGraph | None = None
_COMPILED_STREAMLINED_GRAPH: CompiledConversationalRAGGraph | None = None
_COMPILED_LEGACY_GRAPH: CompiledConversationalRAGGraph | None = None
_GRAPH_LOCK = threading.Lock()


def get_conversational_rag_graph(
    streamlined: bool = True,
) -> CompiledConversationalRAGGraph:
    """Return the cached singleton CompiledStateGraph instance."""
    global _COMPILED_GRAPH, _COMPILED_STREAMLINED_GRAPH, _COMPILED_LEGACY_GRAPH
    if streamlined:
        if _COMPILED_STREAMLINED_GRAPH is None:
            with _GRAPH_LOCK:
                if _COMPILED_STREAMLINED_GRAPH is None:
                    _COMPILED_STREAMLINED_GRAPH = compile_conversational_rag_graph(streamlined=True)
                    _COMPILED_GRAPH = _COMPILED_STREAMLINED_GRAPH
        return _COMPILED_STREAMLINED_GRAPH
    else:
        if _COMPILED_LEGACY_GRAPH is None:
            with _GRAPH_LOCK:
                if _COMPILED_LEGACY_GRAPH is None:
                    _COMPILED_LEGACY_GRAPH = compile_conversational_rag_graph(streamlined=False)
        return _COMPILED_LEGACY_GRAPH


def reset_conversational_rag_graph() -> None:
    """Reset the cached singleton CompiledStateGraph instances (primarily for tests)."""
    global _COMPILED_GRAPH, _COMPILED_STREAMLINED_GRAPH, _COMPILED_LEGACY_GRAPH
    with _GRAPH_LOCK:
        _COMPILED_GRAPH = None
        _COMPILED_STREAMLINED_GRAPH = None
        _COMPILED_LEGACY_GRAPH = None


def run_conversational_rag_graph(
    initial_state: dict[str, Any],
    streamlined: bool = True,
) -> dict[str, Any]:
    """Execute the compiled conversational RAG graph on input state.

    Args:
        initial_state: Partial or complete ConversationalRAGState dictionary.
        streamlined: Whether to run the streamlined pipeline (default True).

    Returns:
        Final state dictionary produced by graph execution.
    """
    app = get_conversational_rag_graph(streamlined=streamlined)
    result = app.invoke(initial_state)
    return dict(result)
