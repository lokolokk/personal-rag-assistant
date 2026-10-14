"""Offline LangGraph StateGraph contract and retrieval runner (Task 6.1 Slices C & D).

Executes the compiled LangGraph state machine inside the isolated_services sandbox
with deterministic fake clients for QueryReformulator, RetrievalGrader, and
GroundedGenerator, ephemeral ChromaDB vector stores, and in-memory BM25 stores.
Provides original-PDF source corpus chunk reconstruction and retrieval diagnostics.
"""

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from app.core.config import get_settings
from app.schemas.chat import ChatMessage
from app.schemas.document import ChunkMetadata, DocumentChunk, ScoredChunk
from app.services.evaluation.answer_quality import (
    apply_semantic_review,
    load_recorded_case_report,
    replay_recorded_case,
)
from app.services.evaluation.grounding_report import (
    CaseEvaluationRecord,
    EvaluationSuite,
    EvidenceLevel,
    FactAssessment,
    FailureCode,
    ResultKind,
    SemanticCaseReview,
    load_suite_manifest,
)
from app.services.evaluation.offline import OfflineNetworkError, isolated_services, offline_network
from app.services.evaluation.report_store import ReportStore, StoredUsage, recover_report
from app.services.evaluation.retrieval import (
    RetrievalReport,
    evaluate_retrieval,
)
from app.services.evaluation.source_dataset import (
    SourceDataset,
)
from app.services.evaluation.usage import UsageRecord
from app.services.generation.graph import (
    get_conversational_rag_graph,
    reset_conversational_rag_graph,
)
from app.services.generation.grounding import (
    FALLBACK_RESPONSE,
    GroundedGenerator,
    RetrievalGrader,
    set_grounded_generator,
    set_retrieval_grader,
)
from app.services.ingestion.chunker import StructuralChunker
from app.services.ingestion.parser import PDFParser
from app.services.retrieval.bm25_store import get_bm25_store
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.reformulator import (
    QueryReformulator,
    set_query_reformulator,
)
from app.services.retrieval.vector_store import get_vector_store


class FakeReformulatorClient:
    """Deterministic mock client for QueryReformulator recording invocations."""

    def __init__(
        self,
        rewrite_fn: Callable[[str], str] | str | None = None,
    ) -> None:
        self._rewrite_fn = rewrite_fn
        self.call_count: int = 0
        self.recorded_calls: list[dict[str, Any]] = []

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        self.recorded_calls.append({"prompt": prompt})
        if callable(self._rewrite_fn):
            return self._rewrite_fn(prompt)
        if isinstance(self._rewrite_fn, str):
            return self._rewrite_fn
        if "User Follow-up:\n" in prompt and "\n\nStandalone Search Query:" in prompt:
            return (
                prompt.split("User Follow-up:\n")[-1]
                .split("\n\nStandalone Search Query:")[0]
                .strip()
            )
        return prompt.strip()


class FakeGraderClient:
    """Deterministic mock client for RetrievalGrader recording invocations."""

    def __init__(
        self,
        grade: (
            Literal["grounded", "insufficient"]
            | Callable[[str, Sequence[Any]], Literal["grounded", "insufficient"]]
        ) = "grounded",
    ) -> None:
        self._grade = grade
        self.call_count: int = 0
        self.recorded_calls: list[dict[str, Any]] = []

    def __call__(self, query: str, chunks: Sequence[Any]) -> Literal["grounded", "insufficient"]:
        self.call_count += 1
        self.recorded_calls.append({"query": query, "chunks": list(chunks)})
        if callable(self._grade):
            return self._grade(query, chunks)
        return self._grade


class FakeGeneratorClient:
    """Deterministic mock client for GroundedGenerator recording invocations."""

    def __init__(
        self,
        response_fn: Callable[[str, Sequence[Any]], str] | str = "",
    ) -> None:
        self._response_fn = response_fn
        self.call_count: int = 0
        self.recorded_calls: list[dict[str, Any]] = []

    def __call__(self, prompt: str, chunks: Sequence[Any]) -> str:
        self.call_count += 1
        self.recorded_calls.append({"prompt": prompt, "chunks": list(chunks)})
        if callable(self._response_fn):
            return self._response_fn(prompt, chunks)
        return self._response_fn


class ContractScenarioResult(BaseModel):
    """Result of an offline LangGraph contract scenario execution."""

    model_config = ConfigDict(extra="forbid", strict=True)

    scenario_id: str
    expected_route: str
    route_taken: str
    visited_nodes: list[str]
    standalone_query: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieval_grade: str
    response_text: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    reformulator_calls: int
    grader_calls: int
    generator_calls: int
    passed: bool
    failure_reason: str | None = None
    execution_error: str | None = None
    duration_seconds: float = 0.0


def create_synthetic_chunk(
    chunk_id: str,
    document_id: str,
    filename: str,
    content: str,
    page_number: int = 1,
    section_title: str = "Section 1",
    is_table: bool = False,
) -> DocumentChunk:
    """Construct a well-formed DocumentChunk for synthetic scenario testing."""
    token_count = max(1, len(content.split()))
    metadata = ChunkMetadata(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=filename,
        page_number=page_number,
        section_title=section_title,
        is_table=is_table,
        token_count=token_count,
        char_start=0,
        char_end=len(content),
    )
    return DocumentChunk(
        chunk_id=chunk_id,
        content=content,
        metadata=metadata,
    )


def _extract_chunk_id(chunk: Any) -> str:
    """Extract string chunk_id from ScoredChunk, DocumentChunk, or dictionary representation."""
    if isinstance(chunk, (DocumentChunk, ScoredChunk)):
        return str(chunk.chunk_id)
    if isinstance(chunk, dict):
        return str(chunk.get("chunk_id", chunk.get("id", "")))
    return str(getattr(chunk, "chunk_id", ""))


def execute_graph_scenario(
    initial_state: dict[str, Any],
    streamlined: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Execute compiled LangGraph and record visited node names and accumulated state."""
    graph = get_conversational_rag_graph(streamlined=streamlined)
    visited_nodes: list[str] = []
    accumulated_state: dict[str, Any] = dict(initial_state)

    for update in graph.stream(initial_state, stream_mode="updates"):
        if isinstance(update, dict):
            for node_name, state_delta in update.items():
                visited_nodes.append(node_name)
                if isinstance(state_delta, dict):
                    accumulated_state.update(state_delta)

    return accumulated_state, visited_nodes


def _run_single_doc_answerable(sandbox_dir: Path) -> ContractScenarioResult:
    """Scenario 1: Single document answerable controlled grounded path."""
    start_time = time.perf_counter()
    with isolated_services(sandbox_dir):
        reset_conversational_rag_graph()
        v_store = get_vector_store()
        b_store = get_bm25_store()

        chunk = create_synthetic_chunk(
            chunk_id="alpha_c1",
            document_id="doc_alpha",
            filename="alpha_paper.pdf",
            content="Alpha achieves 94.5% accuracy on benchmark X with 12ms inference latency.",
            page_number=1,
            section_title="Benchmark Results",
        )
        v_store.add_chunks([chunk])
        b_store.add_chunks([chunk])

        fake_ref = FakeReformulatorClient(
            rewrite_fn="Alpha accuracy on benchmark X in alpha_paper.pdf"
        )
        fake_grader = FakeGraderClient(grade="grounded")
        fake_gen = FakeGeneratorClient(
            response_fn="Alpha achieves 94.5% accuracy on benchmark X [1]."
        )

        set_query_reformulator(QueryReformulator(llm_client=fake_ref))
        set_retrieval_grader(RetrievalGrader(grading_fn=fake_grader))
        set_grounded_generator(GroundedGenerator(generating_fn=fake_gen))

        state_input: dict[str, Any] = {
            "conversation_id": "conv_single_doc",
            "active_document_ids": ["doc_alpha"],
            "active_document_titles": "alpha_paper.pdf",
            "messages": [],
            "user_query": "What is Alpha's accuracy on benchmark X?",
            "standalone_query": "",
            "retrieved_chunks": [],
            "retrieval_grade": "insufficient",
            "response_text": "",
            "citations": [],
        }

        final_state, visited = execute_graph_scenario(state_input)
        duration = time.perf_counter() - start_time

        retrieved_ids = [_extract_chunk_id(c) for c in final_state.get("retrieved_chunks", [])]
        citations = list(final_state.get("citations", []))
        route_taken = visited[-1] if visited else "unknown"

        passed = (
            route_taken == "generate_grounded_response"
            and visited
            == [
                "reformulate_query",
                "scoped_hybrid_retriever",
                "grade_retrieval",
                "generate_grounded_response",
            ]
            and final_state.get("retrieval_grade") == "grounded"
            and retrieved_ids == ["alpha_c1"]
            and len(citations) == 1
            and citations[0].get("chunk_id") == "alpha_c1"
            and citations[0].get("filename") == "alpha_paper.pdf"
            and fake_ref.call_count == 1
            and fake_grader.call_count == 1
            and fake_gen.call_count == 1
        )
        failure_reason = None if passed else "Single document contract expectations not met"

        return ContractScenarioResult(
            scenario_id="contract_single_doc_answerable",
            expected_route="grounded",
            route_taken=route_taken,
            visited_nodes=visited,
            standalone_query=str(final_state.get("standalone_query", "")),
            retrieved_chunk_ids=retrieved_ids,
            retrieval_grade=str(final_state.get("retrieval_grade", "")),
            response_text=str(final_state.get("response_text", "")),
            citations=citations,
            reformulator_calls=fake_ref.call_count,
            grader_calls=fake_grader.call_count,
            generator_calls=fake_gen.call_count,
            passed=passed,
            failure_reason=failure_reason,
            duration_seconds=duration,
        )


def _run_empty_retrieval(sandbox_dir: Path) -> ContractScenarioResult:
    """Scenario 2: Empty retrieval bypasses grader and generator directly to fallback."""
    start_time = time.perf_counter()
    with isolated_services(sandbox_dir):
        reset_conversational_rag_graph()
        v_store = get_vector_store()
        b_store = get_bm25_store()

        other_chunk = create_synthetic_chunk(
            chunk_id="other_c1",
            document_id="doc_other",
            filename="other.pdf",
            content="Cellular biology mitosis and plant chloroplast structures.",
            page_number=2,
            section_title="Botany",
        )
        v_store.add_chunks([other_chunk])
        b_store.add_chunks([other_chunk])

        fake_ref = FakeReformulatorClient(rewrite_fn="quantum teleportation fidelity")
        fake_grader = FakeGraderClient(grade="grounded")
        fake_gen = FakeGeneratorClient(response_fn="Should not be generated [1].")

        set_query_reformulator(QueryReformulator(llm_client=fake_ref))
        set_retrieval_grader(RetrievalGrader(grading_fn=fake_grader))
        set_grounded_generator(GroundedGenerator(generating_fn=fake_gen))

        state_input: dict[str, Any] = {
            "conversation_id": "conv_empty_retrieval",
            "active_document_ids": ["doc_empty_target"],
            "active_document_titles": "empty_target.pdf",
            "messages": [],
            "user_query": "What is the quantum teleportation fidelity?",
            "standalone_query": "",
            "retrieved_chunks": [],
            "retrieval_grade": "insufficient",
            "response_text": "",
            "citations": [],
        }

        final_state, visited = execute_graph_scenario(state_input)
        duration = time.perf_counter() - start_time

        retrieved_ids = [_extract_chunk_id(c) for c in final_state.get("retrieved_chunks", [])]
        citations = list(final_state.get("citations", []))
        route_taken = visited[-1] if visited else "unknown"

        passed = (
            route_taken == "inability_fallback"
            and visited
            == [
                "reformulate_query",
                "scoped_hybrid_retriever",
                "grade_retrieval",
                "inability_fallback",
            ]
            and len(retrieved_ids) == 0
            and final_state.get("retrieval_grade") == "insufficient"
            and final_state.get("response_text") == FALLBACK_RESPONSE
            and len(citations) == 0
            and fake_ref.call_count == 1
            and fake_grader.call_count == 0
            and fake_gen.call_count == 0
        )
        failure_reason = None if passed else "Empty retrieval did not bypass grader and generator"

        return ContractScenarioResult(
            scenario_id="contract_empty_retrieval",
            expected_route="fallback",
            route_taken=route_taken,
            visited_nodes=visited,
            standalone_query=str(final_state.get("standalone_query", "")),
            retrieved_chunk_ids=retrieved_ids,
            retrieval_grade=str(final_state.get("retrieval_grade", "")),
            response_text=str(final_state.get("response_text", "")),
            citations=citations,
            reformulator_calls=fake_ref.call_count,
            grader_calls=fake_grader.call_count,
            generator_calls=fake_gen.call_count,
            passed=passed,
            failure_reason=failure_reason,
            duration_seconds=duration,
        )


def _run_insufficient_context(sandbox_dir: Path) -> ContractScenarioResult:
    """Scenario 3: Non-empty irrelevant context judged insufficient by grader."""
    start_time = time.perf_counter()
    with isolated_services(sandbox_dir):
        reset_conversational_rag_graph()
        v_store = get_vector_store()
        b_store = get_bm25_store()

        chunk = create_synthetic_chunk(
            chunk_id="recipe_c1",
            document_id="doc_alpha",
            filename="recipes.pdf",
            content="The baking recipe batch size is 24 cookies at 180 degrees Celsius.",
            page_number=1,
            section_title="Pastry",
        )
        v_store.add_chunks([chunk])
        b_store.add_chunks([chunk])

        fake_ref = FakeReformulatorClient(rewrite_fn="training batch size of model in recipes.pdf")
        fake_grader = FakeGraderClient(grade="insufficient")
        fake_gen = FakeGeneratorClient(response_fn="Should not be called [1].")

        set_query_reformulator(QueryReformulator(llm_client=fake_ref))
        set_retrieval_grader(RetrievalGrader(grading_fn=fake_grader))
        set_grounded_generator(GroundedGenerator(generating_fn=fake_gen))

        state_input: dict[str, Any] = {
            "conversation_id": "conv_insufficient",
            "active_document_ids": ["doc_alpha"],
            "active_document_titles": "recipes.pdf",
            "messages": [],
            "user_query": "What is the training batch size of the model?",
            "standalone_query": "",
            "retrieved_chunks": [],
            "retrieval_grade": "insufficient",
            "response_text": "",
            "citations": [],
        }

        final_state, visited = execute_graph_scenario(state_input)
        duration = time.perf_counter() - start_time

        retrieved_ids = [_extract_chunk_id(c) for c in final_state.get("retrieved_chunks", [])]
        citations = list(final_state.get("citations", []))
        route_taken = visited[-1] if visited else "unknown"

        passed = (
            route_taken == "inability_fallback"
            and visited
            == [
                "reformulate_query",
                "scoped_hybrid_retriever",
                "grade_retrieval",
                "inability_fallback",
            ]
            and len(retrieved_ids) > 0
            and final_state.get("retrieval_grade") == "insufficient"
            and final_state.get("response_text") == FALLBACK_RESPONSE
            and len(citations) == 0
            and fake_ref.call_count == 1
            and fake_grader.call_count == 1
            and fake_gen.call_count == 0
        )
        failure_reason = None if passed else "Insufficient context did not suppress generator"

        return ContractScenarioResult(
            scenario_id="contract_insufficient_context",
            expected_route="fallback",
            route_taken=route_taken,
            visited_nodes=visited,
            standalone_query=str(final_state.get("standalone_query", "")),
            retrieved_chunk_ids=retrieved_ids,
            retrieval_grade=str(final_state.get("retrieval_grade", "")),
            response_text=str(final_state.get("response_text", "")),
            citations=citations,
            reformulator_calls=fake_ref.call_count,
            grader_calls=fake_grader.call_count,
            generator_calls=fake_gen.call_count,
            passed=passed,
            failure_reason=failure_reason,
            duration_seconds=duration,
        )


def _run_generator_fallback(sandbox_dir: Path) -> ContractScenarioResult:
    """Scenario 4: Grounded route selected, but generator emits standard fallback."""
    start_time = time.perf_counter()
    with isolated_services(sandbox_dir):
        reset_conversational_rag_graph()
        v_store = get_vector_store()
        b_store = get_bm25_store()

        chunk = create_synthetic_chunk(
            chunk_id="lr_c1",
            document_id="doc_alpha",
            filename="experiment.pdf",
            content="Preliminary learning rate experiments evaluated 1e-4 and 1e-3 on dataset D.",
            page_number=3,
            section_title="Ablations",
        )
        v_store.add_chunks([chunk])
        b_store.add_chunks([chunk])

        fake_ref = FakeReformulatorClient(rewrite_fn="final production learning rate")
        fake_grader = FakeGraderClient(grade="grounded")
        fake_gen = FakeGeneratorClient(response_fn=FALLBACK_RESPONSE)

        set_query_reformulator(QueryReformulator(llm_client=fake_ref))
        set_retrieval_grader(RetrievalGrader(grading_fn=fake_grader))
        set_grounded_generator(GroundedGenerator(generating_fn=fake_gen))

        state_input: dict[str, Any] = {
            "conversation_id": "conv_gen_fallback",
            "active_document_ids": ["doc_alpha"],
            "active_document_titles": "experiment.pdf",
            "messages": [],
            "user_query": "What was the final production learning rate?",
            "standalone_query": "",
            "retrieved_chunks": [],
            "retrieval_grade": "insufficient",
            "response_text": "",
            "citations": [],
        }

        final_state, visited = execute_graph_scenario(state_input)
        duration = time.perf_counter() - start_time

        retrieved_ids = [_extract_chunk_id(c) for c in final_state.get("retrieved_chunks", [])]
        citations = list(final_state.get("citations", []))
        route_taken = visited[-1] if visited else "unknown"

        passed = (
            route_taken == "generate_grounded_response"
            and visited
            == [
                "reformulate_query",
                "scoped_hybrid_retriever",
                "grade_retrieval",
                "generate_grounded_response",
            ]
            and final_state.get("retrieval_grade") == "grounded"
            and final_state.get("response_text") == FALLBACK_RESPONSE
            and len(citations) == 0
            and fake_ref.call_count == 1
            and fake_grader.call_count == 1
            and fake_gen.call_count == 1
        )
        failure_reason = None if passed else "Generator fallback route or response mismatched"

        return ContractScenarioResult(
            scenario_id="contract_generator_fallback",
            expected_route="fallback",
            route_taken=route_taken,
            visited_nodes=visited,
            standalone_query=str(final_state.get("standalone_query", "")),
            retrieved_chunk_ids=retrieved_ids,
            retrieval_grade=str(final_state.get("retrieval_grade", "")),
            response_text=str(final_state.get("response_text", "")),
            citations=citations,
            reformulator_calls=fake_ref.call_count,
            grader_calls=fake_grader.call_count,
            generator_calls=fake_gen.call_count,
            passed=passed,
            failure_reason=failure_reason,
            duration_seconds=duration,
        )


def _run_multi_doc_comparison(sandbox_dir: Path) -> ContractScenarioResult:
    """Scenario 5: Two synthetic documents with distinct conflicting facts and metrics."""
    start_time = time.perf_counter()
    with isolated_services(sandbox_dir):
        reset_conversational_rag_graph()
        v_store = get_vector_store()
        b_store = get_bm25_store()

        chunk_a = create_synthetic_chunk(
            chunk_id="ca1",
            document_id="doc_a",
            filename="alpha.pdf",
            content="Alpha architecture uses 12 transformer layers.",
            page_number=2,
            section_title="Model Architecture",
        )
        chunk_b = create_synthetic_chunk(
            chunk_id="cb1",
            document_id="doc_b",
            filename="beta.pdf",
            content="Beta architecture uses 24 transformer layers.",
            page_number=4,
            section_title="Model Architecture",
        )
        v_store.add_chunks([chunk_a, chunk_b])
        b_store.add_chunks([chunk_a, chunk_b])

        fake_ref = FakeReformulatorClient(
            rewrite_fn="Compare transformer layers between alpha.pdf and beta.pdf"
        )
        fake_grader = FakeGraderClient(grade="grounded")
        fake_gen = FakeGeneratorClient(
            response_fn=(
                "In alpha.pdf, Alpha uses 12 layers [1], "
                "whereas in beta.pdf, Beta uses 24 layers [2]."
            )
        )

        set_query_reformulator(QueryReformulator(llm_client=fake_ref))
        set_retrieval_grader(RetrievalGrader(grading_fn=fake_grader))
        set_grounded_generator(GroundedGenerator(generating_fn=fake_gen))

        state_input: dict[str, Any] = {
            "conversation_id": "conv_multi_doc",
            "active_document_ids": ["doc_a", "doc_b"],
            "active_document_titles": "alpha.pdf, beta.pdf",
            "messages": [],
            "user_query": "How many layers do Alpha and Beta architectures use?",
            "standalone_query": "",
            "retrieved_chunks": [],
            "retrieval_grade": "insufficient",
            "response_text": "",
            "citations": [],
        }

        final_state, visited = execute_graph_scenario(state_input)
        duration = time.perf_counter() - start_time

        retrieved_ids = [_extract_chunk_id(c) for c in final_state.get("retrieved_chunks", [])]
        citations = list(final_state.get("citations", []))
        route_taken = visited[-1] if visited else "unknown"

        citation_filenames = {c.get("filename") for c in citations}
        passed = (
            route_taken == "generate_grounded_response"
            and visited
            == [
                "reformulate_query",
                "scoped_hybrid_retriever",
                "grade_retrieval",
                "generate_grounded_response",
            ]
            and set(retrieved_ids) == {"ca1", "cb1"}
            and len(citations) == 2
            and citation_filenames == {"alpha.pdf", "beta.pdf"}
            and fake_ref.call_count == 1
            and fake_grader.call_count == 1
            and fake_gen.call_count == 1
        )
        failure_reason = None if passed else "Multi-doc comparison routing or citations mismatched"

        return ContractScenarioResult(
            scenario_id="contract_multi_doc_comparison",
            expected_route="grounded",
            route_taken=route_taken,
            visited_nodes=visited,
            standalone_query=str(final_state.get("standalone_query", "")),
            retrieved_chunk_ids=retrieved_ids,
            retrieval_grade=str(final_state.get("retrieval_grade", "")),
            response_text=str(final_state.get("response_text", "")),
            citations=citations,
            reformulator_calls=fake_ref.call_count,
            grader_calls=fake_grader.call_count,
            generator_calls=fake_gen.call_count,
            passed=passed,
            failure_reason=failure_reason,
            duration_seconds=duration,
        )


def _run_scoped_distractor(sandbox_dir: Path) -> ContractScenarioResult:
    """Scenario 6: Single-document scope filters out highly relevant out-of-scope distractor."""
    start_time = time.perf_counter()
    with isolated_services(sandbox_dir):
        reset_conversational_rag_graph()
        v_store = get_vector_store()
        b_store = get_bm25_store()

        target_chunk = create_synthetic_chunk(
            chunk_id="target_c1",
            document_id="doc_target",
            filename="target.pdf",
            content="Algorithm Omega achieves 75.0% recall on technical corpus T.",
            page_number=1,
            section_title="Evaluation",
        )
        distractor_chunk = create_synthetic_chunk(
            chunk_id="distractor_c1",
            document_id="doc_distractor",
            filename="distractor.pdf",
            content="Algorithm Omega achieves 99.9% recall on technical corpus T with speed.",
            page_number=1,
            section_title="Evaluation",
        )
        v_store.add_chunks([target_chunk, distractor_chunk])
        b_store.add_chunks([target_chunk, distractor_chunk])

        fake_ref = FakeReformulatorClient(
            rewrite_fn="Algorithm Omega recall on technical corpus T in target.pdf"
        )
        fake_grader = FakeGraderClient(grade="grounded")
        fake_gen = FakeGeneratorClient(
            response_fn="Algorithm Omega achieves 75.0% recall on technical corpus T [1]."
        )

        set_query_reformulator(QueryReformulator(llm_client=fake_ref))
        set_retrieval_grader(RetrievalGrader(grading_fn=fake_grader))
        set_grounded_generator(GroundedGenerator(generating_fn=fake_gen))

        state_input: dict[str, Any] = {
            "conversation_id": "conv_scoped_distractor",
            "active_document_ids": ["doc_target"],
            "active_document_titles": "target.pdf",
            "messages": [],
            "user_query": "What is Algorithm Omega's recall on technical corpus T?",
            "standalone_query": "",
            "retrieved_chunks": [],
            "retrieval_grade": "insufficient",
            "response_text": "",
            "citations": [],
        }

        final_state, visited = execute_graph_scenario(state_input)
        duration = time.perf_counter() - start_time

        retrieved_ids = [_extract_chunk_id(c) for c in final_state.get("retrieved_chunks", [])]
        citations = list(final_state.get("citations", []))
        route_taken = visited[-1] if visited else "unknown"

        passed = (
            route_taken == "generate_grounded_response"
            and visited
            == [
                "reformulate_query",
                "scoped_hybrid_retriever",
                "grade_retrieval",
                "generate_grounded_response",
            ]
            and "distractor_c1" not in retrieved_ids
            and retrieved_ids == ["target_c1"]
            and len(citations) == 1
            and citations[0].get("chunk_id") == "target_c1"
            and citations[0].get("filename") == "target.pdf"
            and fake_ref.call_count == 1
            and fake_grader.call_count == 1
            and fake_gen.call_count == 1
        )
        failure_reason = None if passed else "Distractor chunk was not isolated by document scope"

        return ContractScenarioResult(
            scenario_id="contract_scoped_distractor",
            expected_route="grounded",
            route_taken=route_taken,
            visited_nodes=visited,
            standalone_query=str(final_state.get("standalone_query", "")),
            retrieved_chunk_ids=retrieved_ids,
            retrieval_grade=str(final_state.get("retrieval_grade", "")),
            response_text=str(final_state.get("response_text", "")),
            citations=citations,
            reformulator_calls=fake_ref.call_count,
            grader_calls=fake_grader.call_count,
            generator_calls=fake_gen.call_count,
            passed=passed,
            failure_reason=failure_reason,
            duration_seconds=duration,
        )


def _run_empty_scope(sandbox_dir: Path) -> ContractScenarioResult:
    """Scenario 7: Explicit empty scope searches all documents without error."""
    start_time = time.perf_counter()
    with isolated_services(sandbox_dir):
        reset_conversational_rag_graph()
        v_store = get_vector_store()
        b_store = get_bm25_store()

        chunk_1 = create_synthetic_chunk(
            chunk_id="c1",
            document_id="doc_1",
            filename="doc1.pdf",
            content="Consensus algorithm Raft uses leader election.",
            page_number=1,
            section_title="Overview",
        )
        chunk_2 = create_synthetic_chunk(
            chunk_id="c2",
            document_id="doc_2",
            filename="doc2.pdf",
            content="Consensus algorithm Paxos uses two-phase commits.",
            page_number=1,
            section_title="Overview",
        )
        v_store.add_chunks([chunk_1, chunk_2])
        b_store.add_chunks([chunk_1, chunk_2])

        fake_ref = FakeReformulatorClient(rewrite_fn="consensus algorithm details")
        fake_grader = FakeGraderClient(grade="grounded")
        fake_gen = FakeGeneratorClient(
            response_fn="Raft uses leader election [1] while Paxos uses two-phase commits [2]."
        )

        set_query_reformulator(QueryReformulator(llm_client=fake_ref))
        set_retrieval_grader(RetrievalGrader(grading_fn=fake_grader))
        set_grounded_generator(GroundedGenerator(generating_fn=fake_gen))

        state_input: dict[str, Any] = {
            "conversation_id": "conv_empty_scope",
            "active_document_ids": [],
            "active_document_titles": "",
            "messages": [],
            "user_query": "Tell me about consensus algorithms.",
            "standalone_query": "",
            "retrieved_chunks": [],
            "retrieval_grade": "insufficient",
            "response_text": "",
            "citations": [],
        }

        final_state, visited = execute_graph_scenario(state_input)
        duration = time.perf_counter() - start_time

        retrieved_ids = [_extract_chunk_id(c) for c in final_state.get("retrieved_chunks", [])]
        citations = list(final_state.get("citations", []))
        route_taken = visited[-1] if visited else "unknown"

        passed = (
            route_taken == "generate_grounded_response"
            and visited
            == [
                "reformulate_query",
                "scoped_hybrid_retriever",
                "grade_retrieval",
                "generate_grounded_response",
            ]
            and set(retrieved_ids) == {"c1", "c2"}
            and len(citations) == 2
            and fake_ref.call_count == 1
            and fake_grader.call_count == 1
            and fake_gen.call_count == 1
        )
        failure_reason = None if passed else "Empty scope failed to search across all documents"

        return ContractScenarioResult(
            scenario_id="contract_empty_scope",
            expected_route="grounded",
            route_taken=route_taken,
            visited_nodes=visited,
            standalone_query=str(final_state.get("standalone_query", "")),
            retrieved_chunk_ids=retrieved_ids,
            retrieval_grade=str(final_state.get("retrieval_grade", "")),
            response_text=str(final_state.get("response_text", "")),
            citations=citations,
            reformulator_calls=fake_ref.call_count,
            grader_calls=fake_grader.call_count,
            generator_calls=fake_gen.call_count,
            passed=passed,
            failure_reason=failure_reason,
            duration_seconds=duration,
        )


def _run_history_followup(sandbox_dir: Path) -> ContractScenarioResult:
    """Scenario 8: Multi-turn history follow-up reaches reformulator before retrieval."""
    start_time = time.perf_counter()
    with isolated_services(sandbox_dir):
        reset_conversational_rag_graph()
        v_store = get_vector_store()
        b_store = get_bm25_store()

        chunk = create_synthetic_chunk(
            chunk_id="gamma_c1",
            document_id="doc_gamma",
            filename="gamma.pdf",
            content="Gamma optimizer uses momentum 0.9 and weight decay 0.0001.",
            page_number=3,
            section_title="Hyperparameters",
        )
        v_store.add_chunks([chunk])
        b_store.add_chunks([chunk])

        def verify_prompt_and_rewrite(prompt: str) -> str:
            assert "What is the momentum of Gamma optimizer?" in prompt
            assert "Gamma optimizer uses momentum 0.9." in prompt
            assert "What about its weight decay?" in prompt
            assert "gamma.pdf" in prompt
            return "What is the weight decay of Gamma optimizer in gamma.pdf"

        fake_ref = FakeReformulatorClient(rewrite_fn=verify_prompt_and_rewrite)
        fake_grader = FakeGraderClient(grade="grounded")
        fake_gen = FakeGeneratorClient(response_fn="Gamma optimizer uses weight decay 0.0001 [1].")

        set_query_reformulator(QueryReformulator(llm_client=fake_ref))
        set_retrieval_grader(RetrievalGrader(grading_fn=fake_grader))
        set_grounded_generator(GroundedGenerator(generating_fn=fake_gen))

        history = [
            ChatMessage(role="user", content="What is the momentum of Gamma optimizer?"),
            ChatMessage(role="assistant", content="Gamma optimizer uses momentum 0.9."),
        ]
        state_input: dict[str, Any] = {
            "conversation_id": "conv_history_followup",
            "active_document_ids": ["doc_gamma"],
            "active_document_titles": "gamma.pdf",
            "messages": history,
            "user_query": "What about its weight decay?",
            "standalone_query": "",
            "retrieved_chunks": [],
            "retrieval_grade": "insufficient",
            "response_text": "",
            "citations": [],
        }

        final_state, visited = execute_graph_scenario(state_input)
        duration = time.perf_counter() - start_time

        retrieved_ids = [_extract_chunk_id(c) for c in final_state.get("retrieved_chunks", [])]
        citations = list(final_state.get("citations", []))
        route_taken = visited[-1] if visited else "unknown"

        passed = (
            route_taken == "generate_grounded_response"
            and visited
            == [
                "reformulate_query",
                "scoped_hybrid_retriever",
                "grade_retrieval",
                "generate_grounded_response",
            ]
            and final_state.get("standalone_query")
            == "What is the weight decay of Gamma optimizer in gamma.pdf"
            and retrieved_ids == ["gamma_c1"]
            and len(citations) == 1
            and citations[0].get("filename") == "gamma.pdf"
            and fake_ref.call_count == 1
            and fake_grader.call_count == 1
            and fake_gen.call_count == 1
        )
        failure_reason = None if passed else "History follow-up reformulation flow failed"

        return ContractScenarioResult(
            scenario_id="contract_history_followup",
            expected_route="grounded",
            route_taken=route_taken,
            visited_nodes=visited,
            standalone_query=str(final_state.get("standalone_query", "")),
            retrieved_chunk_ids=retrieved_ids,
            retrieval_grade=str(final_state.get("retrieval_grade", "")),
            response_text=str(final_state.get("response_text", "")),
            citations=citations,
            reformulator_calls=fake_ref.call_count,
            grader_calls=fake_grader.call_count,
            generator_calls=fake_gen.call_count,
            passed=passed,
            failure_reason=failure_reason,
            duration_seconds=duration,
        )


def _run_exception_isolation(sandbox_dir: Path) -> ContractScenarioResult:
    """Scenario 9: Exception isolation and clean service restoration."""
    start_time = time.perf_counter()
    error_message: str | None = None
    with isolated_services(sandbox_dir):
        reset_conversational_rag_graph()
        v_store = get_vector_store()
        b_store = get_bm25_store()

        chunk = create_synthetic_chunk(
            chunk_id="err_c1",
            document_id="doc_err",
            filename="error_test.pdf",
            content="Content for error isolation verification.",
            page_number=1,
            section_title="Error Testing",
        )
        v_store.add_chunks([chunk])
        b_store.add_chunks([chunk])

        def failing_generator(_p: str, _c: Sequence[Any]) -> str:
            raise RuntimeError("Simulated generator fatal failure")

        fake_ref = FakeReformulatorClient(rewrite_fn="query for error test")
        fake_grader = FakeGraderClient(grade="grounded")
        fake_gen = FakeGeneratorClient(response_fn=failing_generator)

        set_query_reformulator(QueryReformulator(llm_client=fake_ref))
        set_retrieval_grader(RetrievalGrader(grading_fn=fake_grader))
        set_grounded_generator(GroundedGenerator(generating_fn=fake_gen))

        state_input: dict[str, Any] = {
            "conversation_id": "conv_error_test",
            "active_document_ids": ["doc_err"],
            "active_document_titles": "error_test.pdf",
            "messages": [],
            "user_query": "Trigger error?",
            "standalone_query": "",
            "retrieved_chunks": [],
            "retrieval_grade": "insufficient",
            "response_text": "",
            "citations": [],
        }

        exception_caught = False
        try:
            execute_graph_scenario(state_input)
        except RuntimeError as err:
            exception_caught = True
            error_message = str(err)

        # Restore working generator and assert graph recovers cleanly in same sandbox
        set_grounded_generator(
            GroundedGenerator(generating_fn=lambda _p, _c: "Recovered cleanly [1].")
        )
        recovery_state, recovery_visited = execute_graph_scenario(state_input)
        recovery_passed = (
            recovery_visited[-1] == "generate_grounded_response"
            and recovery_state.get("response_text") == "Recovered cleanly [1]."
        )

        duration = time.perf_counter() - start_time
        passed = exception_caught and recovery_passed and error_message is not None

        return ContractScenarioResult(
            scenario_id="contract_exception_isolation",
            expected_route="error",
            route_taken="error",
            visited_nodes=["reformulate_query", "scoped_hybrid_retriever", "grade_retrieval"],
            standalone_query="query for error test",
            retrieved_chunk_ids=["err_c1"],
            retrieval_grade="grounded",
            response_text="",
            citations=[],
            reformulator_calls=fake_ref.call_count,
            grader_calls=fake_grader.call_count,
            generator_calls=fake_gen.call_count,
            passed=passed,
            failure_reason=None if passed else "Exception isolation and recovery failed",
            execution_error=error_message,
            duration_seconds=duration,
        )


_SCENARIO_DISPATCH: dict[str, Callable[[Path], ContractScenarioResult]] = {
    "contract_single_doc_answerable": _run_single_doc_answerable,
    "contract_empty_retrieval": _run_empty_retrieval,
    "contract_insufficient_context": _run_insufficient_context,
    "contract_generator_fallback": _run_generator_fallback,
    "contract_multi_doc_comparison": _run_multi_doc_comparison,
    "contract_scoped_distractor": _run_scoped_distractor,
    "contract_empty_scope": _run_empty_scope,
    "contract_history_followup": _run_history_followup,
    "contract_exception_isolation": _run_exception_isolation,
}


def run_contract_scenario(
    scenario_id: str,
    sandbox_root: Path,
) -> ContractScenarioResult:
    """Run an individual offline contract scenario by ID."""
    if scenario_id not in _SCENARIO_DISPATCH:
        raise ValueError(f"Unknown contract scenario ID: '{scenario_id}'")
    runner = _SCENARIO_DISPATCH[scenario_id]
    scenario_dir = sandbox_root / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    return runner(scenario_dir)


def run_all_contract_scenarios(
    sandbox_root: Path,
) -> list[ContractScenarioResult]:
    """Execute all 9 offline LangGraph contract scenarios in sequence."""
    results: list[ContractScenarioResult] = []
    for scenario_id in _SCENARIO_DISPATCH:
        result = run_contract_scenario(scenario_id, sandbox_root)
        results.append(result)
    return results


class RetrievalDiagnosticSummary(BaseModel):
    """Aggregated results and failure diagnoses from source-PDF retrieval evaluation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    suite_id: str
    dataset_id: str
    corpus_document_count: int
    corpus_chunk_count: int
    total_diagnostic_cases: int
    answerable_cases: int
    unanswerable_cases: int
    anchor_hits: int
    extraction_gaps: int
    retrieval_gaps: int
    appropriate_abstentions: int
    scope_leaks: int
    query_policy: str = "annotation_supplied"
    evidence_level: EvidenceLevel = "local_retrieval_diagnostic"
    case_records: list[CaseEvaluationRecord] = Field(default_factory=list)


def reconstruct_source_corpus_chunks(
    dataset: SourceDataset,
    project_root: Path,
) -> list[DocumentChunk]:
    """Parse and structurally chunk all declared source PDFs in the dataset.

    Verifies that:
    1. Each source PDF exists and matches its declared SHA-256 fingerprint.
    2. Parsing and chunking use the exact pinned chunking parameters.
    3. Mapped chunks match expected chunk hashes and page numbers.
    4. Exactly 148 total chunks are produced across the 3 canonical documents.
    """
    root = project_root.resolve()
    all_chunks: list[DocumentChunk] = []

    for document in dataset.documents:
        source_path = (root / document.fixture_path).resolve()
        if not source_path.is_file():
            raise ValueError(f"Source PDF fixture missing: {document.fixture_path}")

        file_bytes = source_path.read_bytes()
        actual_sha256 = hashlib.sha256(file_bytes).hexdigest().lower()
        if actual_sha256 != document.sha256.lower():
            raise ValueError(
                f"Source PDF SHA-256 drift for {document.id} "
                f"(expected {document.sha256}, got {actual_sha256})"
            )

        parser = PDFParser()
        parsed = parser.parse_bytes(file_bytes, document.filename)

        chunker = StructuralChunker(
            target_chunk_size=document.chunking.target_chunk_size,
            chunk_overlap=document.chunking.chunk_overlap,
            encoding_name=document.chunking.encoding,
        )
        doc_chunks = chunker.chunk_document(parsed, document_id=document.id)
        if not doc_chunks:
            raise ValueError(f"Document {document.id} produced zero chunks")

        # Validate chunk mappings for mapped anchors
        chunks_by_id = {c.chunk_id: c for c in doc_chunks}
        for anchor in document.anchors:
            if anchor.chunk_mapping.status == "mapped":
                for chunk_id in anchor.chunk_mapping.chunk_ids:
                    chunk = chunks_by_id.get(chunk_id)
                    if chunk is None:
                        raise ValueError(f"Stale chunk mapping for {anchor.id}: {chunk_id}")
                    if chunk.metadata.page_number != anchor.page_number:
                        raise ValueError(f"Chunk mapping points to another source page: {chunk_id}")
                    expected_hash = anchor.chunk_mapping.chunk_sha256.get(chunk_id)
                    if expected_hash:
                        actual_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
                        if actual_hash != expected_hash:
                            raise ValueError(f"Chunk mapping fingerprint mismatch for {chunk_id}")

        all_chunks.extend(doc_chunks)

    if len(dataset.documents) != 3:
        raise ValueError(
            "Dataset must declare exactly 3 source corpus documents, "
            f"found {len(dataset.documents)}"
        )
    if len(all_chunks) != 148:
        raise ValueError(
            f"Expected exactly 148 source corpus chunks across 3 documents, got {len(all_chunks)}"
        )

    return all_chunks


def build_ephemeral_retriever(
    chunks: Sequence[DocumentChunk],
    sandbox_dir: Path,
) -> HybridRetriever:
    """Index reconstructed chunks into ephemeral VectorStore and BM25Store."""
    settings = get_settings()
    expected_chroma = (sandbox_dir.resolve() / "storage/chroma").resolve()
    if settings.resolved_chroma_dir.resolve() != expected_chroma:
        raise ValueError(
            f"build_ephemeral_retriever must be called inside isolated_services({sandbox_dir})"
        )

    v_store = get_vector_store()
    b_store = get_bm25_store()
    v_store.add_chunks(list(chunks))
    b_store.add_chunks(list(chunks))
    return HybridRetriever(vector_store=v_store, bm25_store=b_store)


def run_source_retrieval_diagnostics(
    suite: EvaluationSuite,
    sandbox_root: Path,
    project_root: Path,
    top_k: int = 5,
) -> tuple[RetrievalDiagnosticSummary, RetrievalReport]:
    """Execute original-PDF retrieval diagnostics over selected suite cases.

    Indexes all 148 reconstructed chunks across the 3 canonical source documents,
    evaluates hybrid retrieval using annotation-supplied queries, and maps
    diagnoses into CaseEvaluationRecord entries with exact failure classification:
    - extraction_gap: source anchor unmapped due to extraction failure (e.g. rgb-06)
    - retrieval_gap: source anchor mapped but not returned in top-k (e.g. rag-02, rag-08, tada-07)
    - appropriate_abstention: genuine absence from source (e.g. tada-08)
    - hit: anchor returned in top-k (e.g. tada-01)
    """
    sandbox_dir = sandbox_root.resolve()
    with isolated_services(sandbox_dir):
        chunks = reconstruct_source_corpus_chunks(suite.source_dataset, project_root)
        retriever = build_ephemeral_retriever(chunks, sandbox_dir)

        report = evaluate_retrieval(
            dataset=suite.source_dataset,
            retriever=retriever,
            k=top_k,
            splits=("diagnostic",),
        )

        cases_by_id = {c.id: c for c in suite.selected_cases}
        case_reports_by_id = {c.case_id: c for c in report.cases}

        case_records: list[CaseEvaluationRecord] = []
        anchor_hits = 0
        extraction_gaps = 0
        retrieval_gaps = 0
        appropriate_abstentions = 0
        scope_leaks = 0

        for case_id in suite.manifest.diagnostic_case_ids:
            case = cases_by_id[case_id]
            case_report = case_reports_by_id[case_id]

            retrieved_chunk_ids = [e.chunk_id for e in case_report.retrieved_evidence]
            has_scope_leak = bool(case_report.scope_leakage_chunk_ids)
            if has_scope_leak:
                scope_leaks += 1

            failure_codes: list[FailureCode] = []
            earliest_stage: FailureCode | None = None
            fact_assessments: list[FactAssessment] = []

            if case.answerability == "unanswerable":
                result_kind: ResultKind = "appropriate_abstention"
                appropriate_abstentions += 1
                retrieval_grade = "insufficient"
            else:
                result_kind = "incomplete"
                is_extraction_gap = bool(case_report.unmapped_fact_ids)
                is_retrieval_gap = bool(case_report.missing_fact_ids) and not is_extraction_gap

                if is_extraction_gap:
                    failure_codes.append("extraction_gap")
                    earliest_stage = "extraction_gap"
                    extraction_gaps += 1
                    retrieval_grade = "insufficient"
                elif is_retrieval_gap:
                    failure_codes.append("retrieval_gap")
                    earliest_stage = "retrieval_gap"
                    retrieval_gaps += 1
                    retrieval_grade = "insufficient"
                else:
                    anchor_hits += 1
                    retrieval_grade = "grounded"

                if has_scope_leak:
                    failure_codes.append("scope_leak")
                    if earliest_stage is None:
                        earliest_stage = "scope_leak"

                for fact in case.required_facts:
                    is_unmapped = fact.id in case_report.unmapped_fact_ids
                    hit = fact.id in case_report.covered_fact_ids
                    fact_assessments.append(
                        FactAssessment(
                            fact_id=fact.id,
                            claim=fact.claim,
                            exact_values=fact.exact_values,
                            unit=fact.unit,
                            qualifications=fact.qualifications,
                            extraction_status="extraction_failure" if is_unmapped else "mapped",
                            anchor_hit=hit,
                            verdict="pass" if hit else "fail",
                            reason=(
                                "Evidence anchor retrieved"
                                if hit
                                else (
                                    "Source anchor extraction failure in corpus chunking"
                                    if is_unmapped
                                    else "Required fact anchor missing from top-k retrieved chunks"
                                )
                            ),
                            citation_indices=[],
                            source_anchor_ids=fact.source_anchor_ids,
                        )
                    )

            case_record = CaseEvaluationRecord(
                case_id=case.id,
                split=case.split,
                exposure_status="diagnostic_exposed_local",
                question=case.question,
                declared_scope=case.scope_document_ids,
                effective_scope=case.scope_document_ids,
                standalone_query=case.retrieval_query,
                result_kind=result_kind,
                retrieval_grade=retrieval_grade,
                retrieved_chunk_ids=retrieved_chunk_ids,
                supplied_chunk_ids=retrieved_chunk_ids,
                response_text=None,
                response_hash=None,
                raw_citation_tags=[],
                citations=[],
                facts=fact_assessments,
                failure_codes=failure_codes,
                earliest_failing_stage=earliest_stage,
                evidence_level="local_retrieval_diagnostic",
                latency_seconds=None,
            )
            case_records.append(case_record)

        summary = RetrievalDiagnosticSummary(
            suite_id=suite.manifest.suite_id,
            dataset_id=suite.source_dataset.dataset_id,
            corpus_document_count=len(suite.source_dataset.documents),
            corpus_chunk_count=len(chunks),
            total_diagnostic_cases=len(suite.manifest.diagnostic_case_ids),
            answerable_cases=sum(
                cases_by_id[cid].answerability == "answerable"
                for cid in suite.manifest.diagnostic_case_ids
            ),
            unanswerable_cases=sum(
                cases_by_id[cid].answerability == "unanswerable"
                for cid in suite.manifest.diagnostic_case_ids
            ),
            anchor_hits=anchor_hits,
            extraction_gaps=extraction_gaps,
            retrieval_gaps=retrieval_gaps,
            appropriate_abstentions=appropriate_abstentions,
            scope_leaks=scope_leaks,
            query_policy="annotation_supplied",
            evidence_level="local_retrieval_diagnostic",
            case_records=case_records,
        )

        return summary, report


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_SUITE = _PROJECT_ROOT / "backend/tests/evaluation/eval_dataset.json"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "backend/.eval-reports/task_6_1"


def _cli_path(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(_PROJECT_ROOT):
        raise ValueError("CLI paths must remain inside the project")
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    """Execute offline work; exit 1 execution, 2 arguments, 3 quality gate failure."""
    parser = argparse.ArgumentParser(
        description="Offline grounding evaluation; no live mode", allow_abbrev=False
    )
    parser.add_argument(
        "--mode", choices=("contracts", "retrieval-diagnostic", "replay"), default="contracts"
    )
    parser.add_argument("--suite", type=Path, default=_DEFAULT_SUITE)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--recorded-report", type=Path)
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--require-quality-pass", action="store_true")
    args = parser.parse_args(argv)
    if args.mode != "replay" and (args.recorded_report or args.review_file):
        parser.error("Recorded report/review arguments require replay mode")
    if args.mode == "replay" and args.recorded_report is None:
        parser.error("Replay requires --recorded-report")
    try:
        for path in (args.suite, args.output_dir, args.recorded_report, args.review_file):
            if path is not None:
                _cli_path(path)
    except ValueError as error:
        parser.error(str(error))

    store: ReportStore | None = None
    try:
        with offline_network():
            suite = load_suite_manifest(args.suite, project_root=_PROJECT_ROOT)
            records: list[CaseEvaluationRecord] = []
            usage: list[StoredUsage] = []
            if args.mode == "contracts":
                keys = [scenario.scenario_id for scenario in suite.manifest.contract_scenarios]
                if not keys or any(key not in _SCENARIO_DISPATCH for key in keys):
                    raise ValueError("Empty/unknown contract scenario selection")
            elif args.mode == "retrieval-diagnostic":
                keys = list(suite.manifest.diagnostic_case_ids)
                if not keys:
                    raise ValueError("Empty diagnostic selection")
            else:
                report = load_recorded_case_report(args.recorded_report)
                case = next(
                    (case for case in suite.selected_cases if case.id == report.case_id), None
                )
                if case is None:
                    raise ValueError("Replay case is outside the diagnostic selection")
                records = [replay_recorded_case(report, arm.name, case) for arm in report.arms]
                if not records:
                    raise ValueError("Recorded report has no arms")
                if args.review_file is not None:
                    reviews = TypeAdapter(list[SemanticCaseReview]).validate_json(
                        args.review_file.read_bytes()
                    )
                    used: set[int] = set()
                    for index, record in enumerate(records):
                        matches = [
                            (i, review)
                            for i, review in enumerate(reviews)
                            if review.case_id == record.case_id
                            and review.run_id == record.run_id
                            and review.answer_sha256 == record.response_hash
                        ]
                        if len(matches) > 1:
                            raise ValueError("Duplicate semantic review binding")
                        if matches:
                            i, review = matches[0]
                            records[index] = apply_semantic_review(record, review)
                            used.add(i)
                    if len(used) != len(reviews):
                        raise ValueError("Review file contains unmatched/stale review")
                keys = [f"{record.case_id}:{record.arm_name}" for record in records]
                for arm in report.arms:
                    attempts = (arm.usage or {}).get("attempts", [])
                    if not isinstance(attempts, list):
                        raise ValueError("Invalid historical usage attempts")
                    for attempt in attempts:
                        observed = TypeAdapter(UsageRecord).validate_python(attempt)
                        usage.append(
                            StoredUsage.from_observed(
                                observed,
                                origin="historical",
                                source_run_id=report.run_id,
                                case_key=f"{report.case_id}:{arm.name}",
                            )
                        )
            store = ReportStore.create(
                project_root=_PROJECT_ROOT,
                output_dir=_cli_path(args.output_dir),
                suite_id=suite.manifest.suite_id,
                planned_case_keys=keys,
            )
            sandbox = store.run_dir / ".sandbox"
            if args.mode == "contracts":
                for key in keys:
                    result = run_contract_scenario(key, sandbox)
                    # Expected injected failures verify a contract; they are not provider answers.
                    record = CaseEvaluationRecord(
                        case_id=key,
                        split="diagnostic",
                        exposure_status="diagnostic_exposed_local",
                        question="",
                        declared_scope=[],
                        effective_scope=[],
                        result_kind="answer",
                        evidence_level="offline_contract",
                        run_id=store.manifest.run_id,
                        claim_support="pass" if result.passed else "fail",
                        failure_codes=[] if result.passed else ["execution_error"],
                        earliest_failing_stage=None if result.passed else "execution_error",
                        retrieved_chunk_ids=result.retrieved_chunk_ids,
                        latency_seconds=result.duration_seconds,
                    )
                    store.append_case(record)
                    if not result.passed:
                        raise ValueError("Offline contract assertion failed")
            elif args.mode == "retrieval-diagnostic":
                diagnostic, _ = run_source_retrieval_diagnostics(suite, sandbox, _PROJECT_ROOT)
                for record in diagnostic.case_records:
                    record.run_id = store.manifest.run_id
                    record.source_dataset_sha256 = suite.manifest.source_manifest_sha256
                    store.append_case(record)
            else:
                for record in records:
                    store.append_case(record)
                for item in usage:
                    store.append_usage(item)
            store.finalize()
            recovered = recover_report(store.run_dir, project_root=_PROJECT_ROOT)
            quality_pass = recovered.complete and recovered.summary.failed_cases == 0
            if args.mode == "replay":
                quality_pass = quality_pass and all(
                    case.successful_provider_answer for case in recovered.cases
                )
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "terminal_state": "completed",
                        "report_dir": str(store.run_dir),
                        "provider_dispatches": 0,
                        "provider_quality_validation_status": "review_pending"
                        if args.mode == "replay"
                        else "not_run",
                        "quality_gate_passed": quality_pass,
                        "excluded_case_ids": suite.excluded_case_ids,
                        "summary": recovered.summary.model_dump(),
                    }
                )
            )
            return 3 if args.require_quality_pass and not quality_pass else 0
    except (Exception, OfflineNetworkError) as error:
        print(
            json.dumps(
                {
                    "terminal_state": "failed",
                    "provider_dispatches": 0,
                    "report_dir": str(store.run_dir) if store else None,
                    "error_type": type(error).__name__,
                }
            )
        )
        print(
            "Offline evaluation failed; inspect retained inputs or recover the journal.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
