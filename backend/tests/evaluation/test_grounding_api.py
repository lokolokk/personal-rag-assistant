"""Real ASGI + graph + hybrid-store continuity; only model outputs are controlled."""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from app.main import create_application
from app.schemas.document import DocumentChunk, DocumentItem, ScoredChunk
from app.services.evaluation.grounding_runner import (
    FakeGeneratorClient,
    FakeGraderClient,
    FakeReformulatorClient,
    create_synthetic_chunk,
)
from app.services.generation.grounding import (
    FALLBACK_RESPONSE,
    GroundedGenerator,
    RetrievalGrader,
    set_grounded_generator,
    set_retrieval_grader,
)
from app.services.ingestion.registry import get_document_registry
from app.services.retrieval.bm25_store import get_bm25_store
from app.services.retrieval.conversation_registry import get_conversation_registry
from app.services.retrieval.hybrid import HybridRetriever, set_hybrid_retriever
from app.services.retrieval.memory import get_memory_manager
from app.services.retrieval.reformulator import QueryReformulator, set_query_reformulator
from app.services.retrieval.vector_store import get_vector_store


@dataclass
class Harness:
    chunks: list[DocumentChunk]
    reformulator: FakeReformulatorClient
    grader: FakeGraderClient
    generator: FakeGeneratorClient
    retrievals: list[tuple[str, list[str] | None, list[ScoredChunk]]]


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    chunks = [
        create_synthetic_chunk(
            "a1", "A", "contract_a.pdf", "Model A exact match is 42.5% on Dataset A.", page_number=2
        ),
        create_synthetic_chunk(
            "b1", "B", "contract_b.pdf", "Model B Macro F1 is 0.52 on Dataset B.", page_number=3
        ),
        create_synthetic_chunk(
            "x1", "X", "distractor.pdf", "Model A exact match is 99% on Dataset X.", page_number=4
        ),
    ]
    vector = get_vector_store()
    sparse = get_bm25_store()
    vector.add_chunks(chunks)
    sparse.add_chunks(chunks)
    registry = get_document_registry()
    for chunk in chunks:
        registry.add_or_update(
            DocumentItem(
                id=chunk.metadata.document_id,
                filename=chunk.metadata.filename,
                page_count=4,
                chunk_count=1,
            )
        )
    retriever = HybridRetriever(vector_store=vector, bm25_store=sparse)
    retrievals: list[tuple[str, list[str] | None, list[ScoredChunk]]] = []
    retrieve = retriever.retrieve

    def observe(
        query: str, document_ids: list[str] | None = None, top_k: int = 5
    ) -> list[ScoredChunk]:
        result = retrieve(query, document_ids=document_ids, top_k=top_k)
        retrievals.append((query, document_ids, result))
        return result

    monkeypatch.setattr(retriever, "retrieve", observe)
    set_hybrid_retriever(retriever)
    reformulator = FakeReformulatorClient("Model A Model B exact match Macro F1 datasets")
    grader = FakeGraderClient("grounded")

    def answer(_prompt: str, context: Sequence[Any]) -> str:
        return " ".join(
            f"{chunk.metadata.filename}: {chunk.content} [{index}]"
            for index, chunk in enumerate(context, 1)
        )

    generator = FakeGeneratorClient(answer)
    set_query_reformulator(QueryReformulator(llm_client=reformulator))
    set_retrieval_grader(RetrievalGrader(grading_fn=grader))
    set_grounded_generator(GroundedGenerator(generating_fn=generator))
    return Harness(chunks, reformulator, grader, generator, retrievals)


async def _send(
    client: httpx.AsyncClient, conversation: str, payload: dict[str, Any]
) -> list[tuple[str, Any]]:
    async with client.stream(
        "POST", f"/api/v1/conversations/{conversation}/messages", json=payload
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        raw = "".join([part async for part in response.aiter_text()])
    events: list[tuple[str, Any]] = []
    for block in raw.strip().split("\n\n"):
        name, data = block.splitlines()
        assert name.startswith("event: ")
        assert data.startswith("data: ")
        events.append((name[7:], json.loads(data[6:])))
    return events


async def _assert_saved(
    client: httpx.AsyncClient, conversation: str, events: list[tuple[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    statuses = [data["status"] for name, data in events if name == "status"]
    assert statuses in (
        ["searching", "streaming"],
        ["reformulating", "searching", "streaming"],
    )
    assert events[-2][0] == "citations"
    assert events[-1] == ("done", {"finishReason": "stop"})
    for name, data in events:
        if name == "token":
            assert set(data) == {"token", "content"}
            assert data["token"] == data["content"]
        if name == "status":
            assert data["status"] == data["state"]
    answer = "".join(data["token"] for name, data in events if name == "token")
    citations: list[dict[str, Any]] = events[-2][1]
    assert set(map(int, re.findall(r"\[(\d+)\]", answer))) == {
        citation["id"] for citation in citations
    }
    response = await client.get(f"/api/v1/conversations/{conversation}/messages")
    assert response.status_code == 200
    assert response.json()[-1]["role"] == "assistant"
    assert response.json()[-1]["content"] == answer
    assert response.json()[-1]["citations"] == citations
    saved = get_conversation_registry().get(conversation)
    assert saved is not None
    assert saved.messages[-1].content == answer
    assert saved.messages[-1].citations == citations
    return answer, citations


@pytest.mark.asyncio
async def test_scope_aliases_history_and_full_stream_continuity(harness: Harness) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_application()), base_url="http://testserver"
    ) as client:
        created = await client.post("/api/v1/conversations", json={"documentIds": ["A"]})
        assert created.status_code == 201
        conversation = created.json()["conversation_id"]
        steps: list[tuple[dict[str, Any], set[str]]] = [
            ({"content": "What is Model A's exact match?", "document_ids": ["A"]}, {"A"}),
            ({"message": "What dataset did it use?"}, {"A"}),
            ({"content": "List all dataset metrics", "documentIds": []}, {"A", "B", "X"}),
            ({"message": "What is Model B's Macro F1?", "document_ids": ["B"]}, {"B"}),
        ]
        first_answer = ""
        for index, (payload, expected_docs) in enumerate(steps):
            events = await _send(client, conversation, payload)
            answer, citations = await _assert_saved(client, conversation, events)
            actual_scope = ["A"] if index < 2 else [] if index == 2 else ["B"]
            assert (
                get_memory_manager().get_or_create_session(conversation).active_document_ids
                == actual_scope
            )
            saved = get_conversation_registry().get(conversation)
            assert saved is not None and saved.document_ids == actual_scope
            query, scope, retrieved = harness.retrievals[-1]
            assert scope == actual_scope
            assert {chunk.metadata.document_id for chunk in retrieved} == expected_docs
            assert {citation["documentId"] for citation in citations} == expected_docs
            by_id = {chunk.chunk_id: chunk for chunk in retrieved}
            for citation in citations:
                chunk = by_id[citation["chunk_id"]]
                assert citation["filename"] == chunk.metadata.filename
                assert citation["pageNumber"] == chunk.metadata.page_number
                assert chunk.content.startswith(citation["quote"])
            if index == 0:
                first_answer = answer
                assert "42.5%" in answer and "99%" not in answer
            if index == 1:
                prompt = harness.reformulator.recorded_calls[-1]["prompt"]
                assert first_answer in prompt
                assert "What dataset did it use?" in prompt
                assert "Active Documents:\ncontract_a.pdf" in prompt
                assert query == "Model A Model B exact match Macro F1 datasets"
        assert harness.generator.call_count == 4
        assert harness.grader.call_count == 0


@pytest.mark.asyncio
async def test_comparison_keeps_distinct_metrics_and_excludes_distractor(harness: Harness) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_application()), base_url="http://testserver"
    ) as client:
        events = await _send(
            client,
            "comparison",
            {
                "message": "Compare Model A and Model B with their datasets and metrics",
                "documentIds": ["A", "B"],
            },
        )
        answer, citations = await _assert_saved(client, "comparison", events)
        assert "contract_a.pdf: Model A exact match is 42.5% on Dataset A." in answer
        assert "contract_b.pdf: Model B Macro F1 is 0.52 on Dataset B." in answer
        assert "99%" not in answer and "distractor.pdf" not in answer
        assert {citation["documentId"] for citation in citations} == {"A", "B"}
        assert {chunk.metadata.document_id for chunk in harness.retrievals[-1][2]} == {"A", "B"}


@pytest.mark.asyncio
async def test_insufficient_route_saves_exact_fallback_without_citations(harness: Harness) -> None:
    harness.generator._response_fn = FALLBACK_RESPONSE
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_application()), base_url="http://testserver"
    ) as client:
        events = await _send(
            client,
            "insufficient",
            {
                "content": "What is the confidence interval?",
                "document_ids": ["A"],
            },
        )
        answer, citations = await _assert_saved(client, "insufficient", events)
        assert answer == FALLBACK_RESPONSE
        assert citations == []
        assert harness.generator.call_count == 1


@pytest.mark.asyncio
async def test_execution_error_has_error_completion_and_no_saved_abstention(
    harness: Harness,
) -> None:
    def fail(_prompt: str, _context: Sequence[Any]) -> str:
        raise RuntimeError("controlled offline generation failure")

    set_grounded_generator(GroundedGenerator(generating_fn=fail))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_application()), base_url="http://testserver"
    ) as client:
        events = await _send(client, "failure", {"content": "Model A", "documentIds": ["A"]})
        assert events[-1] == ("done", {"finishReason": "error"})
        assert events[-2][0] == "error"
        assert events[-2][1]["error"]["code"] == "STREAM_ERROR"
        assert not any(name == "citations" for name, _ in events)
        assert all(
            message.role == "user" for message in get_memory_manager().get_messages("failure")
        )
        assert harness.generator.call_count == 0
