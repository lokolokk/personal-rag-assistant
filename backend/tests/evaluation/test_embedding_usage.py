"""Measure existing Chroma adapter calls without replacing it or re-embedding."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx as asgi_httpx
import httpx2 as httpx
import pytest
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

from app.services.evaluation.usage import estimate_tokens, measure_usage
from app.services.retrieval.bm25_store import BM25Store
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.vector_store import DeterministicEmbeddingFunction, VectorStore
from tests.test_vector_store import create_sample_chunk


def test_ingestion_batches_and_hybrid_query_have_separate_estimated_usage(tmp_path: Path) -> None:
    store = VectorStore(
        persist_dir=tmp_path / "chroma",
        embedding_function=(DeterministicEmbeddingFunction(dimension=64)),
    )
    chunks = [create_sample_chunk(f"c{i}", "doc_a", f"Learning rate {i}") for i in range(35)]
    sparse = BM25Store()
    with measure_usage() as trace:
        store.add_chunks(chunks)
        sparse.add_chunks(chunks)
        HybridRetriever(vector_store=store, bm25_store=sparse).retrieve(
            "Learning rate", document_ids=["doc_a"]
        )
    assert [r.stage for r in trace.records] == [
        "ingestion_embedding",
        "ingestion_embedding",
        "ingestion_embedding",
        "query_embedding",
    ]
    assert [r.input_items for r in trace.records] == [16, 16, 3, 1]
    counts = [estimate_tokens(c.content) for c in chunks]
    assert all(count is not None for count in counts)
    assert sum(r.input_tokens or 0 for r in trace.records[:3]) == sum(c or 0 for c in counts)
    assert trace.records[-1].input_tokens == estimate_tokens("Learning rate")
    assert all(r.source == "local_estimate" and r.local_only for r in trace.records)


def test_precomputed_embeddings_and_empty_operations_do_not_record_calls(tmp_path: Path) -> None:
    store = VectorStore(persist_dir=tmp_path / "chroma")
    with measure_usage() as trace:
        store.add_chunks([])
        assert store.query("Question") == []
        store.add_chunks([create_sample_chunk("c1", "doc_a", "Content")], embeddings=[[1.0, 0.0]])
        store.query("Question", query_embedding=[1.0, 0.0])
    assert trace.records == []


def test_adapter_http_usage_and_retries_are_captured_without_extra_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(503, json={"error": {"message": "Retry"}})
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "fixture-embedding",
                "data": [
                    {"object": "embedding", "index": i, "embedding": [1.0, 0.0, 0.0, 0.0]}
                    for i in range(len(payload["input"]))
                ],
                "usage": {"prompt_tokens": 37, "total_tokens": 37},
            },
        )

    adapter = OpenAIEmbeddingFunction(api_key="offline-transport", model_name="fixture-embedding")
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(api_key="offline-transport", http_client=transport, max_retries=1)
        monkeypatch.setattr(client, "_sleep_for_retry", lambda **_kwargs: None)
        adapter.client = client
        store = VectorStore(persist_dir=tmp_path / "chroma", embedding_function=adapter)
        with measure_usage() as trace:
            store.add_chunks([create_sample_chunk("c1", "doc_a", "Qualified result 37%")])
            store.query("Result 37%", document_ids=["doc_a"])
    assert len(requests) == 3  # One ingestion retry and one query; no accounting-only request.
    assert [r.stage for r in trace.records] == [
        "ingestion_embedding",
        "ingestion_embedding",
        "query_embedding",
    ]
    assert trace.records[0].outcome == "failed"
    assert trace.records[0].total_tokens is None
    assert [r.total_tokens for r in trace.records[1:]] == [37, 37]
    assert all(r.output_tokens == 0 for r in trace.records[1:])
    assert all(r.source == "provider_reported" for r in trace.records[1:])
    assert all(r.retry_visibility == "observed" for r in trace.records)
    assert store._embedding_function is adapter


def test_embedding_failure_is_unknown_consumption_and_does_not_advance_batches() -> None:
    collection = MagicMock()
    collection.upsert.side_effect = RuntimeError("Embedding failed")
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    store = VectorStore(client=client)
    with measure_usage() as trace, pytest.raises(RuntimeError, match="Embedding failed"):
        store.add_chunks([create_sample_chunk(f"c{i}", "doc_a", "Content") for i in range(35)])
    assert len(trace.records) == 1
    assert trace.records[0].stage == "ingestion_embedding"
    assert trace.records[0].source == "unavailable"
    assert trace.records[0].outcome == "failed"
    assert trace.records[0].total_tokens is None
    assert collection.upsert.call_count == 1


@pytest.mark.asyncio
async def test_sse_measurement_captures_all_chat_and_query_stages_and_persists_citations() -> None:
    from types import SimpleNamespace

    from app.main import create_application
    from app.services.generation.grounding import (
        GroundedGenerator,
        RetrievalGrader,
        set_grounded_generator,
        set_retrieval_grader,
    )
    from app.services.retrieval.conversation_registry import get_conversation_registry
    from app.services.retrieval.reformulator import QueryReformulator, set_query_reformulator
    from app.services.retrieval.vector_store import get_vector_store
    from tests.evaluation.test_usage import completion

    store = get_vector_store()
    chunk = create_sample_chunk("evidence", "doc_a", "Learning rate was 0.001.")
    store.add_chunks([chunk])
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        iter(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="The learning rate was 0.001 [1].")
                        )
                    ]
                ),
                SimpleNamespace(choices=[], usage=completion("", 30, 8).usage),
            ]
        ),
    ]
    set_query_reformulator(QueryReformulator(llm_client=client))
    set_retrieval_grader(RetrievalGrader(llm_client=client))
    set_grounded_generator(GroundedGenerator(llm_client=client))
    with measure_usage() as trace:
        async with asgi_httpx.AsyncClient(
            transport=asgi_httpx.ASGITransport(app=create_application()),
            base_url="http://testserver",
        ) as api:
            response = await api.post(
                "/api/v1/conversations/measured/messages",
                json={"content": "Learning rate?", "documentIds": ["doc_a"]},
            )
    assert "event: error" not in response.text
    assert "event: citations" in response.text
    assert [r.stage for r in trace.records] == [
        "query_embedding",
        "generation",
    ]
    assert client.chat.completions.create.call_count == 1
    conversation = get_conversation_registry().get("measured")
    assert conversation is not None
    assert conversation.messages[-1].content == "The learning rate was 0.001 [1]."
    citation = conversation.messages[-1].citations[0]
    assert isinstance(citation, dict)
    assert citation["chunkId"] == "evidence"
    assert citation["documentId"] == "doc_a"
