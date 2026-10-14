"""Usage follows the actual services, including retries and streaming fallbacks."""

import asyncio
import hashlib
import json
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx2 as httpx
import pytest
from openai import OpenAI

from app.services.evaluation.grounding_report import LiveRunPlan
from app.services.evaluation.report_store import LiveReportSummary, ReportStore, recover_report
from app.services.evaluation.usage import (
    DEFAULT_OUTPUT_CAPS,
    LiveBudgetExceeded,
    Stage,
    UsageRecord,
    durable_usage,
    live_token_budget,
    measure_usage,
    observed_completion,
    observed_embeddings,
)
from app.services.generation.graph import run_conversational_rag_graph
from app.services.generation.grounding import (
    GroundedGenerator,
    RetrievalGrader,
    set_grounded_generator,
    set_retrieval_grader,
)
from app.services.retrieval.hybrid import HybridRetriever, set_hybrid_retriever
from app.services.retrieval.reformulator import QueryReformulator, set_query_reformulator
from tests.test_grounding import _create_sample_chunk


def _capture_store(
    tmp_path: Path,
    *,
    maximum_calls: int = 5,
    allowance_tokens: int = 100_000,
) -> ReportStore:
    return ReportStore.create(
        project_root=tmp_path,
        output_dir=tmp_path / "capture",
        suite_id="slice-k",
        run_id="capture-1",
        planned_case_keys=["case-a"],
        live_plan=LiveRunPlan(
            plan_id="task-6-1-five-case",
            dataset_sha256="a" * 64,
            corpus_sha256={"document-a": "b" * 64},
            prompt_sha256={"reformulation": "c" * 64, "grading": "d" * 64, "generation": "e" * 64},
            configuration_sha256="f" * 64,
            destination_host="example.invalid",
            chat_model="test-model",
            embedding_model="test-embedding",
            allowance_tokens=allowance_tokens,
            maximum_calls=maximum_calls,
            output_caps=DEFAULT_OUTPUT_CAPS,
            usage_evidence="synthetic",
        ),
    )


def _capture_response(
    text: str = "Standalone query",
    *,
    usage: dict[str, int] | None = None,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "offline",
            "model": "test-model",
            "object": "chat.completion",
            "created": 0,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "usage": usage
            if usage is not None
            else {
                "prompt_tokens": 30,
                "completion_tokens": 4,
                "total_tokens": 34,
            },
        },
    )


def _bounds() -> dict[Stage, int]:
    return dict.fromkeys(DEFAULT_OUTPUT_CAPS, 10_000)


def test_durable_capture_is_committed_before_fake_transport(tmp_path: Path) -> None:
    store = _capture_store(tmp_path)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        recovered = recover_report(store.run_dir, project_root=tmp_path)
        assert recovered.calls[0].state == "potentially_dispatched"
        assert isinstance(recovered.summary, LiveReportSummary)
        assert recovered.summary.calls.reserved_tokens > 0
        requests.append(request)
        return _capture_response()

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(
            api_key="offline-transport",
            base_url="https://example.invalid/v1",
            http_client=transport,
            max_retries=0,
        )
        with measure_usage(), durable_usage(store, input_token_bounds=_bounds()) as capture:
            with capture.case("case-a"):
                QueryReformulator(llm_client=client, model="test-model").reformulate("Question")
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    assert len(requests) == 1
    assert json.loads(requests[0].content)["max_completion_tokens"] == 246
    assert recovered.calls[0].state == "succeeded"
    assert (
        recovered.calls[0].reservation.payload_sha256
        == hashlib.sha256(requests[0].content).hexdigest()
    )
    assert recovered.calls[0].outcome is not None
    assert recovered.calls[0].outcome.source == "synthetic_usage"
    assert isinstance(recovered.summary, LiveReportSummary)
    assert recovered.summary.current.known_provider_tokens == 0
    assert recovered.summary.calls.synthetic_tokens == 34


def test_durable_capture_independent_call_cap_blocks_second_request(tmp_path: Path) -> None:
    store = _capture_store(tmp_path, maximum_calls=1)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _capture_response()

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(
            api_key="offline-transport",
            base_url="https://example.invalid/v1",
            http_client=transport,
            max_retries=0,
        )
        with pytest.raises(LiveBudgetExceeded, match="call ceiling"):
            with measure_usage(), durable_usage(store, input_token_bounds=_bounds()) as capture:
                with capture.case("case-a"):
                    reformulator = QueryReformulator(llm_client=client, model="test-model")
                    reformulator.reformulate("First")
                    reformulator.reformulate("Second")
    assert len(requests) == 1
    assert len(recover_report(store.run_dir, project_root=tmp_path).calls) == 1


@pytest.mark.parametrize(
    "usage",
    [
        {},
        {"prompt_tokens": 30},
        {"prompt_tokens": 30, "completion_tokens": 4},
        {"prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 100_001},
    ],
)
def test_durable_unknown_or_excess_usage_stops_next_dispatch(
    tmp_path: Path,
    usage: dict[str, int],
) -> None:
    store = _capture_store(tmp_path)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _capture_response(usage=usage)

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(
            api_key="offline-transport",
            base_url="https://example.invalid/v1",
            http_client=transport,
            max_retries=0,
        )
        with pytest.raises(LiveBudgetExceeded):
            with measure_usage(), durable_usage(store, input_token_bounds=_bounds()) as capture:
                with capture.case("case-a"):
                    reformulator = QueryReformulator(llm_client=client, model="test-model")
                    reformulator.reformulate("First")
                    reformulator.reformulate("Second")
    assert len(requests) == 1
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    assert len(recovered.calls) == 1
    assert recovered.calls[0].outcome is not None
    assert isinstance(recovered.summary, LiveReportSummary)
    assert recovered.summary.calls.exact_provider_tokens is None


def test_durable_swallowed_grader_error_blocks_generation(tmp_path: Path) -> None:
    store = _capture_store(tmp_path)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503, json={"error": {"message": "private failure"}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(
            api_key="offline-transport",
            base_url="https://example.invalid/v1",
            http_client=transport,
            max_retries=0,
        )
        with pytest.raises(LiveBudgetExceeded):
            with measure_usage(), durable_usage(store, input_token_bounds=_bounds()) as capture:
                with capture.case("case-a"):
                    chunks = [_create_sample_chunk()]
                    assert (
                        RetrievalGrader(llm_client=client, model="test-model").grade_retrieval(
                            "Attention",
                            chunks,
                        )
                        == "grounded"
                    )
                    GroundedGenerator(llm_client=client, model="test-model").generate(
                        "Attention",
                        chunks,
                    )
    assert len(requests) == 1
    assert recover_report(store.run_dir, project_root=tmp_path).calls[0].state == "failed"


@pytest.mark.parametrize("fault", ["reservation", "intent", "outcome"])
def test_durable_write_fault_blocks_transport_or_later_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    store = _capture_store(tmp_path)
    original = store.append_call
    requests: list[httpx.Request] = []

    def fail_append(event: Any) -> None:
        if (
            event.kind
            == {"reservation": "reservation", "intent": "dispatch_intent", "outcome": "outcome"}[
                fault
            ]
        ):
            raise ValueError("private write detail")
        original(event)

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _capture_response()

    monkeypatch.setattr(store, "append_call", fail_append)
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(
            api_key="offline-transport",
            base_url="https://example.invalid/v1",
            http_client=transport,
            max_retries=0,
        )
        with pytest.raises(LiveBudgetExceeded) as error:
            with measure_usage(), durable_usage(store, input_token_bounds=_bounds()) as capture:
                with capture.case("case-a"):
                    QueryReformulator(llm_client=client, model="test-model").reformulate("First")
        assert "private" not in str(error.value)
    assert len(requests) == (1 if fault == "outcome" else 0)
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    if fault == "outcome":
        assert recovered.calls[0].state == "potentially_dispatched"


@pytest.mark.parametrize("misconfiguration", ["retries", "host", "model", "bounds", "stream"])
def test_durable_invalid_dispatch_rejects_without_transport(
    tmp_path: Path,
    misconfiguration: str,
) -> None:
    store = _capture_store(tmp_path)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _capture_response()

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(
            api_key="offline-transport",
            http_client=transport,
            base_url="https://wrong.invalid/v1"
            if misconfiguration == "host"
            else "https://example.invalid/v1",
            max_retries=1 if misconfiguration == "retries" else 0,
        )
        bounds = _bounds()
        if misconfiguration == "bounds":
            bounds["reformulation"] = 0
        with pytest.raises(LiveBudgetExceeded):
            with measure_usage(), durable_usage(store, input_token_bounds=bounds) as capture:
                with capture.case("case-a"):
                    observed_completion(
                        client,
                        "reformulation",
                        model="wrong" if misconfiguration == "model" else "test-model",
                        messages=[{"role": "user", "content": "Question"}],
                        stream=misconfiguration == "stream",
                    )
    assert requests == []


def test_durable_embedding_usage_is_saved_with_shared_and_case_identity(tmp_path: Path) -> None:
    store = _capture_store(tmp_path)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "test-embedding",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(
            api_key="offline-transport",
            base_url="https://example.invalid/v1",
            http_client=transport,
            max_retries=0,
        )
        with measure_usage(), durable_usage(store, input_token_bounds=_bounds()) as capture:
            observed_embeddings(
                "ingestion_embedding",
                "test-embedding",
                ["Chunk text"],
                lambda: client.embeddings.create(model="test-embedding", input=["Chunk text"]),
                client,
            )
            with capture.case("case-a"):
                observed_embeddings(
                    "query_embedding",
                    "test-embedding",
                    ["Query"],
                    lambda: client.embeddings.create(model="test-embedding", input=["Query"]),
                    client,
                )
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    assert len(requests) == 2
    assert [call.reservation.case_key for call in recovered.calls] == [None, "case-a"]
    assert isinstance(recovered.summary, LiveReportSummary)
    assert recovered.summary.calls.synthetic_tokens == 4


@pytest.mark.parametrize("stage", ["reformulation", "grading", "generation"])
def test_durable_chat_output_caps_are_enforced(tmp_path: Path, stage: Stage) -> None:
    store = _capture_store(tmp_path)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _capture_response()

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(
            api_key="offline-transport",
            base_url="https://example.invalid/v1",
            http_client=transport,
            max_retries=0,
        )
        with measure_usage(), durable_usage(store, input_token_bounds=_bounds()) as capture:
            with capture.case("case-a"):
                observed_completion(
                    client,
                    stage,
                    model="test-model",
                    max_tokens=9000,
                    messages=[{"role": "user", "content": "Question"}],
                )
    assert "max_tokens" not in json.loads(requests[0].content)
    assert (
        json.loads(requests[0].content)["max_completion_tokens"] == DEFAULT_OUTPUT_CAPS[stage] - 10
    )


def test_durable_consistent_excess_usage_is_retained_but_stops_run(tmp_path: Path) -> None:
    store = _capture_store(tmp_path)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _capture_response(
            usage={
                "prompt_tokens": 5996,
                "completion_tokens": 4,
                "total_tokens": 6000,
            }
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(
            api_key="offline-transport",
            base_url="https://example.invalid/v1",
            http_client=transport,
            max_retries=0,
        )
        with pytest.raises(LiveBudgetExceeded, match="exceeded"):
            with measure_usage(), durable_usage(store, input_token_bounds=_bounds()) as capture:
                with capture.case("case-a"):
                    QueryReformulator(llm_client=client, model="test-model").reformulate("Question")
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    assert len(requests) == 1
    assert recovered.calls[0].outcome is not None
    assert recovered.calls[0].outcome.total_tokens == 6000


def test_durable_second_embedding_send_is_blocked(tmp_path: Path) -> None:
    store = _capture_store(tmp_path)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "test-embedding",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(
            api_key="offline-transport",
            base_url="https://example.invalid/v1",
            http_client=transport,
            max_retries=0,
        )

        def two_sends() -> None:
            client.embeddings.create(model="test-embedding", input=["Chunk"])
            client.embeddings.create(model="test-embedding", input=["Chunk"])

        with pytest.raises(LiveBudgetExceeded, match="second request"):
            with measure_usage(), durable_usage(store, input_token_bounds=_bounds()):
                observed_embeddings(
                    "ingestion_embedding", "test-embedding", ["Chunk"], two_sends, client
                )
    assert len(requests) == 1


def test_durable_sdk_prepare_failure_is_not_dispatched(tmp_path: Path) -> None:
    store = _capture_store(tmp_path)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _capture_response()

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(
            api_key="offline-transport",
            base_url="https://example.invalid/v1",
            http_client=transport,
            max_retries=0,
        )
        with pytest.raises(TypeError):
            with measure_usage(), durable_usage(store, input_token_bounds=_bounds()) as capture:
                with capture.case("case-a"):
                    observed_completion(
                        client,
                        "generation",
                        model="test-model",
                        messages=[{"role": "user", "content": "Question"}],
                        unsupported_argument="bad",
                    )
    assert requests == []
    assert recover_report(store.run_dir, project_root=tmp_path).calls[0].state == "not_dispatched"


def test_durable_reusing_existing_capture_is_prohibited(tmp_path: Path) -> None:
    store = _capture_store(tmp_path)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _capture_response()

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(
            api_key="offline-transport",
            base_url="https://example.invalid/v1",
            http_client=transport,
            max_retries=0,
        )
        with measure_usage(), durable_usage(store, input_token_bounds=_bounds()) as capture:
            with capture.case("case-a"):
                QueryReformulator(llm_client=client, model="test-model").reformulate("Question")
        with pytest.raises(LiveBudgetExceeded, match="resume"):
            with measure_usage(), durable_usage(store, input_token_bounds=_bounds()):
                raise AssertionError("Existing capture was resumed")
    assert len(requests) == 1


def test_durable_requires_active_trace_and_observable_sdk(tmp_path: Path) -> None:
    store = _capture_store(tmp_path)
    with pytest.raises(LiveBudgetExceeded, match="measure_usage"):
        with durable_usage(store, input_token_bounds=_bounds()):
            raise AssertionError("No trace was accepted")
    with pytest.raises(LiveBudgetExceeded, match="observable"):
        with measure_usage(), durable_usage(store, input_token_bounds=_bounds()) as capture:
            with capture.case("case-a"):
                QueryReformulator(llm_client=lambda _prompt: "Query").reformulate("Question")


def test_durable_budget_and_stage_bounds_reject_before_send(tmp_path: Path) -> None:
    store = _capture_store(tmp_path, allowance_tokens=1)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _capture_response()

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(
            api_key="offline-transport",
            base_url="https://example.invalid/v1",
            http_client=transport,
            max_retries=0,
        )
        with pytest.raises(LiveBudgetExceeded, match="remaining allowance"):
            with measure_usage(), durable_usage(store, input_token_bounds=_bounds()) as capture:
                with capture.case("case-a"):
                    observed_completion(
                        client,
                        "generation",
                        model="test-model",
                        messages=[{"role": "user", "content": "Question"}],
                    )
    assert requests == []


def test_durable_nested_observations_are_rejected(tmp_path: Path) -> None:
    store = _capture_store(tmp_path)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "test-embedding",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(
            api_key="offline-transport",
            base_url="https://example.invalid/v1",
            http_client=transport,
            max_retries=0,
        )

        def nested_send() -> None:
            client.embeddings.create(model="test-embedding", input=["Chunk"])
            observed_completion(
                client,
                "generation",
                model="test-model",
                messages=[{"role": "user", "content": "Question"}],
            )

        with pytest.raises(LiveBudgetExceeded, match="concurrent"):
            with measure_usage(), durable_usage(store, input_token_bounds=_bounds()):
                observed_embeddings(
                    "ingestion_embedding", "test-embedding", ["Chunk"], nested_send, client
                )
    assert len(requests) == 1


def test_durable_stop_is_restored_between_independent_traces(tmp_path: Path) -> None:
    store = _capture_store(tmp_path)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _capture_response(usage={} if len(requests) == 1 else None)

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(
            api_key="offline-transport",
            base_url="https://example.invalid/v1",
            http_client=transport,
            max_retries=0,
        )
        reformulator = QueryReformulator(llm_client=client, model="test-model")
        with pytest.raises(LiveBudgetExceeded, match="usage unavailable"):
            with measure_usage(), durable_usage(store, input_token_bounds=_bounds()) as capture:
                with capture.case("case-a"):
                    reformulator.reformulate("First")
        fresh_store = _capture_store(tmp_path / "fresh")
        with measure_usage(), durable_usage(fresh_store, input_token_bounds=_bounds()) as capture:
            with capture.case("case-a"):
                reformulator.reformulate("Second")
    assert len(requests) == 2
    assert (
        recover_report(fresh_store.run_dir, project_root=tmp_path / "fresh").calls[0].state
        == "succeeded"
    )


def completion(text: str, input_tokens: int, output_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


def test_actual_graph_accounts_for_each_chat_stage_without_extra_calls() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        completion("What was the learning rate?", 120, 8),
        completion("grounded", 250, 1),
        iter(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="The learning rate was 0.001 [1].")
                        )
                    ]
                ),
                SimpleNamespace(choices=[], usage=completion("", 300, 12).usage),
            ]
        ),
    ]
    set_query_reformulator(QueryReformulator(llm_client=client))
    set_retrieval_grader(RetrievalGrader(llm_client=client))
    set_grounded_generator(GroundedGenerator(llm_client=client))
    retriever = MagicMock(spec=HybridRetriever)
    retriever.retrieve.return_value = [_create_sample_chunk()]
    set_hybrid_retriever(retriever)
    with measure_usage() as trace:
        result = run_conversational_rag_graph(
            {"user_query": "What was the learning rate?"},
            streamlined=False,
        )
    assert result["citations"][0]["chunkId"] == "chunk_1"
    assert [r.stage for r in trace.records] == ["reformulation", "grading", "generation"]
    assert [r.total_tokens for r in trace.records] == [128, 251, 312]
    assert all(r.source == "provider_reported" for r in trace.records)
    assert client.chat.completions.create.call_count == 3


def test_streamlined_graph_accounts_for_only_generation_on_turn_one() -> None:
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
                SimpleNamespace(choices=[], usage=completion("", 300, 12).usage),
            ]
        ),
    ]
    set_query_reformulator(QueryReformulator(llm_client=client))
    set_retrieval_grader(RetrievalGrader(llm_client=client))
    set_grounded_generator(GroundedGenerator(llm_client=client))
    retriever = MagicMock(spec=HybridRetriever)
    retriever.retrieve.return_value = [_create_sample_chunk()]
    set_hybrid_retriever(retriever)
    with measure_usage() as trace:
        result = run_conversational_rag_graph(
            {"user_query": "What was the learning rate?"},
            streamlined=True,
        )
    assert result["citations"][0]["chunkId"] == "chunk_1"
    assert [r.stage for r in trace.records] == ["generation"]
    assert [r.total_tokens for r in trace.records] == [312]
    assert all(r.source == "provider_reported" for r in trace.records)
    assert client.chat.completions.create.call_count == 1


def test_callable_counts_are_estimates_and_failure_is_not_zero() -> None:
    with measure_usage() as trace:
        QueryReformulator(llm_client=lambda _p: "Learning rate?").reformulate("What is it?")
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("Provider failed")
        generator = GroundedGenerator(llm_client=client)
        generator.generate("Learning rate?", [_create_sample_chunk()])
    estimate, failed = trace.records
    assert estimate.source == "local_estimate"
    assert estimate.total_tokens is not None and estimate.total_tokens > 0
    assert failed.source == "unavailable"
    assert failed.total_tokens is None
    assert failed.estimated_input_tokens is not None
    assert failed.outcome == "failed"


def test_stream_usage_only_chunk_and_failed_stream_fallback_are_retained() -> None:
    client = MagicMock()

    def partial_stream() -> Any:
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Partial "))], usage=None
        )
        raise RuntimeError("Stream interrupted")

    client.chat.completions.create.side_effect = [
        partial_stream(),
        completion("The answer [1].", 80, 5),
    ]
    events: list[Any] = []
    with measure_usage() as trace:
        text, citations = GroundedGenerator(llm_client=client).generate(
            "Question", [_create_sample_chunk()], writer=events.append
        )
    assert text == "The answer [1]."
    assert citations[0]["id"] == 1
    assert [r.outcome for r in trace.records] == ["failed", "succeeded"]
    assert trace.records[0].estimated_output_tokens is not None
    assert trace.records[1].total_tokens == 85
    assert client.chat.completions.create.call_count == 2

    client.chat.completions.create.side_effect = None
    client.chat.completions.create.return_value = iter(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Answer [1]."))]
            ),
            SimpleNamespace(choices=[], usage=completion("", 50, 4).usage),
        ]
    )
    with measure_usage() as streaming_trace:
        GroundedGenerator(llm_client=client).generate(
            "Question", [_create_sample_chunk()], writer=events.append
        )
    assert streaming_trace.records[0].total_tokens == 54
    assert streaming_trace.records[0].source == "provider_reported"


def test_sdk_retries_are_observed_with_unknown_consumption_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, json={"error": {"message": "Retry"}})
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "model": "test-model",
                "object": "chat.completion",
                "created": 0,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Query"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 2, "total_tokens": 32},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(api_key="offline-transport", http_client=transport, max_retries=1)
        monkeypatch.setattr(client, "_sleep_for_retry", lambda **_kwargs: None)
        with measure_usage() as trace:
            QueryReformulator(llm_client=client).reformulate("Question")
    assert len(requests) == 2
    assert len(trace.records) == 2
    assert trace.records[0].source == "unavailable"
    assert trace.records[0].outcome == "failed"
    assert trace.records[1].total_tokens == 32
    assert all(r.retry_visibility == "observed" for r in trace.records)


@pytest.mark.asyncio
async def test_concurrent_traces_and_nested_traces_do_not_mix() -> None:
    async def run(text: str) -> list[str]:
        with measure_usage() as trace:
            await asyncio.sleep(0)
            await asyncio.to_thread(QueryReformulator(llm_client=lambda _p: text).reformulate, text)
            return [r.stage for r in trace.records]

    results = await asyncio.gather(run("A"), run("B"))
    assert list(results) == [["reformulation"], ["reformulation"]]
    with measure_usage() as outer:
        with measure_usage() as inner:
            QueryReformulator(llm_client=lambda _p: "Inner").reformulate("Question")
        assert len(inner.records) == 1
        assert outer.records == []
    QueryReformulator(llm_client=lambda _p: "Untraced").reformulate("Question")
    assert outer.records == []


def test_partial_provider_usage_and_no_usage_remain_distinct() -> None:
    client = MagicMock()
    response = completion("Query", 7, 2)
    response.usage = {"prompt_tokens": 7}
    client.chat.completions.create.return_value = response
    with measure_usage() as partial:
        QueryReformulator(llm_client=client).reformulate("Question")
    assert partial.records[0].source == "provider_reported"
    assert partial.records[0].input_tokens == 7
    assert partial.records[0].output_tokens is None
    assert partial.records[0].total_tokens is None
    response.usage = None
    with measure_usage() as missing:
        QueryReformulator(llm_client=client).reformulate("Question")
    assert missing.records[0].source == "local_estimate"
    assert missing.records[0].estimate_method == "cl100k_base:text_only"
    assert client.chat.completions.create.call_count == 2


def test_live_budget_reserves_before_dispatch_and_caps_generation() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "budget-test",
                "model": "test-model",
                "object": "chat.completion",
                "created": 0,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Standalone query"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34},
            },
        )

    budget_limit = 10_000
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(api_key="offline-transport", http_client=transport, max_retries=2)
        with measure_usage() as trace, live_token_budget(budget_limit) as budget:
            QueryReformulator(llm_client=client).reformulate("Question")
    assert len(requests) == 1
    assert json.loads(requests[0].content)["max_tokens"] == 256
    assert budget.reserved_tokens > 0
    assert budget.reserved_tokens <= budget_limit
    assert trace.records[0].total_tokens == 34


def test_live_budget_rejects_before_provider_dispatch() -> None:
    requests: list[httpx.Request] = []

    def unexpected_dispatch(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    with httpx.Client(transport=httpx.MockTransport(unexpected_dispatch)) as transport:
        client = OpenAI(api_key="offline-transport", http_client=transport, max_retries=2)
        with measure_usage(), live_token_budget(1):
            with pytest.raises(LiveBudgetExceeded, match="remaining allowance"):
                QueryReformulator(llm_client=client).reformulate("Question")
    assert requests == []


def test_live_budget_blocks_unaccounted_callable_clients() -> None:
    calls = 0

    def call(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "query"

    with measure_usage(), live_token_budget(10_000):
        with pytest.raises(LiveBudgetExceeded):
            QueryReformulator(llm_client=call).reformulate("Question")
    assert calls == 0


def test_live_budget_stops_an_unreserved_fourth_sdk_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503, json={"error": {"message": "Retry"}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(api_key="offline-transport", http_client=transport, max_retries=3)
        monkeypatch.setattr(client, "_sleep_for_retry", lambda **_kwargs: None)
        with measure_usage(), live_token_budget(100_000):
            with pytest.raises(LiveBudgetExceeded):
                QueryReformulator(llm_client=client).reformulate("Question")
    assert len(requests) == 3


def test_grader_fallback_retains_failed_attempt_and_empty_context_has_no_call() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("Private provider detail")
    grader = RetrievalGrader(llm_client=client)
    with measure_usage() as trace:
        assert grader.grade_retrieval("Attention", [_create_sample_chunk()]) == "grounded"
        assert grader.grade_retrieval("Attention", []) == "insufficient"
    assert len(trace.records) == 1
    assert trace.records[0].outcome == "failed"
    assert trace.records[0].error_type == "RuntimeError"
    assert "Private provider detail" not in repr(trace)


@pytest.mark.asyncio
async def test_shared_sdk_client_retries_are_isolated_across_concurrent_traces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: dict[str, int] = {}
    lock = Lock()

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        label = "A" if "Question_A" in body["messages"][0]["content"] else "B"
        with lock:
            attempts[label] = attempts.get(label, 0) + 1
            attempt = attempts[label]
        if attempt == 1:
            return httpx.Response(503, json={"error": {"message": "Retry"}})
        tokens = 10 if label == "A" else 20
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "model": "test-model",
                "object": "chat.completion",
                "created": 0,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": label},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": tokens,
                    "completion_tokens": 1,
                    "total_tokens": tokens + 1,
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(api_key="offline-transport", http_client=transport, max_retries=1)
        monkeypatch.setattr(client, "_sleep_for_retry", lambda **_kwargs: None)
        reformulator = QueryReformulator(llm_client=client)

        async def run(label: str) -> list[UsageRecord]:
            with measure_usage() as trace:
                await asyncio.to_thread(reformulator.reformulate, f"Question_{label}")
                return trace.records

        a, b = await asyncio.gather(run("A"), run("B"))
        assert [r.total_tokens for r in a] == [None, 11]
        assert [r.total_tokens for r in b] == [None, 21]
        assert attempts == {"A": 2, "B": 2}
