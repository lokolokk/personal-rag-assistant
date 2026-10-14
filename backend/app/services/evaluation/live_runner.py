"""Explicitly budgeted, isolated live pilot for Task 5.9.1.

The default CLI mode is a local pilot preflight. Live mode requires both a numeric
allowance and an explicit dispatch flag, uses only ephemeral retrieval storage, and
never writes conversation history to the application registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from unittest.mock import patch
from urllib.parse import urlsplit

from app.core.config import Settings
from app.schemas.chat import ChatMessage
from app.services.evaluation.answer_quality import (
    SnapshotCommitError,
    _validate_recorded_citation,
    commit_fresh_case,
    evaluate_fresh_case,
)
from app.services.evaluation.grounding_report import (
    QUALITY_OUTPUT_CAPS,
    FreshCaseInput,
    LiveRunPlan,
    review_digest,
)
from app.services.evaluation.report_store import LiveReportManifest, ReportStore, recover_report
from app.services.evaluation.source_dataset import (
    EvaluationCase,
    SourceDataset,
    load_source_dataset,
)
from app.services.evaluation.usage import (
    DEFAULT_OUTPUT_CAPS,
    DurableUsageCapture,
    LiveBudgetExceeded,
    Stage,
    UsageRecord,
    conservative_reservation_tokens,
    durable_usage,
    estimate_tokens,
    live_token_budget,
    measure_usage,
)
from app.services.generation.grounding import (
    FALLBACK_RESPONSE,
    GRADING_PROMPT_TEMPLATE,
    GroundedGenerator,
    RetrievalGrader,
    build_grounding_prompt,
    format_context_chunks,
)
from app.services.retrieval.reformulator import (
    REFORMULATION_PROMPT_TEMPLATE,
    QueryReformulator,
    build_reformulation_prompt,
)

BACKEND_DIR = Path(__file__).resolve().parents[3]
PROJECT_ROOT = BACKEND_DIR.parent
MANIFEST = BACKEND_DIR / "tests" / "evaluation" / "fixtures" / "task_5_9_1_source_dataset.json"

MAX_ATTEMPTS = 1
INPUT_MULTIPLIER = 2
FRAMING_TOKENS_PER_ATTEMPT = 128
CHUNK_BATCH_SIZE = 16
RESULT_COUNT = 5
PILOT_CASE_IDS = (
    "tada-01-single-column-rag",
    "tada-08-confidence-interval-absence",
)

QUALITY_PILOT_CASE_IDS = (
    "tada-01-single-column-rag",
    "rag-02-tqa-split-qualification",
    "rag-08-history-followup-wq",
    "rgb-06-table3-row-extraction-gap",
    "tada-08-confidence-interval-absence",
)
SOURCE_DATASET_SHA256 = "026e0a47f372506c25e3523fc4ad1c05c01f6be189ceeadbab387433a081f282"

PROPOSED_DESTINATION = "example.invalid"


def quality_settings(destination_host: str = PROPOSED_DESTINATION) -> Settings:
    """Build proposed non-secret defaults without reading environment keys or dotenv files."""
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", destination_host):
        raise ValueError("Destination must be a hostname without credentials, port or path")
    return Settings.model_construct(
        OPENAI_API_BASE=f"https://{destination_host}/compatible-mode/v1"
    )


def run_quality_preflight(destination_host: str = PROPOSED_DESTINATION) -> dict[str, Any]:
    """Recompute a five-case disclosure using only local source inputs and prompt builders."""
    settings = quality_settings(destination_host)
    with patch("app.services.ingestion.chunker.get_settings", return_value=settings):
        dataset, chunks = _load_quality_corpus()
    selected = {c.id: c for c in dataset.cases if c.split == "diagnostic"}
    reservations: dict[Stage, int] = dict.fromkeys(QUALITY_OUTPUT_CAPS, 0)
    bounds: dict[Stage, int] = dict.fromkeys(QUALITY_OUTPUT_CAPS, 0)
    batches = 0
    for offset in range(0, len(chunks), CHUNK_BATCH_SIZE):
        tokens = sum(_token_count(c.content) for c in chunks[offset : offset + CHUNK_BATCH_SIZE])
        reservations["ingestion_embedding"] += _stage_reservation(
            "ingestion_embedding", tokens, output_caps=QUALITY_OUTPUT_CAPS
        )
        bounds["ingestion_embedding"] = max(bounds["ingestion_embedding"], tokens)
        batches += 1
    disclosures: list[dict[str, Any]] = []
    for case_id in QUALITY_PILOT_CASE_IDS:
        case = selected[case_id]
        scoped = [c for c in chunks if c.metadata.document_id in case.scope_document_ids]
        longest = sorted(scoped, key=lambda c: _token_count(c.content), reverse=True)[:RESULT_COUNT]
        context = format_context_chunks(longest)
        reformulation = build_reformulation_prompt(
            active_document_titles=_case_titles(case, dataset),
            chat_history=[ChatMessage(role=h.role, content=h.content) for h in case.history],
            user_query=case.question,
        )
        inputs: dict[Stage, int] = {
            "reformulation": _token_count(reformulation),
            "query_embedding": 256,
            "grading": _token_count(
                GRADING_PROMPT_TEMPLATE.format(query=case.question, context=context)
            )
            + 256,
            "generation": _token_count(build_grounding_prompt(context, case.question)) + 256,
        }
        case_reservations = {
            stage: _stage_reservation(stage, tokens, output_caps=QUALITY_OUTPUT_CAPS)
            for stage, tokens in inputs.items()
        }
        for stage, tokens in inputs.items():
            bounds[stage] = max(bounds[stage], tokens)
            reservations[stage] += case_reservations[stage]
        disclosures.append(
            {
                "case_id": case.id,
                "question": case.question,
                "scope": case.scope_document_ids,
                "history": [h.model_dump() for h in case.history],
                "history_sha256": review_digest([h.model_dump() for h in case.history]),
                "context_estimate_tokens": _token_count(context),
                "stage_reserved_tokens": case_reservations,
            }
        )
    total = sum(reservations.values())
    plan = LiveRunPlan(
        plan_id="task-6-1-five-case-v2",
        dataset_sha256=SOURCE_DATASET_SHA256,
        corpus_sha256={d.id: d.sha256 for d in dataset.documents},
        prompt_sha256=quality_prompt_hashes(),
        configuration_sha256=review_digest(quality_configuration(settings)),
        destination_host=destination_host,
        chat_model=settings.chat_model,
        embedding_model=settings.EMBEDDING_MODEL,
        allowance_tokens=total,
        maximum_calls=batches + 4 * len(disclosures),
        output_caps=QUALITY_OUTPUT_CAPS,
        usage_evidence="provider",
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "offline_proposal_not_authorization",
        "provider_dispatches": 0,
        "case_ids": list(QUALITY_PILOT_CASE_IDS),
        "cases": disclosures,
        "corpus_chunks": len(chunks),
        "ingestion_batches": batches,
        "maximum_calls": plan.maximum_calls,
        "stage_reserved_tokens": reservations,
        "total_reserved_tokens": total,
        "planning_baseline_tokens": 175142,
        "planning_baseline_matches": total == 175142,
        "input_token_bounds": bounds,
        "plan": plan.model_dump(),
        "configuration": quality_configuration(settings),
        "payload_classes": [
            "pinned PDF chunk text",
            "canonical questions and frozen history",
            "provider rewrites",
            "ordered Top 5 source context",
        ],
        "held_out_cases_excluded": [c.id for c in dataset.cases if c.split == "held_out"],
        "limitations": [
            "Local cl100k estimates are not billing caps or provider quality",
            "Host/model defaults are proposals, not verified account availability",
            "Any changed preflight requires fresh exact approval; no retries",
            "Historical held-out usage remains unknown and is not added as zero",
        ],
    }
    report["preflight_sha256"] = review_digest(report)
    return report


def validate_quality_approval(
    preflight: dict[str, Any],
    approved: dict[str, Any],
    *,
    approved_sha256: str,
    approved_token_budget: int,
) -> LiveRunPlan:
    """Require the entire freshly recomputed disclosure, including bounds, to agree."""
    if not isinstance(approved, dict):
        raise ValueError("Approved quality disclosure must be an object")
    payload = {key: value for key, value in approved.items() if key != "preflight_sha256"}
    if (
        approved != preflight
        or review_digest(payload) != approved_sha256
        or approved.get("preflight_sha256") != approved_sha256
        or approved_token_budget != preflight["total_reserved_tokens"]
    ):
        raise ValueError("Fresh preflight/approval mismatch; no clients may be constructed")
    return LiveRunPlan.model_validate(preflight["plan"])


def _quality_output_directory(path: Path) -> Path:
    resolved = path.resolve()
    temp_dir = Path(tempfile.gettempdir()).resolve()
    if not (resolved.is_relative_to(PROJECT_ROOT.resolve()) or resolved.is_relative_to(temp_dir)):
        raise ValueError("Quality output must remain inside the project workspace")
    return resolved


def _commit_execution(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError("Existing execution evidence cannot be overwritten")
    pending = path.with_suffix(".pending")
    with pending.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"sha256": review_digest(payload), "result": payload}))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def _quality_clients(
    stack: ExitStack,
    settings: Settings,
    *,
    synthetic: bool,
    stop_after_intent: int | None,
) -> tuple[Any, Any]:
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    from openai import OpenAI

    if synthetic:
        import httpx2 as httpx

        sent = 0
        case_number = 0

        def respond(request: Any) -> Any:
            nonlocal sent, case_number
            sent += 1
            if sent == stop_after_intent:
                os._exit(23)
            body = json.loads(request.content)
            if "input" in body:
                return httpx.Response(
                    200,
                    json={
                        "object": "list",
                        "model": settings.EMBEDDING_MODEL,
                        "data": [
                            {
                                "object": "embedding",
                                "index": i,
                                "embedding": [1.0] + [0.0] * (settings.EMBEDDING_DIMENSION - 1),
                            }
                            for i in range(len(body["input"]))
                        ],
                        "usage": {"prompt_tokens": 4, "total_tokens": 4},
                    },
                )
            cap = body["max_completion_tokens"] + 10
            if cap == 256:
                case_number += 1
                text = "RAG source information and WebQuestions results"
            elif cap == QUALITY_OUTPUT_CAPS["grading"]:
                text = "grounded" if case_number <= 3 else "insufficient"
            else:
                text = "Synthetic source answer [1]."
            return httpx.Response(
                200,
                json={
                    "id": "synthetic",
                    "object": "chat.completion",
                    "created": 0,
                    "model": settings.chat_model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": text},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                },
            )

        transport = stack.enter_context(httpx.Client(transport=httpx.MockTransport(respond)))
        key = "offline-transport"
        client = stack.enter_context(
            OpenAI(
                api_key=key, base_url=settings.OPENAI_API_BASE, http_client=transport, max_retries=0
            )
        )
    else:
        configured_key = os.environ.get("OPENAI_API_KEY")
        if not configured_key:
            from app.core.config import get_settings

            configured_key = get_settings().OPENAI_API_KEY
        if not configured_key:
            raise ValueError("Authorized quality execution requires an environment API key")
        key = configured_key
        client = stack.enter_context(
            OpenAI(api_key=key, base_url=settings.OPENAI_API_BASE, max_retries=0)
        )
    embedding = OpenAIEmbeddingFunction(
        api_key=key, model_name=settings.EMBEDDING_MODEL, api_base=settings.OPENAI_API_BASE
    )
    embedding.client.close()
    embedding.client = client
    return client, embedding


def execute_quality_entry(arguments: argparse.Namespace) -> dict[str, Any]:
    """Validate full disclosure before constructing SDK clients or accessing keys."""
    synthetic = arguments.mode == "quality-fake"
    if not synthetic and (
        not arguments.confirm_live_provider_calls
        or not arguments.plan_file
        or not arguments.approved_plan_sha256
        or arguments.approved_token_budget is None
        or arguments.approved_token_budget < 1
        or arguments.fake_stop_after_intent is not None
    ):
        raise ValueError(
            "Quality live requires complete exact approval before any client/key access"
        )
    host = "example.invalid" if synthetic else (arguments.destination_host or PROPOSED_DESTINATION)
    preflight = run_quality_preflight(host)
    if synthetic:
        plan = LiveRunPlan.model_validate(preflight["plan"]).model_copy(
            update={"usage_evidence": "synthetic"}
        )
    else:
        approved = json.loads(Path(arguments.plan_file).read_text("utf-8"))
        plan = validate_quality_approval(
            preflight,
            approved,
            approved_sha256=arguments.approved_plan_sha256,
            approved_token_budget=arguments.approved_token_budget,
        )
    output = _quality_output_directory(Path(arguments.output_dir))
    effective_root = (
        PROJECT_ROOT if output.is_relative_to(PROJECT_ROOT.resolve()) else output.parent
    )
    store = ReportStore.create(
        project_root=effective_root,
        output_dir=output,
        suite_id="task-6-1-quality",
        run_id=arguments.run_id,
        planned_case_keys=list(QUALITY_PILOT_CASE_IDS),
        live_plan=plan,
    )
    print(json.dumps({"event": "started", "report_dir": str(store.run_dir)}), flush=True)
    settings = quality_settings(host)
    with ExitStack() as stack:
        if synthetic:
            from app.services.evaluation.offline import isolated_services

            stack.enter_context(isolated_services(store.run_dir / "sandbox"))
        chat, embedding = _quality_clients(
            stack, settings, synthetic=synthetic, stop_after_intent=arguments.fake_stop_after_intent
        )
        result = run_quality_pilot(
            store,
            chat_client=chat,
            embedding_function=embedding,
            settings=settings,
            input_token_bounds=preflight["input_token_bounds"],
            confirm_live_provider_calls=arguments.confirm_live_provider_calls,
            streamlined=getattr(arguments, "streamlined", False),
        )
    result["report_dir"] = str(store.run_dir)
    result["preflight_sha256"] = preflight["preflight_sha256"]
    result["provider_dispatches"] = (
        0 if synthetic else result["summary"]["calls"]["potentially_dispatched_calls"]
    )
    _commit_execution(store.run_dir / "execution.json", result)
    return result


def collect_fake_quality_process(
    output_dir: Path,
    run_id: str,
    *,
    stop_after_intent: int | None = None,
    timeout_seconds: int = 45,
    streamlined: bool = False,
) -> dict[str, Any]:
    """Bound one fake subprocess, preserve its handle/output, and never automatically rerun."""
    output = _quality_output_directory(output_dir)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}", run_id):
        raise ValueError("Invalid quality process run ID")
    if not 1 <= timeout_seconds <= 45 or (
        stop_after_intent is not None and not 1 <= stop_after_intent <= 30
    ):
        raise ValueError("Invalid fake process bounds")
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / f"{run_id}.process.json"
    claim_path = output / f"{run_id}.process.claim"
    if receipt_path.exists() or (output / run_id).exists() or claim_path.exists():
        raise ValueError("Existing process evidence cannot be rerun")
    with claim_path.open("x", encoding="utf-8") as claim:
        claim.write("fake-process-launch-reserved; no automatic retry\n")
        claim.flush()
        os.fsync(claim.fileno())
    command = [
        sys.executable,
        "-m",
        "app.services.evaluation.live_runner",
        "--mode",
        "quality-fake",
        "--output-dir",
        str(output),
        "--run-id",
        run_id,
    ]
    if streamlined:
        command.append("--streamlined")
    if stop_after_intent is not None:
        command.extend(["--fake-stop-after-intent", str(stop_after_intent)])
    allowed_env = (
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "TEMP",
        "TMP",
        "TIKTOKEN_CACHE_DIR",
        "PYTHONPATH",
    )
    environment = {key: os.environ[key] for key in allowed_env if key in os.environ}
    environment["OPENAI_API_KEY"] = ""
    process = subprocess.Popen(
        command,
        cwd=BACKEND_DIR,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        _commit_execution(
            output / f"{run_id}.handle.json",
            {
                "process_handle": process.pid,
                "handle_kind": "local_process_pid",
                "command": command,
                "automatic_rerun": False,
                "report_dir": str(output / run_id),
            },
        )
    except OSError:
        process.kill()
        process.communicate(timeout=5)
        raise
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        stdout, stderr = process.communicate(timeout=5)
    result_path = output / run_id / "execution.json"
    collected = False
    if result_path.is_file() and process.returncode == 0 and not timed_out:
        try:
            data = json.loads(result_path.read_text("utf-8"))
            lines = [json.loads(line) for line in stdout.splitlines()]
            collected = (
                data["sha256"] == review_digest(data["result"])
                and len(lines) == 2
                and lines[-1] == data["result"]
                and lines[0] == {"event": "started", "report_dir": str(output / run_id)}
            )
        except (OSError, ValueError, KeyError, TypeError):
            collected = False
    receipt: dict[str, Any] = {
        "kind": "fake_process_collection",
        "process_handle": process.pid,
        "handle_kind": "local_process_pid",
        "command": command,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "execution_status": "collected" if collected else "unknown_execution",
        "stdout": stdout,
        "stdout_sha256": review_digest(stdout),
        "stderr_sha256": review_digest(stderr),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "report_dir": str(output / run_id),
        "provider_dispatches": 0,
        "automatic_rerun": False,
    }
    _commit_execution(receipt_path, receipt)
    return receipt


def quality_configuration(settings: Settings) -> dict[str, Any]:
    """Non-secret execution settings to freeze in a future preflight plan."""
    return {
        "api_base_url": settings.OPENAI_API_BASE,
        "output_limit_parameter": "max_completion_tokens",
        "output_limit_headroom_tokens": 10,
        "quality_output_caps": QUALITY_OUTPUT_CAPS,
        "chat_model": settings.chat_model,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dimension": settings.EMBEDDING_DIMENSION,
        "dense_top_n": settings.DENSE_TOP_N,
        "sparse_top_n": settings.SPARSE_TOP_N,
        "rrf_k": settings.RRF_K,
        "top_k": RESULT_COUNT,
        "batch_size": CHUNK_BATCH_SIZE,
        "temperature": 0.0,
        "stream": False,
        "retries": 0,
    }


def quality_prompt_hashes() -> dict[Stage, str]:
    from app.services.generation.grounding import GROUNDING_PROMPT_TEMPLATE

    return {
        "reformulation": review_digest(REFORMULATION_PROMPT_TEMPLATE),
        "grading": review_digest(GRADING_PROMPT_TEMPLATE),
        "generation": review_digest(GROUNDING_PROMPT_TEMPLATE),
    }


def _load_quality_corpus() -> tuple[SourceDataset, list[Any]]:
    from app.services.evaluation.grounding_runner import reconstruct_source_corpus_chunks

    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != SOURCE_DATASET_SHA256:
        raise ValueError("Quality pilot canonical manifest drift")
    dataset = load_source_dataset(MANIFEST, project_root=PROJECT_ROOT)
    return dataset, reconstruct_source_corpus_chunks(dataset, PROJECT_ROOT)


def _quality_snapshot(
    store: ReportStore,
    capture: DurableUsageCapture,
    dataset: SourceDataset,
    case: EvaluationCase,
    state: dict[str, Any],
    raw_grade: str | None,
    *,
    finished: bool,
    retrieved: bool,
) -> FreshCaseInput:
    plan = capture.plan
    calls = recover_report(store.run_dir, project_root=store.project_root).calls
    chunks = state.get("retrieved_chunks", []) if retrieved else None
    return FreshCaseInput.model_validate(
        {
            "run_id": store.manifest.run_id,
            "case_id": case.id,
            "case_sha256": review_digest(case.model_dump()),
            "source_dataset_sha256": plan.dataset_sha256,
            "source_sha256": {d.id: d.sha256 for d in dataset.documents},
            "history_sha256": review_digest([h.model_dump() for h in case.history]),
            "question": case.question,
            "declared_scope": list(case.scope_document_ids),
            "effective_scope": list(state["active_document_ids"]),
            "standalone_query": state.get("standalone_query") or None,
            "raw_grade": raw_grade,
            "route": ("generate" if state["retrieval_grade"] == "grounded" else "insufficient")
            if finished
            else "incomplete",
            "response_text": state.get("response_text") if finished else None,
            "retrieved_chunks": [c.model_dump() for c in chunks] if chunks is not None else None,
            "supplied_chunks": [c.model_dump() for c in chunks] if chunks is not None else None,
            "rendered_context": format_context_chunks(chunks) if chunks is not None else None,
            "citations": [
                _validate_recorded_citation(c).model_dump() for c in state.get("citations", [])
            ]
            if finished
            else None,
            "request_sha256": {
                c.reservation.stage: c.reservation.payload_sha256
                for c in calls
                if c.reservation.case_key == case.id
            },
            "prompt_sha256": plan.prompt_sha256,
            "settings_sha256": plan.configuration_sha256,
            "absence_review_sha256": review_digest(case.absence_review.model_dump())
            if case.absence_review
            else None,
        }
    )


def run_quality_pilot(
    store: ReportStore,
    *,
    chat_client: Any,
    embedding_function: Any,
    settings: Settings,
    input_token_bounds: dict[Stage, int],
    confirm_live_provider_calls: bool = False,
    streamlined: bool = False,
) -> dict[str, Any]:
    """Run the fixed pilot through real synchronous graph nodes with injected transports.

    The old CLI never selects this path. Provider evidence requires separate explicit
    dispatch authorization; synthetic plans are only for isolated transport tests.
    """
    if not isinstance(store.manifest, LiveReportManifest):
        raise ValueError("Quality pilot requires a frozen live manifest")
    plan = store.manifest.live_plan
    if plan.usage_evidence == "provider" and not confirm_live_provider_calls:
        raise ValueError("Quality pilot requires fresh provider dispatch authorization")
    dataset, chunks = _load_quality_corpus()
    counts = Counter(c.metadata.document_id for c in chunks)
    expected_counts = {
        "eval_2005_11401v4_pdf": 70,
        "eval_2309_01431v2_pdf": 51,
        "eval_TaDA25_2_pdf": 27,
    }
    if counts != expected_counts or len(chunks) != 148:
        raise ValueError("Quality corpus count drift")
    if (
        plan.dataset_sha256 != SOURCE_DATASET_SHA256
        or plan.corpus_sha256 != {d.id: d.sha256 for d in dataset.documents}
        or plan.configuration_sha256 != review_digest(quality_configuration(settings))
        or plan.prompt_sha256 != quality_prompt_hashes()
        or plan.chat_model != settings.chat_model
        or plan.embedding_model != settings.EMBEDDING_MODEL
        or settings.TOP_K != RESULT_COUNT
        or plan.maximum_calls > 30
        or store.manifest.planned_case_keys != list(QUALITY_PILOT_CASE_IDS)
    ):
        raise ValueError("Quality pilot plan/configuration drift")
    for client in (chat_client, embedding_function.client):
        if client.max_retries != 0:
            raise ValueError("Quality pilot requires zero SDK retries")
        if plan.usage_evidence == "synthetic":
            import httpx2 as httpx

            if not isinstance(client._client._transport, httpx.MockTransport):
                raise ValueError("Synthetic quality pilot requires a mock transport")
    cases = {c.id: c for c in dataset.cases if c.split == "diagnostic"}
    selected = [cases[case_id] for case_id in QUALITY_PILOT_CASE_IDS]

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    from app.services.generation.graph import compile_conversational_rag_graph
    from app.services.retrieval.bm25_store import BM25Store
    from app.services.retrieval.hybrid import HybridRetriever
    from app.services.retrieval.vector_store import VectorStore

    vector = VectorStore(
        client=chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False)),
        collection_name="quality_"
        + hashlib.sha256(store.manifest.run_id.encode()).hexdigest()[:20],
        embedding_function=embedding_function,
    )
    sparse = BM25Store()
    hybrid = HybridRetriever(
        vector_store=vector, bm25_store=sparse, settings=settings, max_workers=2
    )
    reformulator = QueryReformulator(llm_client=chat_client, model=plan.chat_model)
    grader = RetrievalGrader(llm_client=chat_client, model=plan.chat_model)
    generator = GroundedGenerator(llm_client=chat_client, model=plan.chat_model)
    timings: dict[str, Any] = {"cases": {}, "ttft_seconds": None, "ingestion_seconds": None}
    status = "completed"
    stopped_reason: str | None = None
    started = monotonic()
    ingest_started = started
    active_case: str | None = None
    case_started = started
    raw_grade: str | None = None

    def retain_grade(response: Any) -> None:
        nonlocal raw_grade
        body = json.loads(response.request.content)
        if (
            body.get("max_completion_tokens") == plan.output_caps["grading"] - 10
            and not response.is_error
        ):
            response.read()
            raw_grade = response.json()["choices"][0]["message"]["content"]

    hooks = chat_client._client.event_hooks["response"]
    original_hooks = {
        id(client): (
            client,
            {name: list(values) for name, values in client._client.event_hooks.items()},
        )
        for client in (chat_client, embedding_function.client)
    }
    hooks.append(retain_grade)

    def synchronous_generation(state: dict[str, Any]) -> dict[str, Any]:
        response, citations = generator.generate(
            standalone_query=state.get("standalone_query") or state["user_query"],
            chunks=state.get("retrieved_chunks", []),
            writer=None,
        )
        output: dict[str, Any] = {
            "response_text": response,
            "citations": citations,
        }
        if state.get("retrieval_grade") != "grounded":
            output["retrieval_grade"] = (
                "insufficient" if response.strip() == FALLBACK_RESPONSE else "grounded"
            )
        return output

    try:
        with ExitStack() as stack:
            _install_live_services(stack, vector, sparse, hybrid, reformulator, grader, generator)
            stack.enter_context(
                patch(
                    "app.services.generation.graph.generate_grounded_response_node",
                    synchronous_generation,
                )
            )
            with (
                measure_usage(),
                durable_usage(store, input_token_bounds=input_token_bounds) as cap,
            ):
                ingest_started = monotonic()
                vector.add_chunks(chunks, batch_size=CHUNK_BATCH_SIZE)
                cap.check()
                sparse.add_chunks(chunks)
                timings["ingestion_seconds"] = monotonic() - ingest_started
                graph = compile_conversational_rag_graph(streamlined=streamlined)
                for case in selected:
                    active_case = case.id
                    case_started = monotonic()
                    stage_times: dict[str, float] = {}
                    timings["cases"][case.id] = {"total_seconds": None, "stages": stage_times}
                    raw_grade = None
                    state: dict[str, Any] = {
                        "conversation_id": f"quality_{store.manifest.run_id}_{case.id}",
                        "active_document_ids": list(case.scope_document_ids),
                        "active_document_titles": _case_titles(case, dataset),
                        "messages": [
                            ChatMessage(role=h.role, content=h.content) for h in case.history
                        ],
                        "user_query": case.question,
                        "standalone_query": "",
                        "retrieved_chunks": [],
                        "retrieval_grade": "insufficient",
                        "response_text": "",
                        "citations": [],
                    }
                    retrieved = False
                    with cap.case(case.id):
                        initial = _quality_snapshot(
                            store,
                            cap,
                            dataset,
                            case,
                            state,
                            raw_grade,
                            finished=False,
                            retrieved=False,
                        )
                        commit_fresh_case(
                            store.run_dir / "inputs" / "initial", initial, capture=cap
                        )
                        stage_started = monotonic()
                        for update in graph.stream(state, stream_mode="updates"):
                            cap.check()
                            for node, delta in update.items():
                                state.update(delta)
                                stage_times[node] = monotonic() - stage_started
                                retrieved = retrieved or node == "scoped_hybrid_retriever"
                                finished = node in (
                                    "generate_grounded_response",
                                    "inability_fallback",
                                )
                                snapshot = _quality_snapshot(
                                    store,
                                    cap,
                                    dataset,
                                    case,
                                    state,
                                    raw_grade,
                                    finished=finished,
                                    retrieved=retrieved,
                                )
                                path = commit_fresh_case(
                                    store.run_dir / "inputs" / node, snapshot, capture=cap
                                )
                                if any(
                                    c.metadata.document_id not in case.scope_document_ids
                                    for c in state["retrieved_chunks"]
                                ):
                                    cap.stop("scope_leak")
                                    cap.check()
                                if finished:
                                    record = evaluate_fresh_case(path)
                                    store.append_case(record)
                                    if "citation_integrity_failure" in record.failure_codes:
                                        cap.stop("citation_integrity_failure")
                                        cap.check()
                                stage_started = monotonic()
                    timings["cases"][case.id] = {
                        "total_seconds": monotonic() - case_started,
                        "stages": stage_times,
                    }
                    active_case = None
    except (LiveBudgetExceeded, SnapshotCommitError):
        status = "stopped"
        stopped_reason = cap.stopped_reason if "cap" in locals() else "capture_failed"
    except Exception:
        status = "failed"
        stopped_reason = "execution_failed"
    finally:
        if timings["ingestion_seconds"] is None:
            timings["ingestion_seconds"] = monotonic() - ingest_started
        if active_case is not None:
            timings["cases"][active_case]["total_seconds"] = monotonic() - case_started
        for client, saved in original_hooks.values():
            for name, values in saved.items():
                client._client.event_hooks[name][:] = values
        vector._client.delete_collection(vector._collection_name)
    timings["run_seconds"] = monotonic() - started
    recovered = recover_report(store.run_dir, project_root=store.project_root)
    summary = store.finalize()
    return {
        "status": status,
        "stopped_reason": stopped_reason,
        "timings": timings,
        "selected_case_ids": list(QUALITY_PILOT_CASE_IDS),
        "completed_case_ids": [c.case_id for c in recovered.cases],
        "summary": summary.model_dump(),
    }


def _token_count(text: str) -> int:
    count = estimate_tokens(text)
    if count is None:
        raise ValueError("Offline preflight requires the verified local cl100k_base cache")
    return count


def parse_source_chunks(dataset: Any) -> list[Any]:
    """Parse and structurally chunk only the source-pinned evaluation PDFs."""
    from app.services.ingestion.chunker import StructuralChunker
    from app.services.ingestion.parser import PDFParser

    parser = PDFParser()
    chunks: list[Any] = []
    for document in _select_pilot_documents(dataset):
        source_path = (PROJECT_ROOT / document.fixture_path).resolve()
        parsed = parser.parse_bytes(source_path.read_bytes(), document.filename)
        chunker = StructuralChunker(
            target_chunk_size=document.chunking.target_chunk_size,
            chunk_overlap=document.chunking.chunk_overlap,
            encoding_name=document.chunking.encoding,
        )
        chunks.extend(chunker.chunk_document(parsed, document_id=document.id))
    return chunks


def _case_titles(case: Any, dataset: Any) -> str:
    documents = {document.id: document for document in dataset.documents}
    selected = case.scope_document_ids or list(documents)
    return ", ".join(documents[document_id].filename for document_id in selected)


def _select_pilot_cases(dataset: Any) -> list[Any]:
    """Return the fixed, source-verified pilot cases and reject stale manifest IDs."""
    cases_by_id = {case.id: case for case in dataset.cases}
    selected: list[Any] = []
    for case_id in PILOT_CASE_IDS:
        case = cases_by_id.get(case_id)
        if case is None or case.split != "diagnostic":
            raise ValueError(f"Configured pilot case is missing or not diagnostic: {case_id}")
        selected.append(case)
    return selected


def _select_pilot_documents(dataset: Any) -> list[Any]:
    """Index only the documents required by the fixed pilot's explicit scopes."""
    cases = _select_pilot_cases(dataset)
    if any(not case.scope_document_ids for case in cases):
        raise ValueError("The reduced pilot requires explicit document scopes")
    required_ids = {document_id for case in cases for document_id in case.scope_document_ids}
    documents = [document for document in dataset.documents if document.id in required_ids]
    if {document.id for document in documents} != required_ids:
        raise ValueError("A pilot scope references a missing source document")
    return documents


def _stage_reservation(
    stage: Stage, prompt_tokens: int, *, output_caps: dict[Stage, int] | None = None
) -> int:
    return conservative_reservation_tokens(
        prompt_tokens,
        (DEFAULT_OUTPUT_CAPS if output_caps is None else output_caps)[stage],
        max_attempts=MAX_ATTEMPTS,
        input_multiplier=INPUT_MULTIPLIER,
        framing_tokens_per_attempt=FRAMING_TOKENS_PER_ATTEMPT,
    )


def build_preflight_report(dataset: Any, chunks: list[Any]) -> dict[str, Any]:
    """Compute a bounded pilot reservation without constructing provider clients."""
    from app.core.config import Settings

    settings = Settings()
    longest_chunks = sorted(chunks, key=lambda chunk: _token_count(chunk.content), reverse=True)[
        :RESULT_COUNT
    ]
    framed_context = format_context_chunks(longest_chunks)
    stage_reservations: dict[str, int] = {
        "ingestion_embedding": 0,
        "query_embedding": 0,
        "reformulation": 0,
        "grading": 0,
        "generation": 0,
    }
    ingestion_calls = 0
    for offset in range(0, len(chunks), CHUNK_BATCH_SIZE):
        batch = chunks[offset : offset + CHUNK_BATCH_SIZE]
        batch_tokens = sum(_token_count(chunk.content) for chunk in batch)
        stage_reservations["ingestion_embedding"] += conservative_reservation_tokens(
            batch_tokens,
            0,
            max_attempts=MAX_ATTEMPTS,
            input_multiplier=INPUT_MULTIPLIER,
            framing_tokens_per_attempt=FRAMING_TOKENS_PER_ATTEMPT,
        )
        ingestion_calls += 1

    diagnostic_cases = [case for case in dataset.cases if case.split == "diagnostic"]
    pilot_cases = _select_pilot_cases(dataset)
    held_out_cases = [case for case in dataset.cases if case.split == "held_out"]
    prompt_case_count = 0
    for case in pilot_cases:
        history = [ChatMessage(role=turn.role, content=turn.content) for turn in case.history]
        reformulation_prompt = build_reformulation_prompt(
            active_document_titles=_case_titles(case, dataset),
            chat_history=history,
            user_query=case.question,
        )
        stage_reservations["reformulation"] += _stage_reservation(
            "reformulation", _token_count(reformulation_prompt)
        )
        stage_reservations["query_embedding"] += conservative_reservation_tokens(
            max(256, _token_count(case.question)),
            0,
            max_attempts=MAX_ATTEMPTS,
            input_multiplier=INPUT_MULTIPLIER,
            framing_tokens_per_attempt=FRAMING_TOKENS_PER_ATTEMPT,
        )

        grading_prompt = GRADING_PROMPT_TEMPLATE.format(
            query=case.retrieval_query,
            context=framed_context,
        )
        generation_prompt = build_grounding_prompt(framed_context, case.retrieval_query)
        # A reformulated query can use its full configured output cap; add that headroom
        # to downstream prompts because preflight does not call a model to predict it.
        stage_reservations["grading"] += _stage_reservation(
            "grading", _token_count(grading_prompt) + DEFAULT_OUTPUT_CAPS["reformulation"]
        )
        stage_reservations["generation"] += _stage_reservation(
            "generation", _token_count(generation_prompt) + DEFAULT_OUTPUT_CAPS["reformulation"]
        )
        prompt_case_count += 1

    return {
        "mode": "offline_preflight",
        "run_scope": "two_case_tada_pilot",
        "dataset_id": dataset.dataset_id,
        "split": "diagnostic",
        "diagnostic_case_count_available": len(diagnostic_cases),
        "case_count": len(pilot_cases),
        "selected_case_ids": [case.id for case in pilot_cases],
        "answerable_case_count": sum(case.answerability == "answerable" for case in pilot_cases),
        "unanswerable_case_count": sum(
            case.answerability == "unanswerable" for case in pilot_cases
        ),
        "held_out_case_count_reserved_and_excluded": len(held_out_cases),
        "held_out_provider_calls": 0,
        "chunk_count": len(chunks),
        "ingestion_batches": ingestion_calls,
        "corpus_documents": [
            {"id": document.id, "filename": document.filename, "sha256": document.sha256}
            for document in _select_pilot_documents(dataset)
        ],
        "model_configuration": {
            "chat_model": settings.chat_model,
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "dense_top_n": settings.DENSE_TOP_N,
            "sparse_top_n": settings.SPARSE_TOP_N,
            "rrf_k": settings.RRF_K,
            "top_k": RESULT_COUNT,
            "embedding_batch_size": CHUNK_BATCH_SIZE,
            "max_attempts_reserved_per_call": MAX_ATTEMPTS,
            "max_retries": MAX_ATTEMPTS - 1,
            "output_caps": DEFAULT_OUTPUT_CAPS,
        },
        "reservation_by_stage_tokens": stage_reservations,
        "total_conservative_reservation_tokens": sum(stage_reservations.values()),
        "reservation_method": (
            "For each call, one attempt of 2x cl100k_base text-only input plus 128 framing "
            "tokens and the stage output cap. Ingestion uses the configured 16-chunk batches. "
            "Grading and generation include the five largest source chunks plus reformulation "
            "output headroom. The two-case pilot indexes only TaDA25_2.pdf and covers one "
            "answerable table question and one unanswerable question."
        ),
        "uncertainties": [
            "Text token estimates do not match provider tokenization or hidden reasoning exactly.",
            (
                "The 2x multiplier and framing allowance are conservative planning estimates, "
                "not a billing guarantee."
            ),
            (
                "The live runner configures zero automatic SDK retries; failed-call consumption "
                "without provider usage remains unknown."
            ),
            (
                "Quality and citation grounding require manual review; preflight performs no "
                "inference."
            ),
        ],
        "prompt_case_count": prompt_case_count,
        "provider_dispatches": 0,
    }


def run_offline_preflight() -> dict[str, Any]:
    """Verify the corpus and runner plan while network dispatch is disabled."""
    from app.services.evaluation.offline import offline_network
    from app.services.evaluation.source_dataset import load_source_dataset

    with offline_network():
        dataset = load_source_dataset(MANIFEST, project_root=PROJECT_ROOT)
        chunks = parse_source_chunks(dataset)
        return build_preflight_report(dataset, chunks)


def _configure_budgetable_client(client: Any, *, label: str) -> None:
    try:
        client.max_retries = MAX_ATTEMPTS - 1
    except (AttributeError, TypeError) as error:
        raise ValueError(f"{label} cannot be configured for one provider attempt") from error
    if getattr(client, "max_retries", None) != MAX_ATTEMPTS - 1:
        raise ValueError(f"{label} must use the runner's zero-retry OpenAI client")
    transport = getattr(client, "_client", None)
    hooks = getattr(transport, "event_hooks", None)
    if (
        not isinstance(hooks, dict)
        or not isinstance(hooks.get("request"), list)
        or not isinstance(hooks.get("response"), list)
    ):
        raise ValueError(f"{label} does not expose request hooks required for retry reservations")


def _usage_summary(records: list[UsageRecord]) -> dict[str, Any]:
    by_stage: dict[str, dict[str, Any]] = {}
    stages = sorted({record.stage for record in records})
    for stage in stages:
        stage_records = [record for record in records if record.stage == stage]
        provider = [record for record in stage_records if record.source == "provider_reported"]
        estimated = [record for record in stage_records if record.source == "local_estimate"]
        by_stage[stage] = {
            "attempt_count": len(stage_records),
            "provider_reported_attempt_count": len(provider),
            "provider_reported_total_tokens_sum": sum(
                record.total_tokens for record in provider if record.total_tokens is not None
            ),
            "provider_reported_total_unknown_attempts": sum(
                record.total_tokens is None for record in provider
            ),
            "provider_reported_input_tokens_sum": sum(
                record.input_tokens for record in provider if record.input_tokens is not None
            ),
            "provider_reported_output_tokens_sum": sum(
                record.output_tokens for record in provider if record.output_tokens is not None
            ),
            "local_estimate_attempt_count": len(estimated),
            "local_estimate_total_tokens_sum": sum(
                record.total_tokens for record in estimated if record.total_tokens is not None
            ),
            "unavailable_attempt_count": sum(
                record.source == "unavailable" for record in stage_records
            ),
            "failed_attempt_count": sum(record.outcome == "failed" for record in stage_records),
            "local_only_attempt_count": sum(record.local_only for record in stage_records),
        }
    return {"by_stage": by_stage, "attempts": [asdict(record) for record in records]}


def _fact_evidence(case: Any, retrieved: list[Any], dataset: Any) -> dict[str, Any]:
    anchor_index = {
        anchor.id: anchor for document in dataset.documents for anchor in document.anchors
    }
    result_ids = {chunk.chunk_id for chunk in retrieved}
    fact_results: list[dict[str, Any]] = []
    for fact in case.required_facts:
        primary_anchors = [anchor_index[anchor_id] for anchor_id in fact.source_anchor_ids]
        primary_hit = all(
            anchor.chunk_mapping.status == "mapped"
            and bool(set(anchor.chunk_mapping.chunk_ids) & result_ids)
            for anchor in primary_anchors
        )
        alternative_hit = any(
            anchor_index[anchor_id].chunk_mapping.status == "mapped"
            and bool(set(anchor_index[anchor_id].chunk_mapping.chunk_ids) & result_ids)
            for anchor_id in fact.acceptable_alternative_evidence_anchor_ids
        )
        fact_results.append(
            {
                "fact_id": fact.id,
                "required_claim": fact.claim,
                "exact_values": fact.exact_values,
                "unit": fact.unit,
                "qualifications": fact.qualifications,
                "source_anchor_ids": fact.source_anchor_ids,
                "evidence_anchor_hit": primary_hit or alternative_hit,
                "extraction_failure": any(
                    anchor.chunk_mapping.status == "extraction_failure"
                    for anchor in primary_anchors
                ),
            }
        )
    return {
        "facts": fact_results,
        "covered_fact_count": sum(fact["evidence_anchor_hit"] for fact in fact_results),
        "required_fact_count": len(fact_results),
        "coverage_kind": "mapped source-anchor coverage proxy; not factual entailment",
    }


def _case_report(case: Any, dataset: Any, result: dict[str, Any]) -> dict[str, Any]:
    chunks = result.get("retrieved_chunks", [])
    citations = result.get("citations", [])
    scope = set(case.scope_document_ids)
    cited_scope_ok = (
        all(citation.get("document_id") in scope for citation in citations) if scope else True
    )
    response = str(result.get("response_text", ""))
    returned_ids = {chunk.chunk_id for chunk in chunks}
    anchor_evidence = _fact_evidence(case, chunks, dataset)
    return {
        "case_id": case.id,
        "split": case.split,
        "answerability": case.answerability,
        "unanswerable_reason": case.unanswerable_reason,
        "absence_review": (
            case.absence_review.model_dump(mode="json") if case.absence_review is not None else None
        ),
        "scope_document_ids": case.scope_document_ids,
        "history_turn_count": len(case.history),
        "question": case.question,
        "retrieval_query_annotation": case.retrieval_query,
        "standalone_query_used": result.get("standalone_query", ""),
        "retrieved_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.metadata.document_id,
                "filename": chunk.metadata.filename,
                "page_number": chunk.metadata.page_number,
                "rrf_score": chunk.score,
                "content": chunk.content,
            }
            for chunk in chunks
        ],
        "scope_leakage_chunk_ids": [
            chunk.chunk_id for chunk in chunks if scope and chunk.metadata.document_id not in scope
        ],
        "anchor_evidence_coverage": anchor_evidence,
        "unnecessary_evidence_chunk_ids": sorted(
            returned_ids
            - {
                chunk_id
                for fact in case.required_facts
                for anchor_id in fact.source_anchor_ids
                + fact.acceptable_alternative_evidence_anchor_ids
                for chunk_id in next(
                    anchor
                    for document in dataset.documents
                    for anchor in document.anchors
                    if anchor.id == anchor_id
                ).chunk_mapping.chunk_ids
            }
        )
        if case.answerability == "answerable"
        else [],
        "response_text": response,
        "is_exact_fallback": response.strip() == FALLBACK_RESPONSE,
        "unsupported_answer_candidate": (
            case.answerability == "unanswerable" and response.strip() != FALLBACK_RESPONSE
        ),
        "answerable_refusal_candidate": (
            case.answerability == "answerable" and response.strip() == FALLBACK_RESPONSE
        ),
        "citations": citations,
        "citation_status": {
            "resolved_citation_count": len(citations),
            "all_resolved_citations_in_scope": cited_scope_ok,
            "grounding_review": "manual_review_required",
        },
        "quality_review": {
            "correctness": "pending_manual_review",
            "completeness": "pending_manual_review",
            "citation_grounding": "pending_manual_review",
            "required_facts": anchor_evidence["facts"],
        },
        "token_success_status": "cannot count as successful until correctness review is recorded",
    }


def _install_live_services(
    stack: ExitStack,
    vector_store: Any,
    bm25_store: Any,
    hybrid_retriever: Any,
    reformulator: Any,
    grader: Any,
    generator: Any,
) -> None:
    import app.services.generation.grounding as grounding_module
    import app.services.retrieval.bm25_store as bm25_module
    import app.services.retrieval.hybrid as hybrid_module
    import app.services.retrieval.reformulator as reformulator_module
    import app.services.retrieval.vector_store as vector_module

    replacements = [
        (vector_module, "_vector_store_instance", vector_store),
        (bm25_module, "_bm25_store_instance", bm25_store),
        (hybrid_module, "_hybrid_retriever_instance", hybrid_retriever),
        (reformulator_module, "_GLOBAL_REFORMULATOR", reformulator),
        (grounding_module, "_GLOBAL_GRADER", grader),
        (grounding_module, "_GLOBAL_GENERATOR", generator),
    ]
    for module, attribute, value in replacements:
        stack.enter_context(patch.object(module, attribute, value))


def run_live_evaluation(
    approved_token_allowance: int,
    *,
    confirm_live_provider_calls: bool,
) -> dict[str, Any]:
    """Run the two-case TaDA diagnostic pilot after explicit live authorization."""
    if not confirm_live_provider_calls or approved_token_allowance < 1:
        raise ValueError(
            "Live mode requires explicit provider-call authorization and a positive allowance"
        )

    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    from app.core.config import get_settings
    from app.services.evaluation.source_dataset import load_source_dataset
    from app.services.generation.graph import run_conversational_rag_graph
    from app.services.retrieval.bm25_store import BM25Store
    from app.services.retrieval.hybrid import HybridRetriever
    from app.services.retrieval.vector_store import VectorStore

    dataset = load_source_dataset(MANIFEST, project_root=PROJECT_ROOT)
    chunks = parse_source_chunks(dataset)
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        raise ValueError("Live evaluation is unavailable until an API key is configured")

    embedding_function = OpenAIEmbeddingFunction(
        api_key=settings.OPENAI_API_KEY,
        model_name=settings.EMBEDDING_MODEL,
        api_base=settings.OPENAI_API_BASE,
    )
    vector_client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    vector_store = VectorStore(
        client=vector_client,
        collection_name="task_5_9_1_live_baseline",
        embedding_function=embedding_function,
    )
    bm25_store = BM25Store()
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_store=bm25_store,
        settings=settings,
        max_workers=2,
    )
    reformulator = QueryReformulator()
    grader = RetrievalGrader()
    generator = GroundedGenerator()
    for label, client in [
        ("reformulator", reformulator.client),
        ("grader", grader.client),
        ("generator", generator.client),
        ("embedding adapter", embedding_function.client),
    ]:
        _configure_budgetable_client(client, label=label)

    diagnostic_cases = [case for case in dataset.cases if case.split == "diagnostic"]
    pilot_cases = _select_pilot_cases(dataset)
    cases: list[dict[str, Any]] = []
    status = "completed"
    failure_type: str | None = None
    budget: Any = None
    ingestion_record_end = 0
    with ExitStack() as stack:
        _install_live_services(
            stack,
            vector_store,
            bm25_store,
            hybrid_retriever,
            reformulator,
            grader,
            generator,
        )
        with (
            measure_usage() as trace,
            live_token_budget(
                approved_token_allowance,
                max_attempts=MAX_ATTEMPTS,
                input_multiplier=INPUT_MULTIPLIER,
                framing_tokens_per_attempt=FRAMING_TOKENS_PER_ATTEMPT,
            ) as active_budget,
        ):
            budget = active_budget
            try:
                vector_store.add_chunks(chunks)
                bm25_store.add_chunks(chunks)
                ingestion_record_end = len(trace.records)
            except LiveBudgetExceeded:
                status = "stopped_at_ingestion_budget_gate"
            except Exception as error:
                status = "ingestion_failed"
                failure_type = type(error).__name__

            if status == "completed":
                for case in pilot_cases:
                    record_start = len(trace.records)
                    reservation_start = len(active_budget.reservations)
                    history = [
                        ChatMessage(role=turn.role, content=turn.content) for turn in case.history
                    ]
                    state = {
                        "conversation_id": f"task_5_9_1_{case.id}",
                        "active_document_ids": list(case.scope_document_ids),
                        "active_document_titles": _case_titles(case, dataset),
                        "messages": history,
                        "user_query": case.question,
                        "standalone_query": "",
                        "retrieved_chunks": [],
                        "retrieval_grade": "insufficient",
                        "response_text": "",
                        "citations": [],
                    }
                    try:
                        result = run_conversational_rag_graph(state, streamlined=False)
                    except LiveBudgetExceeded:
                        status = "stopped_at_case_budget_gate"
                        cases.append({"case_id": case.id, "status": status})
                        break
                    except Exception as error:
                        status = "case_execution_failed"
                        failure_type = type(error).__name__
                        cases.append({"case_id": case.id, "status": status})
                        break
                    case_payload = _case_report(case, dataset, result)
                    case_payload["status"] = "completed"
                    case_payload["usage"] = _usage_summary(trace.records[record_start:])
                    case_payload["budget_reservations"] = [
                        asdict(reservation)
                        for reservation in active_budget.reservations[reservation_start:]
                    ]
                    cases.append(case_payload)

            all_records = list(trace.records)
            ingestion_records = all_records[:ingestion_record_end]
            query_records = all_records[ingestion_record_end:]

    leakage_ids = [
        chunk_id for case in cases for chunk_id in case.get("scope_leakage_chunk_ids", [])
    ]
    return {
        "mode": "live_baseline",
        "status": status,
        "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "dataset_id": dataset.dataset_id,
        "case_split": "diagnostic",
        "run_scope": "two_case_tada_pilot",
        "diagnostic_case_count_available": len(diagnostic_cases),
        "case_count_planned": len(pilot_cases),
        "selected_case_ids": [case.id for case in pilot_cases],
        "case_count_completed": sum(case.get("status") == "completed" for case in cases),
        "provider_dispatch_authorized": confirm_live_provider_calls,
        "approved_token_allowance": approved_token_allowance,
        "budget": {
            "reserved_tokens": budget.reserved_tokens if budget is not None else None,
            "remaining_unreserved_tokens": (
                approved_token_allowance - budget.reserved_tokens if budget is not None else None
            ),
            "reservation_count": len(budget.reservations) if budget is not None else 0,
            "reservation_by_stage_tokens": {
                stage: sum(
                    reservation.reserved_tokens
                    for reservation in budget.reservations
                    if reservation.stage == stage
                )
                for stage in DEFAULT_OUTPUT_CAPS
            }
            if budget is not None
            else {},
            "attempts_reserved_per_call": MAX_ATTEMPTS,
            "input_multiplier": INPUT_MULTIPLIER,
            "framing_tokens_per_attempt": FRAMING_TOKENS_PER_ATTEMPT,
            "output_caps": DEFAULT_OUTPUT_CAPS,
        },
        "model_configuration": {
            "chat_model": settings.chat_model,
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "api_base_host": urlsplit(settings.OPENAI_API_BASE or "").hostname,
            "dense_top_n": settings.DENSE_TOP_N,
            "sparse_top_n": settings.SPARSE_TOP_N,
            "rrf_k": settings.RRF_K,
            "top_k": RESULT_COUNT,
            "embedding_batch_size": CHUNK_BATCH_SIZE,
            "max_retries": MAX_ATTEMPTS - 1,
            "concurrency": 1,
        },
        "corpus_documents": [
            {"id": document.id, "filename": document.filename, "sha256": document.sha256}
            for document in _select_pilot_documents(dataset)
        ],
        "indexed_chunk_count": len(chunks) if ingestion_record_end or status == "completed" else 0,
        "storage_identity": "ephemeral Chroma and in-memory BM25; no library/history persistence",
        "ingestion_usage": _usage_summary(ingestion_records),
        "case_usage": _usage_summary(query_records),
        "quality_summary": {
            "answerable_refusal_candidates": sum(
                case.get("answerable_refusal_candidate", False) for case in cases
            ),
            "unanswerable_unsupported_answer_candidates": sum(
                case.get("unsupported_answer_candidate", False) for case in cases
            ),
            "manual_correctness_reviews_pending": sum(
                case.get("status") == "completed" for case in cases
            ),
            "total_tokens_per_successful_answer": None,
            "successful_answer_note": (
                "Correctness and completeness require manual review; no case counts as a "
                "successful answer until that review is recorded."
            ),
            "scope_leakage_chunk_count": len(leakage_ids),
        },
        "failure_type": failure_type,
        "cases": cases,
        "usage_limitations": [
            (
                "Provider-reported usage, local text-only estimates, and unavailable attempts "
                "remain separate."
            ),
            "Unknown failed-call consumption is not converted to zero.",
            (
                "The reservation is conservative but relies on local token estimates and is "
                "not an exact provider billing cap."
            ),
            "Retrieved text, model answers, and citations require per-case manual factual review.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "preflight",
            "live",
            "quality-preflight",
            "quality-fake",
            "quality-live",
            "quality-smoke",
        ),
        default="preflight",
    )
    parser.add_argument("--approved-token-budget", type=int)
    parser.add_argument("--confirm-live-provider-calls", action="store_true")
    parser.add_argument("--plan-file")
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-id")
    parser.add_argument("--destination-host")
    parser.add_argument("--fake-stop-after-intent", type=int)
    parser.add_argument(
        "--streamlined",
        action="store_true",
        default=False,
        help="Execute streamlined single-pass RAG graph",
    )
    arguments = parser.parse_args(argv)
    quality = arguments.mode.startswith("quality-")
    extended = (
        arguments.plan_file,
        arguments.approved_plan_sha256,
        arguments.output_dir,
        arguments.run_id,
        arguments.destination_host,
        arguments.fake_stop_after_intent,
    )
    if not quality and any(value is not None for value in extended):
        parser.error("Five-case options require an explicit quality mode")
    if quality:
        live = arguments.mode == "quality-live"
        if live:
            if (
                not arguments.confirm_live_provider_calls
                or not arguments.plan_file
                or not arguments.approved_plan_sha256
                or arguments.approved_token_budget is None
                or arguments.approved_token_budget < 1
                or arguments.fake_stop_after_intent is not None
            ):
                parser.error(
                    "Quality live requires exact plan/hash/budget and dispatch confirmation"
                )
        elif (
            arguments.confirm_live_provider_calls
            or arguments.approved_token_budget is not None
            or arguments.plan_file
            or arguments.approved_plan_sha256
        ):
            parser.error("Approval flags are invalid in offline quality modes")
        if arguments.mode == "quality-preflight":
            if (
                arguments.output_dir
                or arguments.run_id
                or arguments.fake_stop_after_intent is not None
            ):
                parser.error("Preflight cannot execute or create run directories")
        else:
            if (
                not arguments.output_dir
                or not arguments.run_id
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}", arguments.run_id)
            ):
                parser.error("Quality execution requires a safe output directory and run ID")
            if (
                arguments.fake_stop_after_intent is not None
                and not 1 <= arguments.fake_stop_after_intent <= 30
            ):
                parser.error("Fake interruption call must be between 1 and 30")
            if not live and arguments.destination_host is not None:
                parser.error("Fake execution has a fixed non-network destination")
        try:
            if arguments.mode == "quality-preflight":
                report = run_quality_preflight(arguments.destination_host or PROPOSED_DESTINATION)
            elif arguments.mode == "quality-smoke":
                report = collect_fake_quality_process(
                    Path(arguments.output_dir),
                    arguments.run_id,
                    stop_after_intent=arguments.fake_stop_after_intent,
                    streamlined=arguments.streamlined,
                )
            else:
                _quality_output_directory(Path(arguments.output_dir))
                report = execute_quality_entry(arguments)
        except (ValueError, OSError):
            parser.error("Quality inputs/configuration/evidence invalid; no automatic retry")
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if arguments.mode == "quality-smoke":
            return 0 if report["execution_status"] == "collected" else 1
        return 0 if report.get("status", "completed") == "completed" else 1
    if arguments.mode == "preflight":
        if arguments.confirm_live_provider_calls:
            parser.error("Provider-call confirmation is invalid in offline preflight mode")
        report = run_offline_preflight()
    else:
        if arguments.approved_token_budget is None or not arguments.confirm_live_provider_calls:
            parser.error(
                "Live mode requires --approved-token-budget and --confirm-live-provider-calls"
            )
        report = run_live_evaluation(
            arguments.approved_token_budget,
            confirm_live_provider_calls=arguments.confirm_live_provider_calls,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
