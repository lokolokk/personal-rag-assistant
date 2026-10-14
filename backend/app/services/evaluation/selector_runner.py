"""Offline preflight for a bounded live comparison of table-aware evidence selection."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from app.schemas.chat import ChatMessage
from app.services.evaluation.evidence_experiment import select_table_evidence_context
from app.services.evaluation.live_runner import (
    CHUNK_BATCH_SIZE,
    FRAMING_TOKENS_PER_ATTEMPT,
    INPUT_MULTIPLIER,
    MANIFEST,
    MAX_ATTEMPTS,
    PROJECT_ROOT,
    _case_titles,
    parse_source_chunks,
)
from app.services.evaluation.usage import (
    DEFAULT_OUTPUT_CAPS,
    LiveBudgetExceeded,
    Stage,
    conservative_reservation_tokens,
    estimate_tokens,
    live_token_budget,
    measure_usage,
)
from app.services.generation.grounding import (
    GRADING_PROMPT_TEMPLATE,
    build_grounding_prompt,
    format_context_chunks,
)
from app.services.retrieval.reformulator import build_reformulation_prompt

CASE_ID = "held-04-tada-llama-single-column"
BASELINE_TOP_K = 5
CANDIDATE_TOP_K = 10
QUERY_EMBEDDING_INPUT_HEADROOM = 256


def _get_live_settings() -> Any:
    from app.core.config import get_settings

    return get_settings()


def _load_preflight_context() -> tuple[Any, Any, list[Any]]:
    from app.services.evaluation.source_dataset import load_source_dataset

    dataset = load_source_dataset(MANIFEST, project_root=PROJECT_ROOT)
    case = next((item for item in dataset.cases if item.id == CASE_ID), None)
    if case is None or case.split != "held_out" or case.answerability != "answerable":
        raise ValueError("Selector comparison case must be a reserved answerable held-out case")
    if not case.scope_document_ids:
        raise ValueError("Selector comparison requires an explicit document scope")
    chunks = [
        chunk
        for chunk in parse_source_chunks(dataset)
        if chunk.metadata.document_id in case.scope_document_ids
    ]
    if not chunks:
        raise ValueError("Selector comparison scope has no parsed source chunks")
    return dataset, case, chunks


def _token_count(text: str) -> int:
    count = estimate_tokens(text)
    if count is None:
        raise ValueError("Offline selector preflight requires local cl100k_base estimates")
    return count


def _reserve(stage: Stage, input_tokens: int) -> int:
    return conservative_reservation_tokens(
        input_tokens,
        DEFAULT_OUTPUT_CAPS[stage],
        max_attempts=MAX_ATTEMPTS,
        input_multiplier=INPUT_MULTIPLIER,
        framing_tokens_per_attempt=FRAMING_TOKENS_PER_ATTEMPT,
    )


def _build_preflight_report(dataset: Any, case: Any, chunks: list[Any]) -> dict[str, Any]:
    settings = _get_live_settings()
    longest_chunks = sorted(chunks, key=lambda chunk: _token_count(chunk.content), reverse=True)
    reservations = dict.fromkeys(DEFAULT_OUTPUT_CAPS, 0)
    calls = dict.fromkeys(DEFAULT_OUTPUT_CAPS, 0)

    for offset in range(0, len(chunks), CHUNK_BATCH_SIZE):
        batch = chunks[offset : offset + CHUNK_BATCH_SIZE]
        batch_tokens = sum(_token_count(chunk.content) for chunk in batch)
        reservations["ingestion_embedding"] += _reserve("ingestion_embedding", batch_tokens)
        calls["ingestion_embedding"] += 1

    reservations["query_embedding"] = _reserve("query_embedding", QUERY_EMBEDDING_INPUT_HEADROOM)
    calls["query_embedding"] = 1

    history = [ChatMessage(role=turn.role, content=turn.content) for turn in case.history]
    reformulation_prompt = build_reformulation_prompt(
        active_document_titles=_case_titles(case, dataset),
        chat_history=history,
        user_query=case.question,
    )
    reservations["reformulation"] = _reserve("reformulation", _token_count(reformulation_prompt))
    calls["reformulation"] = 1

    per_arm: dict[str, dict[str, Any]] = {}
    for arm_name, context_count in (
        ("baseline", BASELINE_TOP_K),
        ("candidate", CANDIDATE_TOP_K),
    ):
        context_chunks = longest_chunks[:context_count]
        context = format_context_chunks(context_chunks)
        context_tokens = _token_count(context)
        grade_prompt = GRADING_PROMPT_TEMPLATE.format(
            query=case.retrieval_query,
            context=context,
        )
        generation_prompt = build_grounding_prompt(context, case.retrieval_query)
        arm_reservations: dict[str, int] = {}
        stage_prompts: list[tuple[Stage, str]] = [
            ("grading", grade_prompt),
            ("generation", generation_prompt),
        ]
        for stage, prompt in stage_prompts:
            prompt_tokens = _token_count(prompt) + DEFAULT_OUTPUT_CAPS["reformulation"]
            reservation = _reserve(stage, prompt_tokens)
            arm_reservations[stage] = reservation
            reservations[stage] += reservation
            calls[stage] += 1
        per_arm[arm_name] = {
            "context_chunk_upper_bound": context_count,
            "context_tokens_upper_bound": context_tokens,
            "stage_reservation_tokens": arm_reservations,
        }

    documents = {document.id: document for document in dataset.documents}
    scoped_documents = [documents[document_id] for document_id in case.scope_document_ids]
    corpus = [
        {
            "document_id": document.id,
            "filename": document.filename,
            "sha256": document.sha256,
            "page_count": document.page_count,
        }
        for document in scoped_documents
    ]
    return {
        "mode": "offline_selector_comparison_preflight",
        "dataset_id": dataset.dataset_id,
        "case_id": case.id,
        "split": case.split,
        "answerability": case.answerability,
        "scope_document_ids": case.scope_document_ids,
        "corpus": corpus[0] if len(corpus) == 1 else corpus,
        "chunk_count": len(chunks),
        "baseline_rule": "production-ranked-top-five-from-shared-top-ten-retrieval",
        "candidate_rule": "top_three_plus_query_matching_table_groups_per_document",
        "retrieval_candidate_count": CANDIDATE_TOP_K,
        "context_reservation_basis": "five/ten longest scoped chunks",
        "arms": per_arm,
        "model_configuration": {
            "chat_model": settings.chat_model,
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "api_base_host": urlsplit(settings.OPENAI_API_BASE or "").hostname,
            "dense_top_n": settings.DENSE_TOP_N,
            "sparse_top_n": settings.SPARSE_TOP_N,
            "rrf_k": settings.RRF_K,
            "baseline_top_k": BASELINE_TOP_K,
            "candidate_top_k": CANDIDATE_TOP_K,
            "max_attempts_per_provider_call": MAX_ATTEMPTS,
            "automatic_retries": MAX_ATTEMPTS - 1,
            "output_caps": DEFAULT_OUTPUT_CAPS,
            "maximum_provider_calls": sum(calls.values()),
            "concurrency": 1,
        },
        "calls_by_stage": calls,
        "reservation_by_stage_tokens": reservations,
        "total_conservative_reservation_tokens": sum(reservations.values()),
        "reservation_formula": (
            f"{MAX_ATTEMPTS} attempt(s): {INPUT_MULTIPLIER}x local cl100k text estimate + "
            f"{FRAMING_TOKENS_PER_ATTEMPT} framing tokens + stage output cap; candidate context "
            "is bounded by ten longest scoped chunks, baseline by five, with reformulation "
            "output headroom added to each grader/generator prompt."
        ),
        "provider_dispatches": 0,
        "uncertainties": [
            "The preflight uses longest scoped chunks because provider retrieval order is "
            "unknown offline.",
            "Local cl100k estimates omit provider framing and hidden reasoning; the multiplier "
            "is not a billing guarantee.",
            "Provider retries are disabled and observable; failed calls without usage remain "
            "unknown.",
            "No response quality or citation correctness is established by preflight.",
        ],
    }


def run_offline_preflight() -> dict[str, Any]:
    """Validate source identity and calculate a paired reservation without provider clients."""
    from app.services.evaluation.offline import offline_network

    with offline_network():
        dataset, case, chunks = _load_preflight_context()
        return _build_preflight_report(dataset, case, chunks)


def _comparison_contexts(retrieved: list[Any], query: str) -> tuple[list[Any], list[Any]]:
    if not retrieved:
        raise ValueError("Live selector comparison requires retrieved evidence")
    baseline = retrieved[:BASELINE_TOP_K]
    candidate = select_table_evidence_context(retrieved[:CANDIDATE_TOP_K], query=query)
    if not baseline or not candidate:
        raise ValueError("Both comparison arms must retain retrieved evidence")
    return baseline, candidate


def run_live_comparison(
    approved_token_allowance: int,
    *,
    confirm_live_provider_calls: bool = False,
) -> dict[str, Any]:
    """Run a paired selector comparison after explicit allowance and dispatch approval."""
    if not confirm_live_provider_calls or approved_token_allowance < 1:
        raise ValueError(
            "Live mode requires explicit provider-call authorization and a positive allowance"
        )

    preflight = run_offline_preflight()
    planned_reservation = preflight["total_conservative_reservation_tokens"]
    if approved_token_allowance < planned_reservation:
        raise ValueError(
            "Approved allowance is below the offline conservative reservation "
            f"of {planned_reservation} tokens"
        )

    settings = _get_live_settings()
    if not settings.OPENAI_API_KEY:
        raise ValueError("Live selector comparison is unavailable until an API key is configured")
    if settings.chat_model != preflight["model_configuration"]["chat_model"]:
        raise ValueError("Chat model changed after offline selector preflight")
    if settings.EMBEDDING_MODEL != preflight["model_configuration"]["embedding_model"]:
        raise ValueError("Embedding model changed after offline selector preflight")

    from dataclasses import asdict
    from datetime import UTC, datetime

    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    from app.schemas.chat import ChatMessage
    from app.services.evaluation.live_runner import (
        _case_report,
        _case_titles,
        _configure_budgetable_client,
        _usage_summary,
    )
    from app.services.generation.grounding import (
        FALLBACK_RESPONSE,
        GroundedGenerator,
        RetrievalGrader,
    )
    from app.services.retrieval.bm25_store import BM25Store
    from app.services.retrieval.hybrid import HybridRetriever
    from app.services.retrieval.reformulator import QueryReformulator
    from app.services.retrieval.vector_store import VectorStore

    dataset, case, chunks = _load_preflight_context()
    embedding_function = OpenAIEmbeddingFunction(
        api_key=settings.OPENAI_API_KEY,
        model_name=settings.EMBEDDING_MODEL,
        api_base=settings.OPENAI_API_BASE,
    )
    vector_store = VectorStore(
        client=chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False)),
        collection_name="task_5_9_1_selector_comparison",
        embedding_function=embedding_function,
    )
    bm25_store = BM25Store()
    retriever = HybridRetriever(vector_store, bm25_store, settings=settings, max_workers=2)
    reformulator = QueryReformulator()
    grader = RetrievalGrader()
    generator = GroundedGenerator()
    for label, client in (
        ("reformulator", reformulator.client),
        ("grader", grader.client),
        ("generator", generator.client),
        ("embedding adapter", embedding_function.client),
    ):
        _configure_budgetable_client(client, label=label)

    cases: list[dict[str, Any]] = []
    status = "completed"
    failure_type: str | None = None
    with (
        measure_usage() as trace,
        live_token_budget(
            approved_token_allowance,
            max_attempts=MAX_ATTEMPTS,
            input_multiplier=INPUT_MULTIPLIER,
            framing_tokens_per_attempt=FRAMING_TOKENS_PER_ATTEMPT,
            output_caps=DEFAULT_OUTPUT_CAPS,
        ) as budget,
    ):
        ingestion_end = 0
        try:
            vector_store.add_chunks(chunks)
            bm25_store.add_chunks(chunks)
            ingestion_end = len(trace.records)
            history = [ChatMessage(role=item.role, content=item.content) for item in case.history]
            standalone_query = reformulator.reformulate(
                user_query=case.question,
                chat_history=history,
                active_document_titles=_case_titles(case, dataset),
            )
            standalone_query = standalone_query.strip() or case.question.strip()
            retrieved = retriever.retrieve(
                query=standalone_query,
                document_ids=case.scope_document_ids,
                top_k=CANDIDATE_TOP_K,
            )
            baseline, candidate = _comparison_contexts(retrieved, standalone_query)
            for arm_name, arm_chunks in (("baseline", baseline), ("candidate", candidate)):
                record_start = len(trace.records)
                reservation_start = len(budget.reservations)
                grade = grader.grade_retrieval(standalone_query, arm_chunks)
                if grade == "grounded":
                    response, citations = generator.generate(standalone_query, arm_chunks)
                    origin = "provider_generation"
                else:
                    response, citations = FALLBACK_RESPONSE, []
                    origin = "application_fallback_after_insufficient_grade"
                arm_records = trace.records[record_start:]
                arm = _case_report(
                    case,
                    dataset,
                    {
                        "standalone_query": standalone_query,
                        "retrieved_chunks": arm_chunks,
                        "response_text": response,
                        "citations": citations,
                    },
                )
                arm.update(
                    {
                        "arm": arm_name,
                        "status": (
                            "provider_call_failed"
                            if any(record.outcome == "failed" for record in arm_records)
                            else "completed"
                        ),
                        "response_origin": origin,
                        "grade": grade,
                        "usage": _usage_summary(arm_records),
                        "budget_reservations": [
                            asdict(item) for item in budget.reservations[reservation_start:]
                        ],
                        "quality_review": "pending_manual_review",
                    }
                )
                cases.append(arm)
        except LiveBudgetExceeded:
            status = "stopped_at_pre_dispatch_budget_gate"
        except Exception as error:
            status = "comparison_execution_failed"
            failure_type = type(error).__name__
        if status == "completed" and any(
            arm.get("status") == "provider_call_failed" for arm in cases
        ):
            status = "provider_calls_failed"
        records = list(trace.records)
        ingestion_records = records[:ingestion_end]
        comparison_records = records[ingestion_end:]

    return {
        "mode": "live_selector_comparison",
        "status": status,
        "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "experiment_id": "task_5_9_1_selector_query_relevant_table_groups_v2",
        "dataset_id": dataset.dataset_id,
        "case_id": case.id,
        "question": case.question,
        "provider_dispatch_authorized": confirm_live_provider_calls,
        "approved_token_allowance": approved_token_allowance,
        "preflight_reservation_tokens": planned_reservation,
        "maximum_provider_calls": preflight["model_configuration"]["maximum_provider_calls"],
        "model_configuration": preflight["model_configuration"],
        "corpus": preflight["corpus"],
        "storage_identity": "ephemeral Chroma and in-memory BM25; no conversation persistence",
        "budget": {
            "reserved_tokens": budget.reserved_tokens,
            "remaining_unreserved_tokens": approved_token_allowance - budget.reserved_tokens,
            "reservation_by_stage_tokens": {
                stage: sum(
                    item.reserved_tokens for item in budget.reservations if item.stage == stage
                )
                for stage in DEFAULT_OUTPUT_CAPS
            },
            "output_caps": DEFAULT_OUTPUT_CAPS,
            "automatic_retries": MAX_ATTEMPTS - 1,
        },
        "ingestion_usage": _usage_summary(ingestion_records),
        "comparison_usage": _usage_summary(comparison_records),
        "arms": cases,
        "selection_calls": 0,
        "failure_type": failure_type,
        "quality_summary": {
            "manual_review_required": True,
            "total_tokens_per_successful_answer": None,
            "success_requires": "manual correctness, completeness, and citation review",
        },
        "provider_dispatches": sum(record.retry_visibility == "observed" for record in records),
        "usage_limitations": [
            "Provider-reported usage, local estimates, and unavailable attempts remain distinct.",
            "Failed attempts without provider usage remain unknown rather than zero.",
            "Reservation estimates are conservative planning, not a billing guarantee.",
        ],
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "live"), default="preflight")
    parser.add_argument("--approved-token-budget", type=int)
    parser.add_argument("--confirm-live-provider-calls", action="store_true")
    args = parser.parse_args()
    if args.mode == "preflight":
        if args.approved_token_budget is not None or args.confirm_live_provider_calls:
            parser.error("Live authorization flags are invalid in offline preflight mode")
        result = run_offline_preflight()
    else:
        if args.approved_token_budget is None or not args.confirm_live_provider_calls:
            parser.error("Live mode requires a budget and --confirm-live-provider-calls")
        result = run_live_comparison(
            args.approved_token_budget,
            confirm_live_provider_calls=args.confirm_live_provider_calls,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
