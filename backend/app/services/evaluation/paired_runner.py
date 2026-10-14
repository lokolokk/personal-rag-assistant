"""Explicitly gated runner for the frozen rank-one context comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.core.config import get_settings
from app.services.evaluation.evidence_experiment import preflight_frozen_response_comparison
from app.services.evaluation.live_runner import (
    FRAMING_TOKENS_PER_ATTEMPT,
    INPUT_MULTIPLIER,
    MANIFEST,
    MAX_ATTEMPTS,
    PROJECT_ROOT,
    _configure_budgetable_client,
    _usage_summary,
    parse_source_chunks,
)
from app.services.evaluation.usage import (
    DEFAULT_OUTPUT_CAPS,
    LiveBudgetExceeded,
    UsageRecord,
    live_token_budget,
    measure_usage,
)
from app.services.generation.grounding import (
    FALLBACK_RESPONSE,
    GroundedGenerator,
    RetrievalGrader,
)

EXPERIMENT_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "evaluation"
    / "fixtures"
    / "task_5_9_1_frozen_evidence_ablation.json"
)


def _load_frozen_comparison() -> tuple[Any, dict[str, Any], Any, list[Any], list[Any]]:
    from app.services.evaluation.source_dataset import load_source_dataset

    dataset = load_source_dataset(MANIFEST, project_root=PROJECT_ROOT)
    experiment = json.loads(EXPERIMENT_FIXTURE.read_text(encoding="utf-8"))
    if dataset.dataset_id != experiment["dataset_id"]:
        raise ValueError("Frozen response comparison dataset identity changed")

    document = next(
        (item for item in dataset.documents if item.id == experiment["corpus"]["document_id"]),
        None,
    )
    if document is None or document.sha256 != experiment["corpus"]["sha256"]:
        raise ValueError("Frozen response comparison source PDF identity changed")

    chunks = parse_source_chunks(dataset)
    if len(chunks) != experiment["corpus"]["chunk_count"]:
        raise ValueError("Frozen response comparison chunk count changed")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    frozen_rows = experiment["baseline_retrieval"]
    baseline_ids = [row["chunk_id"] for row in frozen_rows]
    if [row["rank"] for row in frozen_rows] != list(range(1, len(frozen_rows) + 1)):
        raise ValueError("Frozen baseline retrieval ranks must remain contiguous and ordered")

    baseline_chunks: list[Any] = []
    for row in frozen_rows:
        chunk = chunks_by_id.get(row["chunk_id"])
        if chunk is None:
            raise ValueError(f"Frozen retrieval chunk no longer exists: {row['chunk_id']}")
        fingerprint = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
        if len(chunk.content) != row["content_chars"] or fingerprint != row["source_sha256"]:
            raise ValueError(f"Frozen retrieval content changed: {row['chunk_id']}")
        baseline_chunks.append(chunk)

    candidate_ids = experiment["candidate"]["chunk_ids"]
    if candidate_ids != baseline_ids[:1]:
        raise ValueError("The frozen diagnostic candidate must keep only baseline rank one")
    case = next((item for item in dataset.cases if item.id == experiment["case_id"]), None)
    if case is None or case.split != "diagnostic" or case.answerability != "answerable":
        raise ValueError("Frozen response comparison requires its answerable diagnostic case")
    if case.question != experiment["question"]:
        raise ValueError("Frozen response comparison question changed")

    candidate_id_set = set(candidate_ids)
    candidate_chunks = [chunk for chunk in baseline_chunks if chunk.chunk_id in candidate_id_set]
    return dataset, experiment, case, baseline_chunks, candidate_chunks


def _preflight_report(
    dataset: Any,
    experiment: dict[str, Any],
    case: Any,
    baseline_chunks: list[Any],
) -> dict[str, Any]:
    preflight = preflight_frozen_response_comparison(
        query=case.question,
        baseline_chunks=baseline_chunks,
        candidate_chunk_ids=experiment["candidate"]["chunk_ids"],
    )
    return {
        "mode": "offline_frozen_evidence_preflight",
        "experiment_id": experiment["experiment_id"],
        "run_id": experiment["run_id"],
        "dataset_id": dataset.dataset_id,
        "case_id": case.id,
        "question": case.question,
        "answerability": case.answerability,
        "required_facts": [
            {
                "id": fact.id,
                "claim": fact.claim,
                "exact_values": fact.exact_values,
                "unit": fact.unit,
                "qualifications": fact.qualifications,
            }
            for fact in case.required_facts
        ],
        "corpus": experiment["corpus"],
        "frozen_baseline_chunk_ids": [chunk.chunk_id for chunk in baseline_chunks],
        "candidate_chunk_ids": experiment["candidate"]["chunk_ids"],
        "candidate_rule": experiment["candidate"]["selection_rule"],
        "query_policy": preflight["query_policy"],
        "query_assumption": (
            "The original user question is used for both arms; the live reformulator output "
            "was not retained."
        ),
        "chat_model": preflight["chat_model"],
        "temperature": preflight["temperature"],
        "maximum_provider_calls": preflight["maximum_provider_calls"],
        "preflight": preflight,
        "provider_dispatches": 0,
    }


def run_offline_preflight() -> dict[str, Any]:
    """Verify the pinned case/context and print a reservation without clients."""
    from app.services.evaluation.offline import offline_network

    with offline_network():
        dataset, experiment, case, baseline_chunks, _candidate_chunks = _load_frozen_comparison()
        return _preflight_report(dataset, experiment, case, baseline_chunks)


def run_live_comparison(
    approved_token_allowance: int,
    *,
    confirm_live_provider_calls: bool,
) -> dict[str, Any]:
    """Run both frozen context arms under an explicit cumulative token allowance."""
    if not confirm_live_provider_calls or approved_token_allowance < 1:
        raise ValueError(
            "Live mode requires explicit provider-call authorization and a positive allowance"
        )

    dataset, experiment, case, baseline_chunks, candidate_chunks = _load_frozen_comparison()
    preflight = _preflight_report(dataset, experiment, case, baseline_chunks)
    planned_reservation = preflight["preflight"]["total_conservative_reservation_tokens"]
    if approved_token_allowance < planned_reservation:
        raise ValueError(
            "Approved allowance is below the offline conservative reservation "
            f"of {planned_reservation} tokens"
        )

    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        raise ValueError("Live comparison is unavailable until an API key is configured")
    if settings.chat_model != preflight["chat_model"]:
        raise ValueError("Chat model changed after the offline preflight")
    grader = RetrievalGrader()
    generator = GroundedGenerator()
    for label, client in [("grader", grader.client), ("generator", generator.client)]:
        _configure_budgetable_client(client, label=label)

    arms: list[dict[str, Any]] = []
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
        for arm_name, chunks in (
            ("baseline", baseline_chunks),
            ("candidate", candidate_chunks),
        ):
            arm_preflight = preflight["preflight"]["arms"][arm_name]
            record_start = len(trace.records)
            reservation_start = len(budget.reservations)
            try:
                grade = grader.grade_retrieval(case.question, chunks)
                if grade == "grounded":
                    response, citations = generator.generate(case.question, chunks)
                else:
                    response, citations = FALLBACK_RESPONSE, []
            except LiveBudgetExceeded:
                status = "stopped_at_pre_dispatch_budget_gate"
                arms.append({"name": arm_name, "status": status})
                break
            except Exception as error:
                status = "arm_execution_failed"
                failure_type = type(error).__name__
                arms.append({"name": arm_name, "status": status})
                break

            arm_records = trace.records[record_start:]
            provider_failed = any(record.outcome == "failed" for record in arm_records)
            generation_succeeded = any(
                record.stage == "generation" and record.outcome == "succeeded"
                for record in arm_records
            )
            if provider_failed:
                arm_status = "provider_call_failed"
                response_origin = "application_fallback_after_provider_error"
                quality_review = "not_model_output_due_to_provider_error"
            elif generation_succeeded:
                arm_status = "completed"
                response_origin = "provider_generation"
                quality_review = "manual_review_required"
            else:
                arm_status = "completed"
                response_origin = "application_fallback_after_insufficient_grade"
                quality_review = "not_model_output_after_insufficient_grade"

            arms.append(
                {
                    "name": arm_name,
                    "status": arm_status,
                    "response_origin": response_origin,
                    "chunk_ids": [chunk.chunk_id for chunk in chunks],
                    "context_tokens_local_estimate": arm_preflight["context_tokens"],
                    "preflight_stage_reservations": arm_preflight["stage_reservations"],
                    "grade": grade,
                    "response_text": response,
                    "is_exact_fallback": response.strip() == FALLBACK_RESPONSE,
                    "citations": citations,
                    "citations_in_scope": all(
                        citation.get("document_id") in case.scope_document_ids
                        for citation in citations
                    ),
                    "usage": _usage_summary(arm_records),
                    "budget_reservations": [
                        asdict(reservation)
                        for reservation in budget.reservations[reservation_start:]
                    ],
                    "quality_review": quality_review,
                }
            )

    if status == "completed" and any(arm["status"] == "provider_call_failed" for arm in arms):
        status = "provider_calls_failed"

    trace_records: list[UsageRecord] = list(trace.records)
    provider_generated_count = sum(
        arm.get("response_origin") == "provider_generation" for arm in arms
    )
    review_status = (
        "pending_manual_review"
        if provider_generated_count
        else "not_applicable_no_provider_generated_answer"
    )
    return {
        "mode": "live_frozen_evidence_comparison",
        "status": status,
        "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "experiment_id": experiment["experiment_id"],
        "dataset_id": dataset.dataset_id,
        "case_id": case.id,
        "question": case.question,
        "query_policy": preflight["query_policy"],
        "query_assumption": preflight["query_assumption"],
        "provider_dispatch_authorized": confirm_live_provider_calls,
        "approved_token_allowance": approved_token_allowance,
        "preflight_reservation_tokens": planned_reservation,
        "maximum_provider_calls": preflight["maximum_provider_calls"],
        "corpus": experiment["corpus"],
        "model_configuration": {
            "chat_model": settings.chat_model,
            "api_base_host": urlsplit(settings.OPENAI_API_BASE or "").hostname,
            "temperature": 0.0,
            "maximum_attempts": MAX_ATTEMPTS,
            "automatic_retries": MAX_ATTEMPTS - 1,
        },
        "budget": {
            "reserved_tokens": budget.reserved_tokens,
            "remaining_unreserved_tokens": approved_token_allowance - budget.reserved_tokens,
            "reservation_count": len(budget.reservations),
            "reservation_by_stage_tokens": {
                stage: sum(
                    reservation.reserved_tokens
                    for reservation in budget.reservations
                    if reservation.stage == stage
                )
                for stage in DEFAULT_OUTPUT_CAPS
            },
            "output_caps": DEFAULT_OUTPUT_CAPS,
        },
        "usage": _usage_summary(trace_records),
        "arms": arms,
        "quality_summary": {
            "answerable_refusal_candidates": sum(
                arm.get("response_origin") == "provider_generation"
                and arm.get("is_exact_fallback", False)
                for arm in arms
            ),
            "manual_correctness_reviews_pending": provider_generated_count,
            "total_tokens_per_successful_answer": None,
            "success_requires": "manual correctness, completeness, and citation review",
        },
        "failure_type": failure_type,
        "manual_review": {
            "required_facts": preflight["required_facts"],
            "correctness": review_status,
            "completeness": review_status,
            "citation_grounding": review_status,
            "answerable_refusal": (
                "review_provider_generated_answers_only"
                if provider_generated_count
                else review_status
            ),
            "scope_integrity": "frozen single-document corpus",
        },
        "selection_calls": 0,
        "usage_limitations": [
            "Provider-reported usage, local estimates, and unavailable consumption stay distinct.",
            "A failed call without returned usage remains unknown rather than zero.",
            "The local reservation is conservative planning, not an exact billing cap.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "live"), default="preflight")
    parser.add_argument("--approved-token-budget", type=int)
    parser.add_argument("--confirm-live-provider-calls", action="store_true")
    arguments = parser.parse_args()
    if arguments.mode == "preflight":
        if arguments.confirm_live_provider_calls or arguments.approved_token_budget is not None:
            parser.error("Live authorization flags are invalid in offline preflight mode")
        report = run_offline_preflight()
    else:
        if arguments.approved_token_budget is None or not arguments.confirm_live_provider_calls:
            parser.error(
                "Live mode requires --approved-token-budget and --confirm-live-provider-calls"
            )
        report = run_live_comparison(
            arguments.approved_token_budget,
            confirm_live_provider_calls=arguments.confirm_live_provider_calls,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
