"""Read-only projection of the approved run; no runner or provider dependencies."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from app.services.evaluation.answer_quality import load_fresh_case
from app.services.evaluation.grounding_report import FreshCaseInput, LiveRunPlan, review_digest
from app.services.evaluation.report_store import (
    Digest,
    LiveReportManifest,
    LiveReportSummary,
    RecoveredCall,
    StoredCaseOutcome,
    recover_report,
)
from app.services.evaluation.usage import Stage

BASELINE_RUN_ID = "quality-five-20261008-04"
RUN_ID = "quality-five-20261010-06"
RUN_IDS = (
    "quality-five-20261008-02",
    "quality-five-20261008-03",
    "quality-five-20261008-04",
    "quality-five-20261009-05",
    "quality-five-20261010-06",
)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORT_ROOT = Path("backend/.eval-reports/task_6_1")
REVIEW_PATH = REPORT_ROOT / "full_pipeline_20261008/online-run-review.json"
REVIEW_PATHS = {
    RUN_IDS[1]: REPORT_ROOT / "output_limit_fix_20261008/online-run-review.json",
    BASELINE_RUN_ID: REVIEW_PATH,
}
STAGES: tuple[Stage, ...] = (
    "ingestion_embedding",
    "query_embedding",
    "reformulation",
    "grading",
    "generation",
)
NODES = {
    "reformulation": "reformulate_query",
    "grading": "grade_retrieval",
    "generation": "generate_grounded_response",
}


class Projection(BaseModel):
    model_config = ConfigDict(strict=True)


class StageUsage(Projection):
    stage: Stage
    calls: int
    known_provider_tokens: int
    exact_provider_tokens: int | None
    input_tokens: int | None
    output_tokens: int | None
    unknown_usage_calls: int
    synthetic_tokens: int
    input_estimate_tokens: int
    reserved_tokens: int
    seconds: float | None


class CaseReview(Projection):
    case: str
    verdict: str
    earliest_issue: str | None
    correctness: str | None = None
    qualification: str | None = None
    citation: str | None = None
    review: str | None = None


class SourceReview(Projection):
    run_id: str
    provider_calls: int
    exact_provider_tokens: int
    stages: dict[str, dict[str, int]]
    case_review: list[CaseReview]
    answerable_successes: str
    absence_successes: str
    run_file_sha256: dict[str, Digest]
    automatic_labels_limitation: str
    reasoning_finish_reason: str


class CaseTiming(Projection):
    total_seconds: float
    stages: dict[str, float]


class Timings(Projection):
    cases: dict[str, CaseTiming]
    ttft_seconds: float | None
    ingestion_seconds: float
    run_seconds: float


class Execution(Projection):
    status: str
    selected_case_ids: list[str]
    completed_case_ids: list[str]
    summary: LiveReportSummary
    provider_dispatches: int
    timings: Timings


class RunView(Projection):
    run_id: str
    complete: bool
    summary: LiveReportSummary
    configuration: LiveRunPlan
    stages: list[StageUsage]
    ingestion_tokens: int | None
    question_tokens: int | None
    timings: Timings | None
    automatic_cases: list[StoredCaseOutcome]
    review: SourceReview | None
    review_status: Literal["bound", "missing", "incomplete"]
    review_provenance: str
    verified_snapshots: int
    missing: list[str]
    limitations: list[str]
    questions: list[QuestionView]


class QuestionStage(Projection):
    node: str
    provenance: str
    snapshot: FreshCaseInput | None
    seconds: float | None


class QuestionView(Projection):
    case_id: str
    completed: bool
    stages: list[QuestionStage]
    usage: list[StageUsage]
    calls: list[RecoveredCall]
    tokens: int | None
    seconds: float | None


def _confined(root: Path, relative: str) -> Path:
    # Resolve Windows and POSIX separators, including paths stored by older Windows runs.
    parts = relative.replace("\\", "/").split("/")
    if any(part in ("", ".", "..") or ":" in part for part in parts):
        raise ValueError("Invalid evidence path")
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Escaped evidence path")
    return path


def _stage_usage(stage: Stage, calls: list[RecoveredCall], timings: Timings | None) -> StageUsage:
    selected = [c for c in calls if c.reservation.stage == stage and c.intent is not None]
    known = [
        c.outcome
        for c in selected
        if c.outcome is not None
        and c.outcome.source == "provider_reported"
        and c.outcome.total_tokens is not None
    ]
    unknown = sum(
        c.outcome is None
        or c.outcome.source == "unavailable"
        or c.outcome.total_tokens is None
        or c.state == "outcome_unknown"
        for c in selected
    )
    total = sum(c.total_tokens for c in known if c.total_tokens is not None)
    seconds = None
    if timings is not None:
        if stage == "ingestion_embedding":
            seconds = timings.ingestion_seconds
        elif stage == "reformulation":
            seconds = sum(
                c.stages.get("reformulate_query", 0.0) + c.stages.get("bypass_reformulation", 0.0)
                for c in timings.cases.values()
            )
        elif stage in NODES:
            seconds = sum(c.stages.get(NODES[stage], 0.0) for c in timings.cases.values())
    return StageUsage(
        stage=stage,
        calls=len(selected),
        known_provider_tokens=total,
        exact_provider_tokens=total if not unknown and len(known) == len(selected) else None,
        input_tokens=sum(c.input_tokens for c in known if c.input_tokens is not None)
        if not unknown
        and all(c.input_tokens is not None for c in known)
        and len(known) == len(selected)
        else None,
        output_tokens=sum(c.output_tokens for c in known if c.output_tokens is not None)
        if not unknown
        and all(c.output_tokens is not None for c in known)
        and len(known) == len(selected)
        else None,
        unknown_usage_calls=unknown,
        synthetic_tokens=sum(
            c.outcome.total_tokens or 0
            for c in selected
            if c.outcome is not None and c.outcome.source == "synthetic_usage"
        ),
        input_estimate_tokens=sum(c.reservation.input_tokens_estimate for c in selected),
        reserved_tokens=sum(
            c.reservation.reserved_tokens for c in calls if c.reservation.stage == stage
        ),
        seconds=seconds,
    )


def read_run(run_id: str, *, project_root: Path = PROJECT_ROOT) -> RunView:
    """Only a fixed approved run is readable. Missing artifacts are never fabricated."""
    if not re.fullmatch(r"[a-z0-9-]+", run_id) or run_id not in RUN_IDS:
        raise ValueError("Run is not approved for this station")
    reports = _confined(project_root, REPORT_ROOT.as_posix())
    run = _confined(reports, f"authorized_quality/{run_id}")
    recovered = recover_report(run, project_root=run)
    if not isinstance(recovered.manifest, LiveReportManifest) or not isinstance(
        recovered.summary, LiveReportSummary
    ):
        raise ValueError("Station requires a live durable manifest")
    plan = recovered.manifest.live_plan
    missing = [] if recovered.complete else ["Committed complete report unavailable"]
    execution_path = _confined(run, "execution.json")
    timings = None
    if execution_path.exists():
        envelope = TypeAdapter(dict[str, JsonValue]).validate_json(execution_path.read_bytes())
        if set(envelope) != {"sha256", "result"} or envelope["sha256"] != review_digest(
            envelope["result"]
        ):
            raise ValueError("Invalid execution checksum binding")
        execution = Execution.model_validate(envelope["result"])
        if (
            execution.summary != recovered.summary
            or execution.selected_case_ids != recovered.manifest.planned_case_keys
            or execution.completed_case_ids != [c.key for c in recovered.cases]
            or execution.provider_dispatches != recovered.summary.calls.potentially_dispatched_calls
        ):
            raise ValueError("Execution/report binding mismatch")
        timings = execution.timings
        if execution.status != "completed":
            missing.append("Execution not completed")
    else:
        missing.append("execution.json unavailable")
    verified = 0
    questions: list[QuestionView] = []
    snapshot_paths: set[str] = set()
    for case in recovered.cases:
        timeline: list[QuestionStage] = []
        case_calls = [c for c in recovered.calls if c.reservation.case_key == case.key]
        timing = timings.cases.get(case.case_id) if timings else None
        has_grading = any(c.reservation.stage == "grading" for c in recovered.calls)
        reformulation_node = (
            "bypass_reformulation"
            if (run / f"inputs/bypass_reformulation/{run_id}/{case.case_id}/input.json").exists()
            else "reformulate_query"
        )
        nodes = [
            "initial",
            reformulation_node,
            "scoped_hybrid_retriever",
        ]
        if (
            has_grading
            or (run / f"inputs/grade_retrieval/{run_id}/{case.case_id}/input.json").exists()
        ):
            nodes.append("grade_retrieval")
        generation_snapshot = (
            run / f"inputs/generate_grounded_response/{run_id}/{case.case_id}/input.json"
        )
        fallback_snapshot = run / f"inputs/inability_fallback/{run_id}/{case.case_id}/input.json"
        final_node = (
            "generate_grounded_response"
            if generation_snapshot.exists()
            else (
                "inability_fallback"
                if fallback_snapshot.exists()
                else (
                    "generate_grounded_response"
                    if case.result_kind == "answer"
                    else "inability_fallback"
                )
            )
        )
        nodes.append(final_node)
        for node in nodes:
            relative = f"inputs/{node}/{run_id}/{case.case_id}/input.json"
            snapshot_paths.add(relative)
            path = _confined(run, relative)
            snapshot = None
            if not path.exists():
                missing.append(f"{relative} unavailable")
                timeline.append(
                    QuestionStage(node=node, provenance=relative, snapshot=None, seconds=None)
                )
                continue
            snapshot = load_fresh_case(path)
            payloads = {
                c.reservation.stage: c.reservation.payload_sha256
                for c in recovered.calls
                if c.reservation.case_key == case.key
            }
            if (
                snapshot.run_id != run_id
                or snapshot.case_id != case.case_id
                or snapshot.settings_sha256 != plan.configuration_sha256
                or snapshot.source_dataset_sha256 != plan.dataset_sha256
                or snapshot.source_sha256 != plan.corpus_sha256
                or snapshot.prompt_sha256 != plan.prompt_sha256
                or any(payloads.get(s) != h for s, h in snapshot.request_sha256.items())
                or (
                    node == nodes[-1]
                    and (
                        snapshot.context_sha256 != case.context_sha256
                        or (
                            review_digest(snapshot.response_text or "") != case.answer_sha256
                            if case.answer_sha256 is not None
                            else bool(snapshot.response_text)
                        )
                        or [c.chunk_id for c in snapshot.retrieved_chunks or []]
                        != case.retrieved_chunk_ids
                        or [c.chunk_id for c in snapshot.supplied_chunks or []]
                        != case.supplied_chunk_ids
                    )
                )
            ):
                raise ValueError("Snapshot/artifact binding mismatch")
            verified += 1
            timeline.append(
                QuestionStage(
                    node=node,
                    provenance=relative,
                    snapshot=snapshot,
                    seconds=timing.stages.get(node) if timing else None,
                )
            )
        usage = [_stage_usage(s, case_calls, None) for s in STAGES[1:]]
        questions.append(
            QuestionView(
                case_id=case.case_id,
                completed=True,
                stages=timeline,
                usage=usage,
                calls=case_calls,
                tokens=sum(s.exact_provider_tokens or 0 for s in usage)
                if all(s.exact_provider_tokens is not None for s in usage)
                else None,
                seconds=timing.total_seconds if timing else None,
            )
        )
    stages = [_stage_usage(stage, recovered.calls, timings) for stage in STAGES]
    if (
        sum(s.known_provider_tokens for s in stages)
        != recovered.summary.calls.known_provider_tokens
    ):
        raise ValueError("Stage accounting does not reconcile")
    for key in recovered.manifest.planned_case_keys:
        if any(q.case_id == key for q in questions):
            continue
        calls = [c for c in recovered.calls if c.reservation.case_key == key]
        usage = [_stage_usage(s, calls, None) for s in STAGES[1:]]
        relative = f"inputs/initial/{run_id}/{key}/input.json"
        path = _confined(run, relative)
        snapshot = load_fresh_case(path) if path.exists() else None
        if snapshot is not None:
            if (
                snapshot.run_id != run_id
                or snapshot.case_id != key
                or snapshot.settings_sha256 != plan.configuration_sha256
                or snapshot.source_dataset_sha256 != plan.dataset_sha256
                or snapshot.source_sha256 != plan.corpus_sha256
                or snapshot.prompt_sha256 != plan.prompt_sha256
            ):
                raise ValueError("Interrupted snapshot binding mismatch")
            verified += 1
            snapshot_paths.add(relative)
        timing = timings.cases.get(key) if timings else None
        questions.append(
            QuestionView(
                case_id=key,
                completed=False,
                stages=[
                    QuestionStage(
                        node="initial", provenance=relative, snapshot=snapshot, seconds=None
                    )
                ],
                usage=usage,
                calls=calls,
                tokens=sum(s.exact_provider_tokens or 0 for s in usage)
                if any(c.intent for c in calls)
                and all(s.exact_provider_tokens is not None for s in usage)
                else None,
                seconds=timing.total_seconds if timing else None,
            )
        )
    questions.sort(key=lambda q: recovered.manifest.planned_case_keys.index(q.case_id))
    review_relative = REVIEW_PATHS.get(run_id)
    review_path = _confined(project_root, review_relative.as_posix()) if review_relative else None
    review = None
    review_status: Literal["bound", "missing", "incomplete"] = "missing"
    if review_path is not None and review_path.exists():
        data = TypeAdapter(dict[str, JsonValue]).validate_json(review_path.read_bytes())
        if run_id == RUN_IDS[1]:
            # The older closeout contains cohort findings only; never manufacture case reviews.
            data = {
                **data,
                "case_review": [],
                "absence_successes": data["absence_behavior"],
                "reasoning_finish_reason": data["reasoning_and_finish_reason"],
            }
        candidate = SourceReview.model_validate(data)
        if candidate.run_id != run_id:
            raise ValueError("Review run binding mismatch")
        required = {"calls.jsonl", "cases.jsonl", "manifest.json", "summary.json", "execution.json"}
        bindings = {key.replace("\\", "/") for key in candidate.run_file_sha256}
        if not (required | snapshot_paths).issubset(bindings):
            missing.append("Review artifact bindings unavailable")
        for relative, digest in candidate.run_file_sha256.items():
            path = _confined(run, relative)
            if not path.exists():
                missing.append(f"Review binding unavailable: {relative}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Review artifact binding mismatch")
        if not missing:
            if (
                candidate.provider_calls != recovered.summary.calls.potentially_dispatched_calls
                or candidate.exact_provider_tokens != recovered.summary.calls.exact_provider_tokens
                or (
                    run_id == BASELINE_RUN_ID
                    and (
                        {c.case for c in candidate.case_review} != {c.key for c in recovered.cases}
                        or len(candidate.case_review) != len(recovered.cases)
                    )
                )
                or any(
                    candidate.stages.get(s.stage)
                    != {
                        "calls": s.calls,
                        "input_tokens": s.input_tokens,
                        "output_tokens": s.output_tokens,
                        "total_tokens": s.exact_provider_tokens,
                    }
                    for s in stages
                    if run_id == BASELINE_RUN_ID or s.calls > 0
                )
            ):
                raise ValueError("Review accounting/case binding mismatch")
            review, review_status = candidate, "bound"
        else:
            review_status = "incomplete"
    question = [s.exact_provider_tokens for s in stages if s.stage != "ingestion_embedding"]
    return RunView(
        run_id=run_id,
        complete=recovered.complete and not missing,
        summary=recovered.summary,
        configuration=plan,
        stages=stages,
        ingestion_tokens=stages[0].exact_provider_tokens,
        question_tokens=sum(t for t in question if t is not None)
        if all(t is not None for t in question)
        else None,
        timings=timings,
        automatic_cases=recovered.cases,
        review=review,
        review_status=review_status,
        review_provenance=review_relative.as_posix()
        if review_relative
        else "No bound source closeout",
        verified_snapshots=verified,
        questions=questions,
        missing=list(dict.fromkeys(missing)),
        limitations=[
            "Historical provider usage; viewing dispatches zero calls. Tokens are not charges.",
            "Reservations and local input estimates are not actual consumption.",
            "Finish reasons, reasoning-token splits, exact wire captures, "
            "wider retrieval candidates "
            "and streamed TTFT unavailable. Query embedding time is inside retrieval time.",
            "Evaluation ingestion is not an application baseline. Changed historical settings "
            "do not establish controlled savings.",
            "Earlier failed and historical held-out usage/outcomes remain unknown; never retry "
            "or reconstruct. Combined charges unavailable.",
            "Task 5.9.1 and Task 6.1 quality gate remain open.",
        ],
    )
