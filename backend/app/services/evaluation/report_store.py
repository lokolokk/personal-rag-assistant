"""Durable evaluation journals and honest aggregation; never dispatch or resume calls.

Artifacts intentionally omit free-form prompts, answers, quotes, and exception messages.
Live capture is opt-in; the existing runner is not wired to it in Slice J.
"""

import hashlib
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from app.services.evaluation.grounding_report import (
    CallDispatchIntent,
    CallEvent,
    CallOutcome,
    CallReservation,
    CaseEvaluationRecord,
    EvidenceLevel,
    FactVerdict,
    FailureCode,
    LiveRunPlan,
    ResultKind,
)
from app.services.evaluation.usage import Source, Stage, UsageRecord

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
TokenCount = Annotated[int, Field(ge=0)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ReportManifest(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: Identifier
    suite_id: Identifier
    created_at: str
    planned_case_keys: list[Identifier]


class LiveReportManifest(_StrictModel):
    schema_version: Literal["2.0"] = "2.0"
    run_id: Identifier
    suite_id: Identifier
    created_at: str
    planned_case_keys: list[Identifier]
    live_plan: LiveRunPlan


class RecoveredCall(_StrictModel):
    reservation: CallReservation
    intent: CallDispatchIntent | None = None
    outcome: CallOutcome | None = None
    state: Literal[
        "reserved",
        "potentially_dispatched",
        "not_dispatched",
        "succeeded",
        "failed",
        "outcome_unknown",
    ] = "reserved"


class CallLedgerSummary(_StrictModel):
    reserved_tokens: int
    potentially_dispatched_calls: int
    unknown_usage_calls: int
    known_provider_tokens: int
    exact_provider_tokens: int | None
    synthetic_tokens: int


class _ManifestBinding(_StrictModel):
    kind: Literal["manifest_binding"] = "manifest_binding"
    manifest_sha256: Digest


_MANIFEST: TypeAdapter[ReportManifest | LiveReportManifest] = TypeAdapter(
    Annotated[ReportManifest | LiveReportManifest, Field(discriminator="schema_version")]
)
_CALL_EVENT: TypeAdapter[CallEvent] = TypeAdapter(CallEvent)


class StoredCaseOutcome(_StrictModel):
    case_id: Identifier
    arm_name: Literal["baseline", "candidate"] | None
    source_run_id: Identifier | None
    evidence_level: EvidenceLevel
    result_kind: ResultKind
    failure_codes: list[FailureCode]
    earliest_failing_stage: FailureCode | None
    fact_verdicts: dict[Identifier, FactVerdict]
    claim_support: FactVerdict
    review_complete: bool
    attribution_verdict: FactVerdict | None
    refusal_verdict: FactVerdict | None
    successful_provider_answer: bool
    answer_sha256: Digest | None
    context_sha256: Digest | None
    source_dataset_sha256: Digest | None
    retrieved_chunk_ids: list[Identifier]
    supplied_chunk_ids: list[Identifier]

    @property
    def key(self) -> str:
        return self.case_id if self.arm_name is None else f"{self.case_id}:{self.arm_name}"


class StoredUsage(_StrictModel):
    origin: Literal["current", "historical"]
    source_run_id: Identifier
    call_id: Annotated[int, Field(ge=1)]
    attempt: Annotated[int, Field(ge=1)]
    case_key: Identifier | None
    stage: Stage
    source: Source
    total_tokens: TokenCount | None = None
    local_only: bool = False
    outcome: Literal["dispatched", "succeeded", "failed"] = "succeeded"

    @classmethod
    def from_observed(
        cls,
        record: UsageRecord,
        *,
        origin: Literal["current", "historical"],
        source_run_id: str,
        case_key: str | None,
    ) -> "StoredUsage":
        """Project existing accounting data without serializing model/error/prompt text."""
        return cls(
            origin=origin,
            source_run_id=source_run_id,
            call_id=record.call_id,
            attempt=record.attempt,
            case_key=case_key,
            stage=record.stage,
            source=record.source,
            total_tokens=record.total_tokens,
            local_only=record.local_only,
            outcome=record.outcome,
        )

    @property
    def identity(self) -> tuple[str, str, int, int]:
        return self.origin, self.source_run_id, self.call_id, self.attempt


class UsageTotals(_StrictModel):
    known_provider_tokens: int
    exact_provider_tokens: int | None
    unknown_provider_attempts: int
    unavailable_case_usage: int
    local_estimate_tokens: int
    local_only_attempts: int
    tokens_per_successful_answer: float | None


class ReportSummary(_StrictModel):
    run_id: Identifier
    terminal_state: Literal["completed", "incomplete"]
    total_planned_cases: int
    completed_cases: int
    failed_cases: int
    unreviewed_cases: int
    successful_provider_answers: int
    evidence_counts: dict[EvidenceLevel, int]
    failure_counts: dict[FailureCode, int]
    current: UsageTotals
    historical: UsageTotals
    journal_sha256: Digest
    manifest_sha256: Digest


class LiveReportSummary(ReportSummary):
    schema_version: Literal["2.0"] = "2.0"
    calls: CallLedgerSummary
    calls_sha256: Digest


class _CaseEvent(_StrictModel):
    kind: Literal["case"] = "case"
    outcome: StoredCaseOutcome


class _UsageEvent(_StrictModel):
    kind: Literal["usage"] = "usage"
    usage: StoredUsage


_EVENT: TypeAdapter[_CaseEvent | _UsageEvent] = TypeAdapter(
    Annotated[_CaseEvent | _UsageEvent, Field(discriminator="kind")]
)


class RecoveredReport(_StrictModel):
    manifest: ReportManifest | LiveReportManifest
    cases: list[StoredCaseOutcome]
    usage: list[StoredUsage]
    summary: LiveReportSummary | ReportSummary
    complete: bool
    truncated_final_line: bool
    calls: list[RecoveredCall] = Field(default_factory=list)


def _project_case(record: CaseEvaluationRecord) -> StoredCaseOutcome:
    if (
        record.split == "held_out"
        or record.case_id.startswith("held-")
        or record.exposure_status == "invoked_result_unknown"
    ):
        raise ValueError("Held-out/unknown execution cannot become an offline case result")
    review = record.semantic_review
    decisions = [] if review is None else list(review.facts.values())
    if review is not None:
        decisions.extend(item for item in (review.attribution, review.refusal) if item is not None)
    review_complete = (
        review is not None
        and record.response_text is not None
        and record.response_hash == hashlib.sha256(record.response_text.encode("utf-8")).hexdigest()
        and review.answer_sha256 == record.response_hash
        and review.case_id == record.case_id
        and review.run_id == record.run_id
        and review.context_sha256 == record.context_sha256
        and review.source_dataset_sha256 == record.source_dataset_sha256
        and record.source_dataset_sha256 is not None
        and review.attribution is not None
        and review.refusal is not None
        and set(review.facts) == {fact.fact_id for fact in record.facts}
        and all(item.verdict in ("pass", "fail") for item in decisions)
    )
    success = (
        record.evidence_level == "provider_answer_review"
        and review_complete
        and record.context_sha256 is not None
        and record.result_kind in ("answer", "appropriate_abstention")
        and not record.failure_codes
        and record.claim_support == "pass"
        and all(fact.verdict == "pass" for fact in record.facts)
        and all(item.verdict == "pass" for item in decisions)
    )
    return StoredCaseOutcome(
        case_id=record.case_id,
        source_run_id=record.run_id,
        arm_name=record.arm_name,
        evidence_level=record.evidence_level,
        result_kind=record.result_kind,
        failure_codes=record.failure_codes,
        earliest_failing_stage=record.earliest_failing_stage,
        fact_verdicts={fact.fact_id: fact.verdict for fact in record.facts},
        claim_support=record.claim_support,
        review_complete=review_complete,
        attribution_verdict=(
            review.attribution.verdict
            if review is not None and review.attribution is not None
            else None
        ),
        refusal_verdict=(
            review.refusal.verdict if review is not None and review.refusal is not None else None
        ),
        successful_provider_answer=success,
        answer_sha256=record.response_hash,
        context_sha256=record.context_sha256,
        source_dataset_sha256=record.source_dataset_sha256,
        retrieved_chunk_ids=record.retrieved_chunk_ids,
        supplied_chunk_ids=record.supplied_chunk_ids,
    )


def _totals(usage: list[StoredUsage], cases: list[StoredCaseOutcome]) -> UsageTotals:
    provider = [item for item in usage if not item.local_only]
    known = sum(item.total_tokens or 0 for item in provider if item.source == "provider_reported")
    unknown = sum(
        item.source != "provider_reported"
        or item.total_tokens is None
        or item.outcome == "dispatched"
        for item in provider
    )
    provider_cases = [case for case in cases if case.evidence_level == "provider_answer_review"]
    missing = sum(
        not any(
            item.case_key == case.key and item.source_run_id == case.source_run_id
            for item in provider
        )
        for case in provider_cases
    )
    successes = [case for case in provider_cases if case.successful_provider_answer]
    exact = known if unknown == 0 and missing == 0 else None
    return UsageTotals(
        known_provider_tokens=known,
        exact_provider_tokens=exact,
        unknown_provider_attempts=unknown,
        unavailable_case_usage=missing,
        local_estimate_tokens=sum(
            item.total_tokens or 0 for item in usage if item.source == "local_estimate"
        ),
        local_only_attempts=sum(item.local_only for item in usage),
        tokens_per_successful_answer=(
            exact / len(successes) if exact is not None and successes else None
        ),
    )


def _summary(
    manifest: ReportManifest | LiveReportManifest,
    cases: list[StoredCaseOutcome],
    usage: list[StoredUsage],
    journal: bytes,
    *,
    truncated: bool = False,
) -> ReportSummary:
    evidence: dict[EvidenceLevel, int] = {}
    failures: dict[FailureCode, int] = {}
    for case in cases:
        evidence[case.evidence_level] = evidence.get(case.evidence_level, 0) + 1
        for code in set(case.failure_codes):
            failures[code] = failures.get(code, 0) + 1
    return ReportSummary(
        run_id=manifest.run_id,
        terminal_state=(
            "completed"
            if not truncated and len(cases) == len(manifest.planned_case_keys)
            else "incomplete"
        ),
        total_planned_cases=len(manifest.planned_case_keys),
        completed_cases=len(cases),
        failed_cases=sum(
            bool(case.failure_codes)
            or "fail" in case.fact_verdicts.values()
            or case.claim_support == "fail"
            or case.attribution_verdict == "fail"
            or case.refusal_verdict == "fail"
            or case.result_kind in ("execution_error", "unsupported_answer", "incorrect_refusal")
            for case in cases
        ),
        unreviewed_cases=sum(
            case.evidence_level == "provider_answer_review" and not case.review_complete
            for case in cases
        ),
        successful_provider_answers=sum(case.successful_provider_answer for case in cases),
        evidence_counts=evidence,
        failure_counts=failures,
        current=_totals(
            [item for item in usage if item.origin == "current"],
            [case for case in cases if case.source_run_id == manifest.run_id],
        ),
        historical=_totals(
            [item for item in usage if item.origin == "historical"],
            [case for case in cases if case.source_run_id != manifest.run_id],
        ),
        journal_sha256=hashlib.sha256(journal).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest.model_dump_json().encode("utf-8")).hexdigest(),
    )


def _inside(path: Path, project_root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(project_root.resolve()):
        raise ValueError("Report path must remain inside the project")
    return resolved


def _call_states(
    manifest: LiveReportManifest,
    events: list[CallEvent],
) -> list[RecoveredCall]:
    """Validate transitions, identities and limits independently of a live client."""
    calls: dict[int, RecoveredCall] = {}
    seen: dict[tuple[int, str], CallEvent] = {}
    reserved = 0
    for event in events:
        if event.run_id != manifest.run_id:
            raise ValueError("Call run identity mismatch")
        if event.case_key is None:
            if event.stage != "ingestion_embedding":
                raise ValueError("Only shared ingestion can omit a case key")
        elif event.case_key not in manifest.planned_case_keys:
            raise ValueError("Unplanned call case")
        elif event.stage == "ingestion_embedding":
            raise ValueError("Ingestion must use shared identity")
        key = event.call_id, event.kind
        if key in seen:
            if seen[key] != event:
                raise ValueError("Conflicting duplicate call event")
            continue
        seen[key] = event
        call = calls.get(event.call_id)
        if isinstance(event, CallReservation):
            if event.output_tokens_cap != manifest.live_plan.output_caps[event.stage]:
                raise ValueError("Call output cap differs from frozen manifest")
            reserved += event.reserved_tokens
            if reserved > manifest.live_plan.allowance_tokens:
                raise ValueError("Call reservations exceed frozen allowance")
            if len(calls) >= manifest.live_plan.maximum_calls:
                raise ValueError("Call reservations exceed frozen call ceiling")
            calls[event.call_id] = RecoveredCall(reservation=event)
            continue
        if call is None:
            raise ValueError("Call event requires preceding reservation")
        if event.identity != call.reservation.identity:
            raise ValueError("Call identity changed after reservation")
        if call.outcome is not None:
            raise ValueError("Call is terminal")
        if isinstance(event, CallDispatchIntent):
            call.intent = event
            call.state = "potentially_dispatched"
        else:
            if event.outcome == "not_dispatched":
                if call.intent is not None:
                    raise ValueError("Dispatch intent cannot be undone as not dispatched")
            elif call.intent is None:
                raise ValueError("Call outcome requires preceding dispatch intent")
            expected = (
                "provider_reported"
                if manifest.live_plan.usage_evidence == "provider"
                else "synthetic_usage"
            )
            if event.source not in (expected, "unavailable"):
                raise ValueError("Call usage evidence differs from frozen manifest")
            call.outcome = event
            call.state = event.outcome
    return list(calls.values())


def _read_calls(
    directory: Path,
    manifest: LiveReportManifest,
    project_root: Path,
) -> tuple[list[CallEvent], bytes, bool]:
    try:
        journal = _inside(directory / "calls.jsonl", project_root).read_bytes()
        lines = journal.splitlines(keepends=True)
        binding = _ManifestBinding.model_validate_json(lines[0])
    except (OSError, ValidationError, IndexError):
        raise ValueError("Invalid/missing live call journal binding") from None
    digest = hashlib.sha256(manifest.model_dump_json().encode("utf-8")).hexdigest()
    if not lines[0].endswith(b"\n") or binding.manifest_sha256 != digest:
        raise ValueError("Live manifest does not match call journal binding")
    truncated = not lines[-1].endswith(b"\n")
    events: list[CallEvent] = []
    for number, line in enumerate(lines[1:], 2):
        if truncated and number == len(lines):
            break
        try:
            events.append(_CALL_EVENT.validate_json(line))
        except ValidationError:
            raise ValueError(f"Invalid call journal line {number}") from None
    return events, journal, truncated


def _live_summary(
    summary: ReportSummary,
    calls: list[RecoveredCall],
    journal: bytes,
    cases: list[StoredCaseOutcome],
    manifest: LiveReportManifest,
) -> LiveReportSummary:
    dispatched = [call for call in calls if call.intent is not None]
    unknown = sum(
        call.outcome is None
        or call.state == "outcome_unknown"
        or call.outcome.source == "unavailable"
        or call.outcome.total_tokens is None
        for call in dispatched
    )
    known = sum(
        call.outcome.total_tokens or 0
        for call in dispatched
        if call.outcome is not None and call.outcome.source == "provider_reported"
    )
    synthetic = sum(
        call.outcome.total_tokens or 0
        for call in dispatched
        if call.outcome is not None and call.outcome.source == "synthetic_usage"
    )
    terminal = summary.terminal_state
    if unknown or any(
        call.state in ("reserved", "potentially_dispatched", "outcome_unknown") for call in calls
    ):
        terminal = "incomplete"
    provider_run = manifest.live_plan.usage_evidence == "provider"
    current_cases = [
        case
        for case in cases
        if case.source_run_id == summary.run_id and case.evidence_level == "provider_answer_review"
    ]
    missing = (
        sum(
            not any(call.reservation.case_key == case.key for call in dispatched)
            for case in current_cases
        )
        if provider_run
        else 0
    )
    provider_unknown = unknown if provider_run else 0
    exact = None if provider_unknown or missing else known
    successes = (
        sum(case.successful_provider_answer for case in current_cases) if provider_run else 0
    )
    current = UsageTotals(
        known_provider_tokens=known,
        exact_provider_tokens=exact,
        unknown_provider_attempts=provider_unknown,
        unavailable_case_usage=missing,
        local_estimate_tokens=0,
        local_only_attempts=0,
        tokens_per_successful_answer=exact / successes if exact is not None and successes else None,
    )
    return LiveReportSummary(
        **{**summary.model_dump(), "terminal_state": terminal, "current": current},
        calls=CallLedgerSummary(
            reserved_tokens=sum(call.reservation.reserved_tokens for call in calls),
            potentially_dispatched_calls=len(dispatched),
            unknown_usage_calls=unknown,
            known_provider_tokens=known,
            exact_provider_tokens=None if unknown else known,
            synthetic_tokens=synthetic,
        ),
        calls_sha256=hashlib.sha256(journal).hexdigest(),
    )


class ReportStore:
    """Single-writer run; existing runs are recoverable read-only, never resumed."""

    def __init__(
        self,
        run_dir: Path,
        manifest: ReportManifest | LiveReportManifest,
        project_root: Path,
    ) -> None:
        self.run_dir = run_dir
        self.manifest = manifest
        self.project_root = project_root
        self._finalized = False
        self._write_failed = False
        self._cases_sha256 = hashlib.sha256((run_dir / "cases.jsonl").read_bytes()).hexdigest()
        self._calls_sha256 = (
            hashlib.sha256((run_dir / "calls.jsonl").read_bytes()).hexdigest()
            if isinstance(manifest, LiveReportManifest)
            else None
        )

    @classmethod
    def create(
        cls,
        *,
        project_root: Path,
        output_dir: Path,
        suite_id: str,
        planned_case_keys: list[str],
        run_id: str | None = None,
        live_plan: LiveRunPlan | None = None,
    ) -> "ReportStore":
        stamp = datetime.now(UTC)
        identifier = run_id or f"{stamp:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}"
        if not identifier or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in identifier
        ):
            raise ValueError("Invalid run ID")
        manifest: ReportManifest | LiveReportManifest
        if live_plan is None:
            manifest = ReportManifest(
                run_id=identifier,
                suite_id=suite_id,
                created_at=stamp.isoformat(),
                planned_case_keys=planned_case_keys,
            )
        else:
            live_plan = LiveRunPlan.model_validate(live_plan.model_dump())
            if not planned_case_keys or any(key.startswith("held-") for key in planned_case_keys):
                raise ValueError("Live plans require diagnostic case keys only")
            manifest = LiveReportManifest(
                run_id=identifier,
                suite_id=suite_id,
                created_at=stamp.isoformat(),
                planned_case_keys=planned_case_keys,
                live_plan=live_plan,
            )
        if len(set(planned_case_keys)) != len(planned_case_keys):
            raise ValueError("Duplicate planned case keys")
        base = output_dir if output_dir.is_absolute() else project_root / output_dir
        directory = _inside(base / identifier, project_root)
        try:
            directory.parent.mkdir(parents=True, exist_ok=True)
            directory.mkdir()  # Exclusive reservation: interrupted runs must never be overwritten.
        except FileExistsError:
            raise ValueError("Report run ID already exists; recovery is read-only") from None
        except OSError:
            raise ValueError("Cannot create report directory") from None
        try:
            with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
                stream.write(manifest.model_dump_json())
                stream.flush()
                os.fsync(stream.fileno())
            with (directory / "cases.jsonl").open("xb") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            if isinstance(manifest, LiveReportManifest):
                digest = hashlib.sha256(manifest.model_dump_json().encode("utf-8")).hexdigest()
                binding = _ManifestBinding(manifest_sha256=digest)
                with (directory / "calls.jsonl").open("xb") as stream:
                    stream.write((binding.model_dump_json() + "\n").encode("utf-8"))
                    stream.flush()
                    os.fsync(stream.fileno())
        except OSError:
            raise ValueError("Cannot write initial report manifest/journal") from None
        return cls(directory, manifest, project_root)

    def _live_recovery(self) -> RecoveredReport:
        if self._write_failed:
            raise ValueError("Report write failed; recovery is read-only")
        if self._finalized or (self.run_dir / "summary.json").exists():
            raise ValueError("Report is finalized")
        if not isinstance(self.manifest, LiveReportManifest):
            raise ValueError("Call capture requires a live manifest")
        recovered = recover_report(self.run_dir, project_root=self.project_root)
        if recovered.manifest != self.manifest:
            raise ValueError("Live manifest changed")
        if recovered.truncated_final_line:
            raise ValueError("Incomplete journal; recovery is read-only")
        if (
            not isinstance(recovered.summary, LiveReportSummary)
            or recovered.summary.calls_sha256 != self._calls_sha256
            or recovered.summary.journal_sha256 != self._cases_sha256
        ):
            raise ValueError("Live journal changed; recovery is read-only")
        return recovered

    def append_call(self, event: CallEvent) -> None:
        """Validate then fsync a call event; no provider code is reachable here."""
        self._live_recovery()
        if not isinstance(self.manifest, LiveReportManifest):
            raise ValueError("Call capture requires a live manifest")
        event = _CALL_EVENT.validate_python(event.model_dump())
        events, journal, _ = _read_calls(self.run_dir, self.manifest, self.project_root)
        _call_states(self.manifest, [*events, event])
        if any(prior == event for prior in events):
            return
        addition = (event.model_dump_json() + "\n").encode("utf-8")
        try:
            with _inside(self.run_dir / "calls.jsonl", self.project_root).open("ab") as stream:
                stream.write(addition)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            self._write_failed = True
            raise ValueError("Cannot append/flush call journal; recover read-only") from None
        self._calls_sha256 = hashlib.sha256(journal + addition).hexdigest()

    def assert_dispatch_ready(self, call_id: int) -> None:
        """Validate a committed intent on this writer; does not send a request."""
        recovered = self._live_recovery()
        call = next((item for item in recovered.calls if item.reservation.call_id == call_id), None)
        if call is None or call.intent is None:
            raise ValueError("Dispatch requires committed reservation and dispatch intent")
        if call.outcome is not None:
            raise ValueError("Call already has an outcome")

    def _append(self, event: _CaseEvent | _UsageEvent) -> None:
        if self._write_failed:
            raise ValueError("Report write failed; recovery is read-only")
        if self._finalized or (self.run_dir / "summary.json").exists():
            raise ValueError("Report is finalized; cannot append")
        if isinstance(self.manifest, LiveReportManifest):
            self._live_recovery()
        try:
            with _inside(self.run_dir / "cases.jsonl", self.project_root).open("ab") as stream:
                stream.write((event.model_dump_json() + "\n").encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            self._cases_sha256 = hashlib.sha256(
                (self.run_dir / "cases.jsonl").read_bytes()
            ).hexdigest()
        except OSError:
            self._write_failed = True
            raise ValueError("Cannot append/flush report journal; recover read-only") from None

    def append_case(self, record: CaseEvaluationRecord) -> None:
        if self._finalized:
            raise ValueError("Report is finalized; cannot append")
        outcome = _project_case(record)
        recovered = recover_report(self.run_dir, project_root=self.project_root)
        if recovered.truncated_final_line:
            raise ValueError("Incomplete journal; recovery is read-only")
        if outcome.key not in self.manifest.planned_case_keys:
            raise ValueError("Case is outside the planned report selection")
        if any(case.key == outcome.key for case in recovered.cases):
            raise ValueError("Duplicate case outcome")
        self._append(_CaseEvent(outcome=outcome))

    def append_usage(self, usage: StoredUsage) -> None:
        usage = StoredUsage.model_validate(usage.model_dump())
        if isinstance(self.manifest, LiveReportManifest) and usage.origin == "current":
            raise ValueError("Live current usage must be captured through call events")
        recovered = recover_report(self.run_dir, project_root=self.project_root)
        if self._finalized or (self.run_dir / "summary.json").exists():
            raise ValueError("Report is finalized; cannot append")
        if recovered.truncated_final_line:
            raise ValueError("Incomplete journal; recovery is read-only")
        if usage.case_key is None and usage.stage != "ingestion_embedding":
            raise ValueError("Only shared ingestion can omit a case key")
        if usage.case_key is not None and usage.case_key not in self.manifest.planned_case_keys:
            raise ValueError("Usage case is outside planned selection")
        for prior in recovered.usage:
            if prior.identity == usage.identity:
                if prior != usage:
                    raise ValueError("Conflicting duplicate usage identity")
                return  # Shared ingestion or repeated import counted exactly once.
        self._append(_UsageEvent(usage=usage))

    def finalize(self) -> ReportSummary:
        if self._write_failed:
            raise ValueError("Report write failed; recovery is read-only")
        if self._finalized or (self.run_dir / "summary.json").exists():
            raise ValueError("Report is finalized")
        recovered = recover_report(self.run_dir, project_root=self.project_root)
        if recovered.truncated_final_line:
            raise ValueError("Incomplete journal; recovery is read-only")
        summary = _summary(
            self.manifest,
            recovered.cases,
            recovered.usage,
            (self.run_dir / "cases.jsonl").read_bytes(),
        )
        if isinstance(self.manifest, LiveReportManifest):
            self._live_recovery()
            events, journal, _ = _read_calls(self.run_dir, self.manifest, self.project_root)
            summary = _live_summary(
                summary,
                _call_states(self.manifest, events),
                journal,
                recovered.cases,
                self.manifest,
            )
        temporary = self.run_dir / f".summary-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(summary.model_dump_json())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.run_dir / "summary.json")
        except OSError:
            self._write_failed = True
            raise ValueError("Cannot commit report summary; recover journal read-only") from None
        self._finalized = True
        return summary


def recover_report(run_dir: Path, *, project_root: Path) -> RecoveredReport:
    """Read complete journal lines; never write, dispatch, or resume an interrupted run."""
    directory = _inside(run_dir, project_root)
    try:
        manifest = _MANIFEST.validate_json(
            _inside(directory / "manifest.json", project_root).read_bytes()
        )
        journal = _inside(directory / "cases.jsonl", project_root).read_bytes()
    except (OSError, ValidationError):
        raise ValueError("Invalid/missing report manifest or journal") from None
    if len(set(manifest.planned_case_keys)) != len(manifest.planned_case_keys):
        raise ValueError("Duplicate planned report keys")
    if manifest.run_id != directory.name:
        raise ValueError("Report run identity mismatch")
    cases: list[StoredCaseOutcome] = []
    usage: list[StoredUsage] = []
    lines = journal.splitlines(keepends=True)
    truncated = bool(lines and not lines[-1].endswith(b"\n"))
    for number, line in enumerate(lines, 1):
        if truncated and number == len(lines):
            break
        try:
            event = _EVENT.validate_json(line)
        except ValidationError:
            raise ValueError(f"Invalid report journal line {number}") from None
        if isinstance(event, _CaseEvent):
            if event.outcome.key not in manifest.planned_case_keys:
                raise ValueError(f"Unplanned case on journal line {number}")
            if any(case.key == event.outcome.key for case in cases):
                raise ValueError(f"Duplicate case on journal line {number}")
            cases.append(event.outcome)
        else:
            if isinstance(manifest, LiveReportManifest) and event.usage.origin == "current":
                raise ValueError("Live current usage must be captured through call events")
            if event.usage.case_key is None and event.usage.stage != "ingestion_embedding":
                raise ValueError(f"Unscoped usage on journal line {number}")
            if (
                event.usage.case_key is not None
                and event.usage.case_key not in manifest.planned_case_keys
            ):
                raise ValueError(f"Unplanned usage on journal line {number}")
            if any(item.identity == event.usage.identity for item in usage):
                raise ValueError(f"Duplicate usage on journal line {number}")
            usage.append(event.usage)
    summary = _summary(manifest, cases, usage, journal, truncated=truncated)
    calls: list[RecoveredCall] = []
    if isinstance(manifest, LiveReportManifest):
        events, call_journal, call_truncated = _read_calls(directory, manifest, project_root)
        calls = _call_states(manifest, events)
        truncated = truncated or call_truncated
        summary = _live_summary(summary, calls, call_journal, cases, manifest)
        if truncated:
            summary.terminal_state = "incomplete"
    summary_path = _inside(directory / "summary.json", project_root)
    committed = summary_path.exists()
    if committed:
        try:
            summary_type = (
                LiveReportSummary if isinstance(manifest, LiveReportManifest) else ReportSummary
            )
            stored = summary_type.model_validate_json(summary_path.read_bytes())
        except (OSError, ValidationError):
            raise ValueError("Invalid committed report summary") from None
        if stored != summary:
            raise ValueError("Committed summary does not match journal")
    complete = committed and not truncated and summary.terminal_state == "completed"
    if not committed:
        summary.terminal_state = "incomplete"
    return RecoveredReport(
        manifest=manifest,
        cases=cases,
        usage=usage,
        summary=summary,
        complete=complete,
        truncated_final_line=truncated,
        calls=calls,
    )
