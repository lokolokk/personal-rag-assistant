"""Slice F durability, recovery, privacy, and accounting acceptance tests."""

import hashlib
import json
from pathlib import Path

import pytest

from app.services.evaluation.grounding_report import (
    CallDispatchIntent,
    CallEvent,
    CallOutcome,
    CallReservation,
    CaseEvaluationRecord,
    LiveRunPlan,
    SemanticCaseReview,
    SemanticFactReview,
)
from app.services.evaluation.report_store import (
    LiveReportSummary,
    ReportStore,
    StoredUsage,
    recover_report,
)


def _live_plan() -> LiveRunPlan:
    return LiveRunPlan(
        plan_id="task-6-1-five-case",
        dataset_sha256="a" * 64,
        corpus_sha256={"document-a": "b" * 64},
        prompt_sha256={"reformulation": "c" * 64, "grading": "d" * 64, "generation": "e" * 64},
        configuration_sha256="f" * 64,
        destination_host="example.invalid",
        chat_model="test-chat",
        embedding_model="test-embedding",
        allowance_tokens=1000,
        maximum_calls=3,
        output_caps={
            "reformulation": 256,
            "grading": 32,
            "generation": 2048,
            "query_embedding": 0,
            "ingestion_embedding": 0,
        },
        usage_evidence="synthetic",
    )


def _live_store(tmp_path: Path) -> ReportStore:
    return ReportStore.create(
        project_root=tmp_path,
        output_dir=tmp_path / "live",
        suite_id="suite",
        planned_case_keys=["case-a"],
        run_id="live-1",
        live_plan=_live_plan(),
    )


def _reservation(call_id: int = 1) -> CallReservation:
    return CallReservation(
        run_id="live-1",
        case_key="case-a",
        stage="generation",
        call_id=call_id,
        input_tokens_estimate=10,
        output_tokens_cap=2048,
        reserved_tokens=300,
        payload_sha256="a" * 64,
    )


def _intent(call_id: int = 1) -> CallDispatchIntent:
    return CallDispatchIntent(
        run_id="live-1",
        case_key="case-a",
        stage="generation",
        call_id=call_id,
    )


def _outcome(call_id: int = 1, *, status: str = "succeeded") -> CallOutcome:
    return CallOutcome.model_validate(
        {
            "run_id": "live-1",
            "case_key": "case-a",
            "stage": "generation",
            "call_id": call_id,
            "outcome": status,
            "source": "synthetic_usage",
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 16,
        }
    )


def test_live_reservation_is_not_dispatch_and_recovery_never_writes(tmp_path: Path) -> None:
    store = _live_store(tmp_path)
    store.append_call(_reservation())
    before = (store.run_dir / "calls.jsonl").read_bytes()
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    assert recovered.calls[0].state == "reserved"
    assert isinstance(recovered.summary, LiveReportSummary)
    assert recovered.summary.calls.potentially_dispatched_calls == 0
    assert recovered.summary.calls.exact_provider_tokens == 0
    assert recovered.summary.calls.reserved_tokens == 300
    assert not recovered.complete
    with pytest.raises(ValueError, match="dispatch intent"):
        store.assert_dispatch_ready(1)
    assert (store.run_dir / "calls.jsonl").read_bytes() == before


def test_live_intent_survives_fake_dispatch_without_outcome(tmp_path: Path) -> None:
    store = _live_store(tmp_path)
    store.append_call(_reservation())
    store.append_call(_intent())
    store.assert_dispatch_ready(1)
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    assert recovered.calls[0].state == "potentially_dispatched"
    assert isinstance(recovered.summary, LiveReportSummary)
    assert recovered.summary.calls.unknown_usage_calls == 1
    assert recovered.summary.calls.exact_provider_tokens is None
    assert recovered.summary.calls.reserved_tokens == 300
    assert not recovered.complete


def test_live_outcome_deduplicates_and_preserves_synthetic_evidence(tmp_path: Path) -> None:
    store = _live_store(tmp_path)
    for event in (_reservation(), _intent(), _outcome()):
        store.append_call(event)
    before = (store.run_dir / "calls.jsonl").read_bytes()
    store.append_call(_outcome())
    assert (store.run_dir / "calls.jsonl").read_bytes() == before
    with pytest.raises(ValueError, match="duplicate call event"):
        store.append_call(_outcome().model_copy(update={"total_tokens": 20, "output_tokens": 8}))
    with pytest.raises(ValueError, match="outcome"):
        store.assert_dispatch_ready(1)
    store.append_case(_case())
    summary = store.finalize()
    assert isinstance(summary, LiveReportSummary)
    assert summary.calls.synthetic_tokens == 16
    assert summary.calls.known_provider_tokens == 0
    assert summary.calls.exact_provider_tokens == 0
    assert recover_report(store.run_dir, project_root=tmp_path).complete


def test_live_partial_provider_usage_is_not_filled_with_estimates(tmp_path: Path) -> None:
    plan = _live_plan().model_copy(update={"usage_evidence": "provider"})
    store = ReportStore.create(
        project_root=tmp_path,
        output_dir=tmp_path / "live",
        suite_id="suite",
        planned_case_keys=["case-a"],
        run_id="live-1",
        live_plan=plan,
    )
    store.append_call(_reservation())
    store.append_call(_intent())
    store.append_call(
        CallOutcome(
            run_id="live-1",
            case_key="case-a",
            stage="generation",
            call_id=1,
            outcome="failed",
            source="provider_reported",
            input_tokens=12,
            error_code="provider_error",
        )
    )
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    assert recovered.calls[0].outcome is not None
    assert recovered.calls[0].outcome.input_tokens == 12
    assert recovered.calls[0].outcome.total_tokens is None
    assert isinstance(recovered.summary, LiveReportSummary)
    assert recovered.summary.calls.exact_provider_tokens is None
    assert recovered.summary.calls.unknown_usage_calls == 1
    assert recovered.summary.current.exact_provider_tokens is None


@pytest.mark.parametrize("kind", ["intent", "outcome"])
def test_live_transition_requires_preceding_events(tmp_path: Path, kind: str) -> None:
    store = _live_store(tmp_path)
    with pytest.raises(ValueError, match="reservation"):
        store.append_call(_intent() if kind == "intent" else _outcome())
    store.append_call(_reservation())
    with pytest.raises(ValueError, match="dispatch intent"):
        store.append_call(_outcome())


def test_live_not_dispatched_is_terminal_without_consumption(tmp_path: Path) -> None:
    store = _live_store(tmp_path)
    store.append_call(_reservation())
    store.append_call(
        CallOutcome(
            run_id="live-1",
            case_key="case-a",
            stage="generation",
            call_id=1,
            outcome="not_dispatched",
            source="unavailable",
        )
    )
    with pytest.raises(ValueError, match="terminal"):
        store.append_call(_intent())
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    assert recovered.calls[0].state == "not_dispatched"
    assert isinstance(recovered.summary, LiveReportSummary)
    assert recovered.summary.calls.exact_provider_tokens == 0


@pytest.mark.parametrize(
    "change",
    [
        {"run_id": "another-run"},
        {"case_key": "case-b"},
        {"attempt": 2},
        {"case_key": None},
        {"output_tokens_cap": 999},
        {"reserved_tokens": 1001},
    ],
)
def test_live_unplanned_reservations_reject_before_write(
    tmp_path: Path,
    change: dict[str, object],
) -> None:
    store = _live_store(tmp_path)
    before = (store.run_dir / "calls.jsonl").read_bytes()
    with pytest.raises(ValueError):
        store.append_call(_reservation().model_copy(update=change))
    assert (store.run_dir / "calls.jsonl").read_bytes() == before


def test_live_identity_cannot_change_stage_or_scope(tmp_path: Path) -> None:
    store = _live_store(tmp_path)
    store.append_call(_reservation())
    with pytest.raises(ValueError, match="identity"):
        store.append_call(_intent().model_copy(update={"stage": "grading"}))


@pytest.mark.parametrize("boundary", ["initial", "reservation", "intent", "outcome"])
def test_live_fsync_fault_poisoning_prevents_dispatch_and_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    def fail_fsync(descriptor: int) -> None:
        _ = descriptor
        raise OSError("private exception text")

    if boundary == "initial":
        monkeypatch.setattr("app.services.evaluation.report_store.os.fsync", fail_fsync)
        with pytest.raises(ValueError, match="initial") as error:
            _live_store(tmp_path)
        assert "private" not in str(error.value)
        return
    store = _live_store(tmp_path)
    if boundary in ("intent", "outcome"):
        store.append_call(_reservation())
    if boundary == "outcome":
        store.append_call(_intent())
    fault_events: dict[str, CallEvent] = {
        "reservation": _reservation(),
        "intent": _intent(),
        "outcome": _outcome(),
    }
    event = fault_events[boundary]
    with monkeypatch.context() as patch:
        patch.setattr("app.services.evaluation.report_store.os.fsync", fail_fsync)
        with pytest.raises(ValueError, match="flush") as error:
            store.append_call(event)
    assert "private" not in str(error.value)
    with pytest.raises(ValueError, match="write failed"):
        store.assert_dispatch_ready(1)
    with pytest.raises(ValueError, match="write failed"):
        store.finalize()
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    assert not recovered.complete


def test_live_truncated_outcome_leaves_dispatch_unknown(tmp_path: Path) -> None:
    store = _live_store(tmp_path)
    store.append_call(_reservation())
    store.append_call(_intent())
    with (store.run_dir / "calls.jsonl").open("ab") as stream:
        stream.write(b'{"kind":"outcome",')
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    assert recovered.truncated_final_line
    assert recovered.calls[0].state == "potentially_dispatched"
    with pytest.raises(ValueError, match="read-only"):
        store.append_call(_outcome())
    with pytest.raises(ValueError):
        store.assert_dispatch_ready(1)


def test_live_manifest_and_journal_tampering_reject(tmp_path: Path) -> None:
    store = _live_store(tmp_path)
    for event in (_reservation(), _intent(), _outcome()):
        store.append_call(event)
    store.append_case(_case())
    store.finalize()
    path = store.run_dir / "calls.jsonl"
    path.write_bytes(path.read_bytes().replace(b'"total_tokens":16', b'"total_tokens":17'))
    with pytest.raises(ValueError):
        recover_report(store.run_dir, project_root=tmp_path)


def test_live_manifest_drift_blocks_active_writer(tmp_path: Path) -> None:
    store = _live_store(tmp_path)
    path = store.run_dir / "manifest.json"
    payload = json.loads(path.read_bytes())
    payload["live_plan"]["configuration_sha256"] = "1" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        store.append_call(_reservation())


def test_live_safe_contract_rejects_raw_data_and_held_out_cases(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CallOutcome.model_validate({**_outcome().model_dump(), "exception": "private"})
    with pytest.raises(ValueError):
        ReportStore.create(
            project_root=tmp_path,
            output_dir=tmp_path / "live",
            suite_id="suite",
            planned_case_keys=["held-04-tada-llama-single-column"],
            live_plan=_live_plan(),
        )
    with pytest.raises(ValueError):
        _live_plan().model_copy(update={"destination_host": "https://private?key=value"})
        LiveRunPlan.model_validate(
            {
                **_live_plan().model_dump(),
                "destination_host": "https://private?key=value",
            }
        )


def test_live_schema_one_recovery_unchanged(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_case(_case())
    store.finalize()
    before = {path.name: path.read_bytes() for path in store.run_dir.iterdir()}
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    assert recovered.manifest.schema_version == "1.0"
    assert recovered.calls == []
    assert not (store.run_dir / "calls.jsonl").exists()
    assert {path.name: path.read_bytes() for path in store.run_dir.iterdir()} == before


def test_live_unknown_outcome_retains_known_subtotal_without_exact_total(tmp_path: Path) -> None:
    plan = _live_plan().model_copy(update={"usage_evidence": "provider"})
    store = ReportStore.create(
        project_root=tmp_path,
        output_dir=tmp_path / "live",
        suite_id="suite",
        planned_case_keys=["case-a"],
        run_id="live-1",
        live_plan=plan,
    )
    store.append_call(_reservation())
    store.append_call(_intent())
    store.append_call(
        CallOutcome(
            run_id="live-1",
            case_key="case-a",
            stage="generation",
            call_id=1,
            outcome="outcome_unknown",
            source="provider_reported",
            total_tokens=16,
        )
    )
    store.append_case(_case())
    summary = store.finalize()
    assert isinstance(summary, LiveReportSummary)
    assert summary.current.known_provider_tokens == 16
    assert summary.current.exact_provider_tokens is None
    assert summary.calls.reserved_tokens == 300
    assert summary.terminal_state == "incomplete"
    assert not recover_report(store.run_dir, project_root=tmp_path).complete


@pytest.mark.parametrize("limit", ["calls", "tokens"])
def test_live_cumulative_limits_and_import_deduplication(tmp_path: Path, limit: str) -> None:
    plan = _live_plan().model_copy(
        update={
            "maximum_calls": 1 if limit == "calls" else 3,
            "allowance_tokens": 500 if limit == "tokens" else 1000,
        }
    )
    store = ReportStore.create(
        project_root=tmp_path,
        output_dir=tmp_path / "live",
        suite_id="suite",
        planned_case_keys=["case-a"],
        run_id="live-1",
        live_plan=plan,
    )
    store.append_call(_reservation())
    store.append_call(_reservation())
    store.append_call(_intent())
    store.append_call(_intent())
    before = (store.run_dir / "calls.jsonl").read_bytes()
    with pytest.raises(ValueError, match="ceiling|allowance"):
        store.append_call(_reservation(2))
    assert (store.run_dir / "calls.jsonl").read_bytes() == before
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    assert len(recovered.calls) == 1
    assert isinstance(recovered.summary, LiveReportSummary)
    assert recovered.summary.calls.reserved_tokens == 300


def test_live_shared_ingestion_and_failed_usage_count_once(tmp_path: Path) -> None:
    plan = _live_plan().model_copy(update={"usage_evidence": "provider"})
    store = ReportStore.create(
        project_root=tmp_path,
        output_dir=tmp_path / "live",
        suite_id="suite",
        planned_case_keys=["case-a"],
        run_id="live-1",
        live_plan=plan,
    )
    reserve = _reservation().model_copy(
        update={
            "case_key": None,
            "stage": "ingestion_embedding",
            "output_tokens_cap": 0,
        }
    )
    intent = _intent().model_copy(update={"case_key": None, "stage": "ingestion_embedding"})
    outcome = CallOutcome(
        run_id="live-1",
        case_key=None,
        stage="ingestion_embedding",
        call_id=1,
        outcome="failed",
        source="provider_reported",
        input_tokens=16,
        output_tokens=0,
        total_tokens=16,
        error_code="provider_error",
    )
    for event in (reserve, intent, outcome, outcome):
        store.append_call(event)
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    assert isinstance(recovered.summary, LiveReportSummary)
    assert recovered.summary.current.known_provider_tokens == 16
    assert recovered.summary.current.exact_provider_tokens == 16
    assert recovered.summary.current.tokens_per_successful_answer is None
    assert len(recovered.calls) == 1


def test_live_missing_usage_cannot_finalize_as_complete(tmp_path: Path) -> None:
    store = _live_store(tmp_path)
    store.append_call(_reservation())
    store.append_call(_intent())
    store.append_call(
        CallOutcome(
            run_id="live-1",
            case_key="case-a",
            stage="generation",
            call_id=1,
            outcome="succeeded",
            source="unavailable",
        )
    )
    store.append_case(_case())
    summary = store.finalize()
    assert summary.terminal_state == "incomplete"
    assert not recover_report(store.run_dir, project_root=tmp_path).complete


def test_live_rejects_current_case_usage_in_competing_ledger(tmp_path: Path) -> None:
    store = _live_store(tmp_path)
    usage = _usage("case-a", 16).model_copy(
        update={
            "origin": "current",
            "source_run_id": "live-1",
        }
    )
    with pytest.raises(ValueError, match="call events"):
        store.append_usage(usage)


def test_live_complete_malformed_line_is_not_discarded(tmp_path: Path) -> None:
    store = _live_store(tmp_path)
    with (store.run_dir / "calls.jsonl").open("ab") as stream:
        stream.write(b'{"kind":"outcome","exception":"private"}\n')
    with pytest.raises(ValueError, match="call journal line 2") as error:
        recover_report(store.run_dir, project_root=tmp_path)
    assert "private" not in str(error.value)


def test_live_committed_summary_binds_manifest_and_valid_call_changes(tmp_path: Path) -> None:
    store = _live_store(tmp_path)
    for event in (_reservation(), _intent(), _outcome()):
        store.append_call(event)
    store.append_case(_case())
    store.finalize()
    path = store.run_dir / "calls.jsonl"
    changed = path.read_bytes().replace(b'"total_tokens":16', b'"total_tokens":17')
    path.write_bytes(changed.replace(b'"output_tokens":4', b'"output_tokens":5'))
    with pytest.raises(ValueError, match="Committed summary"):
        recover_report(store.run_dir, project_root=tmp_path)


def test_live_active_writer_cannot_append_after_valid_journal_edit(tmp_path: Path) -> None:
    store = _live_store(tmp_path)
    store.append_call(_reservation())
    path = store.run_dir / "calls.jsonl"
    path.write_bytes(path.read_bytes().replace(b'"reserved_tokens":300', b'"reserved_tokens":301'))
    with pytest.raises(ValueError, match="journal changed"):
        store.append_call(_intent())
    with pytest.raises(ValueError, match="journal changed"):
        store.finalize()


def _case(case_id: str = "case-a", *, reviewed: bool = False) -> CaseEvaluationRecord:
    review = SemanticFactReview(verdict="pass", reason="Checked")
    return CaseEvaluationRecord(
        case_id=case_id,
        split="diagnostic",
        exposure_status="diagnostic_exposed_local",
        question="private full prompt https://private.invalid?credential=hidden",
        declared_scope=[],
        effective_scope=[],
        result_kind="answer",
        response_text="private answer",
        response_hash=hashlib.sha256(b"private answer").hexdigest(),
        run_id="old-run",
        context_sha256="b" * 64,
        source_dataset_sha256="c" * 64,
        evidence_level="provider_answer_review",
        claim_support="pass" if reviewed else "review_required",
        semantic_review=SemanticCaseReview(
            case_id=case_id,
            run_id="old-run",
            answer_sha256=hashlib.sha256(b"private answer").hexdigest(),
            context_sha256="b" * 64,
            source_dataset_sha256="c" * 64,
            facts={},
            attribution=review,
            refusal=review,
        )
        if reviewed
        else None,
    )


def _store(tmp_path: Path, keys: list[str] | None = None) -> ReportStore:
    return ReportStore.create(
        project_root=tmp_path,
        output_dir=tmp_path / "reports",
        suite_id="suite",
        planned_case_keys=keys or ["case-a"],
        run_id="run-1",
    )


def _usage(case_key: str | None, tokens: int | None, call_id: int = 1) -> StoredUsage:
    return StoredUsage(
        origin="historical",
        source_run_id="old-run",
        call_id=call_id,
        attempt=1,
        case_key=case_key,
        stage="generation",
        source="provider_reported",
        total_tokens=tokens,
    )


def test_round_trip_and_pending_quality_are_honest(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_case(_case())
    summary = store.finalize()
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    assert recovered.complete
    assert recovered.summary == summary
    assert summary.completed_cases == 1
    assert summary.unreviewed_cases == 1
    assert summary.successful_provider_answers == 0
    assert summary.historical.tokens_per_successful_answer is None


def test_interruption_preserves_flushed_cases_without_resume(tmp_path: Path) -> None:
    store = _store(tmp_path, ["case-a", "case-b"])
    store.append_case(_case())
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    assert not recovered.complete
    assert recovered.summary.completed_cases == 1
    assert recovered.summary.terminal_state == "incomplete"
    before = (store.run_dir / "cases.jsonl").read_bytes()
    recover_report(store.run_dir, project_root=tmp_path)
    assert (store.run_dir / "cases.jsonl").read_bytes() == before


def test_truncated_last_line_recovers_only_complete_records(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_case(_case())
    with (store.run_dir / "cases.jsonl").open("ab") as stream:
        stream.write(b'{"kind":"case",')
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    assert recovered.summary.completed_cases == 1
    assert recovered.truncated_final_line
    assert not recovered.complete


def test_malformed_complete_line_fails_instead_of_skipping(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with (store.run_dir / "cases.jsonl").open("ab") as stream:
        stream.write(b"{broken}\n{}\n")
    with pytest.raises(ValueError, match="line 1"):
        recover_report(store.run_dir, project_root=tmp_path)


def test_duplicate_stale_and_outside_paths_are_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        _store(tmp_path)
    store.append_case(_case())
    with pytest.raises(ValueError, match="Duplicate"):
        store.append_case(_case())
    store.finalize()
    with pytest.raises(ValueError, match="finalized"):
        store.append_case(_case("case-b"))
    with pytest.raises(ValueError, match="project"):
        ReportStore.create(
            project_root=tmp_path,
            output_dir=tmp_path.parent / "outside",
            suite_id="suite",
            planned_case_keys=["case-a"],
        )


def test_safe_projection_excludes_prompts_urls_and_exception_text(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = _case()
    record.standalone_query = "secret request URL"
    store.append_case(record)
    store.finalize()
    text = "".join(path.read_text(encoding="utf-8") for path in store.run_dir.iterdir())
    for forbidden in ("private", "credential", "hidden", "secret", "https://"):
        assert forbidden not in text


def test_failed_answer_cost_and_shared_ingestion_count_once(tmp_path: Path) -> None:
    store = _store(tmp_path, ["case-a", "case-b"])
    store.append_case(_case(reviewed=True))
    failed = _case("case-b")
    failed.failure_codes = ["wrong_fact"]
    store.append_case(failed)
    store.append_usage(_usage("case-a", 100))
    store.append_usage(_usage("case-b", 50, 2))
    ingestion = _usage(None, 30, 3).model_copy(update={"stage": "ingestion_embedding"})
    store.append_usage(ingestion)
    store.append_usage(ingestion)
    summary = store.finalize()
    assert summary.historical.known_provider_tokens == 180
    assert summary.historical.exact_provider_tokens == 180
    assert summary.historical.tokens_per_successful_answer == 180.0
    assert summary.current.known_provider_tokens == 0
    assert summary.failed_cases == 1
    assert summary.unreviewed_cases == 1


def test_unknown_usage_prevents_exact_totals(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_case(_case(reviewed=True))
    store.append_usage(_usage("case-a", 100))
    store.append_usage(_usage("case-a", None, 2))
    summary = store.finalize()
    assert summary.historical.known_provider_tokens == 100
    assert summary.historical.exact_provider_tokens is None
    assert summary.historical.tokens_per_successful_answer is None


def test_local_estimates_never_enter_provider_totals(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_case(_case(reviewed=True))
    store.append_usage(
        _usage("case-a", 999).model_copy(
            update={
                "source": "local_estimate",
                "local_only": True,
            }
        )
    )
    summary = store.finalize()
    assert summary.historical.known_provider_tokens == 0
    assert summary.historical.local_estimate_tokens == 999
    assert summary.historical.tokens_per_successful_answer is None


def test_atomic_summary_failure_leaves_readable_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    store.append_case(_case())

    def deny_replace(source: object, target: object) -> None:
        _ = source, target
        raise PermissionError("sensitive raw error details")

    monkeypatch.setattr("app.services.evaluation.report_store.os.replace", deny_replace)
    with pytest.raises(ValueError, match="summary") as error:
        store.finalize()
    assert "sensitive" not in str(error.value)
    assert not recover_report(store.run_dir, project_root=tmp_path).complete


def test_tampered_final_summary_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_case(_case())
    store.finalize()
    path = store.run_dir / "summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["completed_cases"] = 999
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="summary"):
        recover_report(store.run_dir, project_root=tmp_path)


def test_missing_case_usage_is_unknown_even_with_another_known_case(tmp_path: Path) -> None:
    store = _store(tmp_path, ["case-a", "case-b"])
    store.append_case(_case(reviewed=True))
    store.append_case(_case("case-b", reviewed=True))
    store.append_usage(_usage("case-a", 100))
    summary = store.finalize()
    assert summary.historical.exact_provider_tokens is None
    assert summary.historical.tokens_per_successful_answer is None


def test_stale_review_never_counts_as_success(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = _case(reviewed=True)
    record.response_text = "altered after review"
    store.append_case(record)
    summary = store.finalize()
    assert summary.successful_provider_answers == 0
    assert summary.unreviewed_cases == 1


def test_append_fsync_error_is_explicit_and_recovery_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)

    def deny_fsync(descriptor: int) -> None:
        _ = descriptor
        raise OSError("private raw system error")

    monkeypatch.setattr("app.services.evaluation.report_store.os.fsync", deny_fsync)
    with pytest.raises(ValueError, match="flush") as error:
        store.append_case(_case())
    assert "private" not in str(error.value)
    recovered = recover_report(store.run_dir, project_root=tmp_path)
    assert recovered.summary.completed_cases == 1
    assert not recovered.complete


def test_usage_identity_conflicts_and_unknown_case_are_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_usage(_usage("case-a", 10))
    with pytest.raises(ValueError, match="Conflicting"):
        store.append_usage(_usage("case-a", 20))
    with pytest.raises(ValueError, match="selection"):
        store.append_usage(_usage("outsider", 20, 2))


def test_current_and_historical_usage_are_distinct(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_case(_case(reviewed=True))
    store.append_usage(_usage("case-a", 100))
    store.append_usage(
        _usage(None, 40, 2).model_copy(
            update={
                "origin": "current",
                "source_run_id": "run-1",
                "stage": "ingestion_embedding",
            }
        )
    )
    summary = store.finalize()
    assert summary.historical.known_provider_tokens == 100
    assert summary.current.known_provider_tokens == 40
    assert summary.current.tokens_per_successful_answer is None


def test_partial_finalization_and_arm_keys(tmp_path: Path) -> None:
    store = _store(tmp_path, ["case-a:baseline", "case-a:candidate"])
    baseline = _case()
    baseline.arm_name = "baseline"
    store.append_case(baseline)
    summary = store.finalize()
    assert summary.completed_cases == 1
    assert summary.terminal_state == "incomplete"
    assert not recover_report(store.run_dir, project_root=tmp_path).complete


def test_duplicate_after_lost_terminal_summary_is_not_reusable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_case(_case())
    store.finalize()
    (store.run_dir / "summary.json").unlink()
    assert not recover_report(store.run_dir, project_root=tmp_path).complete
    with pytest.raises(ValueError, match="already exists"):
        _store(tmp_path)


def test_observed_usage_drops_raw_error_and_model_fields() -> None:
    from app.services.evaluation.usage import UsageRecord

    record = UsageRecord(
        stage="generation",
        model="private provider URL",
        call_id=1,
        error_type="private exception text",
        outcome="failed",
    )
    safe = StoredUsage.from_observed(
        record, origin="historical", source_run_id="old-run", case_key="case-a"
    )
    assert safe.outcome == "failed"
    assert "private" not in safe.model_dump_json()


def test_manifest_tampering_invalidates_terminal_summary(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_case(_case())
    store.finalize()
    path = store.run_dir / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["suite_id"] = "other-suite"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="summary"):
        recover_report(store.run_dir, project_root=tmp_path)


def test_unknown_held_out_cannot_be_saved_as_generated_result(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = _case()
    record.exposure_status = "invoked_result_unknown"
    with pytest.raises(ValueError, match="Held-out|unknown"):
        store.append_case(record)


def test_initial_write_permission_error_is_surfaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_fsync(descriptor: int) -> None:
        _ = descriptor
        raise PermissionError("sensitive raw error")

    monkeypatch.setattr("app.services.evaluation.report_store.os.fsync", deny_fsync)
    with pytest.raises(ValueError, match="initial") as error:
        _store(tmp_path)
    assert "sensitive" not in str(error.value)


def test_faulted_writer_cannot_finalize_uncertain_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)

    def deny_fsync(descriptor: int) -> None:
        _ = descriptor
        raise PermissionError("write failure")

    with monkeypatch.context() as patch:
        patch.setattr("app.services.evaluation.report_store.os.fsync", deny_fsync)
        with pytest.raises(ValueError, match="flush"):
            store.append_case(_case())
    with pytest.raises(ValueError, match="failed"):
        store.finalize()


def test_empty_run_and_unique_auto_ids(tmp_path: Path) -> None:
    first = ReportStore.create(
        project_root=tmp_path, output_dir=Path("reports"), suite_id="suite", planned_case_keys=[]
    )
    second = ReportStore.create(
        project_root=tmp_path, output_dir=Path("reports"), suite_id="suite", planned_case_keys=[]
    )
    assert first.run_dir != second.run_dir
    summary = first.finalize()
    assert summary.completed_cases == 0
    assert summary.historical.tokens_per_successful_answer is None
    assert recover_report(first.run_dir, project_root=tmp_path).complete


def test_case_review_failures_remain_failed_in_summary(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = _case(reviewed=True)
    assert record.semantic_review is not None
    record.semantic_review.attribution = SemanticFactReview(verdict="fail", reason="Wrong source")
    store.append_case(record)
    summary = store.finalize()
    assert summary.failed_cases == 1
    assert summary.unreviewed_cases == 0
    assert summary.successful_provider_answers == 0


def test_review_with_unavailable_context_cannot_count_success(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = _case(reviewed=True)
    record.context_sha256 = None
    assert record.semantic_review is not None
    record.semantic_review.context_sha256 = None
    store.append_case(record)
    summary = store.finalize()
    assert summary.successful_provider_answers == 0


def test_recovery_rejects_unknown_complete_event(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with (store.run_dir / "cases.jsonl").open("ab") as stream:
        stream.write(b'{"kind":"dispatch","payload":"private"}\n')
    with pytest.raises(ValueError, match="line 1"):
        recover_report(store.run_dir, project_root=tmp_path)
