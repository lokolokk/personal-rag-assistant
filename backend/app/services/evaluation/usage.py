"""Opt-in consumption records for actual service calls, never public API payloads.

Local counts use cl100k_base text only, excluding provider chat framing and hidden
reasoning. They are estimates for configured Qwen models, not billed token counts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from math import ceil
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import tiktoken

if TYPE_CHECKING:
    from app.services.evaluation.report_store import ReportStore

Stage = Literal["reformulation", "grading", "generation", "query_embedding", "ingestion_embedding"]
Source = Literal["provider_reported", "local_estimate", "unavailable"]

DEFAULT_OUTPUT_CAPS: dict[Stage, int] = {
    "reformulation": 256,
    "grading": 32,
    "generation": 2048,
    "query_embedding": 0,
    "ingestion_embedding": 0,
}


class LiveBudgetExceeded(BaseException):
    """Stop provider work before dispatch when a live evaluation lacks reserved budget."""


@dataclass(frozen=True)
class BudgetReservation:
    stage: Stage
    input_tokens_estimate: int
    output_tokens_cap: int
    max_attempts: int
    reserved_tokens: int


class LiveTokenBudget:
    """Conservative cumulative reservation ledger for explicitly authorized live runs."""

    def __init__(
        self,
        allowance_tokens: int,
        *,
        max_attempts: int = 3,
        input_multiplier: int = 2,
        framing_tokens_per_attempt: int = 128,
        output_caps: dict[Stage, int] | None = None,
    ) -> None:
        if allowance_tokens < 1 or max_attempts < 1 or input_multiplier < 1:
            raise ValueError("Live budget and reservation limits must be positive")
        self.allowance_tokens = allowance_tokens
        self.max_attempts = max_attempts
        self.input_multiplier = input_multiplier
        self.framing_tokens_per_attempt = framing_tokens_per_attempt
        self.output_caps = dict(DEFAULT_OUTPUT_CAPS if output_caps is None else output_caps)
        if set(self.output_caps) != set(DEFAULT_OUTPUT_CAPS) or any(
            cap < 0 for cap in self.output_caps.values()
        ):
            raise ValueError("Output caps must provide a non-negative cap for every provider stage")
        self.reservations: list[BudgetReservation] = []
        self.reserved_tokens = 0
        self._lock = Lock()

    def reserve(self, stage: Stage, texts: list[str]) -> BudgetReservation:
        counts = [estimate_tokens(text) for text in texts]
        if not counts or any(count is None for count in counts):
            raise LiveBudgetExceeded("Cannot reserve live usage without a local input estimate")
        input_estimate = sum(count for count in counts if count is not None)
        output_cap = self.output_caps[stage]
        reservation = BudgetReservation(
            stage=stage,
            input_tokens_estimate=input_estimate,
            output_tokens_cap=output_cap,
            max_attempts=self.max_attempts,
            reserved_tokens=conservative_reservation_tokens(
                input_estimate,
                output_cap,
                max_attempts=self.max_attempts,
                input_multiplier=self.input_multiplier,
                framing_tokens_per_attempt=self.framing_tokens_per_attempt,
            ),
        )
        with self._lock:
            if self.reserved_tokens + reservation.reserved_tokens > self.allowance_tokens:
                raise LiveBudgetExceeded(
                    "Live evaluation stopped before dispatch: remaining allowance is below "
                    "the conservative reservation for this provider call"
                )
            self.reserved_tokens += reservation.reserved_tokens
            self.reservations.append(reservation)
        return reservation


def conservative_reservation_tokens(
    input_tokens_estimate: int,
    output_tokens_cap: int,
    *,
    max_attempts: int = 3,
    input_multiplier: int = 2,
    framing_tokens_per_attempt: int = 128,
) -> int:
    """Return a text-estimate reservation covering bounded retries and output caps."""
    if (
        input_tokens_estimate < 0
        or output_tokens_cap < 0
        or max_attempts < 1
        or input_multiplier < 1
        or framing_tokens_per_attempt < 0
    ):
        raise ValueError("Reservation inputs must be non-negative and limits must be positive")
    per_attempt = (
        ceil(input_tokens_estimate * input_multiplier)
        + framing_tokens_per_attempt
        + output_tokens_cap
    )
    return per_attempt * max_attempts


_LIVE_BUDGET: ContextVar[LiveTokenBudget | None] = ContextVar(
    "evaluation_live_token_budget", default=None
)


@contextmanager
def live_token_budget(
    allowance_tokens: int,
    *,
    max_attempts: int = 3,
    input_multiplier: int = 2,
    framing_tokens_per_attempt: int = 128,
    output_caps: dict[Stage, int] | None = None,
) -> Iterator[LiveTokenBudget]:
    """Enable a pre-dispatch live budget only inside an active usage-measurement trace."""
    if _TRACE.get() is None:
        raise RuntimeError("Live token budgeting requires an active measure_usage() trace")
    budget = LiveTokenBudget(
        allowance_tokens,
        max_attempts=max_attempts,
        input_multiplier=input_multiplier,
        framing_tokens_per_attempt=framing_tokens_per_attempt,
        output_caps=output_caps,
    )
    token = _LIVE_BUDGET.set(budget)
    try:
        yield budget
    finally:
        _LIVE_BUDGET.reset(token)


@dataclass
class UsageRecord:
    stage: Stage
    model: str
    call_id: int
    attempt: int = 1
    source: Source = "unavailable"
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_input_tokens: int | None = None
    estimated_output_tokens: int | None = None
    estimate_method: str = "cl100k_base:text_only"
    outcome: Literal["dispatched", "succeeded", "failed"] = "dispatched"
    retry_visibility: Literal["observed", "unavailable", "not_applicable"] = "unavailable"
    error_type: str | None = None
    input_items: int = 1
    local_only: bool = False


@dataclass
class UsageTrace:
    records: list[UsageRecord] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def append(self, record: UsageRecord) -> None:
        with self._lock:
            self.records.append(record)


_TRACE: ContextVar[UsageTrace | None] = ContextVar("evaluation_usage", default=None)
_CALL: ContextVar[object | None] = ContextVar("evaluation_call", default=None)
_CAPTURE: ContextVar[DurableUsageCapture | None] = ContextVar("evaluation_capture", default=None)
_CASE_KEY: ContextVar[str | None] = ContextVar("evaluation_capture_case", default=None)


class DurableUsageCapture:
    """Opt-in synchronous capture on a new writer; does not grant live authorization."""

    def __init__(self, store: ReportStore, trace: UsageTrace, bounds: dict[Stage, int]) -> None:
        from app.services.evaluation.report_store import LiveReportManifest, recover_report

        if not isinstance(store.manifest, LiveReportManifest):
            raise LiveBudgetExceeded("Durable capture requires a frozen live manifest")
        if set(bounds) != set(DEFAULT_OUTPUT_CAPS) or any(
            type(value) is not int or value < 0 for value in bounds.values()
        ):
            raise LiveBudgetExceeded("Durable capture requires all stage input bounds")
        recovered = recover_report(store.run_dir, project_root=store.project_root)
        if recovered.calls or recovered.cases or recovered.usage or trace.records:
            raise LiveBudgetExceeded("Durable capture cannot resume or reuse an existing trace")
        self.store = store
        self.plan = store.manifest.live_plan.model_copy(deep=True)
        self.trace = trace
        self._bounds = dict(bounds)
        self._lock = Lock()
        self._next_call_id = 1
        self.stopped_reason: str | None = None

    def check(self) -> None:
        if self.stopped_reason is not None:
            raise LiveBudgetExceeded(f"Durable capture stopped: {self.stopped_reason}")
        if _TRACE.get() is not self.trace:
            self.stop("usage trace changed")
            self.check()

    def stop(self, reason: str) -> None:
        if self.stopped_reason is None:
            self.stopped_reason = reason

    @contextmanager
    def case(self, case_key: str) -> Iterator[None]:
        self.check()
        if case_key not in self.store.manifest.planned_case_keys or _CASE_KEY.get() is not None:
            self.stop("unplanned or nested case scope")
            self.check()
        token = _CASE_KEY.set(case_key)
        try:
            yield
            self.check()
        finally:
            _CASE_KEY.reset(token)

    def begin(self, call: _ObservedCall) -> None:
        self.check()
        if not self._lock.acquire(blocking=False):
            self.stop("concurrent provider calls are prohibited")
            self.check()
        try:
            if self._next_call_id > self.plan.maximum_calls:
                raise LiveBudgetExceeded("Durable capture call ceiling exhausted")
            expected_model = (
                self.plan.embedding_model
                if call.stage.endswith("embedding")
                else self.plan.chat_model
            )
            if call.model != expected_model:
                raise LiveBudgetExceeded("Durable capture model drift")
            if call.input_estimate is None or call.input_estimate > self._bounds[call.stage]:
                raise LiveBudgetExceeded("Durable capture input exceeds frozen stage bound")
            call.case_key = _CASE_KEY.get()
            if (call.stage == "ingestion_embedding") != (call.case_key is None):
                raise LiveBudgetExceeded("Durable capture requires explicit case/shared identity")
            call.call_id = self._next_call_id
            self._next_call_id += 1
        except BaseException:
            self.stop("reservation admission failed")
            self._lock.release()
            raise

    def save_reservation(self, call: _ObservedCall) -> None:
        from app.services.evaluation.grounding_report import CallReservation

        if call.reservation_saved:
            return
        reservation = call.reservation
        if reservation is None:
            raise LiveBudgetExceeded("Durable capture requires an active token reservation")
        self.store.append_call(
            CallReservation(
                run_id=self.store.manifest.run_id,
                case_key=call.case_key,
                stage=call.stage,
                call_id=call.call_id,
                input_tokens_estimate=reservation.input_tokens_estimate,
                output_tokens_cap=reservation.output_tokens_cap,
                reserved_tokens=reservation.reserved_tokens,
                payload_sha256=call.payload_sha256,
            )
        )
        call.reservation_saved = True

    def dispatch(self, call: _ObservedCall, request: Any) -> None:
        from app.services.evaluation.grounding_report import CallDispatchIntent

        self.check()
        if call.requests:
            self.stop("second request attempt prohibited")
            self.check()
        try:
            if request.url.scheme != "https" or request.url.host != self.plan.destination_host:
                raise ValueError("destination drift")
            body = json.loads(request.content)
            if body.get("model") != call.model or body.get("stream"):
                raise ValueError("request configuration drift")
            if call.stage.endswith("embedding"):
                texts = body["input"]
                texts = [texts] if isinstance(texts, str) else texts
            else:
                cap = body.get("max_completion_tokens")
                if (
                    type(cap) is not int
                    or not 0 < cap <= self.plan.output_caps[call.stage] - 10
                    or "max_tokens" in body
                ):
                    raise ValueError("output cap drift")
                texts = ["\n".join(item["content"] for item in body["messages"])]
            if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                raise ValueError("unsupported input payload")
            counts = [estimate_tokens(text) for text in texts]
            if any(count is None for count in counts):
                raise ValueError("unavailable actual input estimate")
            actual = sum(count for count in counts if count is not None)
            if call.input_estimate is None or actual > call.input_estimate:
                raise ValueError("actual input exceeds reservation estimate")
            call.payload_sha256 = hashlib.sha256(request.content).hexdigest()
            self.save_reservation(call)
            call.intent_attempted = True
            self.store.append_call(
                CallDispatchIntent(
                    run_id=self.store.manifest.run_id,
                    case_key=call.case_key,
                    stage=call.stage,
                    call_id=call.call_id,
                )
            )
            self.store.assert_dispatch_ready(call.call_id)
        except (ValueError, OSError, KeyError, TypeError):
            self.stop("request validation or durable write failed")
            self.check()

    def finish(self, call: _ObservedCall, failed: BaseException | None) -> None:
        from app.services.evaluation.grounding_report import CallOutcome

        try:
            self.save_reservation(call)
            status: Literal["not_dispatched", "succeeded", "failed", "outcome_unknown"]
            status = (
                ("failed" if failed is not None else "succeeded")
                if call.requests
                else "outcome_unknown"
                if call.intent_attempted
                else "not_dispatched"
            )
            source: Literal["provider_reported", "synthetic_usage", "unavailable"] = "unavailable"
            counts: dict[str, int | None] = {}
            if call.record.source == "provider_reported" and call.requests:
                source = (
                    "synthetic_usage"
                    if self.plan.usage_evidence == "synthetic"
                    else "provider_reported"
                )
                counts = {
                    key: getattr(call.record, key)
                    for key in (
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                    )
                }
            error_code: Literal["provider_error", "capture_error"] | None = (
                "provider_error" if failed is not None and call.requests else None
            )
            try:
                outcome = CallOutcome(
                    run_id=self.store.manifest.run_id,
                    case_key=call.case_key,
                    stage=call.stage,
                    call_id=call.call_id,
                    outcome=status,
                    source=source,
                    input_tokens=counts.get("input_tokens"),
                    output_tokens=counts.get("output_tokens"),
                    total_tokens=counts.get("total_tokens"),
                    error_code=error_code,
                )
            except ValueError:
                self.stop("inconsistent provider usage")
                outcome = CallOutcome(
                    run_id=self.store.manifest.run_id,
                    case_key=call.case_key,
                    stage=call.stage,
                    call_id=call.call_id,
                    outcome="outcome_unknown",
                    source="unavailable",
                    error_code="capture_error",
                )
            self.store.append_call(outcome)
            if failed is not None:
                self.stop("provider call failed")
            elif call.requests == 0:
                self.stop("SDK call completed without observed dispatch")
            elif outcome.total_tokens is None:
                self.stop("provider total usage unavailable")
            elif call.reservation is not None and (
                outcome.total_tokens > call.reservation.reserved_tokens
                or (outcome.output_tokens or 0) > call.reservation.output_tokens_cap
            ):
                self.stop("measured usage exceeded reserved bounds")
        except (ValueError, OSError):
            self.stop("durable outcome write failed")
            self.check()
        finally:
            self._lock.release()


@contextmanager
def durable_usage(
    store: ReportStore,
    *,
    input_token_bounds: dict[Stage, int],
) -> Iterator[DurableUsageCapture]:
    """Bind one fresh synchronous trace to durable capture; production defaults stay inert."""
    trace = _TRACE.get()
    if trace is None or _CAPTURE.get() is not None or _LIVE_BUDGET.get() is not None:
        raise LiveBudgetExceeded("Durable capture requires a fresh unnested measure_usage trace")
    capture = DurableUsageCapture(store, trace, input_token_bounds)
    token = _CAPTURE.set(capture)
    try:
        with live_token_budget(
            capture.plan.allowance_tokens,
            max_attempts=1,
            output_caps=capture.plan.output_caps,
        ):
            yield capture
            capture.check()
    finally:
        _CAPTURE.reset(token)


@contextmanager
def measure_usage() -> Iterator[UsageTrace]:
    """Collect one run's calls; nested/async/thread contexts restore their parent trace."""
    trace = UsageTrace()
    token = _TRACE.set(trace)
    try:
        yield trace
    finally:
        _TRACE.reset(token)


def estimate_tokens(text: str) -> int | None:
    try:
        return len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))
    except Exception:
        return None


def _value(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _count(obj: Any, key: str) -> int | None:
    value = _value(obj, key)
    return value if type(value) is int and value >= 0 else None


def _apply_usage(record: UsageRecord, usage: Any) -> None:
    input_tokens = _count(usage, "prompt_tokens")
    output_tokens = _count(usage, "completion_tokens")
    total_tokens = _count(usage, "total_tokens")
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return
    if record.stage in ("ingestion_embedding", "query_embedding"):
        output_tokens = 0  # Embeddings have no generated text tokens.
    record.source = "provider_reported"
    record.input_tokens = input_tokens
    record.output_tokens = output_tokens
    record.total_tokens = total_tokens
    if (
        _CAPTURE.get() is None
        and total_tokens is None
        and input_tokens is not None
        and output_tokens is not None
    ):
        record.total_tokens = input_tokens + output_tokens


class _ObservedCall:
    def __init__(
        self,
        trace: UsageTrace,
        stage: Stage,
        model: str,
        prompt: str | list[str],
        local_only: bool = False,
    ) -> None:
        self.trace = trace
        self.stage = stage
        self.model = model
        texts = prompt if isinstance(prompt, list) else [prompt]
        counts = [estimate_tokens(text) for text in texts]
        self.input_estimate = (
            sum(count for count in counts if count is not None)
            if all(count is not None for count in counts)
            else None
        )
        self.input_items = len(texts)
        self.local_only = local_only
        self.capture = _CAPTURE.get()
        self.case_key: str | None = None
        self.reservation_saved = False
        self.intent_attempted = False
        self.payload_sha256 = hashlib.sha256(
            json.dumps(texts, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if self.capture is not None:
            self.capture.begin(self)
        budget = _LIVE_BUDGET.get()
        try:
            self.reservation = budget.reserve(stage, texts) if budget is not None else None
        except BaseException:
            if self.capture is not None:
                self.capture.stop("token reservation failed")
                self.capture._lock.release()
            raise
        self.requests = 0
        with trace._lock:
            if self.capture is None:
                self.call_id = len(trace.records) + 1
            self.record = self._record()
            trace.records.append(self.record)
        self.output = ""

    def _record(self) -> UsageRecord:
        return UsageRecord(
            stage=self.stage,
            model=self.model,
            call_id=self.call_id,
            attempt=self.requests + 1,
            estimated_input_tokens=self.input_estimate,
            input_items=self.input_items,
            local_only=self.local_only,
        )

    def request_hook(self, request: Any) -> None:
        if _CALL.get() is not self:
            return
        if self.capture is not None:
            self.capture.dispatch(self, request)
        if self.reservation is not None and self.requests >= self.reservation.max_attempts:
            raise LiveBudgetExceeded(
                "Live evaluation stopped before an unreserved provider retry dispatch"
            )
        if self.requests:
            self.record.outcome = "failed"
            self.record = self._record()
            self.trace.append(self.record)
        self.requests += 1
        self.record.retry_visibility = "observed"

    def response_hook(self, response: Any) -> None:
        if _CALL.get() is not self:
            return
        if response.status_code >= 400:
            self.record.outcome = "failed"
            self.record.error_type = f"HTTP_{response.status_code}"
            if self.capture is not None:
                self.capture.stop("provider HTTP error")
        elif self.stage not in ("ingestion_embedding", "query_embedding"):
            return
        try:
            response.read()
            _apply_usage(self.record, _value(response.json(), "usage"))
        except Exception as err:
            self.record.error_type = type(err).__name__

    def finish(self, failed: BaseException | None) -> None:
        self.record.outcome = "failed" if failed is not None else "succeeded"
        if failed is not None:
            self.record.error_type = type(failed).__name__
        self.record.estimated_output_tokens = estimate_tokens(self.output)
        if failed is None and self.record.source != "provider_reported" and self.capture is None:
            if self.input_estimate is not None and self.record.estimated_output_tokens is not None:
                self.record.source = "local_estimate"
                self.record.input_tokens = self.input_estimate
                self.record.output_tokens = self.record.estimated_output_tokens
                self.record.total_tokens = self.input_estimate + self.record.estimated_output_tokens
        if self.capture is not None:
            self.capture.finish(self, failed)


_HOOK_LOCK = Lock()


def _request_hook(request: Any) -> None:
    call = _CALL.get()
    if isinstance(call, _ObservedCall):
        call.request_hook(request)


def _response_hook(response: Any) -> None:
    call = _CALL.get()
    if isinstance(call, _ObservedCall):
        call.response_hook(response)


def _require_retry_observation(client: Any) -> None:
    transport = getattr(client, "_client", None)
    hooks = getattr(transport, "event_hooks", None)
    if (
        not isinstance(hooks, dict)
        or not isinstance(hooks.get("request"), list)
        or not isinstance(hooks.get("response"), list)
    ):
        raise LiveBudgetExceeded(
            "Live evaluation requires provider clients with observable request and response hooks"
        )


@contextmanager
def _observe(
    stage: Stage,
    model: str,
    prompt: str | list[str],
    client: Any = None,
    local_only: bool = False,
) -> Iterator[_ObservedCall]:
    trace = _TRACE.get()
    if trace is None:
        raise RuntimeError("Observation requires an active evaluation trace")
    capture = _CAPTURE.get()
    if capture is not None:
        capture.check()
        if getattr(client, "max_retries", None) != 0:
            capture.stop("SDK retries must be zero")
            capture.check()
    if _LIVE_BUDGET.get() is not None:
        try:
            _require_retry_observation(client)
        except LiveBudgetExceeded:
            if capture is not None:
                capture.stop("unobservable SDK client")
            raise
    call = _ObservedCall(trace, stage, model, prompt, local_only)
    token = _CALL.set(call)
    hooks = getattr(getattr(client, "_client", None), "event_hooks", None)
    # Keep one inert dispatcher on a shared SDK client rather than adding/removing
    # per-run callbacks while another thread is iterating the same hook list.
    if isinstance(hooks, dict):
        with _HOOK_LOCK:
            for key, callback in [("request", _request_hook), ("response", _response_hook)]:
                callbacks = hooks.get(key)
                if isinstance(callbacks, list) and callback not in callbacks:
                    callbacks.append(callback)
    failed: BaseException | None = None
    try:
        yield call
    except BaseException as err:
        failed = err
        raise
    finally:
        try:
            call.finish(failed)
        finally:
            _CALL.reset(token)


def observed_text(stage: Stage, model: str, prompt: str, invoke: Callable[[], str]) -> str:
    capture = _CAPTURE.get()
    if capture is not None:
        capture.stop("callable clients are not observable")
        capture.check()
    if _TRACE.get() is None:
        return invoke()
    if _LIVE_BUDGET.get() is not None:
        raise LiveBudgetExceeded(
            "Live evaluation requires an observable SDK client; callable clients are not budgetable"
        )
    with _observe(stage, model, prompt) as call:
        call.record.retry_visibility = "not_applicable"
        call.output = invoke()
        return call.output


def observed_completion(client: Any, stage: Stage, **kwargs: Any) -> Any:
    """Observe the existing SDK call and stream; pass request arguments through unchanged."""
    capture = _CAPTURE.get()
    if capture is not None:
        capture.check()
    if _TRACE.get() is None:
        return client.chat.completions.create(**kwargs)
    if capture is not None and kwargs.get("stream"):
        capture.stop("streaming excluded from synchronous pilot")
        capture.check()
    prompt = "\n".join(m["content"] for m in kwargs["messages"])
    model = str(kwargs["model"])
    budget = _LIVE_BUDGET.get()
    if budget is not None:
        kwargs = dict(kwargs)
        output_cap = budget.output_caps[stage]
        configured_output = kwargs.get("max_tokens")
        kwargs["max_tokens"] = (
            min(configured_output, output_cap)
            if isinstance(configured_output, int) and configured_output >= 0
            else output_cap
        )
    if capture is not None:
        kwargs = dict(kwargs)
        requested = kwargs.pop("max_tokens", capture.plan.output_caps[stage])
        total_cap = capture.plan.output_caps[stage] - 10
        configured_total = kwargs.get("max_completion_tokens", requested)
        kwargs["max_completion_tokens"] = (
            min(configured_total, total_cap)
            if type(configured_total) is int and configured_total > 0
            else total_cap
        )
    if not kwargs.get("stream"):
        with _observe(stage, model, prompt, client) as call:
            response = client.chat.completions.create(**kwargs)
            _apply_usage(call.record, _value(response, "usage"))
            call.output = response.choices[0].message.content or ""
            return response

    def stream() -> Iterator[Any]:
        with _observe(stage, model, prompt, client) as call:
            response = client.chat.completions.create(**kwargs)
            parts: list[str] = []
            try:
                for chunk in response:
                    _apply_usage(call.record, _value(chunk, "usage"))
                    if chunk.choices and chunk.choices[0].delta.content:
                        parts.append(chunk.choices[0].delta.content)
                    yield chunk
            finally:
                call.output = "".join(parts)
                close = getattr(response, "close", None)
                if callable(close):
                    close()

    return stream()


_T = TypeVar("_T")


def observed_embeddings(
    stage: Literal["ingestion_embedding", "query_embedding"],
    model: str,
    texts: list[str],
    invoke: Callable[[], _T],
    client: Any = None,
    local_only: bool = False,
) -> _T:
    """Observe the existing adapter operation, including HTTP usage it otherwise discards.

    Adapters without SDK response hooks retain per-input estimates and explicitly
    unavailable retry visibility. Local deterministic embeddings are flagged separately.
    """
    capture = _CAPTURE.get()
    if capture is not None:
        capture.check()
    if _TRACE.get() is None:
        return invoke()
    if local_only and _LIVE_BUDGET.get() is not None:
        raise LiveBudgetExceeded(
            "Live evaluation requires provider embeddings; local-only vectors are diagnostic"
        )
    with _observe(stage, model, texts, client, local_only) as call:
        if local_only:
            call.record.retry_visibility = "not_applicable"
        return invoke()
