"""Evaluation schemas, typed report models, and suite manifest validation for Task 6.1."""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from app.schemas.document import ChunkMetadata, DocumentChunk
from app.services.evaluation.source_dataset import (
    EvaluationCase,
    SourceDataset,
    load_source_dataset,
)
from app.services.evaluation.usage import DEFAULT_OUTPUT_CAPS, Stage

EvidenceLevel = Literal[
    "offline_contract",
    "local_retrieval_diagnostic",
    "provider_answer_review",
]

ResultKind = Literal[
    "answer",
    "appropriate_abstention",
    "incorrect_refusal",
    "unsupported_answer",
    "execution_error",
    "incomplete",
    "unreviewed",
]

FailureCode = Literal[
    "source_mapping_invalid",
    "extraction_gap",
    "reformulation_failure",
    "retrieval_gap",
    "selection_loss",
    "grader_false_rejection",
    "grader_false_acceptance",
    "generation_omission",
    "wrong_fact",
    "unsupported_claim",
    "citation_integrity_failure",
    "citation_support_failure",
    "scope_leak",
    "attribution_failure",
    "execution_error",
    "incomplete_trace",
]

ProviderQualityStatus = Literal[
    "not_run",
    "partial",
    "review_pending",
    "reviewed",
]

FactVerdict = Literal[
    "pass",
    "fail",
    "review_required",
    "unavailable",
]

RunTerminalState = Literal[
    "running",
    "completed",
    "incomplete",
    "failed",
]

ExposureStatus = Literal[
    "invoked_result_unknown",
    "diagnostic_exposed_local",
    "held_out_unexposed",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


LiveIdentifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")]
LiveDigest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
LiveTokenCount = Annotated[int, Field(ge=0)]


QUALITY_OUTPUT_CAPS: dict[Stage, int] = {**DEFAULT_OUTPUT_CAPS, "grading": 2048, "generation": 4096}


class LiveRunPlan(_StrictModel):
    """Frozen disclosure for future capture; this model grants no dispatch authorization."""

    plan_id: LiveIdentifier
    dataset_sha256: LiveDigest
    corpus_sha256: dict[LiveIdentifier, LiveDigest]
    prompt_sha256: dict[Stage, LiveDigest]
    configuration_sha256: LiveDigest
    destination_host: Annotated[str, Field(pattern=r"^[A-Za-z0-9.-]{1,253}$")]
    chat_model: LiveIdentifier
    embedding_model: LiveIdentifier
    allowance_tokens: Annotated[int, Field(ge=1)]
    maximum_calls: Annotated[int, Field(ge=1)]
    output_caps: dict[Stage, LiveTokenCount]
    max_attempts: Literal[1] = 1
    retries: Literal[0] = 0
    concurrency: Literal[1] = 1
    execution_mode: Literal["synchronous"] = "synchronous"
    usage_evidence: Literal["provider", "synthetic"]

    @model_validator(mode="after")
    def validate_bindings(self) -> Self:
        if not self.corpus_sha256:
            raise ValueError("Live plan requires pinned corpus hashes")
        if set(self.prompt_sha256) != {"reformulation", "grading", "generation"}:
            raise ValueError("Live plan requires every chat prompt hash")
        expected_caps = (
            QUALITY_OUTPUT_CAPS if self.plan_id == "task-6-1-five-case-v2" else DEFAULT_OUTPUT_CAPS
        )
        if self.output_caps != expected_caps:
            raise ValueError("Live plan must preserve its versioned bounded output caps")
        return self


class CallIdentity(_StrictModel):
    run_id: LiveIdentifier
    case_key: LiveIdentifier | None
    stage: Stage
    call_id: Annotated[int, Field(ge=1)]
    attempt: Literal[1] = 1

    @property
    def identity(self) -> tuple[str, str | None, Stage, int, int]:
        return self.run_id, self.case_key, self.stage, self.call_id, self.attempt


class CallReservation(CallIdentity):
    kind: Literal["reservation"] = "reservation"
    input_tokens_estimate: LiveTokenCount
    output_tokens_cap: LiveTokenCount
    reserved_tokens: Annotated[int, Field(ge=1)]
    payload_sha256: LiveDigest


class CallDispatchIntent(CallIdentity):
    """Committed before transport; cannot prove provider receipt after a crash."""

    kind: Literal["dispatch_intent"] = "dispatch_intent"


class CallOutcome(CallIdentity):
    kind: Literal["outcome"] = "outcome"
    outcome: Literal["not_dispatched", "succeeded", "failed", "outcome_unknown"]
    source: Literal["provider_reported", "synthetic_usage", "unavailable"]
    input_tokens: LiveTokenCount | None = None
    output_tokens: LiveTokenCount | None = None
    total_tokens: LiveTokenCount | None = None
    error_code: Literal["provider_error", "transport_error", "capture_error"] | None = None
    provider_request_sha256: LiveDigest | None = None

    @model_validator(mode="after")
    def validate_usage(self) -> Self:
        counts = (self.input_tokens, self.output_tokens, self.total_tokens)
        if self.source == "unavailable" and any(value is not None for value in counts):
            raise ValueError("Unavailable usage cannot contain token counts")
        if self.outcome == "not_dispatched" and (
            self.source != "unavailable" or self.provider_request_sha256 is not None
        ):
            raise ValueError("Not-dispatched calls cannot contain provider evidence")
        if all(value is not None for value in counts):
            if self.total_tokens != (self.input_tokens or 0) + (self.output_tokens or 0):
                raise ValueError("Provider usage fields disagree")
        return self


CallEvent = Annotated[
    CallReservation | CallDispatchIntent | CallOutcome, Field(discriminator="kind")
]


class ContractScenario(_StrictModel):
    scenario_id: str
    name: str
    description: str
    is_synthetic: bool = True
    expected_route: str | None = None


class CaseExposureNote(_StrictModel):
    case_id: str
    exposure_status: ExposureStatus
    note: str


class EvaluationSuiteManifest(_StrictModel):
    schema_version: str
    suite_id: str
    source_manifest: str
    source_manifest_sha256: str
    diagnostic_case_ids: list[str]
    contract_scenarios: list[ContractScenario]
    case_exposure_notes: list[CaseExposureNote]


class EvaluationSuite(_StrictModel):
    manifest: EvaluationSuiteManifest
    source_dataset: SourceDataset
    selected_cases: list[EvaluationCase]
    excluded_case_ids: list[str]


class FactAssessment(_StrictModel):
    fact_id: str
    claim: str
    exact_values: list[str]
    unit: str | None = None
    qualifications: list[str] = Field(default_factory=list)
    extraction_status: Literal["mapped", "extraction_failure"]
    anchor_hit: bool
    verdict: FactVerdict
    reason: str | None = None
    citation_indices: list[int] = Field(default_factory=list)
    source_anchor_ids: list[str] = Field(default_factory=list)


class CitationPayloadItem(_StrictModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    section_title: str | None = None
    quote: str | None = None


class SemanticFactReview(_StrictModel):
    verdict: FactVerdict
    reason: str = Field(min_length=1)
    supporting_citation_tags: list[int] = Field(default_factory=list)
    supporting_anchor_ids: list[str] = Field(default_factory=list)


class SemanticCaseReview(_StrictModel):
    case_id: str
    run_id: str
    answer_sha256: str
    context_sha256: str | None
    source_dataset_sha256: str
    facts: dict[str, SemanticFactReview]
    attribution: SemanticFactReview | None = None
    refusal: SemanticFactReview | None = None


def review_digest(value: JsonValue) -> str:
    """Hash exact UTF-8 text or canonical ordered JSON without lossy previews."""
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RetainedMetadata(ChunkMetadata):
    model_config = ConfigDict(extra="forbid", strict=True)


class RetainedChunk(DocumentChunk):
    model_config = ConfigDict(extra="forbid", strict=True)
    metadata: RetainedMetadata
    score: float | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.chunk_id != self.metadata.chunk_id:
            raise ValueError("Retained chunk identity disagrees with metadata")
        return self


class FreshCaseInput(_StrictModel):
    """Local-only exact pilot inputs. No headers, credentials or raw history are retained."""

    schema_version: Literal[1] = 1
    run_id: LiveIdentifier
    case_id: LiveIdentifier
    case_sha256: LiveDigest
    source_dataset_sha256: LiveDigest
    source_sha256: dict[LiveIdentifier, LiveDigest]
    history_sha256: LiveDigest
    question: str
    declared_scope: list[str]
    effective_scope: list[str]
    standalone_query: str | None
    raw_grade: str | None
    route: Literal["generate", "insufficient", "error", "incomplete"]
    response_text: str | None
    retrieved_chunks: list[RetainedChunk] | None
    supplied_chunks: list[RetainedChunk] | None
    rendered_context: str | None
    citations: list[CitationPayloadItem] | None
    request_sha256: dict[Stage, LiveDigest]
    prompt_sha256: dict[Stage, LiveDigest]
    settings_sha256: LiveDigest
    absence_review_sha256: LiveDigest | None = None

    @model_validator(mode="after")
    def validate_chunk_lists(self) -> Self:
        for chunks in (self.retrieved_chunks, self.supplied_chunks):
            if chunks is not None and len({c.chunk_id for c in chunks}) != len(chunks):
                raise ValueError("Duplicate retained chunk identities")
        return self

    @property
    def context_sha256(self) -> str | None:
        if self.supplied_chunks is None or self.rendered_context is None:
            return None
        return review_digest(
            {
                "chunks": [c.model_dump() for c in self.supplied_chunks],
                "rendered_context": self.rendered_context,
            }
        )


class FreshCaseReview(SemanticCaseReview):
    """Complete source review bound to one atomically committed fresh input."""

    schema_version: Literal[1] = 1
    snapshot_sha256: LiveDigest
    extra_claims: SemanticFactReview | None = None
    all_claims_reviewed: bool = False
    sufficient_supplied_evidence: bool = False
    absence_review_sha256: LiveDigest | None = None


class RecordedArmData(_StrictModel):
    name: Literal["baseline", "candidate"]
    status: Literal["completed", "failed", "incomplete"]
    response_text: str | None = None
    citations: list[CitationPayloadItem] | None = None
    chunk_ids: list[str] | None = None
    usage: dict[str, JsonValue] | None = None
    grade: str | None = None
    full_chunks: list[DocumentChunk] | None = None


class RecordedCaseReport(_StrictModel):
    case_id: str
    run_id: str
    question: str
    dataset_id: str
    arms: list[RecordedArmData]
    source_dataset_sha256: str | None = None
    standalone_query: str | None = None


class CaseEvaluationRecord(_StrictModel):
    case_id: str
    split: Literal["diagnostic", "held_out"]
    exposure_status: str
    question: str
    declared_scope: list[str]
    effective_scope: list[str]
    standalone_query: str | None = None
    result_kind: ResultKind
    retrieval_grade: str | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    supplied_chunk_ids: list[str] = Field(default_factory=list)
    response_text: str | None = None
    response_hash: str | None = None
    raw_citation_tags: list[int] = Field(default_factory=list)
    citations: list[CitationPayloadItem] = Field(default_factory=list)
    facts: list[FactAssessment] = Field(default_factory=list)
    failure_codes: list[FailureCode] = Field(default_factory=list)
    earliest_failing_stage: FailureCode | None = None
    evidence_level: EvidenceLevel
    latency_seconds: float | None = None
    run_id: str | None = None
    arm_name: Literal["baseline", "candidate"] | None = None
    context_sha256: str | None = None
    source_dataset_sha256: str | None = None
    claim_support: FactVerdict = "review_required"
    semantic_review: FreshCaseReview | SemanticCaseReview | None = None
    fresh_input_sha256: LiveDigest | None = None
    absence_review_sha256: LiveDigest | None = None


class EvaluationRunSummary(_StrictModel):
    schema_version: str
    suite_id: str
    run_id: str
    created_at: str
    evidence_level: EvidenceLevel
    terminal_state: RunTerminalState
    harness_implementation_verified: bool
    provider_quality_validation_status: ProviderQualityStatus
    selected_case_ids: list[str]
    excluded_case_ids: list[str]
    total_planned_cases: int
    completed_cases: int
    failed_cases: int
    unreviewed_cases: int


def load_suite_manifest(
    manifest_path: Path,
    *,
    project_root: Path,
) -> EvaluationSuite:
    """Load and strictly validate the Task 6.1 suite selection manifest.

    Guarantees that:
    1. All file paths remain strictly inside the project root.
    2. The canonical dataset file's SHA-256 matches the manifest checksum.
    3. Selected diagnostic cases exist in the canonical dataset and are marked 'diagnostic'.
    4. Held-out cases are never included in the default diagnostic selection.
    5. The prior unrecovered held-out invocation remains documented with 'invoked_result_unknown'.
    6. No network or provider client is constructed during loading.
    """
    root = project_root.resolve()
    resolved_manifest_path = manifest_path.resolve()

    if not resolved_manifest_path.is_file():
        raise ValueError(f"Suite manifest path is missing: {manifest_path}")

    try:
        manifest_data = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
        manifest = EvaluationSuiteManifest.model_validate(manifest_data)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise ValueError(f"Invalid evaluation suite manifest: {error}") from error

    source_manifest_path = (root / manifest.source_manifest).resolve()
    if not source_manifest_path.is_relative_to(root) or not source_manifest_path.is_file():
        raise ValueError(
            "Canonical source manifest is outside the project or missing: "
            f"{manifest.source_manifest}"
        )

    actual_sha256 = hashlib.sha256(source_manifest_path.read_bytes()).hexdigest().lower()
    if actual_sha256 != manifest.source_manifest_sha256.lower():
        raise ValueError(
            "Canonical source manifest SHA-256 drift "
            f"(expected {manifest.source_manifest_sha256}, found {actual_sha256})"
        )

    source_dataset = load_source_dataset(source_manifest_path, project_root=root)

    # Validate diagnostic case selections
    if len(manifest.diagnostic_case_ids) != len(set(manifest.diagnostic_case_ids)):
        raise ValueError("Duplicate case IDs in diagnostic_case_ids")

    canonical_by_id = {case.id: case for case in source_dataset.cases}
    selected_cases: list[EvaluationCase] = []

    for case_id in manifest.diagnostic_case_ids:
        if case_id not in canonical_by_id:
            raise ValueError(
                f"Unknown diagnostic case ID '{case_id}' not found in canonical dataset"
            )

        case = canonical_by_id[case_id]
        if case.split == "held_out":
            raise ValueError(
                f"Held-out case '{case_id}' cannot be included in diagnostic suite selection"
            )
        if case.split != "diagnostic":
            raise ValueError(
                f"Case '{case_id}' has unexpected split '{case.split}', must be 'diagnostic'"
            )
        selected_cases.append(case)

    # Validate contract scenarios
    scenario_ids = [s.scenario_id for s in manifest.contract_scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Duplicate scenario ID in contract_scenarios")

    # Validate exposure notes: Session 44 held-out case must be marked invoked_result_unknown
    notes_by_id = {n.case_id: n for n in manifest.case_exposure_notes}
    held_out_unknown = "held-04-tada-llama-single-column"
    if (
        held_out_unknown not in notes_by_id
        or notes_by_id[held_out_unknown].exposure_status != "invoked_result_unknown"
    ):
        raise ValueError(
            f"Missing required exposure note for '{held_out_unknown}' "
            "with status 'invoked_result_unknown'"
        )

    # Compute excluded cases
    selected_id_set = set(manifest.diagnostic_case_ids)
    excluded_case_ids = [c.id for c in source_dataset.cases if c.id not in selected_id_set]

    return EvaluationSuite(
        manifest=manifest,
        source_dataset=source_dataset,
        selected_cases=selected_cases,
        excluded_case_ids=excluded_case_ids,
    )
