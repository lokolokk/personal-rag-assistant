"""Answer and citation integrity assessment for Task 6.1 (Slice B).

Validates:
1. Raw inline citation tags ([1], [2]) directly from response text.
2. Ghost citations (out-of-range indices like [99] dropped by production resolver).
3. Citation payload metadata accuracy (chunk_id, filename, page, document_id).
4. Consistency across snake_case and camelCase payload aliases.
5. Full chunk evidence support (avoiding false rejection from 200-char quote truncation).
6. Required fact completeness (rejecting answers omitting '5-shot' or other qualifications).
7. Swapped metric/model associations (e.g. Macro vs Micro F1).
8. Execution-error provenance vs appropriate model abstention.
9. Scope integrity and retrieval leakage.
"""

import hashlib
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from app.schemas.document import DocumentChunk, ScoredChunk
from app.services.evaluation.grounding_report import (
    CaseEvaluationRecord,
    CitationPayloadItem,
    EvidenceLevel,
    FactAssessment,
    FactVerdict,
    FailureCode,
    FreshCaseInput,
    FreshCaseReview,
    RecordedArmData,
    RecordedCaseReport,
    ResultKind,
    SemanticCaseReview,
    SemanticFactReview,
    review_digest,
)
from app.services.evaluation.source_dataset import EvaluationCase, RequiredFact
from app.services.generation.grounding import (
    FALLBACK_RESPONSE,
    _extract_chunk_attributes,
    format_context_chunks,
)

if TYPE_CHECKING:
    from app.services.evaluation.usage import DurableUsageCapture

_FAILURE_STAGE_ORDER: list[FailureCode] = [
    "source_mapping_invalid",
    "extraction_gap",
    "scope_leak",
    "retrieval_gap",
    "selection_loss",
    "grader_false_rejection",
    "grader_false_acceptance",
    "wrong_fact",
    "generation_omission",
    "unsupported_claim",
    "citation_integrity_failure",
    "citation_support_failure",
    "attribution_failure",
    "execution_error",
    "incomplete_trace",
]


def _extract_chunk_content(chunk: Any) -> str:
    """Extract content string from chunk object or dict."""
    if isinstance(chunk, (DocumentChunk, ScoredChunk)):
        return chunk.content
    if isinstance(chunk, dict):
        return str(chunk.get("content", ""))
    return str(getattr(chunk, "content", ""))


def _normalize_citation_payload_item(item: Any) -> tuple[CitationPayloadItem, bool]:
    """Normalize a citation item into CitationPayloadItem and detect alias conflicts.

    Returns:
        tuple of (CitationPayloadItem, has_alias_conflict: bool)
    """
    has_alias_conflict = False
    if isinstance(item, CitationPayloadItem):
        return item, False

    if isinstance(item, dict):
        chunk_id = str(item.get("chunk_id", item.get("chunkId", "")))
        doc_id = str(item.get("document_id", item.get("documentId", "")))
        filename = str(item.get("filename", "unknown.pdf"))
        raw_page = item.get("page_number", item.get("pageNumber", item.get("page", 1)))
        page_number = int(raw_page) if raw_page is not None else 1
        section_title = item.get("section_title", item.get("sectionTitle"))
        quote = item.get("quote")

        # Check for conflicting aliases
        if "chunk_id" in item and "chunkId" in item and item["chunk_id"] != item["chunkId"]:
            has_alias_conflict = True
        if (
            "document_id" in item
            and "documentId" in item
            and item["document_id"] != item["documentId"]
        ):
            has_alias_conflict = True
        if (
            "page_number" in item
            and "pageNumber" in item
            and item["page_number"] != item["pageNumber"]
        ):
            has_alias_conflict = True

        return (
            CitationPayloadItem(
                chunk_id=chunk_id,
                document_id=doc_id,
                filename=filename,
                page_number=page_number,
                section_title=str(section_title) if section_title is not None else None,
                quote=str(quote) if quote is not None else None,
            ),
            has_alias_conflict,
        )

    # Fallback attribute extraction
    return (
        CitationPayloadItem(
            chunk_id=str(getattr(item, "chunk_id", getattr(item, "chunkId", ""))),
            document_id=str(getattr(item, "document_id", getattr(item, "documentId", ""))),
            filename=str(getattr(item, "filename", "unknown.pdf")),
            page_number=int(getattr(item, "page_number", getattr(item, "pageNumber", 1))),
            section_title=getattr(item, "section_title", None),
            quote=getattr(item, "quote", None),
        ),
        False,
    )


def _check_swapped_associations(text: str, fact: RequiredFact) -> bool:
    """Detect if numbers are swapped across named metrics (e.g. Micro F1 vs Macro F1)."""
    text_lower = text.lower()

    if "micro" in text_lower and "macro" in text_lower and len(fact.exact_values) >= 2:
        val0 = fact.exact_values[0].lower()
        val1 = fact.exact_values[1].lower()
        m0 = re.search(r"[0-9.]+%?", val0)
        m1 = re.search(r"[0-9.]+%?", val1)
        num0 = m0.group(0) if m0 else val0
        num1 = m1.group(0) if m1 else val1

        micro_expected = num0
        macro_expected = num1
        for qual in list(fact.qualifications) + list(fact.exact_values):
            ql = qual.lower()
            if "macro" in ql and num0 in ql:
                macro_expected = num0
                micro_expected = num1
                break

        if (
            micro_expected in text_lower
            and macro_expected in text_lower
            and micro_expected != macro_expected
        ):
            p_micro = text_lower.find("micro")
            p_macro = text_lower.find("macro")

            first_idx, second_idx = (p_micro, p_macro) if p_micro < p_macro else (p_macro, p_micro)
            first_metric = "micro" if p_micro < p_macro else "macro"

            # Find delimiter between first and second metric
            between_text = text_lower[first_idx:second_idx]
            split_rel = -1
            for delim in [",", ";", " and ", " vs ", " versus ", " while "]:
                pos = between_text.rfind(delim)
                if pos != -1:
                    split_rel = pos + (len(delim) if delim.endswith(" ") else 1)
                    break

            split_point = (
                first_idx + split_rel if split_rel != -1 else (first_idx + second_idx) // 2
            )

            clause_first = text_lower[:split_point]
            clause_second = text_lower[split_point:]

            if first_metric == "micro":
                if macro_expected in clause_first and micro_expected not in clause_first:
                    return True
                if micro_expected in clause_second and macro_expected not in clause_second:
                    return True
            else:
                if micro_expected in clause_first and macro_expected not in clause_first:
                    return True
                if macro_expected in clause_second and micro_expected not in clause_second:
                    return True

    return False


def evaluate_answer_quality(
    case_id: str,
    question: str,
    response_text: str,
    supplied_chunks: Sequence[Any],
    citations_payload: Sequence[Any],
    required_facts: Sequence[RequiredFact],
    *,
    case_answerability: Literal["answerable", "unanswerable"] = "answerable",
    split: Literal["diagnostic", "held_out"] = "diagnostic",
    exposure_status: str = "diagnostic_exposed_local",
    declared_scope: Sequence[str] = (),
    effective_scope: Sequence[str] = (),
    standalone_query: str | None = None,
    execution_route: str | None = None,
    error_message: str | None = None,
    evidence_level: EvidenceLevel = "offline_contract",
    latency_seconds: float | None = None,
    semantic_reviews: dict[str, FactVerdict] | None = None,
) -> CaseEvaluationRecord:
    """Evaluate response correctness, citation integrity, and failure root causes.

    This evaluator parses raw citation brackets independently of the production
    resolver to capture dropped out-of-bounds tags (e.g. [99]).
    """
    failure_codes: list[FailureCode] = []
    normalized_text = response_text.strip()
    is_exact_fallback = normalized_text == FALLBACK_RESPONSE.strip()

    # Phase 1: Determine Result Kind & Provenance
    if execution_route == "error" or error_message is not None:
        result_kind: ResultKind = "execution_error"
        failure_codes.append("execution_error")
    elif is_exact_fallback:
        if case_answerability == "unanswerable":
            result_kind = "appropriate_abstention"
        else:
            result_kind = "incorrect_refusal"
            failure_codes.append("grader_false_rejection")
    else:
        if case_answerability == "unanswerable":
            result_kind = "unsupported_answer"
            failure_codes.append("unsupported_claim")
        else:
            result_kind = "answer"

    # Phase 2: Citation Tag & Payload Inspection
    raw_tag_matches = re.findall(r"\[(\d+)\]", response_text)
    raw_citation_tags = [int(m) for m in raw_tag_matches]
    unique_tags = list(dict.fromkeys(raw_citation_tags))

    normalized_citations: list[CitationPayloadItem] = []
    alias_conflict_found = False

    for item in citations_payload:
        norm_item, conflict = _normalize_citation_payload_item(item)
        normalized_citations.append(norm_item)
        if conflict:
            alias_conflict_found = True

    if alias_conflict_found:
        if "citation_integrity_failure" not in failure_codes:
            failure_codes.append("citation_integrity_failure")

    # Check ghost citations (out-of-range 1-based indices)
    num_supplied = len(supplied_chunks)
    for tag in unique_tags:
        if tag < 1 or tag > num_supplied:
            if "citation_integrity_failure" not in failure_codes:
                failure_codes.append("citation_integrity_failure")

    # Check missing citations on answerable case
    if result_kind == "answer" and not unique_tags:
        if "citation_integrity_failure" not in failure_codes:
            failure_codes.append("citation_integrity_failure")

    # Check payload accuracy against supplied chunks
    for i, tag in enumerate(unique_tags):
        if 1 <= tag <= num_supplied:
            target_chunk = supplied_chunks[tag - 1]
            (
                act_chunk_id,
                act_doc_id,
                act_filename,
                act_page,
                _,
                _,
            ) = _extract_chunk_attributes(target_chunk)

            if i < len(normalized_citations):
                cit = normalized_citations[i]
                if (
                    cit.chunk_id != act_chunk_id
                    or cit.document_id != act_doc_id
                    or cit.filename != act_filename
                    or cit.page_number != act_page
                ):
                    if "citation_integrity_failure" not in failure_codes:
                        failure_codes.append("citation_integrity_failure")

    # Phase 3: Scope Enforcement (Detect Leakage)
    if declared_scope:
        declared_set = set(declared_scope)
        for chunk in supplied_chunks:
            _, chunk_doc, _, _, _, _ = _extract_chunk_attributes(chunk)
            if chunk_doc and chunk_doc not in declared_set:
                if "scope_leak" not in failure_codes:
                    failure_codes.append("scope_leak")

        for cit in normalized_citations:
            if cit.document_id and cit.document_id not in declared_set:
                if "scope_leak" not in failure_codes:
                    failure_codes.append("scope_leak")

    # Phase 4: Fact Assessment and Evidence Sufficiency
    # Gather full content of all valid cited chunks
    valid_cited_contents = [
        _extract_chunk_content(supplied_chunks[tag - 1])
        for tag in unique_tags
        if 1 <= tag <= num_supplied
    ]
    consolidated_cited_content = "\n".join(valid_cited_contents)

    fact_assessments: list[FactAssessment] = []

    for fact in required_facts:
        fact_reason: str | None = None
        fact_verdict: FactVerdict = "pass"
        values_present = all(val.lower() in normalized_text.lower() for val in fact.exact_values)

        # Check qualifications (e.g. '5-shot')
        qualifications_present = True
        for qual in fact.qualifications:
            clean_qual = qual.lower()
            if "shot" in clean_qual and clean_qual not in normalized_text.lower():
                qualifications_present = False
                fact_reason = f"Missing required qualification: {qual}"
                if "generation_omission" not in failure_codes:
                    failure_codes.append("generation_omission")
            elif clean_qual not in normalized_text.lower() and ":" not in clean_qual:
                qualifications_present = False
                fact_reason = f"Missing required qualification: {qual}"
                if "generation_omission" not in failure_codes:
                    failure_codes.append("generation_omission")

        # Check swapped associations
        if values_present and _check_swapped_associations(normalized_text, fact):
            fact_verdict = "fail"
            fact_reason = "Swapped metric/value association in response text"
            if "wrong_fact" not in failure_codes:
                failure_codes.append("wrong_fact")

        # Check claim-level citation support in full chunk content
        claim_supported_by_citations = True
        if valid_cited_contents:
            for val in fact.exact_values:
                if val.lower() not in consolidated_cited_content.lower():
                    claim_supported_by_citations = False
                    fact_reason = f"Cited context does not contain required value '{val}'"
                    if "citation_support_failure" not in failure_codes:
                        failure_codes.append("citation_support_failure")
                    break
        elif result_kind == "answer":
            claim_supported_by_citations = False
            fact_reason = "Answer has no valid citations to support claims"

        # Determine anchor hit
        anchor_hit = False
        if valid_cited_contents:
            for tag in unique_tags:
                if 1 <= tag <= num_supplied:
                    c_id, _, _, _, _, _ = _extract_chunk_attributes(supplied_chunks[tag - 1])
                    if c_id in fact.source_anchor_ids or any(
                        val.lower() in _extract_chunk_content(supplied_chunks[tag - 1]).lower()
                        for val in fact.exact_values
                    ):
                        anchor_hit = True
                        break

        # Final verdict assignment
        if (
            not values_present
            or not qualifications_present
            or not claim_supported_by_citations
            or fact_verdict == "fail"
        ):
            fact_verdict = "fail"
        elif semantic_reviews and fact.id in semantic_reviews:
            fact_verdict = semantic_reviews[fact.id]

        fact_assessments.append(
            FactAssessment(
                fact_id=fact.id,
                claim=fact.claim,
                exact_values=fact.exact_values,
                unit=fact.unit,
                qualifications=fact.qualifications,
                extraction_status="mapped",
                anchor_hit=anchor_hit,
                verdict=fact_verdict,
                reason=fact_reason,
                citation_indices=unique_tags,
                source_anchor_ids=fact.source_anchor_ids,
            )
        )

    # Determine earliest failing stage in pipeline order
    earliest_stage: FailureCode | None = None
    for stage in _FAILURE_STAGE_ORDER:
        if stage in failure_codes:
            earliest_stage = stage
            break

    response_hash = (
        hashlib.sha256(response_text.encode("utf-8")).hexdigest() if response_text else None
    )

    retrieved_chunk_ids = [_extract_chunk_attributes(c)[0] for c in supplied_chunks]

    return CaseEvaluationRecord(
        case_id=case_id,
        split=split,
        exposure_status=exposure_status,
        question=question,
        declared_scope=list(declared_scope),
        effective_scope=list(effective_scope) or list(declared_scope),
        standalone_query=standalone_query,
        result_kind=result_kind,
        retrieval_grade=execution_route,
        retrieved_chunk_ids=retrieved_chunk_ids,
        supplied_chunk_ids=retrieved_chunk_ids,
        response_text=response_text,
        response_hash=response_hash,
        raw_citation_tags=raw_citation_tags,
        citations=normalized_citations,
        facts=fact_assessments,
        failure_codes=failure_codes,
        earliest_failing_stage=earliest_stage,
        evidence_level=evidence_level,
        latency_seconds=latency_seconds,
    )


_SOURCE_DATASET_SHA256 = "026e0a47f372506c25e3523fc4ad1c05c01f6be189ceeadbab387433a081f282"
_PAIRED_REPORT_FIELDS = {
    "mode",
    "status",
    "run_id",
    "experiment_id",
    "dataset_id",
    "case_id",
    "question",
    "query_policy",
    "query_assumption",
    "provider_dispatch_authorized",
    "approved_token_allowance",
    "preflight_reservation_tokens",
    "maximum_provider_calls",
    "corpus",
    "model_configuration",
    "budget",
    "usage",
    "arms",
    "quality_summary",
    "failure_type",
    "manual_review",
    "selection_calls",
    "usage_limitations",
}
_PAIRED_ARM_FIELDS = {
    "name",
    "status",
    "response_origin",
    "chunk_ids",
    "context_tokens_local_estimate",
    "preflight_stage_reservations",
    "grade",
    "response_text",
    "is_exact_fallback",
    "citations",
    "citations_in_scope",
    "usage",
    "budget_reservations",
    "quality_review",
}


def _validate_recorded_citation(raw: object) -> CitationPayloadItem:
    """Require the inspected historical citation aliases to agree without coercion."""
    if not isinstance(raw, dict):
        raise ValueError("Recorded citation must be an object")
    allowed = {
        "id",
        "citationIndex",
        "chunk_id",
        "chunkId",
        "document_id",
        "documentId",
        "filename",
        "page_number",
        "pageNumber",
        "page",
        "section_title",
        "sectionTitle",
        "quote",
        "sourceText",
    }
    if set(raw) - allowed:
        raise ValueError("Unexpected recorded citation fields")
    for names in (
        ("chunk_id", "chunkId"),
        ("document_id", "documentId"),
        ("page_number", "pageNumber", "page"),
        ("section_title", "sectionTitle"),
        ("quote", "sourceText"),
        ("id", "citationIndex"),
    ):
        values = [raw[name] for name in names if name in raw]
        if values and any(value != values[0] for value in values):
            raise ValueError("Conflicting citation aliases")
    for name in ("chunk_id", "document_id", "filename"):
        if not isinstance(raw.get(name), str) or not raw[name]:
            raise ValueError(f"Recorded citation requires nonempty string '{name}'")
    if type(raw.get("page_number")) is not int or raw["page_number"] < 1:
        raise ValueError("Recorded citation requires positive integer page_number")
    for name in ("section_title", "quote"):
        if raw.get(name) is not None and not isinstance(raw[name], str):
            raise ValueError(f"Recorded citation '{name}' must be text or null")
    return CitationPayloadItem.model_validate(
        {
            name: raw.get(name)
            for name in (
                "chunk_id",
                "document_id",
                "filename",
                "page_number",
                "section_title",
                "quote",
            )
        }
    )


def load_recorded_case_report(path: Path) -> RecordedCaseReport:
    """Adapt only the retained Task 5.9.1 frozen-evidence paired report, offline.

    Citation previews never become full context. Historical usage is preserved verbatim,
    and neither old dispatch authorization nor manual-review summaries authorize execution.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) - _PAIRED_REPORT_FIELDS:
            raise ValueError("Unexpected paired-report fields")
        if data.get("mode") != "live_frozen_evidence_comparison":
            raise ValueError("Expected live_frozen_evidence_comparison artifact")
        if data.get("status") != "completed":
            raise ValueError("Only completed retained paired reports can be adapted")
        arms: list[RecordedArmData] = []
        for item in data["arms"]:
            if not isinstance(item, dict) or set(item) - _PAIRED_ARM_FIELDS:
                raise ValueError("Unexpected paired-arm fields")
            citations = None
            if item.get("citations") is not None:
                citations = []
                for raw in item["citations"]:
                    citations.append(_validate_recorded_citation(raw))
            arms.append(
                RecordedArmData.model_validate(
                    {
                        key: item.get(key)
                        for key in (
                            "name",
                            "status",
                            "response_text",
                            "chunk_ids",
                            "usage",
                            "grade",
                        )
                    }
                    | {"citations": citations}
                )
            )
        if [arm.name for arm in arms] != ["baseline", "candidate"]:
            raise ValueError("Expected exactly baseline and candidate arms in that order")
        return RecordedCaseReport.model_validate(
            {key: data[key] for key in ("case_id", "run_id", "question", "dataset_id")}
            | {"arms": arms}
        )
    except (
        OSError,
        json.JSONDecodeError,
        ValidationError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(f"Invalid recorded report '{path.name}': {error}") from error


def replay_recorded_case(
    report: RecordedCaseReport,
    arm_name: Literal["baseline", "candidate"],
    case: EvaluationCase,
) -> CaseEvaluationRecord:
    """Assess retained text against canonical facts without recreating missing traces."""
    if case.split == "held_out" or report.case_id.startswith("held-"):
        raise ValueError("Held-out replay prohibited; invoked_result_unknown remains unknown")
    if report.case_id != case.id or report.question != case.question:
        raise ValueError("Recorded case/question does not match canonical case")
    dataset_path = Path(__file__).resolve().parents[3] / (
        "tests/evaluation/fixtures/task_5_9_1_source_dataset.json"
    )
    dataset_bytes = dataset_path.read_bytes()
    if hashlib.sha256(dataset_bytes).hexdigest() != _SOURCE_DATASET_SHA256:
        raise ValueError("Canonical source dataset SHA-256 drift")
    dataset = json.loads(dataset_bytes)
    canonical = next((item for item in dataset["cases"] if item["id"] == case.id), None)
    if canonical != case.model_dump() or report.dataset_id != dataset["dataset_id"]:
        # Compare parsed schemas because historical annotations omit default fields.
        if canonical is None or EvaluationCase.model_validate(canonical) != case:
            raise ValueError("Replay requires unchanged canonical facts")
        if report.dataset_id != dataset["dataset_id"]:
            raise ValueError("Recorded dataset identity mismatch")
    if report.source_dataset_sha256 not in (None, _SOURCE_DATASET_SHA256):
        raise ValueError("Recorded source dataset SHA-256 drift")
    arms = [arm for arm in report.arms if arm.name == arm_name]
    if len(arms) != 1:
        raise ValueError("Recorded arm is missing or duplicated")
    arm = arms[0]
    if arm.status != "completed" or arm.response_text is None:
        raise ValueError("Recorded arm has no completed answer; execution remains unknown")
    chunks = arm.full_chunks
    if chunks is not None and [chunk.chunk_id for chunk in chunks] != arm.chunk_ids:
        raise ValueError("Full context order does not match recorded chunk IDs")
    # Composite source annotations are split into independently checkable qualifications.
    facts = [
        fact.model_copy(
            update={
                "exact_values": [
                    match.group()
                    if (match := re.search(r"\d+(?:\.\d+)?%?", value.replace("F1", "F-score")))
                    else value
                    for value in fact.exact_values
                ],
                "qualifications": [
                    re.sub(r"\s+Table \d+\.?$", "", part.strip().rstrip("."))
                    for qual in fact.qualifications
                    for part in qual.split(";")
                ],
            }
        )
        for fact in case.required_facts
    ]
    record = evaluate_answer_quality(
        case_id=case.id,
        question=case.question,
        response_text=arm.response_text,
        supplied_chunks=chunks or [],
        citations_payload=arm.citations or [],
        required_facts=facts,
        case_answerability=case.answerability,
        declared_scope=case.scope_document_ids,
        standalone_query=report.standalone_query,
        evidence_level="provider_answer_review",
    )
    if chunks is None:
        # Validate tag bounds using retained IDs without pretending to have source text.
        count = len(arm.chunk_ids or [])
        bad_tags = any(tag < 1 or tag > count for tag in record.raw_citation_tags)
        record.failure_codes = [
            code
            for code in record.failure_codes
            if code not in ("citation_support_failure", "citation_integrity_failure")
        ]
        valid_tags = list(dict.fromkeys(record.raw_citation_tags))
        payload_matches = arm.citations is not None and len(arm.citations) == len(valid_tags)
        if payload_matches and arm.citations is not None:
            payload_matches = all(
                1 <= tag <= count and citation.chunk_id == (arm.chunk_ids or [])[tag - 1]
                for tag, citation in zip(valid_tags, arm.citations, strict=True)
            )
        if bad_tags or (
            record.result_kind in ("answer", "unsupported_answer")
            and (not record.raw_citation_tags or not payload_matches)
        ):
            record.failure_codes.append("citation_integrity_failure")
        record.failure_codes.append("incomplete_trace")
        for fact, expected in zip(record.facts, facts, strict=True):
            missing_values = any(
                value.lower() not in arm.response_text.lower() for value in expected.exact_values
            )
            missing = any(
                q.lower() not in arm.response_text.lower()
                for q in expected.qualifications
                if ":" not in q
            )
            fact.verdict = (
                "fail"
                if missing or missing_values or "wrong_fact" in record.failure_codes
                else "unavailable"
            )
            fact.reason = (
                "Missing required qualification (including 5-shot)"
                if missing
                else "Full recorded context unavailable"
            )
            fact.anchor_hit = False
    else:
        if len(arm.citations or []) != len(set(record.raw_citation_tags)):
            if "citation_integrity_failure" not in record.failure_codes:
                record.failure_codes.append("citation_integrity_failure")
            for fact in record.facts:
                fact.verdict = "fail"
                fact.reason = "Recorded citation payload is incomplete"
        record.claim_support = "review_required"
        for fact in record.facts:
            if fact.verdict == "pass":
                fact.verdict = "review_required"
                fact.reason = "Structural checks passed; source semantic review required"
    record.claim_support = "unavailable" if chunks is None else "review_required"
    record.run_id = report.run_id
    record.arm_name = arm_name
    record.context_sha256 = (
        hashlib.sha256(
            json.dumps(
                [chunk.model_dump() for chunk in chunks],
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if chunks is not None
        else None
    )
    for assessment, canonical_fact in zip(record.facts, case.required_facts, strict=True):
        assessment.exact_values = list(canonical_fact.exact_values)
        assessment.qualifications = list(canonical_fact.qualifications)
        assessment.source_anchor_ids = list(
            dict.fromkeys(
                canonical_fact.source_anchor_ids
                + canonical_fact.acceptable_alternative_evidence_anchor_ids
            )
        )
    record.source_dataset_sha256 = _SOURCE_DATASET_SHA256
    record.supplied_chunk_ids = list(arm.chunk_ids or [])
    record.retrieved_chunk_ids = []  # The original retrieval trace was not retained.
    record.earliest_failing_stage = next(
        (stage for stage in _FAILURE_STAGE_ORDER if stage in record.failure_codes), None
    )
    return record


def apply_semantic_review(
    record: CaseEvaluationRecord,
    review: SemanticCaseReview,
    *,
    snapshot_path: Path | None = None,
) -> CaseEvaluationRecord:
    """Apply a source review only to its exact retained answer/context/dataset binding."""
    actual_answer_hash = hashlib.sha256((record.response_text or "").encode("utf-8")).hexdigest()
    if (
        review.case_id != record.case_id
        or review.run_id != record.run_id
        or review.answer_sha256 != actual_answer_hash
        or record.response_hash != actual_answer_hash
        or review.context_sha256 != record.context_sha256
        or review.source_dataset_sha256 != record.source_dataset_sha256
        or review.source_dataset_sha256 != _SOURCE_DATASET_SHA256
    ):
        raise ValueError("Semantic review binding does not match retained case provenance")
    if record.split == "held_out":
        raise ValueError("Held-out review prohibited; invoked_result_unknown remains unknown")
    if isinstance(review, FreshCaseReview) and record.fresh_input_sha256 is None:
        raise ValueError("Fresh review requires a fresh committed snapshot binding")
    if record.fresh_input_sha256 is not None:
        if not isinstance(review, FreshCaseReview) or snapshot_path is None:
            raise ValueError("Fresh review requires its committed snapshot")
        snapshot = load_fresh_case(snapshot_path)
        expected = evaluate_fresh_case(snapshot_path)
        if (
            review.snapshot_sha256 != record.fresh_input_sha256
            or review.snapshot_sha256 != review_digest(snapshot.model_dump())
            or review.absence_review_sha256 != expected.absence_review_sha256
            or record.model_dump(exclude={"semantic_review"})
            != expected.model_dump(exclude={"semantic_review"})
        ):
            raise ValueError("Fresh review binding does not match committed snapshot")
        return _apply_fresh_review(record, review)
    facts_by_id = {fact.fact_id: fact for fact in record.facts}
    if set(review.facts) - set(facts_by_id):
        raise ValueError("Semantic review contains unknown fact IDs")
    result = record.model_copy(deep=True)
    for fact in result.facts:
        decision = review.facts.get(fact.fact_id)
        if decision is None:
            continue
        if set(decision.supporting_citation_tags) - set(record.raw_citation_tags) or set(
            decision.supporting_anchor_ids
        ) - set(fact.source_anchor_ids):
            raise ValueError("Semantic review references unavailable citations or anchors")
        if decision.verdict == "pass" and record.claim_support != "unavailable":
            if not decision.supporting_citation_tags or not decision.supporting_anchor_ids:
                raise ValueError("Passing source review requires citations and source anchors")
        if fact.verdict not in ("fail", "unavailable"):
            fact.verdict = decision.verdict
            fact.reason = decision.reason
        if decision.verdict == "fail" and "wrong_fact" not in result.failure_codes:
            result.failure_codes.append("wrong_fact")
            fact.verdict = "fail"
            fact.reason = decision.reason
    case_decisions: list[tuple[SemanticFactReview | None, FailureCode]] = [
        (review.attribution, "attribution_failure"),
        (review.refusal, "grader_false_rejection"),
    ]
    for decision, code in case_decisions:
        if decision is not None and decision.verdict == "fail" and code not in result.failure_codes:
            result.failure_codes.append(code)
    result.semantic_review = review.model_copy(deep=True)
    if record.claim_support != "unavailable":
        result.claim_support = (
            "pass"
            if result.facts and all(fact.verdict == "pass" for fact in result.facts)
            else "fail"
            if any(fact.verdict == "fail" for fact in result.facts)
            else "review_required"
        )
    result.earliest_failing_stage = next(
        (stage for stage in _FAILURE_STAGE_ORDER if stage in result.failure_codes), None
    )
    return result


class SnapshotCommitError(BaseException):
    """Stop dispatch even when production code catches ordinary provider exceptions."""


def commit_fresh_case(
    root: Path, snapshot: FreshCaseInput, *, capture: "DurableUsageCapture | None" = None
) -> Path:
    """Exclusively commit a local input; failure poisons a supplied active capture."""
    try:
        snapshot = FreshCaseInput.model_validate(snapshot.model_dump())
        directory = root / snapshot.run_id / snapshot.case_id
        directory.mkdir(parents=True, exist_ok=False)
        pending = directory / "input.pending"
        envelope = {"sha256": review_digest(snapshot.model_dump()), "input": snapshot.model_dump()}
        with pending.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(envelope, sort_keys=True, ensure_ascii=False))
            handle.flush()
            os.fsync(handle.fileno())
        path = directory / "input.json"
        os.replace(pending, path)
        return path
    except (OSError, ValueError):
        if capture is not None:
            capture.stop("snapshot_commit_failed")
        raise SnapshotCommitError("Fresh snapshot commit failed; dispatch must stop") from None


def load_fresh_case(path: Path) -> FreshCaseInput:
    """Read only complete committed snapshots, rejecting mutation and partial writes."""
    try:
        if path.name != "input.json":
            raise ValueError("Uncommitted input")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {"sha256", "input"}:
            raise ValueError("Unexpected envelope fields")
        snapshot = FreshCaseInput.model_validate(data["input"])
        if data["sha256"] != review_digest(snapshot.model_dump()):
            raise ValueError("Snapshot checksum mismatch")
        if path.parent.name != snapshot.case_id or path.parent.parent.name != snapshot.run_id:
            raise ValueError("Snapshot path identity mismatch")
        _fresh_canonical_case(snapshot)
        return snapshot
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError("Invalid or missing fresh snapshot") from error


def _fresh_canonical_case(snapshot: FreshCaseInput) -> EvaluationCase:
    path = Path(__file__).resolve().parents[3] / (
        "tests/evaluation/fixtures/task_5_9_1_source_dataset.json"
    )
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != _SOURCE_DATASET_SHA256:
        raise ValueError("Canonical source dataset drift")
    data = json.loads(content)
    case = next((c for c in data["cases"] if c["id"] == snapshot.case_id), None)
    if case is None:
        raise ValueError("Unknown fresh case")
    canonical = EvaluationCase.model_validate(case)
    if canonical.split == "held_out" or snapshot.case_id.startswith("held-"):
        raise ValueError("Held-out invocation remains unknown")
    absence_hash = (
        review_digest(canonical.absence_review.model_dump()) if canonical.absence_review else None
    )
    if (
        snapshot.source_dataset_sha256 != _SOURCE_DATASET_SHA256
        or snapshot.case_sha256 != review_digest(canonical.model_dump())
        or snapshot.history_sha256 != review_digest([h.model_dump() for h in canonical.history])
        or snapshot.question != canonical.question
        or snapshot.declared_scope != canonical.scope_document_ids
        or snapshot.source_sha256 != {d["id"]: d["sha256"] for d in data["documents"]}
        or snapshot.absence_review_sha256 != absence_hash
    ):
        raise ValueError("Fresh canonical binding mismatch")
    return canonical


def evaluate_fresh_case(path: Path) -> CaseEvaluationRecord:
    """Assess exact committed inputs; source semantic acceptance remains a separate review."""
    snapshot = load_fresh_case(path)
    case = _fresh_canonical_case(snapshot)
    facts = [
        fact.model_copy(
            update={
                "exact_values": [
                    match.group()
                    if (match := re.search(r"\d+(?:\.\d+)?%?", value.replace("F1", "F-score")))
                    else value
                    for value in fact.exact_values
                ],
                "qualifications": [
                    re.sub(r"\s+Table \d+\.?$", "", part.strip().rstrip("."))
                    for qual in fact.qualifications
                    for part in qual.split(";")
                ],
            }
        )
        for fact in case.required_facts
    ]
    record = evaluate_answer_quality(
        case_id=case.id,
        question=case.question,
        response_text=snapshot.response_text or "",
        supplied_chunks=snapshot.supplied_chunks or [],
        citations_payload=snapshot.citations or [],
        required_facts=facts,
        case_answerability=case.answerability,
        declared_scope=snapshot.declared_scope,
        effective_scope=snapshot.effective_scope,
        standalone_query=snapshot.standalone_query,
        execution_route=snapshot.route,
        evidence_level="provider_answer_review",
    )
    record.run_id = snapshot.run_id
    record.retrieval_grade = snapshot.raw_grade
    record.context_sha256 = snapshot.context_sha256
    record.source_dataset_sha256 = snapshot.source_dataset_sha256
    record.fresh_input_sha256 = review_digest(snapshot.model_dump())
    record.absence_review_sha256 = snapshot.absence_review_sha256
    record.retrieved_chunk_ids = [c.chunk_id for c in snapshot.retrieved_chunks or []]
    complete = (
        snapshot.response_text is not None
        and snapshot.retrieved_chunks is not None
        and snapshot.context_sha256 is not None
        and snapshot.citations is not None
        and snapshot.standalone_query is not None
        and snapshot.raw_grade is not None
        and snapshot.route not in ("incomplete", "error")
        and "grading" in snapshot.request_sha256
        and "grading" in snapshot.prompt_sha256
        and (
            not case.history
            or (
                "reformulation" in snapshot.request_sha256
                and "reformulation" in snapshot.prompt_sha256
            )
        )
        and (
            snapshot.route != "generate"
            or ("generation" in snapshot.request_sha256 and "generation" in snapshot.prompt_sha256)
        )
    )
    if not complete:
        record.failure_codes.append("incomplete_trace")
        record.claim_support = "unavailable"
    elif snapshot.rendered_context != format_context_chunks(snapshot.supplied_chunks or []):
        record.failure_codes.append("incomplete_trace")
    retrieved = {c.chunk_id: c for c in snapshot.retrieved_chunks or []}
    if any(retrieved.get(c.chunk_id) != c for c in snapshot.supplied_chunks or []):
        if "incomplete_trace" not in record.failure_codes:
            record.failure_codes.append("incomplete_trace")
    if snapshot.effective_scope != snapshot.declared_scope or any(
        c.metadata.document_id not in snapshot.effective_scope
        for c in (snapshot.retrieved_chunks or []) + (snapshot.supplied_chunks or [])
    ):
        if "scope_leak" not in record.failure_codes:
            record.failure_codes.append("scope_leak")
    if len(snapshot.citations or []) != len(set(record.raw_citation_tags)):
        if "citation_integrity_failure" not in record.failure_codes:
            record.failure_codes.append("citation_integrity_failure")
    mapped = {
        a["id"]: a["chunk_mapping"]["status"]
        for d in json.loads(
            (
                Path(__file__).resolve().parents[3]
                / "tests/evaluation/fixtures/task_5_9_1_source_dataset.json"
            ).read_text("utf-8")
        )["documents"]
        for a in d["anchors"]
    }
    extraction_gap = any(
        mapped[a] == "extraction_failure" for f in case.required_facts for a in f.source_anchor_ids
    )
    if extraction_gap:
        record.failure_codes.append("extraction_gap")
    if record.result_kind == "incorrect_refusal":
        record.failure_codes = [c for c in record.failure_codes if c != "grader_false_rejection"]
        if not extraction_gap:
            record.failure_codes.append("retrieval_gap")
    for assessment, fact in zip(record.facts, case.required_facts, strict=True):
        assessment.exact_values = list(fact.exact_values)
        assessment.qualifications = list(fact.qualifications)
        assessment.source_anchor_ids = list(
            dict.fromkeys(fact.source_anchor_ids + fact.acceptable_alternative_evidence_anchor_ids)
        )
        if extraction_gap:
            assessment.extraction_status = "extraction_failure"
        if assessment.verdict == "pass":
            assessment.verdict = "review_required"
    record.earliest_failing_stage = next(
        (code for code in _FAILURE_STAGE_ORDER if code in record.failure_codes), None
    )
    return record


def _apply_fresh_review(
    record: CaseEvaluationRecord, review: FreshCaseReview
) -> CaseEvaluationRecord:
    result = record.model_copy(deep=True)
    facts_by_id = {f.fact_id: f for f in record.facts}
    if set(review.facts) - set(facts_by_id):
        raise ValueError("Semantic review contains unknown fact IDs")
    for fact in result.facts:
        decision = review.facts.get(fact.fact_id)
        if decision is None:
            continue
        if set(decision.supporting_citation_tags) - set(record.raw_citation_tags) or set(
            decision.supporting_anchor_ids
        ) - set(fact.source_anchor_ids):
            raise ValueError("Semantic review references unavailable citations or anchors")
        if decision.verdict == "pass" and (
            not decision.supporting_citation_tags or not decision.supporting_anchor_ids
        ):
            raise ValueError("Passing source review requires citations and source anchors")
        if fact.verdict not in ("fail", "unavailable") or decision.verdict == "fail":
            fact.verdict, fact.reason = decision.verdict, decision.reason
        if decision.verdict == "fail" and "wrong_fact" not in result.failure_codes:
            result.failure_codes.append("wrong_fact")
    case_decisions: list[tuple[SemanticFactReview | None, FailureCode]] = [
        (review.extra_claims, "unsupported_claim"),
        (review.attribution, "attribution_failure"),
    ]
    for decision, code in case_decisions:
        if decision is not None and decision.verdict == "fail" and code not in result.failure_codes:
            result.failure_codes.append(code)
    if (
        review.refusal is not None
        and review.refusal.verdict == "fail"
        and review.sufficient_supplied_evidence
        and record.result_kind == "incorrect_refusal"
    ):
        result.failure_codes.append("grader_false_rejection")
    absence = (
        record.result_kind == "appropriate_abstention"
        and record.absence_review_sha256 is not None
        and not record.facts
        and not record.citations
        and not record.raw_citation_tags
    )
    complete = (
        review.all_claims_reviewed
        and set(review.facts) == set(facts_by_id)
        and all(f.verdict == "pass" for f in result.facts)
        and all(
            d is not None and d.verdict == "pass"
            for d in (review.extra_claims, review.attribution, review.refusal)
        )
        and (bool(result.facts) or absence)
    )
    result.claim_support = (
        "unavailable"
        if record.claim_support == "unavailable"
        else "fail"
        if result.failure_codes
        else "pass"
        if complete
        else "review_required"
    )
    result.semantic_review = review.model_copy(deep=True)
    result.earliest_failing_stage = next(
        (code for code in _FAILURE_STAGE_ORDER if code in result.failure_codes), None
    )
    return result
