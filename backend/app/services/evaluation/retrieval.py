"""Evaluate source-anchored evidence returned by the existing scoped hybrid retriever."""

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.document import ScoredChunk
from app.services.evaluation.source_dataset import EvaluationCase, SourceAnchor, SourceDataset
from app.services.evaluation.usage import estimate_tokens
from app.services.generation.grounding import format_context_chunks

CaseSplit = Literal["diagnostic", "held_out"]


class Retriever(Protocol):
    """Minimal interface implemented by the production HybridRetriever."""

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[ScoredChunk]: ...


class _ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievalFailureKind(StrEnum):
    """Taxonomy of retrieval failures for offline IR diagnostics."""

    MISSED_CUTOFF = "missed_cutoff"
    ZERO_RECALL = "zero_recall"
    LOW_PRECISION = "low_precision"
    NONE = "none"


class MultiKRankingMetrics(_ReportModel):
    """Multi-cutoff information retrieval ranking metrics across arbitrary K values."""

    relevant_chunk_count: int
    returned_chunk_count: int
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    mrr: float
    mrr_at_k: dict[int, float] = Field(default_factory=dict)


class RetrievalDiagnosticResult(_ReportModel):
    """Diagnostic failure classification and retrieval ranking summary."""

    failure_kind: RetrievalFailureKind
    is_failure: bool
    primary_k: int
    diagnostic_k: int
    precision_threshold: float
    relevant_count: int
    returned_count: int
    hits_in_primary: int
    hits_in_diagnostic: int
    precision_at_primary: float
    recall_at_primary: float
    recall_at_diagnostic: float
    first_relevant_rank: int | None = None
    metrics: MultiKRankingMetrics | None = None
    detail: str = ""


class RankingMetrics(_ReportModel):
    relevant_chunk_count: int
    returned_relevant_chunk_count: int
    returned_result_count: int
    returned_result_relevant_count: int
    precision_at_k: float
    returned_result_precision: float | None
    recall: float
    mrr: float
    ndcg: float


class RetrievedEvidence(_ReportModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    rrf_score: float
    content_sha256: str
    relevant_anchor_ids: list[str]
    is_relevant: bool


class CaseRetrievalReport(_ReportModel):
    case_id: str
    split: CaseSplit
    answerability: Literal["answerable", "unanswerable"]
    scope_document_ids: list[str]
    retrieval_query: str
    ranking_status: Literal["scored", "unanswerable", "source_evidence_unmapped"]
    ranking_metrics: RankingMetrics | None
    relevant_anchor_ids: list[str]
    relevant_chunk_ids: list[str]
    retrieved_evidence: list[RetrievedEvidence]
    covered_fact_ids: list[str]
    missing_fact_ids: list[str]
    unmapped_fact_ids: list[str]
    unnecessary_evidence_chunk_ids: list[str]
    unanswerable_retrieved_chunk_ids: list[str]
    scope_leakage_chunk_ids: list[str]
    out_of_scope_evidence_chunk_ids: list[str]
    context_tokens: int | None
    overlapping_content_tokens: int | None
    failures: list[str]


class AggregateRetrievalMetrics(_ReportModel):
    total_cases: int
    answerable_cases: int
    unanswerable_cases: int
    scored_answerable_cases: int
    precision_at_k_macro: float | None
    returned_result_precision_macro: float | None
    returned_precision_case_count: int
    recall_macro: float | None
    mrr_macro: float | None
    ndcg_macro: float | None
    required_fact_count: int
    covered_fact_count: int
    required_fact_evidence_coverage: float | None
    mean_context_tokens: float | None
    p95_context_tokens: int | None
    mean_overlapping_content_tokens: float | None
    unnecessary_evidence_chunk_count: int
    scope_leakage_chunk_count: int
    out_of_scope_evidence_chunk_count: int
    unanswerable_nonempty_retrieval_cases: int


class CorpusDocumentIdentity(_ReportModel):
    document_id: str
    filename: str
    sha256: str
    page_count: int


class RetrievalConfiguration(_ReportModel):
    dense_top_n: int
    sparse_top_n: int
    rrf_k: int
    top_k: int
    embedding_identity: str


class RetrievalReport(_ReportModel):
    dataset_id: str
    top_k: int
    case_splits: list[CaseSplit]
    corpus_documents: list[CorpusDocumentIdentity]
    retrieval_configuration: RetrievalConfiguration | None
    query_policy: str
    relevance_judgment: str
    metric_definitions: list[str]
    token_estimate_method: str
    cases: list[CaseRetrievalReport]
    aggregate: AggregateRetrievalMetrics


def calculate_ranking_metrics(
    *, relevant_chunk_ids: set[str], returned_chunk_ids: Sequence[str], k: int
) -> RankingMetrics:
    """Calculate binary chunk relevance metrics with independent K/returned denominators."""
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant_chunk_ids:
        raise ValueError("Ranking metrics require at least one annotated relevant chunk")

    result_ids = list(returned_chunk_ids)
    hits = [chunk_id in relevant_chunk_ids for chunk_id in result_ids[:k]]
    relevant_returned = sum(hits)
    returned_relevant_count = sum(chunk_id in relevant_chunk_ids for chunk_id in result_ids)
    first_relevant_rank = next(
        (rank for rank, is_relevant in enumerate(hits, start=1) if is_relevant), None
    )
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, is_relevant in enumerate(hits, start=1) if is_relevant
    )
    ideal_relevant_count = min(len(relevant_chunk_ids), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))

    return RankingMetrics(
        relevant_chunk_count=len(relevant_chunk_ids),
        returned_relevant_chunk_count=relevant_returned,
        returned_result_count=len(result_ids),
        returned_result_relevant_count=returned_relevant_count,
        precision_at_k=relevant_returned / k,
        returned_result_precision=(
            returned_relevant_count / len(result_ids) if result_ids else None
        ),
        recall=relevant_returned / len(relevant_chunk_ids),
        mrr=1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0,
        ndcg=dcg / ideal_dcg if ideal_dcg else 0.0,
    )


def calculate_multi_k_ranking_metrics(
    relevant_chunk_ids: set[str] | Sequence[str],
    returned_chunk_ids: Sequence[str],
    k_values: Sequence[int] = (1, 3, 5, 10, 20),
) -> MultiKRankingMetrics:
    """Calculate multi-cutoff precision, recall, NDCG, and MRR ranking metrics.

    Handles empty candidates, unannotated relevant sets, zero hits, and candidate deduplication.
    """
    for k in k_values:
        if k < 1:
            raise ValueError(f"k cutoffs must be positive integers, got {k}")

    seen: set[str] = set()
    deduped_returned: list[str] = []
    for chunk_id in returned_chunk_ids:
        if chunk_id not in seen:
            seen.add(chunk_id)
            deduped_returned.append(chunk_id)

    relevant_set = set(relevant_chunk_ids)
    relevant_count = len(relevant_set)
    returned_count = len(deduped_returned)

    precision_at_k: dict[int, float] = {}
    recall_at_k: dict[int, float] = {}
    ndcg_at_k: dict[int, float] = {}
    mrr_at_k: dict[int, float] = {}

    first_relevant_rank: int | None = None
    for rank, chunk_id in enumerate(deduped_returned, start=1):
        if chunk_id in relevant_set:
            first_relevant_rank = rank
            break

    overall_mrr = 1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0

    for k in k_values:
        candidates_at_k = deduped_returned[:k]
        hits = [chunk_id in relevant_set for chunk_id in candidates_at_k]
        hit_count = sum(hits)

        precision_at_k[k] = hit_count / k
        recall_at_k[k] = hit_count / relevant_count if relevant_count > 0 else 0.0

        dcg = sum(1.0 / math.log2(rank + 1) for rank, is_rel in enumerate(hits, start=1) if is_rel)
        ideal_count = min(relevant_count, k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcg_at_k[k] = dcg / ideal_dcg if ideal_dcg > 0.0 else 0.0

        first_k = next((rank for rank, is_rel in enumerate(hits, start=1) if is_rel), None)
        mrr_at_k[k] = 1.0 / first_k if first_k is not None else 0.0

    return MultiKRankingMetrics(
        relevant_chunk_count=relevant_count,
        returned_chunk_count=returned_count,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        ndcg_at_k=ndcg_at_k,
        mrr=overall_mrr,
        mrr_at_k=mrr_at_k,
    )


def diagnose_retrieval_failure(
    relevant_chunk_ids: set[str] | Sequence[str],
    returned_chunk_ids: Sequence[str],
    primary_k: int = 5,
    diagnostic_k: int = 20,
    precision_threshold: float = 0.2,
) -> RetrievalDiagnosticResult:
    """Diagnose retrieval failure taxonomy against primary and diagnostic cutoffs.

    Taxonomy:
    - missed_cutoff: target chunk present in diagnostic_k (top 20) but missed primary_k (top 5).
    - zero_recall: target chunk entirely absent from candidates.
    - low_precision: target chunk present in primary_k, but precision <= precision_threshold.
    - none: high precision in primary_k, or no annotated targets.
    """
    if primary_k < 1 or diagnostic_k < 1:
        raise ValueError("Cutoff values must be positive integers")
    if primary_k > diagnostic_k:
        raise ValueError(f"primary_k ({primary_k}) must be <= diagnostic_k ({diagnostic_k})")

    seen: set[str] = set()
    deduped_returned: list[str] = []
    for chunk_id in returned_chunk_ids:
        if chunk_id not in seen:
            seen.add(chunk_id)
            deduped_returned.append(chunk_id)

    relevant_set = set(relevant_chunk_ids)
    relevant_count = len(relevant_set)
    returned_count = len(deduped_returned)

    primary_candidates = deduped_returned[:primary_k]
    diagnostic_candidates = deduped_returned[:diagnostic_k]

    hits_primary = [cid in relevant_set for cid in primary_candidates]
    hits_diagnostic = [cid in relevant_set for cid in diagnostic_candidates]

    hit_count_primary = sum(hits_primary)
    hit_count_diagnostic = sum(hits_diagnostic)

    precision_at_primary = hit_count_primary / primary_k
    recall_at_primary = hit_count_primary / relevant_count if relevant_count > 0 else 0.0
    recall_at_diagnostic = hit_count_diagnostic / relevant_count if relevant_count > 0 else 0.0

    first_rank = next(
        (rank for rank, cid in enumerate(deduped_returned, start=1) if cid in relevant_set),
        None,
    )

    if relevant_count == 0:
        failure_kind = RetrievalFailureKind.NONE
        is_failure = False
        detail = "No annotated relevant chunks; not a retrieval failure."
    elif hit_count_diagnostic == 0:
        failure_kind = RetrievalFailureKind.ZERO_RECALL
        is_failure = True
        detail = f"Target chunk(s) entirely absent from top {diagnostic_k} candidates."
    elif hit_count_primary == 0:
        failure_kind = RetrievalFailureKind.MISSED_CUTOFF
        is_failure = True
        detail = (
            f"Target chunk(s) present in top {diagnostic_k} (first rank: {first_rank}) "
            f"but missed top {primary_k} cutoff."
        )
    elif precision_at_primary <= precision_threshold:
        failure_kind = RetrievalFailureKind.LOW_PRECISION
        is_failure = True
        detail = (
            f"Target chunk(s) retrieved in top {primary_k}, but precision "
            f"({precision_at_primary:.2f}) <= threshold ({precision_threshold:.2f})."
        )
    else:
        failure_kind = RetrievalFailureKind.NONE
        is_failure = False
        detail = f"Target chunk(s) retrieved with adequate precision ({precision_at_primary:.2f})."

    metrics = calculate_multi_k_ranking_metrics(
        relevant_chunk_ids=relevant_set,
        returned_chunk_ids=deduped_returned,
        k_values=(primary_k, diagnostic_k),
    )

    return RetrievalDiagnosticResult(
        failure_kind=failure_kind,
        is_failure=is_failure,
        primary_k=primary_k,
        diagnostic_k=diagnostic_k,
        precision_threshold=precision_threshold,
        relevant_count=relevant_count,
        returned_count=returned_count,
        hits_in_primary=hit_count_primary,
        hits_in_diagnostic=hit_count_diagnostic,
        precision_at_primary=precision_at_primary,
        recall_at_primary=recall_at_primary,
        recall_at_diagnostic=recall_at_diagnostic,
        first_relevant_rank=first_rank,
        metrics=metrics,
        detail=detail,
    )


def _anchor_index(dataset: SourceDataset) -> dict[str, SourceAnchor]:
    return {anchor.id: anchor for document in dataset.documents for anchor in document.anchors}


def _case_relevant_anchor_ids(case: EvaluationCase) -> set[str]:
    return {
        anchor_id
        for fact in case.required_facts
        for anchor_id in fact.source_anchor_ids + fact.acceptable_alternative_evidence_anchor_ids
    }


def _mapped_chunk_ids(anchor_ids: set[str], anchors: dict[str, SourceAnchor]) -> set[str]:
    return {
        chunk_id
        for anchor_id in anchor_ids
        if anchors[anchor_id].chunk_mapping.status == "mapped"
        for chunk_id in anchors[anchor_id].chunk_mapping.chunk_ids
    }


def _fact_coverage(
    case: EvaluationCase, anchors: dict[str, SourceAnchor], returned_ids: set[str]
) -> tuple[list[str], list[str], list[str]]:
    covered: list[str] = []
    missing: list[str] = []
    unmapped: list[str] = []
    for fact in case.required_facts:
        source_anchors = [anchors[anchor_id] for anchor_id in fact.source_anchor_ids]
        primary_route = all(
            anchor.chunk_mapping.status == "mapped"
            and bool(set(anchor.chunk_mapping.chunk_ids) & returned_ids)
            for anchor in source_anchors
        )
        alternative_route = any(
            anchors[anchor_id].chunk_mapping.status == "mapped"
            and bool(set(anchors[anchor_id].chunk_mapping.chunk_ids) & returned_ids)
            for anchor_id in fact.acceptable_alternative_evidence_anchor_ids
        )
        if primary_route or alternative_route:
            covered.append(fact.id)
        else:
            missing.append(fact.id)
        if any(anchor.chunk_mapping.status == "extraction_failure" for anchor in source_anchors):
            unmapped.append(fact.id)
    return covered, missing, unmapped


def _overlap_text_tokens(chunks: Sequence[ScoredChunk]) -> list[str]:
    """Estimate repeated adjacent chunk text within a document page, in source order."""
    page_groups: dict[tuple[str, int], list[ScoredChunk]] = defaultdict(list)
    for chunk in chunks:
        page_groups[(chunk.metadata.document_id, chunk.metadata.page_number)].append(chunk)

    overlaps: list[str] = []
    for page_chunks in page_groups.values():
        page_chunks.sort(key=lambda chunk: chunk.metadata.char_start)
        for left, right in zip(page_chunks, page_chunks[1:], strict=False):
            left_tokens = re.findall(r"\S+", left.content.casefold())
            right_tokens = re.findall(r"\S+", right.content.casefold())
            max_overlap = min(len(left_tokens), len(right_tokens))
            for overlap_size in range(max_overlap, 0, -1):
                if left_tokens[-overlap_size:] == right_tokens[:overlap_size]:
                    overlaps.extend(right_tokens[:overlap_size])
                    break
    return overlaps


def _average(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile_95(values: Sequence[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def evaluate_retrieval(
    dataset: SourceDataset,
    retriever: Retriever,
    *,
    k: int = 5,
    splits: Sequence[CaseSplit] = ("diagnostic",),
    retrieval_configuration: RetrievalConfiguration | None = None,
) -> RetrievalReport:
    """Run annotated queries against the supplied retriever and score source-anchored evidence.

    Callers must load the dataset with ``load_source_dataset`` first. That verifies original
    PDFs, fingerprints, current chunk mappings, and the local tokenizer cache without network.
    """
    if k < 1:
        raise ValueError("k must be positive")
    selected_splits = list(dict.fromkeys(splits))
    if not selected_splits:
        raise ValueError("At least one case split must be selected")
    selected_cases = [case for case in dataset.cases if case.split in selected_splits]
    if not selected_cases:
        raise ValueError("The selected case splits have no cases in the dataset")
    anchors = _anchor_index(dataset)
    reports: list[CaseRetrievalReport] = []
    for case in selected_cases:
        results = retriever.retrieve(
            query=case.retrieval_query,
            document_ids=case.scope_document_ids or None,
            top_k=k,
        )
        returned_ids = {chunk.chunk_id for chunk in results}
        relevant_anchor_ids = _case_relevant_anchor_ids(case)
        relevant_chunk_ids = _mapped_chunk_ids(relevant_anchor_ids, anchors)
        unmapped_anchor_ids = {
            anchor_id
            for anchor_id in relevant_anchor_ids
            if anchors[anchor_id].chunk_mapping.status == "extraction_failure"
        }
        covered_fact_ids, missing_fact_ids, unmapped_fact_ids = _fact_coverage(
            case, anchors, returned_ids
        )
        out_of_scope_chunk_ids = _mapped_chunk_ids(set(case.out_of_scope_anchor_ids), anchors)
        scope = set(case.scope_document_ids)
        scope_leaks = [
            chunk.chunk_id for chunk in results if scope and chunk.metadata.document_id not in scope
        ]
        out_of_scope_hits = [
            chunk.chunk_id for chunk in results if chunk.chunk_id in out_of_scope_chunk_ids
        ]
        ranking_status: Literal["scored", "unanswerable", "source_evidence_unmapped"]
        ranking_metrics: RankingMetrics | None
        if case.answerability == "unanswerable":
            ranking_status = "unanswerable"
            ranking_metrics = None
        elif relevant_chunk_ids:
            ranking_status = "scored"
            ranking_metrics = calculate_ranking_metrics(
                relevant_chunk_ids=relevant_chunk_ids,
                returned_chunk_ids=[chunk.chunk_id for chunk in results],
                k=k,
            )
        else:
            ranking_status = "source_evidence_unmapped"
            ranking_metrics = None

        context_tokens = estimate_tokens(format_context_chunks(results) if results else "")
        overlap_tokens = _overlap_text_tokens(results)
        overlap_count = estimate_tokens(" ".join(overlap_tokens)) if overlap_tokens else 0
        failures: list[str] = []
        if case.answerability == "answerable" and not results:
            failures.append("no_results")
        if missing_fact_ids:
            failures.append("required_fact_evidence_missing")
        if unmapped_anchor_ids:
            failures.append("source_extraction_failure")
        if scope_leaks:
            failures.append("scope_leakage")
        if out_of_scope_hits:
            failures.append("out_of_scope_evidence_retrieved")

        evidence = [
            RetrievedEvidence(
                chunk_id=chunk.chunk_id,
                document_id=chunk.metadata.document_id,
                filename=chunk.metadata.filename,
                page_number=chunk.metadata.page_number,
                rrf_score=chunk.score,
                content_sha256=hashlib.sha256(chunk.content.encode("utf-8")).hexdigest(),
                relevant_anchor_ids=sorted(
                    anchor_id
                    for anchor_id in relevant_anchor_ids
                    if chunk.chunk_id in anchors[anchor_id].chunk_mapping.chunk_ids
                ),
                is_relevant=chunk.chunk_id in relevant_chunk_ids,
            )
            for chunk in results
        ]
        reports.append(
            CaseRetrievalReport(
                case_id=case.id,
                split=case.split,
                answerability=case.answerability,
                scope_document_ids=case.scope_document_ids,
                retrieval_query=case.retrieval_query,
                ranking_status=ranking_status,
                ranking_metrics=ranking_metrics,
                relevant_anchor_ids=sorted(relevant_anchor_ids),
                relevant_chunk_ids=sorted(relevant_chunk_ids),
                retrieved_evidence=evidence,
                covered_fact_ids=covered_fact_ids,
                missing_fact_ids=missing_fact_ids,
                unmapped_fact_ids=unmapped_fact_ids,
                unnecessary_evidence_chunk_ids=(
                    sorted(returned_ids - relevant_chunk_ids)
                    if case.answerability == "answerable"
                    else []
                ),
                unanswerable_retrieved_chunk_ids=(
                    sorted(returned_ids) if case.answerability == "unanswerable" else []
                ),
                scope_leakage_chunk_ids=scope_leaks,
                out_of_scope_evidence_chunk_ids=out_of_scope_hits,
                context_tokens=context_tokens,
                overlapping_content_tokens=overlap_count,
                failures=failures,
            )
        )

    ranked = [case.ranking_metrics for case in reports if case.ranking_metrics is not None]
    returned_precisions = [
        metric.returned_result_precision
        for metric in ranked
        if metric.returned_result_precision is not None
    ]
    answerable = [case for case in reports if case.answerability == "answerable"]
    unanswerable = [case for case in reports if case.answerability == "unanswerable"]
    required_fact_count = sum(len(case.required_facts) for case in selected_cases)
    covered_fact_count = sum(len(case.covered_fact_ids) for case in reports)
    context_values = [case.context_tokens for case in reports if case.context_tokens is not None]
    overlap_values = [
        case.overlapping_content_tokens
        for case in reports
        if case.overlapping_content_tokens is not None
    ]
    aggregate = AggregateRetrievalMetrics(
        total_cases=len(reports),
        answerable_cases=len(answerable),
        unanswerable_cases=len(unanswerable),
        scored_answerable_cases=len(ranked),
        precision_at_k_macro=_average([metric.precision_at_k for metric in ranked]),
        returned_result_precision_macro=_average(returned_precisions),
        returned_precision_case_count=len(returned_precisions),
        recall_macro=_average([metric.recall for metric in ranked]),
        mrr_macro=_average([metric.mrr for metric in ranked]),
        ndcg_macro=_average([metric.ndcg for metric in ranked]),
        required_fact_count=required_fact_count,
        covered_fact_count=covered_fact_count,
        required_fact_evidence_coverage=(
            covered_fact_count / required_fact_count if required_fact_count else None
        ),
        mean_context_tokens=_average(context_values),
        p95_context_tokens=_percentile_95(context_values),
        mean_overlapping_content_tokens=_average(overlap_values),
        unnecessary_evidence_chunk_count=sum(
            len(case.unnecessary_evidence_chunk_ids) for case in reports
        ),
        scope_leakage_chunk_count=sum(len(case.scope_leakage_chunk_ids) for case in reports),
        out_of_scope_evidence_chunk_count=sum(
            len(case.out_of_scope_evidence_chunk_ids) for case in reports
        ),
        unanswerable_nonempty_retrieval_cases=sum(
            bool(case.unanswerable_retrieved_chunk_ids) for case in reports
        ),
    )
    return RetrievalReport(
        dataset_id=dataset.dataset_id,
        top_k=k,
        case_splits=selected_splits,
        corpus_documents=[
            CorpusDocumentIdentity(
                document_id=document.id,
                filename=document.filename,
                sha256=document.sha256,
                page_count=document.page_count,
            )
            for document in dataset.documents
        ],
        retrieval_configuration=retrieval_configuration,
        query_policy="Uses the frozen retrieval_query field; does not run LLM query reformulation.",
        relevance_judgment=(
            "Binary chunk relevance comes from mapped source anchors attached to required facts "
            "or their explicitly accepted alternatives. Unanswerable cases have no positive "
            "answer-supporting chunks and are retained with ranking metrics marked not applicable."
        ),
        metric_definitions=[
            "Precision@K divides relevant results by K, including missing result slots.",
            (
                "Returned-result precision uses the returned count; empty cases are excluded from "
                "its macro average."
            ),
            "Recall divides relevant retrieved chunks by mapped relevant chunks for that case.",
            "MRR uses the first relevant rank; NDCG uses binary relevance with log discount.",
            "Ranking aggregates are macro averages over answerable cases with mapped positives.",
            (
                "Fact coverage requires every primary anchor group or an accepted alternative; "
                "it is an anchor-coverage proxy, not an entailment judge."
            ),
        ],
        token_estimate_method="cl100k_base text-only estimate of the framed context",
        cases=reports,
        aggregate=aggregate,
    )
