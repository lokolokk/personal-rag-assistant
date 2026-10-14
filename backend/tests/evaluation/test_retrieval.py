"""Ranking, evidence coverage, and scope metrics for the real hybrid retriever."""

import math

import chromadb
import pytest

from app.core.config import Settings
from app.schemas.document import (
    ChunkMetadata,
    DocumentChunk,
    ScoredChunk,
)
from app.services.evaluation.retrieval import (
    MultiKRankingMetrics,
    RetrievalConfiguration,
    RetrievalDiagnosticResult,
    RetrievalFailureKind,
    calculate_multi_k_ranking_metrics,
    calculate_ranking_metrics,
    diagnose_retrieval_failure,
    evaluate_retrieval,
)
from app.services.evaluation.source_dataset import (
    AbsenceDocumentReview,
    AbsenceReview,
    ChunkingIdentity,
    ChunkMapping,
    EvaluationCase,
    RequiredFact,
    SourceAnchor,
    SourceDataset,
    SourceDocument,
)
from app.services.retrieval.bm25_store import BM25Store
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.vector_store import DeterministicEmbeddingFunction, VectorStore


def test_ranking_metrics_use_k_and_returned_result_denominators() -> None:
    metrics = calculate_ranking_metrics(
        relevant_chunk_ids={"relevant-1", "relevant-2"},
        returned_chunk_ids=["other", "relevant-1"],
        k=3,
    )

    assert metrics.precision_at_k == pytest.approx(1 / 3)
    assert metrics.returned_result_precision == pytest.approx(1 / 2)
    assert metrics.recall == pytest.approx(1 / 2)
    assert metrics.mrr == pytest.approx(1 / 2)
    assert metrics.ndcg == pytest.approx((1 / 1.5849625) / (1 + 1 / 1.5849625))


class _FixedRetriever:
    def __init__(self, results: list[ScoredChunk]) -> None:
        self.results = results
        self.scopes: list[list[str] | None] = []

    def retrieve(
        self, query: str, document_ids: list[str] | None = None, top_k: int | None = None
    ) -> list[ScoredChunk]:
        _ = query
        self.scopes.append(document_ids)
        return self.results[:top_k]


def _chunk(chunk_id: str, document_id: str, content: str, start: int) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        content=content,
        score=0.03,
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document_id,
            filename=f"{document_id}.pdf",
            page_number=2,
            section_title="Results",
            token_count=len(content.split()),
            char_start=start,
            char_end=start + len(content),
        ),
    )


def _dataset() -> SourceDataset:
    documents = [
        SourceDocument(
            id="doc-a",
            filename="a.pdf",
            title="A",
            fixture_path="a.pdf",
            sha256="a" * 64,
            page_count=1,
            chunking=ChunkingIdentity(
                encoding="cl100k_base", target_chunk_size=600, chunk_overlap=100
            ),
            anchors=[
                SourceAnchor(
                    id="answer-anchor",
                    page_number=1,
                    locator="Table 1 result",
                    source_quote="Exact result 42.5 percent",
                    chunk_mapping=ChunkMapping(
                        status="mapped",
                        chunk_ids=["support"],
                        chunk_sha256={"support": "b" * 64},
                    ),
                )
            ],
        ),
        SourceDocument(
            id="doc-b",
            filename="b.pdf",
            title="B",
            fixture_path="b.pdf",
            sha256="c" * 64,
            page_count=1,
            chunking=ChunkingIdentity(
                encoding="cl100k_base", target_chunk_size=600, chunk_overlap=100
            ),
            anchors=[],
        ),
    ]
    diagnostic_case = EvaluationCase(
        id="case-a",
        split="diagnostic",
        question="What is the exact result?",
        retrieval_query="exact result",
        scope_document_ids=["doc-a"],
        answerability="answerable",
        required_facts=[
            RequiredFact(
                id="fact-a",
                claim="Report the exact value.",
                exact_values=["42.5%"],
                unit="percent",
                source_anchor_ids=["answer-anchor"],
            )
        ],
    )
    held_out_case = diagnostic_case.model_copy(update={"id": "held-case", "split": "held_out"})
    unanswerable_case = EvaluationCase(
        id="unanswerable-case",
        split="diagnostic",
        question="What confidence interval is reported?",
        retrieval_query="confidence interval result",
        scope_document_ids=["doc-a"],
        answerability="unanswerable",
        unanswerable_reason="The source reports no confidence interval.",
        absence_review=AbsenceReview(
            documents=[
                AbsenceDocumentReview(
                    document_id="doc-a",
                    pages_reviewed=[1],
                    search_terms=["confidence interval"],
                    method="Source PDF page review",
                )
            ],
            note="The only source page contains no interval.",
        ),
    )
    return SourceDataset(
        dataset_id="metric-test",
        created_at="2026-10-06",
        held_out_frozen_before_tuning=True,
        documents=documents,
        cases=[diagnostic_case, unanswerable_case, held_out_case],
    )


def test_retrieval_report_tracks_fact_hits_unnecessary_evidence_and_scope_leaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.evaluation.retrieval.estimate_tokens", lambda text: len(text.split())
    )
    retriever = _FixedRetriever(
        [
            _chunk("support", "doc-a", "Exact result 42.5 percent in Table 1", 0),
            _chunk("leak", "doc-b", "Out of scope evidence", 0),
            _chunk("unneeded", "doc-a", "unrelated note", 100),
        ]
    )

    report = evaluate_retrieval(_dataset(), retriever, k=3)

    case = report.cases[0]
    assert report.case_splits == ["diagnostic"]
    assert report.aggregate.total_cases == 2
    assert case.ranking_metrics is not None
    assert case.ranking_metrics.precision_at_k == pytest.approx(1 / 3)
    assert case.ranking_metrics.returned_result_precision == pytest.approx(1 / 3)
    assert case.ranking_metrics.recall == 1
    assert case.covered_fact_ids == ["fact-a"]
    assert case.unnecessary_evidence_chunk_ids == ["leak", "unneeded"]
    assert case.scope_leakage_chunk_ids == ["leak"]
    assert case.context_tokens is not None and case.context_tokens > 0
    assert report.aggregate.total_cases == 2
    negative_case = report.cases[1]
    assert negative_case.answerability == "unanswerable"
    assert negative_case.ranking_metrics is None
    assert negative_case.unanswerable_retrieved_chunk_ids
    assert report.aggregate.unanswerable_nonempty_retrieval_cases == 1
    assert retriever.scopes == [["doc-a"], ["doc-a"]]


def test_actual_chroma_bm25_rrf_evaluation_uses_isolated_scoped_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.evaluation.retrieval.estimate_tokens", lambda text: len(text.split())
    )
    chunks = [
        _chunk("support", "doc-a", "Exact result 42.5 percent in Table 1", 0),
        _chunk("unneeded", "doc-a", "A separate discussion without the requested value", 100),
        _chunk("other-doc", "doc-b", "Exact result 900 percent in another paper", 0),
    ]
    stored_chunks = [
        DocumentChunk(chunk_id=chunk.chunk_id, content=chunk.content, metadata=chunk.metadata)
        for chunk in chunks
    ]
    vector_store = VectorStore(
        client=chromadb.EphemeralClient(),
        collection_name="task_5_9_1_retrieval_test",
        embedding_function=DeterministicEmbeddingFunction(dimension=32),
    )
    vector_store.add_chunks(stored_chunks)
    bm25_store = BM25Store()
    bm25_store.add_chunks(stored_chunks)
    settings = Settings(DENSE_TOP_N=3, SPARSE_TOP_N=3, RRF_K=60, TOP_K=5)
    retriever = HybridRetriever(vector_store, bm25_store, settings=settings)

    report = evaluate_retrieval(
        _dataset(),
        retriever,
        k=2,
        retrieval_configuration=RetrievalConfiguration(
            dense_top_n=3,
            sparse_top_n=3,
            rrf_k=60,
            top_k=2,
            embedding_identity="deterministic local-only test embedding",
        ),
    )

    case = report.cases[0]
    assert case.scope_leakage_chunk_ids == []
    assert "support" in [item.chunk_id for item in case.retrieved_evidence]
    assert all(item.document_id == "doc-a" for item in case.retrieved_evidence)
    assert report.cases[1].ranking_status == "unanswerable"
    assert report.cases[1].unanswerable_retrieved_chunk_ids
    assert report.retrieval_configuration is not None
    assert report.retrieval_configuration.embedding_identity.startswith("deterministic")


def test_multi_k_ranking_metrics_multi_cutoffs() -> None:
    relevant = {"c1", "c3", "c7"}
    returned = ["c1", "noise1", "c3", "noise2", "noise3", "noise4", "c7", "noise5"]
    metrics = calculate_multi_k_ranking_metrics(
        relevant_chunk_ids=relevant,
        returned_chunk_ids=returned,
        k_values=(1, 3, 5, 10, 20),
    )

    assert isinstance(metrics, MultiKRankingMetrics)
    assert metrics.relevant_chunk_count == 3
    assert metrics.returned_chunk_count == 8

    # K = 1: returned ["c1"], hits: 1
    assert metrics.precision_at_k[1] == pytest.approx(1.0)
    assert metrics.recall_at_k[1] == pytest.approx(1 / 3)
    assert metrics.ndcg_at_k[1] == pytest.approx(1.0)

    # K = 3: returned ["c1", "noise1", "c3"], hits: 2
    assert metrics.precision_at_k[3] == pytest.approx(2 / 3)
    assert metrics.recall_at_k[3] == pytest.approx(2 / 3)
    dcg_3 = 1.0 / math.log2(2) + 0.0 + 1.0 / math.log2(4)
    idcg_3 = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    assert metrics.ndcg_at_k[3] == pytest.approx(dcg_3 / idcg_3)

    # K = 5: returned ["c1", "noise1", "c3", "noise2", "noise3"], hits: 2
    assert metrics.precision_at_k[5] == pytest.approx(2 / 5)
    assert metrics.recall_at_k[5] == pytest.approx(2 / 3)

    # K = 10: returned 8 items (all 3 hits included), hits: 3
    assert metrics.precision_at_k[10] == pytest.approx(3 / 10)
    assert metrics.recall_at_k[10] == pytest.approx(3 / 3)

    # K = 20: returned 8 items, hits: 3
    assert metrics.precision_at_k[20] == pytest.approx(3 / 20)
    assert metrics.recall_at_k[20] == pytest.approx(1.0)

    # MRR: first hit is at rank 1 -> 1.0
    assert metrics.mrr == pytest.approx(1.0)


def test_exact_logarithmic_discount_formula_for_ndcg() -> None:
    # 2 relevant chunks, returned at rank 2 and rank 4
    relevant = {"doc_b", "doc_d"}
    returned = ["noise_a", "doc_b", "noise_c", "doc_d", "noise_e"]
    metrics = calculate_multi_k_ranking_metrics(
        relevant_chunk_ids=relevant,
        returned_chunk_ids=returned,
        k_values=(5,),
    )
    expected_dcg = (1.0 / math.log2(3)) + (1.0 / math.log2(5))
    expected_idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
    expected_ndcg = expected_dcg / expected_idcg
    assert metrics.ndcg_at_k[5] == pytest.approx(expected_ndcg)


def test_mrr_reciprocal_rank_calculation() -> None:
    # First hit at rank 1 -> 1.0
    metrics_rank1 = calculate_multi_k_ranking_metrics(
        relevant_chunk_ids={"target"},
        returned_chunk_ids=["target", "noise"],
    )
    assert metrics_rank1.mrr == pytest.approx(1.0)

    # First hit at rank 3 -> 1/3 ≈ 0.33
    metrics_rank3 = calculate_multi_k_ranking_metrics(
        relevant_chunk_ids={"target"},
        returned_chunk_ids=["noise1", "noise2", "target", "noise3"],
    )
    assert metrics_rank3.mrr == pytest.approx(1 / 3)
    assert metrics_rank3.mrr == pytest.approx(0.33, abs=0.01)

    # Missing target chunk entirely -> 0.0
    metrics_missing = calculate_multi_k_ranking_metrics(
        relevant_chunk_ids={"target"},
        returned_chunk_ids=["noise1", "noise2", "noise3"],
    )
    assert metrics_missing.mrr == 0.0


def test_failure_taxonomy_classification() -> None:
    # Target chunk in top 20 (rank 7) but missed top 5 -> "missed_cutoff"
    candidates_missed = (
        [f"noise_{i}" for i in range(6)] + ["target_chunk"] + [f"tail_{i}" for i in range(13)]
    )
    res_missed = diagnose_retrieval_failure(
        relevant_chunk_ids={"target_chunk"},
        returned_chunk_ids=candidates_missed,
        primary_k=5,
        diagnostic_k=20,
        precision_threshold=0.2,
    )
    assert isinstance(res_missed, RetrievalDiagnosticResult)
    assert res_missed.failure_kind == RetrievalFailureKind.MISSED_CUTOFF
    assert res_missed.failure_kind.value == "missed_cutoff"
    assert res_missed.is_failure is True
    assert res_missed.hits_in_primary == 0
    assert res_missed.hits_in_diagnostic == 1

    # Target chunk entirely absent from candidates -> "zero_recall"
    candidates_absent = [f"noise_{i}" for i in range(25)]
    res_absent = diagnose_retrieval_failure(
        relevant_chunk_ids={"target_chunk"},
        returned_chunk_ids=candidates_absent,
        primary_k=5,
        diagnostic_k=20,
    )
    assert res_absent.failure_kind == RetrievalFailureKind.ZERO_RECALL
    assert res_absent.failure_kind.value == "zero_recall"
    assert res_absent.is_failure is True
    assert res_absent.hits_in_primary == 0
    assert res_absent.hits_in_diagnostic == 0

    # Target chunk present in top 5, but low signal-to-noise ratio -> "low_precision"
    candidates_low_snr = ["target_chunk", "noise_1", "noise_2", "noise_3", "noise_4"]
    res_low_snr = diagnose_retrieval_failure(
        relevant_chunk_ids={"target_chunk"},
        returned_chunk_ids=candidates_low_snr,
        primary_k=5,
        diagnostic_k=20,
        precision_threshold=0.2,
    )
    assert res_low_snr.failure_kind == RetrievalFailureKind.LOW_PRECISION
    assert res_low_snr.failure_kind.value == "low_precision"
    assert res_low_snr.is_failure is True
    assert res_low_snr.precision_at_primary == pytest.approx(0.2)

    # Target chunk present with high signal-to-noise ratio -> "none" (no failure)
    candidates_high_snr = ["target_1", "target_2", "noise_1", "noise_2", "noise_3"]
    res_high_snr = diagnose_retrieval_failure(
        relevant_chunk_ids={"target_1", "target_2"},
        returned_chunk_ids=candidates_high_snr,
        primary_k=5,
        diagnostic_k=20,
        precision_threshold=0.2,
    )
    assert res_high_snr.failure_kind == RetrievalFailureKind.NONE
    assert res_high_snr.failure_kind.value == "none"
    assert res_high_snr.is_failure is False
    assert res_high_snr.precision_at_primary == pytest.approx(0.4)


def test_edge_cases_empty_no_annotations_and_rank_ties() -> None:
    # 1. Empty candidates
    res_empty_cands = diagnose_retrieval_failure(
        relevant_chunk_ids={"target"},
        returned_chunk_ids=[],
        primary_k=5,
        diagnostic_k=20,
    )
    assert res_empty_cands.failure_kind == RetrievalFailureKind.ZERO_RECALL
    assert res_empty_cands.is_failure is True

    metrics_empty_cands = calculate_multi_k_ranking_metrics(
        relevant_chunk_ids={"target"},
        returned_chunk_ids=[],
        k_values=(1, 5),
    )
    assert metrics_empty_cands.precision_at_k[1] == 0.0
    assert metrics_empty_cands.recall_at_k[5] == 0.0
    assert metrics_empty_cands.ndcg_at_k[5] == 0.0
    assert metrics_empty_cands.mrr == 0.0

    # 2. No annotated relevant chunks
    res_no_rel = diagnose_retrieval_failure(
        relevant_chunk_ids=[],
        returned_chunk_ids=["chunk_1", "chunk_2"],
        primary_k=5,
        diagnostic_k=20,
    )
    assert res_no_rel.failure_kind == RetrievalFailureKind.NONE
    assert res_no_rel.is_failure is False

    metrics_no_rel = calculate_multi_k_ranking_metrics(
        relevant_chunk_ids=[],
        returned_chunk_ids=["chunk_1", "chunk_2"],
        k_values=(1, 5),
    )
    assert metrics_no_rel.precision_at_k[1] == 0.0
    assert metrics_no_rel.recall_at_k[5] == 0.0
    assert metrics_no_rel.ndcg_at_k[5] == 0.0
    assert metrics_no_rel.mrr == 0.0

    # 3. Zero hits
    metrics_zero = calculate_multi_k_ranking_metrics(
        relevant_chunk_ids={"target"},
        returned_chunk_ids=["noise_1", "noise_2"],
        k_values=(1, 5),
    )
    assert metrics_zero.precision_at_k[1] == 0.0
    assert metrics_zero.recall_at_k[5] == 0.0
    assert metrics_zero.ndcg_at_k[5] == 0.0
    assert metrics_zero.mrr == 0.0

    # 4. Rank ties / duplicate candidates in returned list
    metrics_ties = calculate_multi_k_ranking_metrics(
        relevant_chunk_ids={"target_1", "target_2"},
        returned_chunk_ids=["target_1", "target_1", "target_2", "target_2"],
        k_values=(2,),
    )
    assert metrics_ties.returned_chunk_count == 2
    assert metrics_ties.precision_at_k[2] == pytest.approx(1.0)
    assert metrics_ties.recall_at_k[2] == pytest.approx(1.0)
    assert metrics_ties.mrr == pytest.approx(1.0)
