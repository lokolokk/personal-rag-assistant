"""Unit and integration tests for Scoped Hybrid Retriever Core (Layer 1) & LangGraph Dispatcher."""

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.schemas.chat import ChatMessage, ConversationalRAGState
from app.schemas.document import ChunkMetadata, DocumentChunk, ScoredChunk
from app.services.retrieval.bm25_store import BM25Store
from app.services.retrieval.hybrid import (
    HybridRetriever,
    compute_rrf_scores,
    get_hybrid_retriever,
    scoped_hybrid_retriever_node,
    set_hybrid_retriever,
)
from app.services.retrieval.vector_store import DeterministicEmbeddingFunction, VectorStore


def _create_chunk(chunk_id: str, doc_id: str, content: str) -> DocumentChunk:
    """Helper to create a well-formed DocumentChunk for testing."""
    return DocumentChunk(
        chunk_id=chunk_id,
        content=content,
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            document_id=doc_id,
            filename=f"{doc_id}.pdf",
            page_number=1,
            section_title="Introduction",
            is_table=False,
            token_count=len(content.split()),
            char_start=0,
            char_end=len(content),
        ),
    )


def _create_scored_chunk(chunk_id: str, doc_id: str, content: str, score: float) -> ScoredChunk:
    """Helper to create a ScoredChunk for testing."""
    chunk = _create_chunk(chunk_id, doc_id, content)
    return ScoredChunk(
        chunk_id=chunk.chunk_id,
        content=chunk.content,
        metadata=chunk.metadata,
        score=score,
    )


class TestRRFScoring:
    """Tests for the pure Reciprocal Rank Fusion (RRF) algorithm."""

    def test_rrf_scoring_math_exact(self) -> None:
        """Verify exact RRF score computation with smoothing constant k=60."""
        # Chunk 1: rank 1 in dense (0.01639344), rank 1 in sparse (0.01639344) -> ~0.03278688
        # Chunk 2: rank 2 in dense (0.01612903), absent in sparse -> ~0.01612903
        # Chunk 3: absent in dense, rank 2 in sparse (0.01612903) -> ~0.01612903
        c1 = _create_scored_chunk("c1", "doc1", "Dense and sparse match", 0.95)
        c2 = _create_scored_chunk("c2", "doc1", "Dense only match", 0.85)
        c3 = _create_scored_chunk("c3", "doc1", "Sparse only match", 5.2)

        dense_candidates = [c1, c2]
        sparse_candidates = [c1, c3]

        rrf_scores = compute_rrf_scores(
            dense_candidates=dense_candidates,
            sparse_candidates=sparse_candidates,
            k=60,
        )

        expected_c1 = (1.0 / (60 + 1)) + (1.0 / (60 + 1))  # 2 / 61
        expected_c2 = 1.0 / (60 + 2)  # 1 / 62
        expected_c3 = 1.0 / (60 + 2)  # 1 / 62

        assert pytest.approx(rrf_scores["c1"], rel=1e-6) == expected_c1
        assert pytest.approx(rrf_scores["c2"], rel=1e-6) == expected_c2
        assert pytest.approx(rrf_scores["c3"], rel=1e-6) == expected_c3
        assert rrf_scores["c1"] > rrf_scores["c2"]

    def test_rrf_custom_k(self) -> None:
        """Verify RRF computation with a custom smoothing constant k."""
        c1 = _create_scored_chunk("c1", "doc1", "Content 1", 0.9)
        rrf_scores = compute_rrf_scores(
            dense_candidates=[c1],
            sparse_candidates=[],
            k=20,
        )
        assert pytest.approx(rrf_scores["c1"], rel=1e-6) == 1.0 / (20 + 1)

    def test_rrf_empty_candidate_pools(self) -> None:
        """Verify RRF computation when both candidate pools are empty."""
        rrf_scores = compute_rrf_scores(
            dense_candidates=[],
            sparse_candidates=[],
            k=60,
        )
        assert rrf_scores == {}


class TestHybridRetrieverCore:
    """Tests for the Layer 1 HybridRetriever engine."""

    def test_dependency_injection_and_singleton(self) -> None:
        """Verify HybridRetriever supports injecting custom stores and singleton access."""
        mock_vector = MagicMock(spec=VectorStore)
        mock_bm25 = MagicMock(spec=BM25Store)

        retriever = HybridRetriever(vector_store=mock_vector, bm25_store=mock_bm25)
        assert retriever.vector_store is mock_vector
        assert retriever.bm25_store is mock_bm25

        # Test global singleton getter and setter
        set_hybrid_retriever(retriever)
        assert get_hybrid_retriever() is retriever

        # Clean up singleton
        set_hybrid_retriever(None)

    def test_parallel_execution_of_stores(self) -> None:
        """Verify that dense and sparse stores are queried concurrently in parallel."""
        mock_vector = MagicMock(spec=VectorStore)
        mock_bm25 = MagicMock(spec=BM25Store)

        def slow_vector_query(*args: Any, **kwargs: Any) -> list[ScoredChunk]:
            _ = (args, kwargs)
            time.sleep(0.05)
            return [_create_scored_chunk("c_dense", "doc1", "Dense content", 0.9)]

        def slow_bm25_query(*args: Any, **kwargs: Any) -> list[ScoredChunk]:
            _ = (args, kwargs)
            time.sleep(0.05)
            return [_create_scored_chunk("c_sparse", "doc1", "Sparse content", 4.0)]

        mock_vector.query.side_effect = slow_vector_query
        mock_bm25.query.side_effect = slow_bm25_query

        retriever = HybridRetriever(vector_store=mock_vector, bm25_store=mock_bm25)

        start_time = time.perf_counter()
        results = retriever.retrieve("test query", top_k=5)
        duration = time.perf_counter() - start_time

        # If executed sequentially: >= 0.10s. If concurrent: ~0.05s (< 0.09s).
        assert duration < 0.095
        assert len(results) == 2
        mock_vector.query.assert_called_once()
        mock_bm25.query.assert_called_once()

    def test_candidate_merging_and_top_k_ranking(self) -> None:
        """Verify candidate pools are merged without dropping unique items and ranked by RRF."""
        mock_vector = MagicMock(spec=VectorStore)
        mock_bm25 = MagicMock(spec=BM25Store)

        # 6 distinct chunks total
        chunks = [_create_chunk(f"c{i}", "doc1", f"Content {i}") for i in range(1, 7)]

        # Dense: c1 (rank 1), c2 (rank 2), c3 (rank 3), c4 (rank 4)
        mock_vector.query.return_value = [
            _create_scored_chunk("c1", "doc1", chunks[0].content, 0.9),
            _create_scored_chunk("c2", "doc1", chunks[1].content, 0.8),
            _create_scored_chunk("c3", "doc1", chunks[2].content, 0.7),
            _create_scored_chunk("c4", "doc1", chunks[3].content, 0.6),
        ]

        # Sparse: c2 (rank 1), c5 (rank 2), c1 (rank 3), c6 (rank 4)
        mock_bm25.query.return_value = [
            _create_scored_chunk("c2", "doc1", chunks[1].content, 10.0),
            _create_scored_chunk("c5", "doc1", chunks[4].content, 8.0),
            _create_scored_chunk("c1", "doc1", chunks[0].content, 6.0),
            _create_scored_chunk("c6", "doc1", chunks[5].content, 4.0),
        ]

        retriever = HybridRetriever(vector_store=mock_vector, bm25_store=mock_bm25)
        top_results = retriever.retrieve("query", top_k=5)

        # Exactly top_k=5 chunks returned
        assert len(top_results) == 5

        # c1 score: 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
        # c2 score: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
        # c2 has the highest combined RRF score, followed by c1
        assert top_results[0].chunk_id == "c2"
        assert top_results[1].chunk_id == "c1"

        # Verify scores are sorted in descending order
        scores = [r.score for r in top_results]
        assert scores == sorted(scores, reverse=True)

    def test_strict_document_scoping_single_document(self) -> None:
        """Verify strict document scoping passes single doc_id filter to both stores."""
        mock_vector = MagicMock(spec=VectorStore)
        mock_bm25 = MagicMock(spec=BM25Store)

        retriever = HybridRetriever(vector_store=mock_vector, bm25_store=mock_bm25)
        mock_vector.query.return_value = [
            _create_scored_chunk("c1", "doc_selected", "Selected content", 0.9)
        ]
        mock_bm25.query.return_value = []

        results = retriever.retrieve("query", document_ids=["doc_selected"], top_k=5)

        # Check call arguments to both stores
        mock_vector.query.assert_called_once_with(
            query_text="query",
            document_ids=["doc_selected"],
            top_n=10,
        )
        mock_bm25.query.assert_called_once_with(
            query_text="query",
            document_ids=["doc_selected"],
            top_n=10,
        )

        assert len(results) == 1
        assert results[0].metadata.document_id == "doc_selected"

    def test_strict_document_scoping_multi_document(self) -> None:
        """Verify strict document scoping passes multiple doc_ids to both stores."""
        mock_vector = MagicMock(spec=VectorStore)
        mock_bm25 = MagicMock(spec=BM25Store)

        retriever = HybridRetriever(vector_store=mock_vector, bm25_store=mock_bm25)
        mock_vector.query.return_value = []
        mock_bm25.query.return_value = []

        retriever.retrieve("compare papers", document_ids=["doc_a", "doc_b"], top_k=5)

        mock_vector.query.assert_called_once_with(
            query_text="compare papers",
            document_ids=["doc_a", "doc_b"],
            top_n=10,
        )
        mock_bm25.query.assert_called_once_with(
            query_text="compare papers",
            document_ids=["doc_a", "doc_b"],
            top_n=10,
        )

    def test_global_retrieval_when_document_ids_empty_or_none(self) -> None:
        """Verify global search is executed when document_ids is None or empty list."""
        mock_vector = MagicMock(spec=VectorStore)
        mock_bm25 = MagicMock(spec=BM25Store)

        retriever = HybridRetriever(vector_store=mock_vector, bm25_store=mock_bm25)
        mock_vector.query.return_value = []
        mock_bm25.query.return_value = []

        # Empty list should pass None / empty list indicating global search
        retriever.retrieve("global query", document_ids=[], top_k=5)
        mock_vector.query.assert_called_with(
            query_text="global query",
            document_ids=None,
            top_n=10,
        )
        mock_bm25.query.assert_called_with(
            query_text="global query",
            document_ids=None,
            top_n=10,
        )

    def test_empty_query_returns_empty_list(self) -> None:
        """Verify empty or whitespace query immediately returns empty list."""
        mock_vector = MagicMock(spec=VectorStore)
        mock_bm25 = MagicMock(spec=BM25Store)

        retriever = HybridRetriever(vector_store=mock_vector, bm25_store=mock_bm25)

        assert retriever.retrieve("") == []
        assert retriever.retrieve("   \n\t  ") == []
        mock_vector.query.assert_not_called()
        mock_bm25.query.assert_not_called()

    def test_zero_matches_from_both_stores(self) -> None:
        """Verify retriever gracefully returns empty list when both stores yield no matches."""
        mock_vector = MagicMock(spec=VectorStore)
        mock_bm25 = MagicMock(spec=BM25Store)
        mock_vector.query.return_value = []
        mock_bm25.query.return_value = []

        retriever = HybridRetriever(vector_store=mock_vector, bm25_store=mock_bm25)
        results = retriever.retrieve("nonexistent phrase", top_k=5)
        assert results == []

    def test_custom_settings_override(self) -> None:
        """Verify custom Settings override dense_top_n, sparse_top_n, and rrf_k."""
        mock_vector = MagicMock(spec=VectorStore)
        mock_bm25 = MagicMock(spec=BM25Store)
        mock_vector.query.return_value = []
        mock_bm25.query.return_value = []

        custom_settings = Settings(
            DENSE_TOP_N=15,
            SPARSE_TOP_N=20,
            RRF_K=30,
            TOP_K=8,
        )

        retriever = HybridRetriever(
            vector_store=mock_vector,
            bm25_store=mock_bm25,
            settings=custom_settings,
        )

        retriever.retrieve("custom settings query")

        mock_vector.query.assert_called_once_with(
            query_text="custom settings query",
            document_ids=None,
            top_n=15,
        )
        mock_bm25.query.assert_called_once_with(
            query_text="custom settings query",
            document_ids=None,
            top_n=20,
        )


class TestLangGraphNodeDispatcher:
    """Tests for the Layer 2 LangGraph scoped_hybrid_retriever_node."""

    def test_scoped_hybrid_retriever_node_contract(self) -> None:
        """Verify the node dispatches standalone_query and returns state delta dict."""
        mock_retriever = MagicMock(spec=HybridRetriever)
        expected_chunk = _create_scored_chunk("c1", "doc1", "Found content", 0.032)
        mock_retriever.retrieve.return_value = [expected_chunk]

        set_hybrid_retriever(mock_retriever)

        state: ConversationalRAGState = {
            "conversation_id": "conv_123",
            "active_document_ids": ["doc1"],
            "active_document_titles": "doc1.pdf",
            "messages": [ChatMessage(role="user", content="Original question")],
            "user_query": "Original question",
            "standalone_query": "Standalone query for doc1",
            "retrieved_chunks": [],
            "retrieval_grade": "grounded",
            "response_text": "",
            "citations": [],
        }

        output = scoped_hybrid_retriever_node(state)

        # Must return dict with 'retrieved_chunks'
        assert isinstance(output, dict)
        assert "retrieved_chunks" in output
        assert output["retrieved_chunks"] == [expected_chunk]

        mock_retriever.retrieve.assert_called_once_with(
            query="Standalone query for doc1",
            document_ids=["doc1"],
            top_k=5,
        )

        set_hybrid_retriever(None)

    def test_node_fallback_to_user_query_if_standalone_empty(self) -> None:
        """Verify node falls back to user_query if standalone_query is empty or missing."""
        mock_retriever = MagicMock(spec=HybridRetriever)
        mock_retriever.retrieve.return_value = []
        set_hybrid_retriever(mock_retriever)

        state: ConversationalRAGState = {
            "conversation_id": "conv_456",
            "active_document_ids": [],
            "active_document_titles": "",
            "messages": [],
            "user_query": "Raw user question",
            "standalone_query": "",
            "retrieved_chunks": [],
            "retrieval_grade": "grounded",
            "response_text": "",
            "citations": [],
        }

        output = scoped_hybrid_retriever_node(state)
        assert output["retrieved_chunks"] == []

        mock_retriever.retrieve.assert_called_once_with(
            query="Raw user question",
            document_ids=[],
            top_k=5,
        )

        set_hybrid_retriever(None)


class TestHybridRetrieverIntegration:
    """Integration tests executing HybridRetriever with real VectorStore and BM25Store."""

    @pytest.fixture
    def real_stores(self, tmp_path: Path) -> tuple[VectorStore, BM25Store]:
        """Create real local VectorStore and in-memory BM25Store."""
        chroma_dir = tmp_path / "chroma_test"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        vector_store = VectorStore(
            persist_dir=chroma_dir,
            collection_name="hybrid_test_collection",
            embedding_function=DeterministicEmbeddingFunction(dimension=128),
        )
        bm25_store = BM25Store()
        return vector_store, bm25_store

    def test_end_to_end_scoped_isolation_zero_leakage(
        self, real_stores: tuple[VectorStore, BM25Store]
    ) -> None:
        """Verify strict document scoping returns 0% leakage from unselected documents."""
        vector_store, bm25_store = real_stores

        doc1_chunks = [
            _create_chunk("c1_1", "doc_alpha", "Quantum computing and entanglement circuits."),
            _create_chunk("c1_2", "doc_alpha", "Superconducting qubits maintain coherence."),
        ]
        doc2_chunks = [
            _create_chunk(
                "c2_1", "doc_beta", "Quantum computing algorithms for portfolio optimization."
            ),
            _create_chunk("c2_2", "doc_beta", "Photonic qubits exhibit low decoherence rates."),
        ]

        vector_store.add_chunks(doc1_chunks + doc2_chunks)
        bm25_store.add_chunks(doc1_chunks + doc2_chunks)

        retriever = HybridRetriever(vector_store=vector_store, bm25_store=bm25_store)

        # Scoped strictly to doc_alpha
        results_alpha = retriever.retrieve("quantum computing qubits", document_ids=["doc_alpha"])
        assert len(results_alpha) > 0
        for chunk in results_alpha:
            assert chunk.metadata.document_id == "doc_alpha"
            assert chunk.metadata.document_id != "doc_beta"

        # Scoped strictly to doc_beta
        results_beta = retriever.retrieve("quantum computing qubits", document_ids=["doc_beta"])
        assert len(results_beta) > 0
        for chunk in results_beta:
            assert chunk.metadata.document_id == "doc_beta"
            assert chunk.metadata.document_id != "doc_alpha"

    def test_end_to_end_technical_keyword_boost(
        self, real_stores: tuple[VectorStore, BM25Store]
    ) -> None:
        """Verify exact acronym and hyperparameter matches are boosted via RRF fusion."""
        vector_store, bm25_store = real_stores

        chunks = [
            _create_chunk(
                "general_chunk", "doc1", "We train our language models using large web corpora."
            ),
            _create_chunk(
                "hyperparam_chunk",
                "doc1",
                "The model achieves BLEU-4 score of 41.8 with lr=0.0003.",
            ),
            _create_chunk(
                "eval_chunk",
                "doc1",
                "Evaluation on machine translation benchmarks demonstrates accuracy.",
            ),
        ]

        vector_store.add_chunks(chunks)
        bm25_store.add_chunks(chunks)

        retriever = HybridRetriever(vector_store=vector_store, bm25_store=bm25_store)

        results = retriever.retrieve("BLEU-4 41.8 0.0003", top_k=3)
        assert len(results) > 0
        # The hyperparameter chunk with exact match must be the top ranked item
        assert results[0].chunk_id == "hyperparam_chunk"
        assert results[0].score > 0.0
