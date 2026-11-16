"""Scoped Hybrid Retriever Core (Layer 1) & LangGraph Dispatcher Node.

Executes concurrent dense semantic retrieval (ChromaDB) and sparse lexical retrieval
(BM25Okapi), fuses ranked candidate pools using Reciprocal Rank Fusion (RRF, k=60),
and exposes a deterministic tool interface for LangGraph state graph dispatch.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any

from app.core.config import Settings, get_settings
from app.schemas.chat import ConversationalRAGState
from app.schemas.document import ScoredChunk
from app.services.retrieval.bm25_store import BM25Store, get_bm25_store
from app.services.retrieval.continuation import assemble_continuation_context
from app.services.retrieval.table_context import prioritize_table_context
from app.services.retrieval.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


def compute_rrf_scores(
    dense_candidates: list[ScoredChunk],
    sparse_candidates: list[ScoredChunk],
    k: int = 60,
) -> dict[str, float]:
    """Calculate Reciprocal Rank Fusion (RRF) scores across dense and sparse candidate lists.

    Formula:
        RRF_Score(d) = sum_{m in {dense, sparse}} (1.0 / (k + rank_m(d)))

    where rank_m(d) is the 1-based rank position of chunk d in list m. If chunk d does
    not appear in a candidate list, its reciprocal contribution for that list is 0.0.
    """
    scores: dict[str, float] = {}

    for rank_idx, chunk in enumerate(dense_candidates, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (1.0 / (k + rank_idx))

    for rank_idx, chunk in enumerate(sparse_candidates, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (1.0 / (k + rank_idx))

    return scores


class HybridRetriever:
    """Layer 1 Native Engine executing scoped hybrid search and Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_store: BM25Store | None = None,
        settings: Settings | None = None,
        max_workers: int = 2,
    ) -> None:
        self._settings = settings or get_settings()
        self._vector_store = vector_store or get_vector_store()
        self._bm25_store = bm25_store or get_bm25_store()
        self._max_workers = max_workers

    @property
    def vector_store(self) -> VectorStore:
        """Return the underlying dense vector store instance."""
        return self._vector_store

    @property
    def bm25_store(self) -> BM25Store:
        """Return the underlying sparse BM25 store instance."""
        return self._bm25_store

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Fuse hybrid anchors and assemble at most K bounded evidence chunks.

        Args:
            query: User search query or reformulated standalone query.
            document_ids: Optional document IDs filter. None or empty list searches all docs.
            top_k: Maximum context chunks to return. Defaults to settings.TOP_K.

        Returns:
            RRF-ordered anchors with eligible direct list continuations prioritized.
            Added continuations have score zero: they are evidence neighbors, not
            independently ranked hits. Private DEBUG trace retains the fused ranks.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        target_top_k = top_k if top_k is not None else self._settings.TOP_K
        filter_doc_ids = document_ids if document_ids else None

        dense_top_n = self._settings.DENSE_TOP_N
        sparse_top_n = self._settings.SPARSE_TOP_N
        rrf_k = self._settings.RRF_K

        # Concurrently dispatch dense and sparse searches
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_dense = executor.submit(
                copy_context().run,
                self._vector_store.query,
                query_text=cleaned_query,
                document_ids=filter_doc_ids,
                top_n=dense_top_n,
            )
            future_sparse = executor.submit(
                copy_context().run,
                self._bm25_store.query,
                query_text=cleaned_query,
                document_ids=filter_doc_ids,
                top_n=sparse_top_n,
            )
            dense_results = future_dense.result()
            sparse_results = future_sparse.result()

        if not dense_results and not sparse_results:
            return []

        # Map unique chunk objects by ID (preserving first-seen candidate payload)
        unique_chunks: dict[str, ScoredChunk] = {}
        for chunk in dense_results:
            unique_chunks[chunk.chunk_id] = chunk
        for chunk in sparse_results:
            if chunk.chunk_id not in unique_chunks:
                unique_chunks[chunk.chunk_id] = chunk

        # Compute reciprocal rank fusion scores
        rrf_scores = compute_rrf_scores(
            dense_candidates=dense_results,
            sparse_candidates=sparse_results,
            k=rrf_k,
        )

        # Build fused scored chunks with updated RRF scores
        fused_chunks: list[ScoredChunk] = []
        for chunk_id, chunk in unique_chunks.items():
            fused_score = rrf_scores.get(chunk_id, 0.0)
            fused_chunks.append(
                ScoredChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    score=fused_score,
                )
            )

        # Sort descending by RRF score with deterministic secondary sort on chunk_id
        fused_chunks.sort(key=lambda c: (c.score, c.chunk_id), reverse=True)

        anchors = fused_chunks[: max(0, target_top_k)]
        context_candidates = (
            [chunk for chunk in fused_chunks if chunk.metadata.is_table] if target_top_k > 0 else []
        )
        context_candidates.extend(
            chunk
            for chunk in anchors
            if not chunk.metadata.is_table and chunk.content.rstrip().endswith(":")
        )
        pages = {
            (chunk.metadata.document_id, chunk.metadata.page_number) for chunk in context_candidates
        }
        available = self._bm25_store.get_chunks_for_pages(pages) if pages else []
        anchors = prioritize_table_context(
            cleaned_query,
            fused_chunks,
            available,
            target_top_k,
            max_tokens=self._settings.CHUNK_SIZE * target_top_k,
        )
        context = assemble_continuation_context(
            anchors, available, target_top_k, self._settings.CHUNK_SIZE * target_top_k
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "retrieval dense=%s sparse=%s fused=%s final_context=%s",
                [(i, c.chunk_id, c.score) for i, c in enumerate(dense_results, 1)],
                [(i, c.chunk_id, c.score) for i, c in enumerate(sparse_results, 1)],
                [(i, c.chunk_id, c.score) for i, c in enumerate(fused_chunks, 1)],
                [c.chunk_id for c in context],
            )
        return context


# Global cached singleton instance for HybridRetriever
_hybrid_retriever_instance: HybridRetriever | None = None


def get_hybrid_retriever() -> HybridRetriever:
    """Return the global singleton instance of HybridRetriever."""
    global _hybrid_retriever_instance
    if _hybrid_retriever_instance is None:
        _hybrid_retriever_instance = HybridRetriever()
    return _hybrid_retriever_instance


def set_hybrid_retriever(retriever: HybridRetriever | None) -> None:
    """Set or reset the global singleton instance of HybridRetriever (for testing/DI)."""
    global _hybrid_retriever_instance
    _hybrid_retriever_instance = retriever


def scoped_hybrid_retriever_node(
    state: ConversationalRAGState,
) -> dict[str, Any]:
    """Pure LangGraph node function dispatching query to Layer 1 HybridRetriever.

    Adheres strictly to docs/DESIGN.md Section 3.2 (Node 2) and LangGraph node contracts.
    Reads standalone_query (or user_query fallback) and active_document_ids, invokes Layer 1,
    and returns state update delta {"retrieved_chunks": list[ScoredChunk]}.
    """
    retriever = get_hybrid_retriever()
    query = state.get("standalone_query", "").strip()
    if not query:
        query = state.get("user_query", "").strip()

    active_doc_ids = state.get("active_document_ids")

    chunks = retriever.retrieve(
        query=query,
        document_ids=active_doc_ids,
        top_k=5,
    )

    return {"retrieved_chunks": chunks}
