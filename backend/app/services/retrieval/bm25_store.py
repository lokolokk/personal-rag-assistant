"""BM25 sparse lexical indexer for exact keyword and technical retrieval."""

import math
import re

from rank_bm25 import BM25Okapi

from app.schemas.document import DocumentChunk, ScoredChunk


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens, preserving stop-words and numbers."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def build_bm25_index(tokenized_corpus: list[list[str]]) -> BM25Okapi:
    """Build a BM25Okapi index configured with Lucene's non-negative IDF formulation.

    Lucene / Elasticsearch formulation:
        IDF(q) = ln(1.0 + (N - n(q) + 0.5) / (n(q) + 0.5))

    Eliminates the negative or zero IDF floor problem of original Robertson 1994 BM25
    when terms appear in small corpora (N <= 2) or in more than half of documents,
    ensuring documents with matched terms always achieve positive relevance scores.
    """
    bm25 = BM25Okapi(tokenized_corpus)
    for word in bm25.idf:
        freq = sum(1 for doc_freq in bm25.doc_freqs if word in doc_freq)
        bm25.idf[word] = math.log(1.0 + (bm25.corpus_size - freq + 0.5) / (freq + 0.5))
    return bm25


class BM25Store:
    """In-memory BM25 lexical index with dynamic sub-corpus scoping and document lifecycle."""

    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}
        self._doc_chunks: dict[str, set[str]] = {}
        self._bm25: BM25Okapi | None = None
        self._corpus_chunk_ids: list[str] = []

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Add or update document chunks and invalidate cached index."""
        if not chunks:
            return

        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            doc_id = chunk.metadata.document_id
            if doc_id not in self._doc_chunks:
                self._doc_chunks[doc_id] = set()
            self._doc_chunks[doc_id].add(chunk.chunk_id)

        # Invalidate cached full index
        self._bm25 = None
        self._corpus_chunk_ids = []

    def delete_document(self, document_id: str) -> None:
        """Remove all chunks associated with a document_id and invalidate index."""
        chunk_ids_to_remove = self._doc_chunks.pop(document_id, set())
        for chunk_id in chunk_ids_to_remove:
            self._chunks.pop(chunk_id, None)

        self._bm25 = None
        self._corpus_chunk_ids = []

    def clear(self) -> None:
        """Clear all stored chunks and indices."""
        self._chunks.clear()
        self._doc_chunks.clear()
        self._bm25 = None
        self._corpus_chunk_ids = []

    def count(self) -> int:
        """Return total number of chunks stored in memory."""
        return len(self._chunks)

    def query(
        self,
        query_text: str,
        document_ids: list[str] | None = None,
        top_n: int = 10,
    ) -> list[ScoredChunk]:
        """Perform lexical BM25 search over all chunks or scoped to document_ids."""
        query_tokens = tokenize_bm25(query_text)
        if not query_tokens:
            return []

        candidates: list[tuple[DocumentChunk, float]] = []

        if document_ids:
            # Dynamic sub-corpus pre-filtering
            candidate_ids: list[str] = []
            for doc_id in document_ids:
                if doc_id in self._doc_chunks:
                    candidate_ids.extend(self._doc_chunks[doc_id])

            if not candidate_ids:
                return []

            candidate_chunks = [self._chunks[cid] for cid in candidate_ids]
            tokenized_sub_corpus = [tokenize_bm25(c.content) for c in candidate_chunks]
            sub_bm25 = build_bm25_index(tokenized_sub_corpus)
            scores = sub_bm25.get_scores(query_tokens)

            for chunk, score in zip(candidate_chunks, scores, strict=False):
                candidates.append((chunk, float(score)))
        else:
            # Global corpus search
            if not self._chunks:
                return []

            if self._bm25 is None or len(self._corpus_chunk_ids) != len(self._chunks):
                self._corpus_chunk_ids = list(self._chunks.keys())
                tokenized_corpus = [
                    tokenize_bm25(self._chunks[cid].content) for cid in self._corpus_chunk_ids
                ]
                self._bm25 = build_bm25_index(tokenized_corpus)

            scores = self._bm25.get_scores(query_tokens)
            for i, chunk_id in enumerate(self._corpus_chunk_ids):
                candidates.append((self._chunks[chunk_id], float(scores[i])))

        # Filter out zero-score matches (no keyword overlap)
        matched = [(chunk, score) for chunk, score in candidates if score > 0.0]
        matched.sort(key=lambda item: item[1], reverse=True)

        return [
            ScoredChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
                score=score,
            )
            for chunk, score in matched[:top_n]
        ]


_bm25_store_instance: BM25Store | None = None


def get_bm25_store() -> BM25Store:
    """Return the global singleton instance of the BM25Store."""
    global _bm25_store_instance
    if _bm25_store_instance is None:
        _bm25_store_instance = BM25Store()
    return _bm25_store_instance


def set_bm25_store(store: BM25Store | None) -> None:
    """Set or reset the global singleton instance of the BM25Store (for testing/DI)."""
    global _bm25_store_instance
    _bm25_store_instance = store
