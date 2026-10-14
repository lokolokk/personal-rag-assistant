"""Tests for BM25 sparse indexer (lexical retrieval)."""

import pytest

from app.schemas.document import ChunkMetadata, DocumentChunk, ScoredChunk
from app.services.retrieval.bm25_store import BM25Store, tokenize_bm25


def create_sample_chunk(
    chunk_id: str,
    document_id: str,
    content: str,
    filename: str = "paper.pdf",
    page_number: int = 1,
    section_title: str = "Method",
    is_table: bool = False,
) -> DocumentChunk:
    """Helper to create a typed DocumentChunk."""
    return DocumentChunk(
        chunk_id=chunk_id,
        content=content,
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document_id,
            filename=filename,
            page_number=page_number,
            section_title=section_title,
            is_table=is_table,
            token_count=len(content.split()),
            char_start=0,
            char_end=len(content),
        ),
    )


@pytest.fixture
def bm25_store() -> BM25Store:
    """Create an empty in-memory BM25Store."""
    return BM25Store()


def test_bm25_tokenize_stopword_preservation() -> None:
    """Verify regex alphanumeric tokenization preserves stopwords, numbers, and acronyms."""
    text = "In Table 1, the TaDA25_2 model achieves an F1 of 0.76 with DoDuo."
    tokens = tokenize_bm25(text)

    # Stopwords like 'in', 'the', 'an', 'of', 'with' must be preserved
    assert "in" in tokens
    assert "the" in tokens
    assert "of" in tokens
    assert "with" in tokens

    # Exact numbers and acronyms must be preserved
    assert "table" in tokens
    assert "1" in tokens
    assert "0" in tokens
    assert "76" in tokens
    assert "doduo" in tokens
    # 'TaDA25_2' splits into alphanumeric tokens: tada25, 2
    assert "tada25" in tokens
    assert "2" in tokens


def test_bm25_add_and_count(bm25_store: BM25Store) -> None:
    """Test adding chunks and retrieving count."""
    chunks = [
        create_sample_chunk("c1", "doc1", "BERT: Pre-training of Deep Bidirectional Transformers"),
        create_sample_chunk(
            "c2", "doc1", "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
        ),
        create_sample_chunk("c3", "doc2", "Electra: Pre-training Text Encoders as Discriminators"),
    ]
    bm25_store.add_chunks(chunks)
    assert bm25_store.count() == 3


def test_bm25_exact_keyword_and_acronym_retrieval(bm25_store: BM25Store) -> None:
    """Test that rare acronyms and table numbers rank exact matches #1."""
    chunks = [
        create_sample_chunk(
            "c1",
            "doc1",
            "Table 1: Benchmark results on TaDA25 dataset. DoDuo achieves 0.76 accuracy.",
            is_table=True,
        ),
        create_sample_chunk(
            "c2",
            "doc1",
            "General background on natural language processing and transformer architectures.",
        ),
        create_sample_chunk(
            "c3",
            "doc2",
            "Table 2: Comparison of standard baseline methods without any deep learning.",
            is_table=True,
        ),
    ]
    bm25_store.add_chunks(chunks)

    # Query for exact acronym 'DoDuo'
    results = bm25_store.query("DoDuo", top_n=3)
    assert len(results) > 0
    top = results[0]
    assert isinstance(top, ScoredChunk)
    assert top.chunk_id == "c1"
    assert top.score > 0.0

    # Query for exact table reference 'Table 1'
    table_results = bm25_store.query("Table 1", top_n=3)
    assert len(table_results) > 0
    assert table_results[0].chunk_id == "c1"


def test_bm25_single_doc_scoping(bm25_store: BM25Store) -> None:
    """Test pre-filtering sub-corpus to a single document_id with zero candidate leakage."""
    chunks = [
        create_sample_chunk("c1", "doc1", "Convolutional neural network for image segmentation"),
        create_sample_chunk("c2", "doc2", "Convolutional neural network for image segmentation"),
        create_sample_chunk("c3", "doc3", "Convolutional neural network for image segmentation"),
    ]
    bm25_store.add_chunks(chunks)

    results = bm25_store.query(
        query_text="image segmentation",
        document_ids=["doc2"],
        top_n=5,
    )
    assert len(results) == 1
    assert results[0].chunk_id == "c2"
    assert results[0].metadata.document_id == "doc2"


def test_bm25_multi_doc_scoping(bm25_store: BM25Store) -> None:
    """Test scoping across multiple documents excludes unselected documents."""
    chunks = [
        create_sample_chunk("c1", "doc1", "Evaluation metrics and loss curves"),
        create_sample_chunk("c2", "doc2", "Evaluation metrics and validation loss"),
        create_sample_chunk("c3", "doc3", "Evaluation metrics and test loss"),
    ]
    bm25_store.add_chunks(chunks)

    results = bm25_store.query(
        query_text="evaluation metrics loss",
        document_ids=["doc1", "doc2"],
        top_n=5,
    )
    returned_docs = {r.metadata.document_id for r in results}
    assert returned_docs.issubset({"doc1", "doc2"})
    assert "doc3" not in returned_docs


def test_bm25_global_query(bm25_store: BM25Store) -> None:
    """Test query across all documents when document_ids is None or empty."""
    chunks = [
        create_sample_chunk("c1", "doc1", "Attention is all you need paper"),
        create_sample_chunk("c2", "doc2", "Attention mechanism in vision transformers"),
        create_sample_chunk("c3", "doc3", "Attention mechanisms in sequence models"),
    ]
    bm25_store.add_chunks(chunks)

    results_none = bm25_store.query("attention", document_ids=None, top_n=5)
    results_empty = bm25_store.query("attention", document_ids=[], top_n=5)

    assert len(results_none) == 3
    assert len(results_empty) == 3


def test_bm25_delete_document_and_rebuild(bm25_store: BM25Store) -> None:
    """Test document deletion removes chunks and rebuilds lexical index."""
    chunks = [
        create_sample_chunk("c1", "doc1", "Target keyword unique_token_xyz in doc1"),
        create_sample_chunk("c2", "doc1", "Second chunk in doc1"),
        create_sample_chunk("c3", "doc2", "Target keyword unique_token_xyz in doc2"),
    ]
    bm25_store.add_chunks(chunks)
    assert bm25_store.count() == 3

    # Delete doc1
    bm25_store.delete_document("doc1")
    assert bm25_store.count() == 1

    # Query for unique token should only return doc2
    results = bm25_store.query("unique_token_xyz", top_n=5)
    assert len(results) == 1
    assert results[0].chunk_id == "c3"
    assert results[0].metadata.document_id == "doc2"


def test_bm25_empty_query_or_zero_match(bm25_store: BM25Store) -> None:
    """Test handling of empty corpus, empty query text, or zero matching terms."""
    # Empty store
    assert bm25_store.query("anything") == []

    # Store with content, query with zero matching terms
    chunk = create_sample_chunk("c1", "doc1", "Quantum computing and qubits")
    bm25_store.add_chunks([chunk])

    # Unmatched query
    assert bm25_store.query("mitochondria photosynthesis") == []

    # Empty string query
    assert bm25_store.query("") == []

    # Nonexistent document_id
    assert bm25_store.query("quantum", document_ids=["nonexistent"]) == []
