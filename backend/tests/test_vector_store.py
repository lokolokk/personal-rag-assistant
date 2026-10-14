"""Tests for ChromaDB vector store (dense semantic retrieval)."""

from pathlib import Path

import pytest

from app.schemas.document import ChunkMetadata, DocumentChunk, ScoredChunk
from app.services.retrieval.vector_store import DeterministicEmbeddingFunction, VectorStore


def create_sample_chunk(
    chunk_id: str,
    document_id: str,
    content: str,
    filename: str = "test.pdf",
    page_number: int = 1,
    section_title: str = "Introduction",
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
def temp_chroma_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for ChromaDB persistence."""
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chroma_dir


@pytest.fixture
def vector_store(temp_chroma_dir: Path) -> VectorStore:
    """Create a test VectorStore instance with deterministic embedding function."""
    embedding_fn = DeterministicEmbeddingFunction(dimension=128)
    return VectorStore(
        persist_dir=temp_chroma_dir,
        collection_name="test_chunks",
        embedding_function=embedding_fn,
    )


def test_vector_store_initialization(temp_chroma_dir: Path) -> None:
    """Test that VectorStore initializes properly and creates the persistent directory."""
    store = VectorStore(
        persist_dir=temp_chroma_dir,
        collection_name="test_init",
        embedding_function=DeterministicEmbeddingFunction(dimension=64),
    )
    assert store.count() == 0
    assert temp_chroma_dir.exists()


def test_vector_store_add_and_count(vector_store: VectorStore) -> None:
    """Test adding chunks to VectorStore and verifying count."""
    chunks = [
        create_sample_chunk("c1", "doc1", "Transformer neural network architecture"),
        create_sample_chunk("c2", "doc1", "Self-attention mechanism and feedforward layers"),
        create_sample_chunk("c3", "doc2", "Convolutional neural network image recognition"),
    ]
    vector_store.add_chunks(chunks)
    assert vector_store.count() == 3


def test_vector_store_query_semantic(vector_store: VectorStore) -> None:
    """Test dense query returning ScoredChunk with cosine similarity scores."""
    chunks = [
        create_sample_chunk("c1", "doc1", "Multi-head attention improves translation quality"),
        create_sample_chunk("c2", "doc1", "Gradient descent optimizer and learning rate schedule"),
        create_sample_chunk("c3", "doc2", "Marine biology coral reef conservation techniques"),
    ]
    vector_store.add_chunks(chunks)

    results = vector_store.query(
        query_text="multi-head attention translation",
        top_n=2,
    )
    assert len(results) <= 2
    assert len(results) > 0
    top_result = results[0]
    assert isinstance(top_result, ScoredChunk)
    assert top_result.chunk_id == "c1"
    assert top_result.metadata.document_id == "doc1"
    assert top_result.metadata.filename == "test.pdf"
    # Similarity should be positive for high semantic overlap
    assert top_result.score > 0.5


def test_vector_store_single_doc_scoping(vector_store: VectorStore) -> None:
    """Test query strictly filtered to a single document_id."""
    chunks = [
        create_sample_chunk("c1", "doc1", "Deep learning optimization methods"),
        create_sample_chunk("c2", "doc2", "Deep learning optimization methods identical content"),
        create_sample_chunk("c3", "doc3", "Deep learning optimization methods third doc"),
    ]
    vector_store.add_chunks(chunks)

    # Scoped strictly to doc2
    results = vector_store.query(
        query_text="deep learning optimization",
        document_ids=["doc2"],
        top_n=10,
    )
    assert len(results) == 1
    assert results[0].chunk_id == "c2"
    assert results[0].metadata.document_id == "doc2"


def test_vector_store_multi_doc_scoping(vector_store: VectorStore) -> None:
    """Test query filtered to multiple document_ids ($in clause)."""
    chunks = [
        create_sample_chunk("c1", "doc1", "Reinforcement learning from human feedback"),
        create_sample_chunk("c2", "doc2", "Proximal policy optimization reinforcement learning"),
        create_sample_chunk("c3", "doc3", "Unrelated genetic algorithms paper"),
    ]
    vector_store.add_chunks(chunks)

    # Scoped to doc1 and doc2
    results = vector_store.query(
        query_text="reinforcement learning",
        document_ids=["doc1", "doc2"],
        top_n=10,
    )
    returned_doc_ids = {r.metadata.document_id for r in results}
    assert returned_doc_ids.issubset({"doc1", "doc2"})
    assert "doc3" not in returned_doc_ids


def test_vector_store_global_query(vector_store: VectorStore) -> None:
    """Test query without document_ids searches all documents."""
    chunks = [
        create_sample_chunk("c1", "doc1", "Attention is all you need paper"),
        create_sample_chunk("c2", "doc2", "BERT pre-training of transformers"),
        create_sample_chunk("c3", "doc3", "GPT language models few shot learners"),
    ]
    vector_store.add_chunks(chunks)

    results_none = vector_store.query(query_text="transformers", document_ids=None, top_n=5)
    results_empty = vector_store.query(query_text="transformers", document_ids=[], top_n=5)

    assert len(results_none) == len(results_empty)
    doc_ids = {r.metadata.document_id for r in results_none}
    assert len(doc_ids) > 1


def test_vector_store_delete_document(vector_store: VectorStore) -> None:
    """Test deleting all chunks for a specific document_id."""
    chunks = [
        create_sample_chunk("c1", "doc1", "Document 1 first chunk"),
        create_sample_chunk("c2", "doc1", "Document 1 second chunk"),
        create_sample_chunk("c3", "doc2", "Document 2 sole chunk"),
    ]
    vector_store.add_chunks(chunks)
    assert vector_store.count() == 3

    vector_store.delete_document("doc1")
    assert vector_store.count() == 1

    # Querying doc1 should return nothing
    doc1_results = vector_store.query("document", document_ids=["doc1"], top_n=5)
    assert len(doc1_results) == 0

    # Querying doc2 should still return doc2 chunk
    doc2_results = vector_store.query("document", document_ids=["doc2"], top_n=5)
    assert len(doc2_results) == 1
    assert doc2_results[0].chunk_id == "c3"


def test_vector_store_empty_query_or_no_match(vector_store: VectorStore) -> None:
    """Test graceful handling of empty collection or non-matching filter."""
    empty_results = vector_store.query("anything", top_n=5)
    assert empty_results == []

    chunk = create_sample_chunk("c1", "doc1", "Some content")
    vector_store.add_chunks([chunk])

    no_match_results = vector_store.query("anything", document_ids=["nonexistent"], top_n=5)
    assert no_match_results == []


def test_vector_store_explicit_embeddings(temp_chroma_dir: Path) -> None:
    """Test adding chunks and querying with explicit embedding vectors."""
    store = VectorStore(persist_dir=temp_chroma_dir, collection_name="test_explicit")
    chunks = [
        create_sample_chunk("c1", "doc1", "Alpha"),
        create_sample_chunk("c2", "doc1", "Beta"),
    ]
    # Pass explicit 4-dimensional unit vectors
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    store.add_chunks(chunks, embeddings=embeddings)
    assert store.count() == 2

    # Query with vector matching c1 exactly
    results = store.query(
        query_text="",
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        top_n=1,
    )
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert pytest.approx(results[0].score, abs=0.01) == 1.0


def test_end_to_end_ingestion_pipeline(temp_chroma_dir: Path) -> None:
    """Verify Checkpoint 2: PDF -> Parser -> Chunker -> Dual Index (Chroma + BM25)."""
    from app.services.ingestion.chunker import StructuralChunker
    from app.services.ingestion.parser import PDFParser
    from app.services.retrieval.bm25_store import BM25Store

    fixture_path = Path(__file__).parent / "fixtures" / "sample_table.pdf"
    assert fixture_path.exists()

    # Step 1: Parse PDF with layout and table awareness
    parser = PDFParser()
    parsed_doc = parser.parse_file(fixture_path)
    parsed_doc.document_id = "doc_pipeline_test"
    assert parsed_doc.page_count > 0

    # Step 2: Chunk document structurally
    chunker = StructuralChunker(target_chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(parsed_doc)
    assert len(chunks) > 0

    # Verify at least one table chunk was isolated
    table_chunks = [c for c in chunks if c.metadata.is_table]
    assert len(table_chunks) > 0

    # Step 3: Dual-index into VectorStore and BM25Store
    vstore = VectorStore(
        persist_dir=temp_chroma_dir,
        collection_name="pipeline_test_chunks",
        embedding_function=DeterministicEmbeddingFunction(dimension=128),
    )
    bstore = BM25Store()

    vstore.add_chunks(chunks)
    bstore.add_chunks(chunks)

    assert vstore.count() == len(chunks)
    assert bstore.count() == len(chunks)

    # Step 4: Verify dense semantic retrieval
    semantic_results = vstore.query("latency inference benchmarks", top_n=3)
    assert len(semantic_results) > 0
    assert semantic_results[0].metadata.document_id == "doc_pipeline_test"

    # Step 5: Verify sparse lexical retrieval for exact table headers
    lexical_results = bstore.query("Model Architecture Latency", top_n=3)
    assert len(lexical_results) > 0
    assert lexical_results[0].metadata.is_table is True
    assert "Model Architecture" in lexical_results[0].content


def test_vector_store_reset(vector_store: VectorStore) -> None:
    """Verify reset() empties the collection completely."""
    chunks = [
        create_sample_chunk("c1", "doc1", "Alpha chunk content"),
        create_sample_chunk("c2", "doc1", "Beta chunk content"),
    ]
    vector_store.add_chunks(chunks)
    assert vector_store.count() == 2

    vector_store.reset()
    assert vector_store.count() == 0


def test_vector_store_openai_compatible_initialization(
    monkeypatch: pytest.MonkeyPatch, temp_chroma_dir: Path
) -> None:
    """Verify VectorStore passes api_base and api_key to OpenAIEmbeddingFunction."""
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-12345")
    monkeypatch.setenv("OPENAI_API_BASE", "https://custom-dashscope-endpoint/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")

    # Clear cached settings so monkeypatched env is loaded
    from app.core.config import get_settings

    get_settings.cache_clear()

    vstore = VectorStore(
        persist_dir=temp_chroma_dir,
        collection_name="test_openai_compatible",
    )
    assert isinstance(vstore._embedding_function, OpenAIEmbeddingFunction)
    assert vstore._embedding_function.api_base == "https://custom-dashscope-endpoint/v1"
    assert vstore._embedding_function.model_name == "qwen3.7-text-embedding"

    get_settings.cache_clear()


def test_vector_store_add_chunks_batching(
    vector_store: VectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify add_chunks batches upsert calls in slices of <= 16 chunks."""
    from unittest.mock import MagicMock

    # Generate 35 chunks (> 20 chunk provider limit)
    chunks = [
        create_sample_chunk(f"batch_chunk_{i}", "doc_batch", f"Batch content for chunk {i}")
        for i in range(35)
    ]

    original_upsert = vector_store._collection.upsert
    mock_upsert = MagicMock(wraps=original_upsert)
    monkeypatch.setattr(vector_store._collection, "upsert", mock_upsert)

    vector_store.add_chunks(chunks)

    # Should be batched into ceil(35 / 16) = 3 calls (16, 16, 3)
    assert mock_upsert.call_count == 3
    for call_args in mock_upsert.call_args_list:
        batch_ids = call_args.kwargs.get("ids") or call_args.args[0]
        assert len(batch_ids) <= 16

    assert vector_store.count() == 35


def test_vector_store_add_chunks_batching_with_embeddings(
    vector_store: VectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify add_chunks batches precomputed embeddings synchronously with chunk slices."""
    from unittest.mock import MagicMock

    chunks = [
        create_sample_chunk(f"emb_chunk_{i}", "doc_emb", f"Emb content {i}") for i in range(25)
    ]
    fake_embeddings = [[0.1] * 128 for _ in range(25)]

    original_upsert = vector_store._collection.upsert
    mock_upsert = MagicMock(wraps=original_upsert)
    monkeypatch.setattr(vector_store._collection, "upsert", mock_upsert)

    vector_store.add_chunks(chunks, embeddings=fake_embeddings, batch_size=10)

    # 25 chunks with batch_size=10 -> 3 calls (10, 10, 5)
    assert mock_upsert.call_count == 3
    for call_args in mock_upsert.call_args_list:
        batch_embs = call_args.kwargs.get("embeddings")
        assert batch_embs is not None
        assert len(batch_embs) <= 10
