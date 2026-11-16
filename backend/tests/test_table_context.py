"""General table evidence must survive hybrid fusion and context cutoffs."""

from unittest.mock import MagicMock

from app.schemas.document import ChunkMetadata, DocumentChunk, ScoredChunk
from app.services.retrieval.table_context import prioritize_table_context
from app.services.workspaces import RequestBM25Store


def _chunk(
    chunk_id: str,
    content: str,
    rank_score: float,
    *,
    document: str = "paper",
    start: int = 0,
    table: bool = False,
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        content=content,
        score=rank_score,
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document,
            filename=f"{document}.pdf",
            page_number=5,
            section_title="Results",
            is_table=table,
            token_count=len(content.split()),
            char_start=start,
            char_end=start + len(content),
        ),
    )


def test_single_retriever_table_hit_is_promoted_with_its_caption() -> None:
    caption = _chunk("caption", "Table 2: Dataset split sizes", 0.0, start=100)
    table = _chunk(
        "table",
        "| Fold | Train | Dev | Test | All |\n| Train | 809 | 300 | 300 |",
        0.014,
        start=130,
        table=True,
    )
    fused = [
        _chunk(f"consensus-{rank}", f"Shared result {rank}.", 1 / rank) for rank in range(1, 9)
    ]
    fused.insert(7, table)  # A dense rank-8 hit outside the five-chunk RRF cutoff.

    selected = prioritize_table_context(
        "How many claims are in the training, development and test splits?",
        fused,
        [caption, table],
        max_chunks=5,
    )

    selected_ids = [chunk.chunk_id for chunk in selected]
    assert "table" in selected_ids
    assert "caption" in selected_ids
    assert len(selected_ids) == len(set(selected_ids)) == 5


def test_unmatched_table_does_not_displace_higher_ranked_evidence() -> None:
    table = _chunk("table", "| Optimizer | Learning rate |", 0.01, start=10, table=True)
    ranked = [_chunk(f"hit-{i}", "Disease risk and treatment result.", 1 / i) for i in range(1, 7)]
    ranked.append(table)

    selected = prioritize_table_context("What was the treatment outcome?", ranked, [table], 5)

    assert [chunk.chunk_id for chunk in selected] == [f"hit-{i}" for i in range(1, 6)]


def test_one_common_word_does_not_promote_a_table_for_a_longer_query() -> None:
    table = _chunk("table", "| Method | Test | Outcome |", 0.01, start=10, table=True)
    ranked = [_chunk(f"hit-{i}", "A method improves the result.", 1 / i) for i in range(1, 6)]
    ranked.append(table)

    selected = prioritize_table_context("Why does this method improve?", ranked, [table], 5)

    assert [chunk.chunk_id for chunk in selected] == [f"hit-{i}" for i in range(1, 6)]


def test_table_promotion_preserves_other_documents_represented_in_top_k() -> None:
    caption = _chunk("caption", "Table 1: Training data splits", 0, document="a", start=10)
    table = _chunk(
        "table",
        "| Train | Dev | Test |\n| 100 | 20 | 20 |",
        0.01,
        document="a",
        start=40,
        table=True,
    )
    ranked = [
        _chunk("a-anchor", "Overview of the training splits.", 0.03, document="a"),
        _chunk("b-anchor", "Independent evidence about a related result.", 0.029, document="b"),
        _chunk("c-anchor", "Another paper's reported finding.", 0.028, document="c"),
        _chunk("a-filler", "Lower ranked evidence from paper a.", 0.02, document="a"),
        _chunk("b-filler", "Lower ranked evidence from paper b.", 0.019, document="b"),
        table,
    ]

    selected = prioritize_table_context(
        "How many training, development and test items?", ranked, [caption, table], 5
    )

    ids = {chunk.chunk_id for chunk in selected}
    assert {"b-anchor", "c-anchor", "table"} <= ids
    assert "caption" in ids
    assert len(selected) == 5


def test_best_matching_table_wins_over_an_unrelated_table_on_same_page() -> None:
    caption_a = _chunk("caption-a", "Table 1: Error rates by optimizer", 0, start=10)
    table_a = _chunk("table-a", "| Optimizer | Error | Adam | 2.1 |", 0.02, start=40, table=True)
    caption_b = _chunk("caption-b", "Table 2: Dataset split totals", 0, start=100)
    table_b = _chunk(
        "table-b",
        "| Fold | Train | Dev | Test |\n| Train | 809 | 300 | 300 |",
        0.01,
        start=130,
        table=True,
    )
    ranked = [
        _chunk(f"hit-{rank}", "Shared scientific background.", 1 / rank) for rank in range(1, 7)
    ] + [table_a, table_b]

    selected = prioritize_table_context(
        "How many claims are in the training, development and test splits?",
        ranked,
        [caption_a, table_a, caption_b, table_b],
        5,
    )

    ids = {chunk.chunk_id for chunk in selected}
    assert {"caption-b", "table-b"} <= ids
    assert "table-a" not in ids and "caption-a" not in ids


def test_selected_chunk_contains_the_terms_that_matched_within_split_table() -> None:
    caption = _chunk("caption", "Table 3: Model evaluation", 0, start=10)
    generic_rows = _chunk("generic-rows", "| Model | Precision |", 0.03, start=40, table=True)
    matching_rows = _chunk(
        "matching-rows", "| Development precision | 0.91 |", 0.01, start=70, table=True
    )
    ranked = [
        _chunk(f"hit-{rank}", "General evaluation discussion.", 1 / rank) for rank in range(1, 7)
    ] + [generic_rows, matching_rows]

    selected = prioritize_table_context(
        "What is the development precision?",
        ranked,
        [caption, generic_rows, matching_rows],
        5,
    )

    assert "matching-rows" in {chunk.chunk_id for chunk in selected}


def test_repeated_table_numbers_keep_distinct_caption_groups() -> None:
    caption_a = _chunk("caption-a", "Table 1: Optimizer error rate", 0, start=10)
    table_a = _chunk("table-a", "| Adam | Error |", 0.03, start=40, table=True)
    caption_b = _chunk("caption-b", "Table 1: Development precision", 0, start=100)
    table_b = _chunk("table-b", "| Development precision | 0.91 |", 0.01, start=130, table=True)
    ranked = [
        _chunk(f"hit-{rank}", "General evaluation discussion.", 1 / rank) for rank in range(1, 7)
    ] + [table_a, table_b]

    selected = prioritize_table_context(
        "What is the development precision?",
        ranked,
        [caption_a, table_a, caption_b, table_b],
        5,
    )

    ids = {chunk.chunk_id for chunk in selected}
    assert {"caption-b", "table-b"} <= ids
    assert "caption-a" not in ids


def test_request_bm25_page_lookup_delegates_to_its_workspace_bundle() -> None:
    bundle = MagicMock()
    caption = DocumentChunk(
        chunk_id="caption",
        content="Table 2: Dataset split sizes",
        metadata=_chunk("caption", "Table 2: Dataset split sizes", 0).metadata,
    )
    bundle.bm25.get_chunks_for_pages.return_value = [caption]

    result = RequestBM25Store(bundle).get_chunks_for_pages({("paper", 5)})

    assert result == [caption]
    bundle.bm25.get_chunks_for_pages.assert_called_once_with({("paper", 5)})
