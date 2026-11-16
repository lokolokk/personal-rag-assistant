"""Approved targeted checks for table captions, headers and bounded context."""

import pytest
import tiktoken

from app.schemas.document import ChunkMetadata, ScoredChunk
from app.services.retrieval.table_context import prioritize_table_context


def chunk(
    name: str,
    text: str,
    start: int,
    *,
    table: bool = False,
    document: str = "research",
    page: int = 2,
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=name,
        content=text,
        score=0.01,
        metadata=ChunkMetadata(
            chunk_id=name,
            document_id=document,
            filename=f"{document}.pdf",
            page_number=page,
            section_title="Results",
            is_table=table,
            token_count=1,
            char_start=start,
            char_end=start + len(text),
        ),
    )


def evidence() -> tuple[ScoredChunk, ScoredChunk, ScoredChunk]:
    header = chunk(
        "header", "TASK A\nTASK B\n\nTraining data\nP\nR\nF1\nACC.\n\nMixed\n1\n2\n3\n4", 100
    )
    table = chunk(
        "rows",
        "| Encoder | P | R | F1 | ACC. |\n| --- | --- | --- | --- | --- |\n"
        "| Model-A | 80 | 70 | 75 | 66 |\n| Model-B | 70 | 80 | 74 | 88 |",
        header.metadata.char_end + 1,
        table=True,
    )
    caption = chunk(
        "caption", "Table 7: Encoder comparison for TASK A and TASK B", table.metadata.char_end + 1
    )
    return header, table, caption


def ranked(table: ScoredChunk) -> list[ScoredChunk]:
    return [chunk(f"background-{i}", "General background.", 2000 + 30 * i) for i in range(5)] + [
        table
    ]


@pytest.mark.parametrize("position", ["above", "below", "embedded"])
def test_caption_positions_preserve_original_sources(position: str) -> None:
    _, table, caption = evidence()
    if position == "above":
        caption = chunk(
            "caption", caption.content, table.metadata.char_start - len(caption.content) - 1
        )
    elif position == "embedded":
        caption = chunk(
            "caption", "Baseline-only\n10\n20\n\n" + caption.content, caption.metadata.char_start
        )
    selected = prioritize_table_context(
        "Table 7 encoder accuracy", ranked(table), [table, caption], 5
    )
    by_id = {item.chunk_id: item for item in selected}
    assert {"rows", "caption"} <= by_id.keys()
    for original in (table, caption):
        assert by_id[original.chunk_id].content == original.content
        assert by_id[original.chunk_id].metadata == original.metadata
    assert len(selected) == len(by_id) == 5


@pytest.mark.parametrize("matching", [True, False])
def test_adjacent_header_requires_matching_column_sequence(matching: bool) -> None:
    header, table, caption = evidence()
    if not matching:
        header = header.model_copy(
            update={"content": header.content.replace("P\nR\nF1\nACC.", "P\nR\nF1\nLOSS")}
        )
    selected = prioritize_table_context(
        "Table 7 encoder accuracy", ranked(table), [header, table, caption], 5
    )
    ids = [item.chunk_id for item in selected]
    assert ("header" in ids) is matching
    assert {"rows", "caption"} <= set(ids)
    if matching:
        assert ids.index("header") < ids.index("rows") < ids.index("caption")


@pytest.mark.parametrize("barrier", ["prose", "competing-caption", "other-table"])
def test_ambiguous_or_interrupted_caption_is_not_associated(barrier: str) -> None:
    _, table, caption = evidence()
    if barrier == "competing-caption":
        extra = chunk(
            "other-caption", "Table 8: Another comparison", table.metadata.char_start - 40
        )
    else:
        extra = chunk(
            "barrier", "A new section discusses a separate experiment.", table.metadata.char_end + 1
        )
        if barrier == "other-table":
            extra = chunk(
                "barrier",
                "| Optimizer | Loss |\n| SGD | 0.1 |",
                extra.metadata.char_start,
                table=True,
            )
        caption = chunk("caption", caption.content, extra.metadata.char_end + 1)
    selected = prioritize_table_context(
        "Table 7 encoder accuracy", ranked(table), [table, extra, caption], 5
    )
    assert "caption" not in {item.chunk_id for item in selected}


@pytest.mark.parametrize("boundary", ["document", "page", "filename"])
def test_caption_and_header_do_not_cross_source_boundaries(boundary: str) -> None:
    header, table, caption = evidence()
    changes: dict[str, str | int] = {
        "document_id": "other",
        "page_number": 3,
        "filename": "other.pdf",
    }
    field = {"document": "document_id", "page": "page_number", "filename": "filename"}[boundary]
    neighbors = [
        item.model_copy(
            update={"metadata": item.metadata.model_copy(update={field: changes[field]})}
        )
        for item in (header, caption)
    ]
    selected = prioritize_table_context(
        "Table 7 encoder accuracy", ranked(table), [table, *neighbors], 5
    )
    assert not {"header", "caption"} & {item.chunk_id for item in selected}


@pytest.mark.parametrize("budget", ["chunks", "tokens"])
def test_evidence_group_is_not_partially_promoted_over_budget(budget: str) -> None:
    header, table, caption = evidence()
    selected = prioritize_table_context(
        "Table 7 encoder accuracy",
        ranked(table),
        [header, table, caption],
        2 if budget == "chunks" else 5,
        max_tokens=1 if budget == "tokens" else 1000,
    )
    assert not {"header", "rows", "caption"} & {item.chunk_id for item in selected}


def test_other_document_anchor_and_exact_group_budget_are_preserved() -> None:
    header, table, caption = evidence()
    other = chunk("other-source", "Independent evidence.", 0, document="another")
    encoding = tiktoken.get_encoding("cl100k_base")
    limit = sum(len(encoding.encode(item.content)) for item in (header, table, caption, other))
    selected = prioritize_table_context(
        "Table 7 encoder accuracy",
        [other, *ranked(table)],
        [header, table, caption],
        4,
        max_tokens=limit,
    )
    assert {item.chunk_id for item in selected} == {"header", "rows", "caption", "other-source"}
    assert sum(len(encoding.encode(item.content)) for item in selected) == limit
