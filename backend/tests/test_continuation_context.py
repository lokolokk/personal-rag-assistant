"""General list evidence stays scoped, attributable and bounded after retrieval."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.schemas.document import ChunkMetadata, DocumentChunk, ScoredChunk
from app.services.retrieval.continuation import assemble_continuation_context


@pytest.mark.asyncio
async def test_live_graph_expands_only_own_existing_workspace_and_resolves_citation(
    tmp_path: Path,
) -> None:
    from app.api.v1.live import create_live_application
    from app.schemas.document import DocumentItem
    from tests.test_invited_rag import SyntheticProvider

    class EvidenceProvider(SyntheticProvider):
        def generate(self, prompt: str) -> Iterator[str]:
            assert "OWNER_INSPECTION" in prompt
            assert "FOREIGN_INSPECTION" not in prompt
            self.generation_calls += 1
            yield "Synthetic continuation answer [2]."

    provider = EvidenceProvider()
    app = create_live_application(tmp_path, provider)
    headers = {}
    for owner, label in ((True, "OWNER_INSPECTION"), (False, "FOREIGN_INSPECTION")):
        workspace, invitation = app.state.access.create_invitation()
        grant = app.state.access.redeem(invitation)
        bundle = app.state.manager.get(workspace)
        intro = chunk("intro", "The workflow has these phases:", 0)
        answer = chunk(
            "list",
            f"1. {label} examines state.\n2. VALIDATION checks health.",
            intro.metadata.char_end + 1,
        )
        filler = [
            chunk(
                f"other-{i}",
                "Workflow phases are described in an independent discussion.",
                1000 + i * 100,
            )
            for i in range(6)
        ]
        chunks = [intro, answer, *filler]
        bundle.documents.add_or_update(
            DocumentItem(
                id="manual",
                filename="manual.pdf",
                page_count=1,
                chunk_count=len(chunks),
            )
        )
        bundle.vectors.add_chunks(
            chunks,
            embeddings=[
                [1.0] + [0.0] * 1023,
                [-1.0] + [0.0] * 1023,
                *[[0.9, 0.1] + [0.0] * 1022 for _ in filler],
            ],
        )
        bundle.bm25.add_chunks(chunks)
        if owner:
            headers = {"Authorization": "Bearer " + grant.session_token}

    # Dense ranking is intentionally controlled, not semantic-provider evidence.
    provider.embeddings = lambda texts: [[1.0] + [0.0] * 1023 for _ in texts]
    async with AsyncClient(transport=ASGITransport(app), base_url="https://test") as client:
        created = await client.post("/api/v1/conversations", json={}, headers=headers)
        conversation = created.json()["conversationId"]
        response = await client.post(
            f"/api/v1/conversations/{conversation}/messages",
            headers=headers,
            json={"content": "workflow phases", "documentIds": ["manual"]},
        )
        assert '"finishReason": "stop"' in response.text, response.text
        history = await client.get(
            f"/api/v1/conversations/{conversation}/messages", headers=headers
        )
        citation = history.json()[-1]["citations"][0]
        assert citation["chunkId"] == "list" and citation["pageNumber"] == 1
        assert "OWNER_INSPECTION" in citation["quote"]
    assert provider.generation_calls == 1


def chunk(
    name: str,
    text: str,
    start: int,
    *,
    document: str = "manual",
    page: int = 1,
    table: bool = False,
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=name,
        content=text,
        score=0.03,
        metadata=ChunkMetadata(
            chunk_id=name,
            document_id=document,
            filename=f"{document}.pdf",
            page_number=page,
            section_title="Section",
            is_table=table,
            token_count=1,
            char_start=start,
            char_end=start + len(text),
        ),
    )


@pytest.mark.parametrize(
    "introduction,body",
    [
        (
            "The recovery workflow has three phases:",
            "1. INSPECTION examines state.\n2. RESTORATION restores files.\n"
            "3. VALIDATION checks health.",
        ),
        (
            "The access process contains these stages:",
            "1. AUTHENTICATION verifies identity.\n2. AUTHORIZATION checks rights.",
        ),
        (
            "The review requires the following evidence:",
            "- A reproducible procedure.\n- An independently verified result.",
        ),
    ],
)
def test_continuation_replaces_lowest_anchor_without_extra_chunks(
    introduction: str,
    body: str,
) -> None:
    anchor = chunk("intro", introduction, 0)
    answer = chunk("list", body, anchor.metadata.char_end + 1)
    other = chunk("other", "An independently ranked passage.", 500)
    result = assemble_continuation_context([anchor, other], [answer], 2, 300)
    assert [c.chunk_id for c in result] == ["intro", "list"]
    assert result[1].content == answer.content and result[1].metadata == answer.metadata


@pytest.mark.parametrize("boundary", ["document", "page", "gap", "table", "heading"])
def test_expansion_never_crosses_scope_or_unrelated_boundaries(boundary: str) -> None:
    anchor = chunk("intro", "The workflow has these phases:", 0)
    answer = chunk(
        "next",
        "1. INSPECTION examines state.\n2. VALIDATION checks health.",
        anchor.metadata.char_end + (100 if boundary == "gap" else 1),
        document="foreign" if boundary == "document" else "manual",
        page=2 if boundary == "page" else 1,
        table=boundary == "table",
    )
    if boundary == "heading":
        answer = chunk("next", "2. Methods\nA genuine new section.", anchor.metadata.char_end + 1)
    assert assemble_continuation_context([anchor], [answer], 2, 300) == [anchor]


def test_budget_counts_actual_text_and_deduplicates_sources() -> None:
    anchor = chunk("intro", "Required evidence:", 0)
    answer = chunk("list", "- A result.\n- A repeatable method.", anchor.metadata.char_end + 1)
    assert assemble_continuation_context([anchor], [answer], 2, 5) == [anchor]
    result = assemble_continuation_context([anchor, answer], [answer, answer], 2, 100)
    assert [c.chunk_id for c in result] == ["intro", "list"]
    assert assemble_continuation_context([anchor], [answer], 1, 100) == [anchor]


def test_trace_records_ids_and_decisions_without_source_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    anchor = chunk("intro", "Confidential process steps:", 0)
    answer = chunk(
        "list", "- First private step.\n- Second private step.", anchor.metadata.char_end + 1
    )
    with caplog.at_level(logging.DEBUG, logger="app.services.retrieval.continuation"):
        assemble_continuation_context([anchor], [answer], 2, 100)
    assert "list" in caplog.text
    assert "Confidential" not in caplog.text and "private step" not in caplog.text


def test_empty_or_zero_budget_returns_no_context() -> None:
    anchor = chunk("intro", "Evidence:", 0)
    available: list[DocumentChunk] = []
    assert assemble_continuation_context([], available, 5, 100) == []
    assert assemble_continuation_context([anchor], available, 0, 100) == []
    assert assemble_continuation_context([anchor], available, 5, 0) == []


def test_continuation_does_not_evict_another_documents_only_anchor() -> None:
    anchor = chunk("intro", "The workflow has these phases:", 0)
    answer = chunk(
        "list",
        "1. INSPECTION checks state.\n2. VALIDATION checks health.",
        anchor.metadata.char_end + 1,
    )
    comparison = chunk(
        "comparison", "Independent evidence from the second document.", 0, document="second"
    )
    result = assemble_continuation_context([anchor, comparison], [answer], 2, 300)
    assert [c.chunk_id for c in result] == ["intro", "comparison"]


def test_untrusted_long_number_cannot_break_list_detection() -> None:
    from app.services.ingestion.structure import introduced_list_items

    assert introduced_list_items(["9" * 5000 + ". Malformed input"], "Items:") == set()
