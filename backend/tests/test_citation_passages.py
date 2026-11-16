"""Provider-free stored passage access and historical preview contracts."""

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.live import create_live_application
from app.schemas.chat import ChatMessage, ConversationSummaryItem
from app.schemas.document import ChunkMetadata, DocumentChunk, DocumentItem
from app.services.generation.grounding import resolve_citations


@pytest.mark.asyncio
async def test_full_passage_auth_ownership_and_deleted_sources(tmp_path: Path) -> None:
    app = create_live_application(tmp_path)  # No provider exists in this application.
    access = app.state.access
    owner, invitation = access.create_invitation()
    _, foreign_invitation = access.create_invitation()
    headers = {"Authorization": "Bearer " + access.redeem(invitation).session_token}
    foreign = {"Authorization": "Bearer " + access.redeem(foreign_invitation).session_token}
    bundle = app.state.manager.get(owner)
    text = "  Original stored passage\n" + "source content " * 40 + "\n<script>literal</script>  "
    chunk = DocumentChunk(
        chunk_id="doc_source_p7_c3",
        content=text,
        metadata=ChunkMetadata(
            chunk_id="doc_source_p7_c3",
            document_id="doc_source",
            filename="source.pdf",
            page_number=7,
            section_title="Results",
            token_count=100,
            char_start=0,
            char_end=len(text),
        ),
    )
    bundle.documents.add_or_update(DocumentItem(id="doc_source", filename="source.pdf"))
    bundle.vectors.add_chunks([chunk], embeddings=[[0.1] * 1024])
    endpoint = "/api/v1/chunks/doc_source_p7_c3"
    async with AsyncClient(transport=ASGITransport(app), base_url="https://test") as client:
        assert (await client.get(endpoint)).status_code == 401
        assert (await client.get(endpoint, headers=foreign)).status_code == 404
        result = await client.get(endpoint, headers=headers)
        assert result.status_code == 200
        assert result.json() == {
            "chunkId": chunk.chunk_id,
            "text": text,
            "filename": "source.pdf",
            "pageNumber": 7,
        }
        assert result.headers["cache-control"] == "no-store"
        assert (
            await client.get(endpoint + "?workspace=" + owner, headers=foreign)
        ).status_code == 422
        missing = await client.get("/api/v1/chunks/missing", headers=headers)
        assert missing.status_code == 404
        bundle.documents.remove("doc_source")  # Even orphaned vectors must stay inaccessible.
        deleted = await client.get(endpoint, headers=headers)
        assert deleted.status_code == 404
        assert deleted.json() == missing.json()
        assert text not in deleted.text


@pytest.mark.asyncio
async def test_history_marks_actual_shortening_without_rewriting_saved_citations(
    tmp_path: Path,
) -> None:
    app = create_live_application(tmp_path)
    owner, invitation = app.state.access.create_invitation()
    headers = {"Authorization": "Bearer " + app.state.access.redeem(invitation).session_token}
    bundle = app.state.manager.get(owner)
    texts = ["x" * 240, "y" * 200 + "..."]  # Natural ellipsis is not proof of truncation.
    citations = []
    for i, text in enumerate(texts):
        cid = f"doc_source_p1_c{i}"
        chunk = DocumentChunk(
            chunk_id=cid,
            content=text,
            metadata=ChunkMetadata(
                chunk_id=cid,
                document_id="doc_source",
                filename="source.pdf",
                page_number=1,
                section_title="",
                token_count=60,
                char_start=0,
                char_end=len(text),
            ),
        )
        bundle.vectors.add_chunks([chunk], embeddings=[[0.1] * 1024])
        citations.append(
            {"id": i + 1, "chunk_id": cid, "quote": text[:200] + "..." if i == 0 else text}
        )
    bundle.documents.add_or_update(DocumentItem(id="doc_source", filename="source.pdf"))
    message = ChatMessage(role="assistant", content="Answer [1] [2]", citations=citations)
    bundle.conversations.add_or_update(
        ConversationSummaryItem(id="conv_source", messages=[message])
    )
    async with AsyncClient(transport=ASGITransport(app), base_url="https://test") as client:
        result = await client.get("/api/v1/conversations/conv_source/messages", headers=headers)
    returned = result.json()[0]["citations"]
    assert [c["isTruncated"] for c in returned] == [True, False]
    assert [{k: v for k, v in c.items() if k != "isTruncated"} for c in returned] == citations
    assert "isTruncated" not in bundle.conversations.get("conv_source").messages[0].citations[0]


def test_new_citation_has_explicit_preview_flag() -> None:
    for text, expected in [("a" * 201, True), ("short...", False)]:
        citation = resolve_citations("Answer [1]", [{"chunk_id": "chunk", "content": text}])[0]
        assert citation["isTruncated"] is expected
        assert citation["sourceText"] == (text[:200] + "..." if expected else text)
