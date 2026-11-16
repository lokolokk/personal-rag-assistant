"""An invalid provider citation must never become a saved live answer."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.live import create_live_application
from app.services.access import AccessStore
from tests.test_invited_rag import SyntheticProvider


class GhostProvider(SyntheticProvider):
    def generate(self, prompt: str) -> Iterator[str]:
        _ = prompt
        yield "Unverified synthetic answer [99]."


@pytest.mark.asyncio
async def test_invalid_citation_is_not_saved_as_answer(tmp_path: Path) -> None:
    app = create_live_application(tmp_path, GhostProvider())
    access = AccessStore(tmp_path / "control.sqlite3")
    _, invitation = access.create_invitation()
    headers = {"Authorization": "Bearer " + access.redeem(invitation).session_token}
    pdf = (Path(__file__).parent / "fixtures/sample_singlecolumn.pdf").read_bytes()
    async with AsyncClient(transport=ASGITransport(app), base_url="https://test") as client:
        uploaded = await client.post(
            "/api/v1/documents/upload",
            headers=headers,
            files={"file": ("same.pdf", pdf, "application/pdf")},
        )
        assert uploaded.status_code == 201
        created = await client.post("/api/v1/conversations", headers=headers, json={})
        conversation = created.json()["conversationId"]
        result = await client.post(
            f"/api/v1/conversations/{conversation}/messages",
            headers=headers,
            json={"content": "research"},
        )
        assert '"code": "GROUNDING_INVALID"' in result.text
        history = await client.get(
            f"/api/v1/conversations/{conversation}/messages", headers=headers
        )
        assert len(history.json()) == 1
        assert history.json()[0]["role"] == "user"
