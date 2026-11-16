"""Live API ownership checks happen before any provider dispatch."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.live import create_live_application
from app.services.access import AccessStore


class BlockedProvider:
    """Every provider entry point fails the test if ownership checks dispatch it."""

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError(f"Unexpected embeddings dispatch for {len(texts)} texts")

    def reformulate(self, prompt: str) -> str:
        raise AssertionError(f"Unexpected reformulation dispatch for {len(prompt)} characters")

    def generate(self, prompt: str) -> Iterator[str]:
        raise AssertionError(f"Unexpected generation dispatch for {len(prompt)} characters")


@pytest.mark.asyncio
async def test_unauthorized_operations_do_not_dispatch_provider(tmp_path: Path) -> None:
    app = create_live_application(tmp_path, provider=BlockedProvider())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        requests = [
            ("GET", "/api/v1/documents"),
            ("DELETE", "/api/v1/documents/forged"),
            ("GET", "/api/v1/conversations"),
            ("POST", "/api/v1/conversations"),
            ("GET", "/api/v1/conversations/forged/messages"),
            ("POST", "/api/v1/conversations/forged/messages"),
            ("DELETE", "/api/v1/conversations/forged"),
        ]
        for method, path in requests:
            response = await client.request(method, path, json={})
            assert response.status_code == 401, (method, path, response.text)


@pytest.mark.asyncio
async def test_foreign_conversation_is_invisible_and_cannot_be_mutated(tmp_path: Path) -> None:
    app = create_live_application(tmp_path, provider=BlockedProvider())
    access = AccessStore(tmp_path / "control.sqlite3")
    _, first_invitation = access.create_invitation()
    _, second_invitation = access.create_invitation()
    first_headers = {"Authorization": f"Bearer {access.redeem(first_invitation).session_token}"}
    second_headers = {"Authorization": f"Bearer {access.redeem(second_invitation).session_token}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        created = await client.post("/api/v1/conversations", json={}, headers=first_headers)
        assert created.status_code == 201, created.text
        conversation_id = created.json()["conversationId"]
        foreign_path = f"/api/v1/conversations/{conversation_id}"

        listing = await client.get("/api/v1/conversations", headers=second_headers)
        assert listing.status_code == 200
        assert conversation_id not in listing.text
        for method, suffix, body in [
            ("GET", "/messages", {}),
            ("POST", "/messages", {"message": "What does it say?", "documentIds": []}),
            ("PATCH", "", {"title": "stolen"}),
            ("DELETE", "", {}),
        ]:
            response = await client.request(
                method, foreign_path + suffix, json=body, headers=second_headers
            )
            assert response.status_code == 404, (method, response.text)

        preserved = await client.get(foreign_path + "/messages", headers=first_headers)
        assert preserved.status_code == 200


@pytest.mark.asyncio
async def test_revoked_session_cannot_list_existing_workspace(tmp_path: Path) -> None:
    app = create_live_application(tmp_path, provider=BlockedProvider())
    access = AccessStore(tmp_path / "control.sqlite3")
    workspace, invitation = access.create_invitation()
    headers = {"Authorization": f"Bearer {access.redeem(invitation).session_token}"}
    access.revoke(workspace)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
        for path in ["/api/v1/documents", "/api/v1/conversations", "/api/v1/access/session"]:
            response = await client.get(path, headers=headers)
            assert response.status_code == 401
