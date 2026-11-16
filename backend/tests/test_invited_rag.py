"""Synthetic dense/BM25/graph acceptance, distinct from online semantic evidence."""

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.live import create_live_application
from app.services.access import AccessStore
from app.services.retrieval.vector_store import DeterministicEmbeddingFunction


class SyntheticProvider:
    def __init__(self) -> None:
        self.embedding_calls = 0
        self.generation_calls = 0
        self.reformulation_calls = 0

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        self.embedding_calls += 1
        return [[float(x) for x in v] for v in DeterministicEmbeddingFunction(1024)(texts)]

    def reformulate(self, prompt: str) -> str:
        self.reformulation_calls += 1
        return prompt.split("User Follow-up:\n")[-1].split("\n\nStandalone Search Query:")[0]

    def generate(self, prompt: str) -> Iterator[str]:
        _ = prompt
        self.generation_calls += 1
        yield "Synthetic answer [1]."


@pytest.mark.asyncio
async def test_identical_filenames_scoped_dense_sparse_chat_and_reuse(tmp_path: Path) -> None:
    provider = SyntheticProvider()
    app = create_live_application(tmp_path, provider)
    access = AccessStore(tmp_path / "control.sqlite3")
    a, invite_a = access.create_invitation()
    b, invite_b = access.create_invitation()
    ah = {"Authorization": "Bearer " + access.redeem(invite_a).session_token}
    bh = {"Authorization": "Bearer " + access.redeem(invite_b).session_token}
    pdf = (Path(__file__).parent / "fixtures/sample_singlecolumn.pdf").read_bytes()
    async with AsyncClient(transport=ASGITransport(app), base_url="https://test") as client:
        first = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("same.pdf", pdf, "application/pdf")},
            headers=ah,
        )
        second = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("same.pdf", pdf, "application/pdf")},
            headers=bh,
        )
        assert first.status_code == second.status_code == 201, (first.text, second.text)
        da, db = first.json()["id"], second.json()["id"]
        assert da != db
        for method in ("GET", "DELETE"):
            assert (
                await client.request(method, f"/api/v1/documents/{da}", headers=bh)
            ).status_code == 404
        ca = (await client.post("/api/v1/conversations", json={}, headers=ah)).json()[
            "conversationId"
        ]
        cb = (await client.post("/api/v1/conversations", json={}, headers=bh)).json()[
            "conversationId"
        ]
        before = provider.embedding_calls
        reused = access.redeem(invite_a)
        documents = await client.get(
            "/api/v1/documents", headers={"Authorization": "Bearer " + reused.session_token}
        )
        assert documents.json()[0]["id"] == da
        assert provider.embedding_calls == before
        mixed = await client.post(
            f"/api/v1/conversations/{ca}/messages",
            json={"content": "Explain", "documentIds": [da, db]},
            headers=ah,
        )
        assert mixed.status_code == 404
        assert provider.embedding_calls == before
        responses = await asyncio.gather(
            *[
                client.post(
                    f"/api/v1/conversations/{cid}/messages",
                    json={"content": "Introduction research", "documentIds": []},
                    headers=headers,
                )
                for cid, headers in ((ca, ah), (cb, bh))
            ]
        )
        for response, own, foreign in zip(responses, (da, db), (db, da), strict=True):
            assert response.status_code == 200
            assert "finishReason" in response.text
            assert '"finishReason": "error"' not in response.text, response.text
            assert own in response.text and foreign not in response.text
        assert app.state.manager.get(a).memory.list_sessions() == [ca]
        assert app.state.manager.get(b).memory.list_sessions() == [cb]
        history = await client.get(f"/api/v1/conversations/{ca}/messages", headers=ah)
        assert len(history.json()) == 2
        assert history.json()[1]["citations"][0]["documentId"] == da
        listing = await client.get("/api/v1/conversations", headers=ah)
        assert listing.json()[0]["messageCount"] == 2


@pytest.mark.asyncio
async def test_live_without_provider_never_ingests_fixture_vectors(tmp_path: Path) -> None:
    app = create_live_application(tmp_path)
    access = AccessStore(tmp_path / "control.sqlite3")
    workspace, secret = access.create_invitation()
    headers = {"Authorization": "Bearer " + access.redeem(secret).session_token}
    async with AsyncClient(transport=ASGITransport(app), base_url="https://test") as client:
        response = await client.post(
            "/api/v1/documents/upload",
            headers=headers,
            files={"file": ("file.pdf", b"%PDF-", "application/pdf")},
        )
        assert response.status_code == 503
        assert app.state.manager.get(workspace).vectors.count() == 0


@pytest.mark.asyncio
async def test_cors_exact_origin_and_secret_error_redaction(tmp_path: Path) -> None:
    app = create_live_application(tmp_path, origins=["https://approved.pages.dev"])
    async with AsyncClient(transport=ASGITransport(app), base_url="https://test") as client:
        for origin, status in (
            ("https://approved.pages.dev", 200),
            ("https://foreign.pages.dev", 400),
        ):
            result = await client.options(
                "/api/v1/documents",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "authorization,ngrok-skip-browser-warning",
                },
            )
            assert result.status_code == status
        secret = "short-private-test-input"
        result = await client.post("/api/v1/access/redeem", json={"invitation": secret})
        assert result.status_code == 422
        assert secret not in result.text
        assert (await client.get("/docs")).status_code == 404


@pytest.mark.asyncio
async def test_failed_startup_blocks_all_live_workspaces(tmp_path: Path) -> None:
    app = create_live_application(tmp_path, SyntheticProvider())
    access = app.state.access
    a, secret = access.create_invitation()
    bundle = app.state.manager.get(a)
    (bundle.root / "documents.json").write_text("corrupt", encoding="utf-8")
    other = create_live_application(tmp_path, SyntheticProvider())
    token = access.redeem(secret).session_token
    async with other.router.lifespan_context(other):
        async with AsyncClient(transport=ASGITransport(other), base_url="https://test") as client:
            assert (await client.get("/ready")).status_code == 503
            result = await client.get(
                "/api/v1/access/session", headers={"Authorization": "Bearer " + token}
            )
            assert result.status_code == 503
