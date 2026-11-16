"""Provider configuration, token bounds, cancellation and no automatic replay."""

import threading
from pathlib import Path

import pytest

from app.core.exceptions import AppException
from app.services.access import AccessStore
from app.services.limits import UsageLimits
from app.services.live_provider import BoundProvider, estimated_tokens
from tests.test_invited_rag import SyntheticProvider


def test_revoked_access_blocks_all_provider_calls(tmp_path: Path) -> None:
    access = AccessStore(tmp_path / "control.sqlite3")
    workspace, secret = access.create_invitation()
    provider = SyntheticProvider()
    bound = BoundProvider(provider, access, UsageLimits(access), access.redeem(secret))
    access.revoke(workspace)
    with pytest.raises(AppException):
        bound.embeddings(["quartz"])
    with pytest.raises(AppException):
        bound.reformulate("quartz")
    with pytest.raises(AppException):
        list(bound.generate("quartz"))
    assert (
        provider.embedding_calls == provider.reformulation_calls == provider.generation_calls == 0
    )


def test_generation_and_embedding_cannot_replay(tmp_path: Path) -> None:
    access = AccessStore(tmp_path / "control.sqlite3")
    _, secret = access.create_invitation()
    provider = SyntheticProvider()
    bound = BoundProvider(provider, access, UsageLimits(access), access.redeem(secret))
    list(bound.generate("quartz"))
    with pytest.raises(AppException):
        list(bound.generate("quartz"))
    bound.embeddings(["quartz"])
    with pytest.raises(AppException):
        bound.embeddings(["quartz"])
    assert provider.generation_calls == provider.embedding_calls == 1


def test_deadline_and_cancel_block_dispatch(tmp_path: Path) -> None:
    access = AccessStore(tmp_path / "control.sqlite3")
    _, secret = access.create_invitation()
    provider = SyntheticProvider()
    bound = BoundProvider(provider, access, UsageLimits(access), access.redeem(secret))
    bound.deadline = 0
    with pytest.raises(AppException):
        bound.embeddings(["quartz"])
    other = BoundProvider(provider, access, UsageLimits(access), access.redeem(secret))
    other.cancelled = threading.Event()
    other.cancelled.set()
    with pytest.raises(AppException):
        other.embeddings(["quartz"])
    assert provider.embedding_calls == 0


def test_upload_budget_admitted_before_first_batch_and_reserved_once(tmp_path: Path) -> None:
    access = AccessStore(tmp_path / "control.sqlite3")
    workspace, secret = access.create_invitation()
    provider = SyntheticProvider()
    limits = UsageLimits(access)
    texts = ["quartz"] * 32
    total = sum(estimated_tokens(text) for text in texts)
    limits.reserve(workspace, "embedding", 200_000 - total + 1)
    bound = BoundProvider(provider, access, limits, access.redeem(secret), upload=True)
    bound.prepare_embeddings(texts)
    assert provider.embedding_calls == 0
    with access.transaction() as db:
        amount = db.execute(
            "SELECT amount FROM usage WHERE scope=? AND kind='embedding'", (workspace,)
        ).fetchone()[0]
    assert amount == 200_001

    other_workspace, other_secret = access.create_invitation()
    other = BoundProvider(provider, access, limits, access.redeem(other_secret), upload=True)
    other.prepare_embeddings(texts)
    other.embeddings(texts[:16])
    other.embeddings(texts[16:])
    assert provider.embedding_calls == 2
    with access.transaction() as db:
        amount = db.execute(
            "SELECT amount FROM usage WHERE scope=? AND kind='embedding'", (other_workspace,)
        ).fetchone()[0]
    assert amount == total


@pytest.mark.asyncio
async def test_upload_provider_preflight_rejects_without_any_dispatch(tmp_path: Path) -> None:
    from httpx import ASGITransport, AsyncClient

    from app.api.v1.live import create_live_application

    class BudgetRejectingProvider(SyntheticProvider):
        def check_embedding_budget(self, tokens: int, calls: int) -> None:
            assert tokens > 0 and calls > 0
            raise AppException("Insufficient upload embedding allowance.", "EMBEDDING_BUDGET", 429)

    provider = BudgetRejectingProvider()
    app = create_live_application(tmp_path, provider)
    _, invitation = app.state.access.create_invitation()
    grant = app.state.access.redeem(invitation)
    pdf = (Path(__file__).parent / "fixtures/sample_singlecolumn.pdf").read_bytes()
    async with AsyncClient(transport=ASGITransport(app), base_url="https://test") as client:
        response = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("sample.pdf", pdf, "application/pdf")},
            headers={"Authorization": "Bearer " + grant.session_token},
        )
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "EMBEDDING_BUDGET"
    assert response.json()["detail"] == "Insufficient upload embedding allowance."
    assert provider.embedding_calls == 0
