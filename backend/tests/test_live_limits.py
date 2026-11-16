"""Durable admission restrictions cannot be reset by restarting."""

from pathlib import Path

import pytest

from app.core.exceptions import AppException
from app.services.access import AccessStore
from app.services.limits import UsageLimits


def test_daily_reservations_survive_restart(tmp_path: Path) -> None:
    access = AccessStore(tmp_path / "control.sqlite3")
    workspace, _ = access.create_invitation()
    limits = UsageLimits(access)
    limits.reserve(workspace, "input", 200_000)
    with pytest.raises(AppException, match="limit"):
        UsageLimits(AccessStore(access.path)).reserve(workspace, "input", 1)


def test_global_budget_is_shared(tmp_path: Path) -> None:
    access = AccessStore(tmp_path / "control.sqlite3")
    a, _ = access.create_invitation()
    b, _ = access.create_invitation()
    limits = UsageLimits(access)
    limits.reserve(a, "output", 20_000)
    with pytest.raises(AppException):
        limits.reserve(a, "output", 1)
    limits.reserve(b, "output", 1024)


def test_embedding_tokens_are_uncapped_but_audited_across_restart(tmp_path: Path) -> None:
    access = AccessStore(tmp_path / "control.sqlite3")
    workspace, _ = access.create_invitation()
    UsageLimits(access).reserve(workspace, "embedding", 2_000_000)
    UsageLimits(AccessStore(access.path)).reserve(workspace, "embedding", 3_000_000)
    with access.transaction() as db:
        rows = db.execute("SELECT scope,amount FROM usage WHERE kind='embedding'").fetchall()
    assert dict(rows) == {workspace: 5_000_000, "*": 5_000_000}
    with pytest.raises(ValueError, match="Negative"):
        UsageLimits(access).reserve(workspace, "embedding", -1)


def test_operation_slots_release_and_reject_same_workspace(tmp_path: Path) -> None:
    access = AccessStore(tmp_path / "control.sqlite3")
    a, _ = access.create_invitation()
    limits = UsageLimits(access)
    with limits.operation(a):
        with pytest.raises(AppException):
            with limits.operation(a):
                pytest.fail("Second concurrent operation admitted")
    with limits.operation(a):
        assert limits.active_count == 1
    assert limits.active_count == 0


@pytest.mark.asyncio
async def test_invalid_uploads_do_not_exhaust_daily_attempt_allowance(tmp_path: Path) -> None:
    from httpx import ASGITransport, AsyncClient

    from app.api.v1.live import create_live_application

    app = create_live_application(tmp_path)
    _, invitation = app.state.access.create_invitation()
    token = app.state.access.redeem(invitation).session_token
    async with AsyncClient(transport=ASGITransport(app), base_url="https://test") as client:
        statuses = [
            (
                await client.post(
                    "/api/v1/documents/upload",
                    json={},
                    headers={"Authorization": "Bearer " + token},
                )
            ).status_code
            for _ in range(6)
        ]
    assert statuses == [422] * 6
    with app.state.access.transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM usage WHERE kind='upload'").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_ordinary_actions_ignore_old_usage_counters_without_resetting_them(
    tmp_path: Path,
) -> None:
    from datetime import datetime, timedelta, timezone

    from httpx import ASGITransport, AsyncClient

    from app.api.v1.live import create_live_application
    from app.services.limits import PUBLIC_LIMITS

    app = create_live_application(tmp_path)
    workspace, invitation = app.state.access.create_invitation()
    token = app.state.access.redeem(invitation).session_token
    now = datetime.now(timezone(timedelta(hours=8)))
    with app.state.access.transaction() as db:
        for kind, period, amount in (
            ("request", now.strftime("%Y-%m-%dT%H:%M"), 1000),
            ("upload", now.strftime("%Y-%m-%d"), 1000),
            ("chat", now.strftime("%Y-%m-%d"), 1000),
        ):
            db.execute("INSERT INTO usage VALUES(?,?,?,?)", (period, workspace, kind, amount))
    headers = {"Authorization": "Bearer " + token}
    async with AsyncClient(transport=ASGITransport(app), base_url="https://test") as client:
        assert (await client.get("/api/v1/conversations", headers=headers)).status_code == 200
        assert (
            await client.post("/api/v1/conversations", json={}, headers=headers)
        ).status_code == 201
        assert (
            await client.post("/api/v1/documents/upload", json={}, headers=headers)
        ).status_code == 422
    with app.state.access.transaction() as db:
        amounts = dict(db.execute("SELECT kind,amount FROM usage WHERE scope=?", (workspace,)))
    assert amounts == {"request": 1000, "upload": 1000, "chat": 1000}
    assert "uploadsPerDay" not in PUBLIC_LIMITS and "chatsPerDay" not in PUBLIC_LIMITS
