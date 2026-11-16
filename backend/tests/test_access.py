"""Invitation capabilities remain reusable, bounded and revocable across restarts."""

from pathlib import Path

import pytest

from app.services.access import AccessDenied, AccessLimitExceeded, AccessStore


def test_same_invitation_reuses_workspace_with_distinct_sessions(tmp_path: Path) -> None:
    store = AccessStore(tmp_path / "control.sqlite3")
    workspace_id, invitation = store.create_invitation()

    first = store.redeem(invitation)
    second = store.redeem(invitation)

    assert first.workspace_id == second.workspace_id == workspace_id
    assert first.session_token != second.session_token
    assert len(invitation) >= 43
    assert len(first.session_token) >= 43
    assert store.validate(first.session_token).workspace_id == workspace_id


@pytest.mark.parametrize("invalid", ["", "invalid", "../control.sqlite3", "a" * 4096])
def test_invalid_invitation_and_session_are_rejected(tmp_path: Path, invalid: str) -> None:
    store = AccessStore(tmp_path / "control.sqlite3")
    store.create_invitation()

    with pytest.raises(AccessDenied):
        store.redeem(invalid)
    with pytest.raises(AccessDenied):
        store.validate(invalid)


def test_revocation_invalidates_all_sessions_and_future_redemption(tmp_path: Path) -> None:
    store = AccessStore(tmp_path / "control.sqlite3")
    workspace_id, invitation = store.create_invitation()
    sessions = [store.redeem(invitation), store.redeem(invitation)]

    store.revoke(workspace_id)

    with pytest.raises(AccessDenied):
        store.redeem(invitation)
    for session in sessions:
        with pytest.raises(AccessDenied):
            store.validate(session.session_token)


def test_rotation_preserves_workspace_and_invalidates_old_access(tmp_path: Path) -> None:
    store = AccessStore(tmp_path / "control.sqlite3")
    workspace_id, invitation = store.create_invitation()
    old_session = store.redeem(invitation)

    replacement = store.rotate(workspace_id)

    assert replacement != invitation
    assert store.redeem(replacement).workspace_id == workspace_id
    with pytest.raises(AccessDenied):
        store.redeem(invitation)
    with pytest.raises(AccessDenied):
        store.validate(old_session.session_token)


def test_access_state_and_revocation_persist_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "control.sqlite3"
    store = AccessStore(path)
    workspace_id, invitation = store.create_invitation()
    session = store.redeem(invitation)

    reopened = AccessStore(path)
    assert reopened.validate(session.session_token).workspace_id == workspace_id
    assert reopened.redeem(invitation).workspace_id == workspace_id
    reopened.revoke(workspace_id)

    restarted = AccessStore(path)
    with pytest.raises(AccessDenied):
        restarted.redeem(invitation)
    with pytest.raises(AccessDenied):
        restarted.validate(session.session_token)


def test_persisted_files_do_not_contain_raw_capabilities(tmp_path: Path) -> None:
    store = AccessStore(tmp_path / "control.sqlite3")
    _, invitation = store.create_invitation()
    session = store.redeem(invitation)

    for path in tmp_path.iterdir():
        if path.is_file():
            contents = path.read_bytes()
            assert invitation.encode() not in contents
            assert session.session_token.encode() not in contents


def test_five_workspace_limit_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "control.sqlite3"
    store = AccessStore(path)
    workspace_ids = {store.create_invitation()[0] for _ in range(5)}
    assert len(workspace_ids) == 5

    with pytest.raises(AccessLimitExceeded):
        AccessStore(path).create_invitation()


def test_session_limit_does_not_invalidate_existing_sessions(tmp_path: Path) -> None:
    store = AccessStore(tmp_path / "control.sqlite3")
    _, invitation = store.create_invitation()
    sessions = [store.redeem(invitation) for _ in range(10)]

    with pytest.raises(AccessLimitExceeded):
        store.redeem(invitation)
    for session in sessions:
        assert store.validate(session.session_token).workspace_id == session.workspace_id


def test_session_expires_at_one_hour_and_invitation_remains_reusable(tmp_path: Path) -> None:
    now = [1000.0]
    store = AccessStore(tmp_path / "control.sqlite3", clock=lambda: now[0])
    workspace_id, invitation = store.create_invitation()
    session = store.redeem(invitation)
    assert session.expires_at == 4600.0

    now[0] = 4600.0
    with pytest.raises(AccessDenied):
        store.validate(session.session_token)
    assert store.redeem(invitation).workspace_id == workspace_id


def test_session_cannot_outlive_invitation(tmp_path: Path) -> None:
    now = [1000.0]
    store = AccessStore(tmp_path / "control.sqlite3", clock=lambda: now[0])
    _, invitation = store.create_invitation(invitation_ttl=60)
    session = store.redeem(invitation)
    assert session.expires_at == 1060.0

    now[0] = 1060.0
    with pytest.raises(AccessDenied):
        store.validate(session.session_token)
    with pytest.raises(AccessDenied):
        store.redeem(invitation)


def test_default_invitation_expires_after_thirty_days(tmp_path: Path) -> None:
    now = [1000.0]
    store = AccessStore(tmp_path / "control.sqlite3", clock=lambda: now[0])
    _, invitation = store.create_invitation()
    now[0] += 30 * 86400

    with pytest.raises(AccessDenied):
        store.redeem(invitation)


def test_revoking_one_workspace_preserves_another(tmp_path: Path) -> None:
    store = AccessStore(tmp_path / "control.sqlite3")
    first_id, first_invitation = store.create_invitation()
    second_id, second_invitation = store.create_invitation()
    first = store.redeem(first_invitation)
    second = store.redeem(second_invitation)

    store.revoke(first_id)

    with pytest.raises(AccessDenied):
        store.validate(first.session_token)
    assert store.validate(second.session_token).workspace_id == second_id
    assert store.redeem(second_invitation).workspace_id == second_id
