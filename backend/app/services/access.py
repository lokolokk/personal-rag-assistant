"""Local capability administration and durable, hashed browser sessions."""

import hashlib
import secrets
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import AppException


class AccessDenied(AppException):
    def __init__(self, code: str = "ACCESS_INVALID") -> None:
        super().__init__("Access is invalid, expired or revoked.", code, 401)


class AccessLimitExceeded(AppException):
    def __init__(self) -> None:
        super().__init__("Access limit reached.", "ACCESS_LIMIT", 429)


@dataclass(frozen=True)
class SessionGrant:
    session_token: str
    workspace_id: str
    expires_at: float


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AccessStore:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.clock = clock
        populated = (path.parent / "workspaces").exists() and any(
            (path.parent / "workspaces").iterdir()
        )
        if path.is_symlink() or (not path.exists() and populated):
            raise OSError("Missing or unsafe live control database")
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            if (
                populated
                and db.execute("SELECT name FROM sqlite_master WHERE name='invitations'").fetchone()
                is None
            ):
                raise OSError("Missing live control schema")
            schema = """
                CREATE TABLE IF NOT EXISTS format(version INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS invitations(
                  workspace TEXT PRIMARY KEY, hash TEXT UNIQUE NOT NULL,
                  expires REAL NOT NULL, enabled INTEGER NOT NULL, generation INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions(
                  hash TEXT PRIMARY KEY, workspace TEXT NOT NULL, expires REAL NOT NULL,
                  generation INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS usage(
                  period TEXT NOT NULL, scope TEXT NOT NULL, kind TEXT NOT NULL,
                  amount INTEGER NOT NULL, PRIMARY KEY(period,scope,kind));
            """
            for statement in schema.split(";"):
                if statement.strip():
                    db.execute(statement)
            versions = db.execute("SELECT version FROM format").fetchall()
            if not versions:
                db.execute("INSERT INTO format VALUES(1)")
            elif versions != [(1,)]:
                raise AppException("Unsupported live data format.", "STORAGE_UNAVAILABLE", 503)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def workspace_ids(self) -> list[str]:
        with self.transaction() as db:
            return [str(row[0]) for row in db.execute("SELECT workspace FROM invitations")]

    def create_invitation(
        self, invitation_ttl: float = 30 * 86400, *, display_name: str | None = None
    ) -> tuple[str, str]:
        if invitation_ttl <= 0 or invitation_ttl > 30 * 86400:
            raise ValueError("Invitation lifetime must be within 30 days")
        secret = secrets.token_urlsafe(32)
        workspace = uuid.uuid4().hex
        with self.transaction() as db:
            if db.execute("SELECT COUNT(*) FROM invitations").fetchone()[0] >= 5:
                raise AccessLimitExceeded()
            db.execute(
                "INSERT INTO invitations VALUES(?,?,?,?,?)",
                (workspace, digest(secret), self.clock() + invitation_ttl, 1, 1),
            )
            if display_name is not None:
                db.execute("INSERT INTO workspace_profiles VALUES(?,?)", (workspace, display_name))
        return workspace, secret

    def redeem(self, secret: str) -> SessionGrant:
        if len(secret) != 43:
            raise AccessDenied()
        now = self.clock()
        token = secrets.token_urlsafe(32)
        with self.transaction() as db:
            row = db.execute(
                "SELECT workspace,expires,enabled,generation FROM invitations WHERE hash=?",
                (digest(secret),),
            ).fetchone()
            if row is None:
                raise AccessDenied()
            if not row[2]:
                raise AccessDenied("ACCESS_REVOKED")
            if row[1] <= now:
                raise AccessDenied("ACCESS_EXPIRED")
            db.execute("DELETE FROM sessions WHERE expires<=?", (now,))
            count = db.execute(
                "SELECT COUNT(*) FROM sessions WHERE workspace=? AND generation=?", (row[0], row[3])
            ).fetchone()[0]
            total = db.execute(
                """SELECT COUNT(*) FROM sessions s JOIN invitations i
                ON s.workspace=i.workspace
                WHERE i.enabled=1 AND s.generation=i.generation"""
            ).fetchone()[0]
            if count >= 10 or total >= 50:
                raise AccessLimitExceeded()
            expires = min(now + 3600, float(row[1]))
            db.execute(
                "INSERT INTO sessions VALUES(?,?,?,?)", (digest(token), row[0], expires, row[3])
            )
        return SessionGrant(token, str(row[0]), expires)

    def validate(self, token: str) -> SessionGrant:
        if len(token) != 43:
            raise AccessDenied()
        with self.transaction() as db:
            row = db.execute(
                """SELECT s.workspace,s.expires,s.generation,i.expires,i.enabled,i.generation
                FROM sessions s JOIN invitations i ON s.workspace=i.workspace WHERE s.hash=?""",
                (digest(token),),
            ).fetchone()
            if row is None:
                raise AccessDenied()
            if not row[4] or row[2] != row[5]:
                raise AccessDenied("ACCESS_REVOKED")
            if min(row[1], row[3]) <= self.clock():
                raise AccessDenied("ACCESS_EXPIRED")
            return SessionGrant(token, str(row[0]), float(row[1]))

    def end_session(self, token: str) -> None:
        with self.transaction() as db:
            db.execute("DELETE FROM sessions WHERE hash=?", (digest(token),))

    def revoke(self, workspace: str) -> None:
        with self.transaction() as db:
            result = db.execute(
                "UPDATE invitations SET enabled=0,generation=generation+1 WHERE workspace=?",
                (workspace,),
            )
            if result.rowcount != 1:
                raise ValueError("Unknown workspace")

    def rotate(self, workspace: str) -> str:
        secret = secrets.token_urlsafe(32)
        with self.transaction() as db:
            result = db.execute(
                """UPDATE invitations SET hash=?,enabled=1,expires=?,generation=generation+1
                WHERE workspace=?""",
                (digest(secret), self.clock() + 30 * 86400, workspace),
            )
            if result.rowcount != 1:
                raise ValueError("Unknown workspace")
        return secret
