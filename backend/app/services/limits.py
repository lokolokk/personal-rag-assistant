"""Persistent provider budgets, authentication throttling and operation slots."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from app.core.exceptions import AppException
from app.services.access import AccessStore

LIMITS: dict[str, tuple[int | None, int | None]] = {
    "input": (200_000, 1_000_000),
    "output": (20_000, 100_000),
    "embedding": (None, None),
    "redeem": (5, 30),
}


class UsageLimits:
    def __init__(self, access: AccessStore) -> None:
        self.access = access
        self._lock = threading.Lock()
        self._active: set[str] = set()
        self._upload_active = False

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def reserve(self, workspace: str, kind: str, amount: int) -> None:
        if amount < 0:
            raise ValueError("Negative reservation")
        now = datetime.fromtimestamp(self.access.clock(), timezone(timedelta(hours=8)))
        period = now.strftime("%Y-%m-%dT%H:%M" if kind == "redeem" else "%Y-%m-%d")
        local_cap, total_cap = LIMITS[kind]
        with self.access.transaction() as db:
            for scope, cap in ((workspace, local_cap), ("*", total_cap)):
                row = db.execute(
                    "SELECT amount FROM usage WHERE period=? AND scope=? AND kind=?",
                    (period, scope, kind),
                ).fetchone()
                if cap is not None and (int(row[0]) if row else 0) + amount > cap:
                    raise AppException("Usage limit reached.", "QUOTA_EXCEEDED", 429)
            for scope in (workspace, "*"):
                db.execute(
                    """INSERT INTO usage VALUES(?,?,?,?) ON CONFLICT(period,scope,kind)
                    DO UPDATE SET amount=amount+excluded.amount""",
                    (period, scope, kind, amount),
                )
            # Keep daily audit counters; bounded minute counters cannot grow indefinitely.
            db.execute(
                "DELETE FROM usage WHERE kind='redeem' AND period<?",
                ((now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M"),),
            )

    @contextmanager
    def operation(self, workspace: str, upload: bool = False) -> Iterator[None]:
        with self._lock:
            if (
                workspace in self._active
                or len(self._active) >= 2
                or (upload and self._upload_active)
            ):
                raise AppException("Another operation is active. Try again.", "BUSY", 429)
            self._active.add(workspace)
            if upload:
                self._upload_active = True
        try:
            yield
        finally:
            with self._lock:
                self._active.remove(workspace)
                if upload:
                    self._upload_active = False


PUBLIC_LIMITS: dict[str, int] = {
    "pdfBytes": 10 * 1024 * 1024,
    "pdfPages": 100,
    "documents": 10,
    "pdfStorageBytes": 100 * 1024 * 1024,
    "chunks": 10_000,
    "conversations": 50,
    "turns": 100,
    "queryCharacters": 8000,
}
