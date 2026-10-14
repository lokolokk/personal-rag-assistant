"""All ordinary tests are offline, including collection and singleton defaults."""

from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from app.services.evaluation.offline import isolated_services

_SESSION_SANDBOX = ExitStack()


def pytest_sessionstart(session: pytest.Session) -> None:
    _ = session
    root = _SESSION_SANDBOX.enter_context(TemporaryDirectory(prefix="rag_pytest_"))
    _SESSION_SANDBOX.enter_context(isolated_services(Path(root)))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _ = (session, exitstatus)
    _SESSION_SANDBOX.close()


@pytest.fixture(autouse=True)
def offline_test_services(tmp_path: Path) -> Iterator[None]:
    """Each test gets fresh services and paths, even if a prior fixture resets them."""
    with isolated_services(tmp_path):
        yield
