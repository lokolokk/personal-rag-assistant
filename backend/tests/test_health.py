"""Provider-free liveness/readiness checks using real isolated local stores."""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_application
from app.schemas.document import ChunkMetadata, DocumentChunk
from app.services.retrieval.bm25_store import get_bm25_store
from app.services.retrieval.vector_store import get_vector_store


def test_empty_startup_is_ready_and_health_is_unchanged() -> None:
    application = create_application()
    with TestClient(application) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "project": get_settings().PROJECT_NAME,
        }
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}
        assert get_bm25_store().count() == 0
    assert not list(get_settings().resolved_storage_dir.rglob(".ready-*"))


def test_populated_startup_hydrates_bm25() -> None:
    get_vector_store().add_chunks(
        [
            DocumentChunk(
                chunk_id="synthetic-chunk",
                content="synthetic persistence evidence",
                metadata=ChunkMetadata(
                    chunk_id="synthetic-chunk",
                    document_id="synthetic",
                    filename="synthetic.pdf",
                    page_number=1,
                    section_title="Test",
                    token_count=3,
                    char_start=0,
                    char_end=30,
                ),
            )
        ]
    )
    with TestClient(create_application()) as client:
        assert client.get("/ready").status_code == 200
        assert get_bm25_store().query("persistence")[0].chunk_id == "synthetic-chunk"


@pytest.mark.parametrize("boundary", ["get_vector_store", "get_bm25_store"])
def test_failed_initialization_stays_unready(boundary: str) -> None:
    with patch(f"app.main.{boundary}", side_effect=RuntimeError("private/path/secret")):
        with TestClient(create_application()) as client:
            response = client.get("/ready")
            assert response.status_code == 503
            assert response.json() == {"status": "not_ready"}
            assert client.get("/health").status_code == 200


def test_unwritable_storage_stays_unready() -> None:
    with patch("tempfile._mkstemp_inner", side_effect=PermissionError("private location")):
        with TestClient(create_application()) as client:
            assert client.get("/ready").status_code == 503
            assert client.get("/health").status_code == 200


def test_directory_creation_failure_stays_unready() -> None:
    with patch.object(Path, "mkdir", side_effect=PermissionError("private location")):
        with TestClient(create_application()) as client:
            assert client.get("/ready").status_code == 503


def test_readiness_is_per_application_and_cleared_on_shutdown() -> None:
    application = create_application()
    other = create_application()
    with TestClient(application) as client:
        assert client.get("/ready").status_code == 200
        assert TestClient(other).get("/ready").status_code == 503
    assert TestClient(application).get("/ready").status_code == 503


def test_startup_and_health_never_embed_or_construct_providers() -> None:
    with (
        patch("openai.OpenAI", side_effect=AssertionError("provider constructed")),
        patch("openai.AsyncOpenAI", side_effect=AssertionError("provider constructed")),
        patch(
            "app.services.retrieval.vector_store.DeterministicEmbeddingFunction.__call__",
            autospec=True,
            side_effect=AssertionError("health embedded text"),
        ),
    ):
        with TestClient(create_application()) as client:
            for _ in range(3):
                assert client.get("/health").status_code == 200
                assert client.get("/ready").status_code == 200
