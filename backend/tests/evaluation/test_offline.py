"""Safety checks run before the application suites or inspection scripts."""

import socket
from pathlib import Path

import httpx
import pytest

from app.core.config import get_settings
from app.services.evaluation.offline import OfflineNetworkError, isolated_services
from app.services.ingestion.registry import get_document_registry
from app.services.retrieval.conversation_registry import get_conversation_registry
from app.services.retrieval.memory import get_memory_manager
from app.services.retrieval.reformulator import get_query_reformulator
from app.services.retrieval.vector_store import DeterministicEmbeddingFunction, get_vector_store


def test_credentials_storage_and_singletons_are_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "configured-outside-sandbox")
    original_memory = get_memory_manager()
    original_memory.add_message("outside", "user", "Keep this history")
    with isolated_services(tmp_path / "evaluation"):
        settings = get_settings()
        assert settings.OPENAI_API_KEY is None
        assert settings.resolved_chroma_dir == tmp_path / "evaluation/storage/chroma"
        assert get_memory_manager().list_sessions() == []
        assert get_document_registry().get_all() == []
        assert get_conversation_registry().get_all() == []
        assert isinstance(get_vector_store()._embedding_function, DeterministicEmbeddingFunction)
        assert get_query_reformulator().reformulate("What is its score?", "Some history") == (
            "What is its score?"
        )
        get_memory_manager().add_message("inside", "user", "Temporary")
    assert get_memory_manager() is original_memory
    assert original_memory.list_sessions() == ["outside"]


def test_network_is_blocked_before_dispatch(tmp_path: Path) -> None:
    with isolated_services(tmp_path):
        with pytest.raises(OfflineNetworkError):
            httpx.get("https://example.invalid")
        with socket.socket() as sock, pytest.raises(OfflineNetworkError):
            sock.connect(("192.0.2.1", 443))


def test_dns_is_blocked_before_resolution(tmp_path: Path) -> None:
    with isolated_services(tmp_path), pytest.raises(OfflineNetworkError):
        socket.getaddrinfo("example.invalid", 443)


@pytest.mark.asyncio
async def test_async_network_is_blocked_but_asgi_upload_and_chat_work(tmp_path: Path) -> None:
    with isolated_services(tmp_path):
        async with httpx.AsyncClient() as external_client:
            with pytest.raises(OfflineNetworkError):
                await external_client.get("https://example.invalid")

        from app.main import create_application

        app = create_application()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            pdf = Path(__file__).parents[1] / "fixtures/sample_table.pdf"
            response = await client.post(
                "/api/v1/documents/upload", files={"file": (pdf.name, pdf.read_bytes())}
            )
            assert response.status_code == 201
            assert get_vector_store().count() > 0
            chat = await client.post(
                "/api/v1/conversations/offline/messages",
                json={"content": "What are the model latencies?", "documentIds": []},
            )
            assert "event: done" in chat.text
            assert "event: error" not in chat.text
            conversation = get_conversation_registry().get("offline")
            assert conversation is not None
            assert len(conversation.messages) == 2
            assert (tmp_path / "storage/conversations.json").exists()


def test_suite_default_blocks_credentials_enabled_after_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "configured-during-test")
    get_settings.cache_clear()
    from app.services.retrieval.reformulator import QueryReformulator

    with pytest.raises(OfflineNetworkError):
        QueryReformulator().reformulate("An unexpected live query")


def test_suite_default_starts_with_fresh_services() -> None:
    assert get_memory_manager().list_sessions() == []
    assert get_conversation_registry().get_all() == []
