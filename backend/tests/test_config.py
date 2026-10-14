"""Tests for application settings and environment configuration."""

from pathlib import Path

import pytest

from app.core.config import Settings, get_settings


def test_default_settings() -> None:
    """Verify default configuration values match project specifications."""
    settings = Settings()

    assert settings.PROJECT_NAME == "Conversational Document RAG"
    assert settings.API_V1_STR == "/api/v1"
    assert settings.ENVIRONMENT == "development"
    assert "http://localhost:5173" in settings.CORS_ORIGINS

    # Model & Embedding defaults
    assert settings.LLM_MODEL == "qwen3.8-flash"
    assert settings.chat_model == "qwen3.8-flash"
    assert settings.EMBEDDING_MODEL == "qwen3.7-text-embedding"
    assert settings.EMBEDDING_DIMENSION == 1024

    # Chunking & Retrieval hyperparameters
    assert settings.CHUNK_SIZE == 600
    assert settings.CHUNK_OVERLAP == 100
    assert settings.RRF_K == 60
    assert settings.DENSE_TOP_N == 10
    assert settings.SPARSE_TOP_N == 10
    assert settings.TOP_K == 5
    assert settings.MEMORY_WINDOW_TURNS == 6

    # Grounding fallback
    assert (
        settings.INABILITY_FALLBACK_MESSAGE
        == "Based on the provided document(s), I cannot find information to answer this question."
    )


def test_settings_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that environment variables cleanly override defaults."""
    monkeypatch.setenv("PROJECT_NAME", "Custom RAG Title")
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "150")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-mock-key")
    monkeypatch.setenv("ENVIRONMENT", "testing")

    settings = Settings()

    assert settings.PROJECT_NAME == "Custom RAG Title"
    assert settings.CHUNK_SIZE == 800
    assert settings.CHUNK_OVERLAP == 150
    assert settings.OPENAI_API_KEY == "sk-test-mock-key"
    assert settings.ENVIRONMENT == "testing"


def test_cors_origins_parsing_comma_separated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CORS origins can be parsed from a comma-separated string."""
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000, https://app.example.com")
    settings = Settings()
    assert settings.CORS_ORIGINS == [
        "http://localhost:3000",
        "https://app.example.com",
    ]


def test_cors_origins_parsing_json_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CORS origins can be parsed from a JSON array string."""
    monkeypatch.setenv("CORS_ORIGINS", '["http://localhost:3000", "https://app.example.com"]')
    settings = Settings()
    assert settings.CORS_ORIGINS == [
        "http://localhost:3000",
        "https://app.example.com",
    ]


def test_path_resolution(tmp_path: Path) -> None:
    """Verify relative paths resolve against BASE_DIR and ensure_directories creates dirs."""
    test_storage = tmp_path / "custom_storage"
    test_upload = test_storage / "uploads"
    test_chroma = test_storage / "chroma"

    settings = Settings(
        BASE_DIR=tmp_path,
        STORAGE_DIR=Path("custom_storage"),
        UPLOAD_DIR=Path("custom_storage/uploads"),
        CHROMA_PERSIST_DIR=Path("custom_storage/chroma"),
    )

    assert settings.resolved_storage_dir == test_storage
    assert settings.resolved_upload_dir == test_upload
    assert settings.resolved_chroma_dir == test_chroma

    settings.ensure_directories()
    assert test_upload.exists()
    assert test_upload.is_dir()
    assert test_chroma.exists()
    assert test_chroma.is_dir()


def test_get_settings_cached() -> None:
    """Verify get_settings returns a cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
