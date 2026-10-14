"""Application configuration using Pydantic Settings."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class FlexibleEnvSettingsSource(EnvSettingsSource):
    """Env source that falls back to raw string when complex JSON decoding fails."""

    def decode_complex_value(self, field_name: str, field: FieldInfo, value: Any) -> Any:
        _ = (field_name, field)
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value


class FlexibleDotEnvSettingsSource(DotEnvSettingsSource):
    """DotEnv source that falls back to raw string when complex JSON decoding fails."""

    def decode_complex_value(self, field_name: str, field: FieldInfo, value: Any) -> Any:
        _ = (field_name, field)
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value


class Settings(BaseSettings):
    """Centralized application settings loaded from environment and defaults."""

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Project metadata
    PROJECT_NAME: str = "Conversational Document RAG"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"

    # CORS Configuration
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Base & Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    STORAGE_DIR: Path = Path("storage")
    UPLOAD_DIR: Path = Path("storage/uploads")
    CHROMA_PERSIST_DIR: Path = Path("storage/chroma")

    # LLM & Embedding Configuration
    OPENAI_API_KEY: str | None = None
    OPENAI_API_BASE: str | None = None
    LLM_MODEL: str = "qwen3.8-flash"
    EMBEDDING_MODEL: str = "qwen3.7-text-embedding"
    EMBEDDING_DIMENSION: int = 1024

    @property
    def chat_model(self) -> str:
        """Alias for LLM_MODEL matching LangGraph node specifications."""
        return self.LLM_MODEL

    # Ingestion & Chunking Hyperparameters
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100

    # Hybrid Retrieval & RRF Hyperparameters
    RRF_K: int = 60
    DENSE_TOP_N: int = 10
    SPARSE_TOP_N: int = 10
    TOP_K: int = 5

    # Conversational Session Memory
    MEMORY_WINDOW_TURNS: int = 6

    # Grounding & Fallback
    INABILITY_FALLBACK_MESSAGE: str = (
        "Based on the provided document(s), I cannot find information to answer this question."
    )

    @field_validator("OPENAI_API_KEY", "OPENAI_API_BASE", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        """Convert empty or whitespace-only string values to None."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> list[str]:
        """Parse CORS origins from a comma-separated string, JSON list, or list."""
        if isinstance(v, str):
            stripped = v.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                parsed: object = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed]
            return [item.strip() for item in stripped.split(",") if item.strip()]
        if isinstance(v, list):
            return [str(item).strip() for item in v]
        return []

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Inject flexible env sources that handle comma-separated string lists."""
        _ = (env_settings, dotenv_settings)
        env_file = settings_cls.model_config.get("env_file")
        return (
            init_settings,
            FlexibleEnvSettingsSource(settings_cls),
            FlexibleDotEnvSettingsSource(settings_cls, env_file=env_file),
            file_secret_settings,
        )

    def resolve_path(self, path: Path) -> Path:
        """Resolve a relative storage path against BASE_DIR."""
        if path.is_absolute():
            return path
        return self.BASE_DIR / path

    @property
    def resolved_storage_dir(self) -> Path:
        """Absolute path to the root storage directory."""
        return self.resolve_path(self.STORAGE_DIR)

    @property
    def resolved_upload_dir(self) -> Path:
        """Absolute path to the document uploads directory."""
        return self.resolve_path(self.UPLOAD_DIR)

    @property
    def resolved_chroma_dir(self) -> Path:
        """Absolute path to the ChromaDB persistence directory."""
        return self.resolve_path(self.CHROMA_PERSIST_DIR)

    def ensure_directories(self) -> None:
        """Ensure all required persistent storage directories exist on disk."""
        self.resolved_upload_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_chroma_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of the application settings."""
    return Settings()
