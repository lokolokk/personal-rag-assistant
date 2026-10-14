"""Cloud LLM streaming client and OpenAI-compatible completions interface (Task 4.2).

Provides unified synchronous and streaming completions for Qwen / OpenAI-compatible models
with support for dependency injection in offline testing.
"""

import logging
import threading
from collections.abc import Iterator
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import LLMConfigurationError

logger = logging.getLogger(__name__)

_SENTINEL_DEFAULT_KEY = object()


class ChatLLMClient:
    """OpenAI-compatible chat completion client with native token streaming."""

    def __init__(
        self,
        api_key: str | None | object = _SENTINEL_DEFAULT_KEY,
        api_base: str | None = None,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        """Initialize ChatLLMClient.

        Args:
            api_key: Optional explicit API key, or None to disable.
            api_base: Optional OpenAI-compatible base URL.
            model: Default chat model identifier.
            client: Injected OpenAI client instance, mock, or callable.
        """
        settings = get_settings()
        self._model = model or settings.chat_model

        if client is not None:
            self._client = client
        else:
            if api_key is _SENTINEL_DEFAULT_KEY:
                resolved_key = settings.OPENAI_API_KEY
            elif isinstance(api_key, str) and api_key.strip():
                resolved_key = api_key.strip()
            else:
                resolved_key = None

            resolved_base = (
                api_base.strip()
                if isinstance(api_base, str) and api_base.strip()
                else settings.OPENAI_API_BASE
            )

            if resolved_key:
                import openai

                client_kwargs: dict[str, Any] = {"api_key": resolved_key}
                if resolved_base:
                    client_kwargs["base_url"] = resolved_base
                self._client = openai.OpenAI(**client_kwargs)
            else:
                self._client = None

    @property
    def client(self) -> Any | None:
        """Return the underlying client instance."""
        return self._client

    @property
    def model(self) -> str:
        """Return the default model identifier."""
        return self._model

    def complete_chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Execute a blocking chat completion call.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            model: Optional model override.
            temperature: Sampling temperature (default 0.0).

        Returns:
            Completed response string.
        """
        target_model = model or self._model

        if self._client is None:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is not configured and no LLM client was provided."
            )

        if hasattr(self._client, "chat"):
            response = self._client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        elif callable(self._client):
            prompt = "\n".join(m.get("content", "") for m in messages)
            return str(self._client(prompt))

        raise LLMConfigurationError(f"Unsupported client type: {type(self._client)}")

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """Execute a streaming chat completion call yielding individual token strings.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            model: Optional model override.
            temperature: Sampling temperature (default 0.0).

        Yields:
            Token strings as they arrive from the LLM provider.
        """
        target_model = model or self._model

        if self._client is None:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is not configured and no LLM client was provided."
            )

        if hasattr(self._client, "chat"):
            response_stream = self._client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        elif callable(self._client):
            prompt = "\n".join(m.get("content", "") for m in messages)
            full_text = str(self._client(prompt))
            for word in full_text.split(" "):
                yield word + " "
        else:
            raise LLMConfigurationError(f"Unsupported client type: {type(self._client)}")


_GLOBAL_LLM_CLIENT: ChatLLMClient | None = None
_LLM_LOCK = threading.Lock()


def get_llm_client() -> ChatLLMClient:
    """Return the cached singleton ChatLLMClient instance."""
    global _GLOBAL_LLM_CLIENT
    if _GLOBAL_LLM_CLIENT is None:
        with _LLM_LOCK:
            if _GLOBAL_LLM_CLIENT is None:
                _GLOBAL_LLM_CLIENT = ChatLLMClient()
    return _GLOBAL_LLM_CLIENT


def set_llm_client(client: ChatLLMClient | None) -> None:
    """Set or reset the cached singleton ChatLLMClient instance."""
    global _GLOBAL_LLM_CLIENT
    with _LLM_LOCK:
        _GLOBAL_LLM_CLIENT = client
