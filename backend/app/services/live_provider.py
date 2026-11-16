"""Explicit Qwen dispatch with no SDK retries or offline embedding fallback."""

import math
import threading
import time
from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

import openai
import tiktoken

from app.core.config import Settings
from app.core.exceptions import AppException
from app.services.access import AccessStore, SessionGrant
from app.services.limits import UsageLimits


class Provider(Protocol):
    def embeddings(self, texts: list[str]) -> list[list[float]]: ...
    def reformulate(self, prompt: str) -> str: ...
    def generate(self, prompt: str) -> Iterator[str]: ...


@runtime_checkable
class EmbeddingBudget(Protocol):
    def check_embedding_budget(self, tokens: int, calls: int) -> None: ...


class QwenProvider:
    def __init__(self, settings: Settings) -> None:
        if not settings.OPENAI_API_KEY or not settings.OPENAI_API_BASE:
            raise AppException(
                "Live provider configuration is incomplete.", "PROVIDER_UNAVAILABLE", 503
            )
        self.settings = settings
        self.client = openai.OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
            max_retries=0,
            timeout=60,
        )

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.settings.EMBEDDING_MODEL,
            input=texts,
            dimensions=1024,
            encoding_format="float",
        )
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    def _complete(self, prompt: str, stream: bool, cap: int) -> Any:
        return self.client.chat.completions.create(
            model=self.settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=stream,
            max_tokens=cap,
            extra_body={"enable_thinking": False},
        )

    def reformulate(self, prompt: str) -> str:
        return str(self._complete(prompt, False, 512).choices[0].message.content or "")

    def generate(self, prompt: str) -> Iterator[str]:
        stream = self._complete(prompt, True, 1024)
        try:
            for item in stream:
                if item.choices and item.choices[0].delta.content:
                    yield str(item.choices[0].delta.content)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()


def estimated_tokens(text: str) -> int:
    # Qwen's tokenizer differs: conservative local estimate, never an exact bill.
    return (
        math.ceil(
            len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=())) * 1.2
        )
        + 32
    )


class BoundProvider:
    def __init__(
        self,
        provider: Provider | None,
        access: AccessStore,
        limits: UsageLimits,
        grant: SessionGrant,
        upload: bool = False,
    ) -> None:
        self.provider = provider
        self.access = access
        self.limits = limits
        self.grant = grant
        self.deadline = time.monotonic() + (180 if upload else 120)
        self.upload = upload
        self._prepared_embedding_tokens: int | None = None
        self.calls: dict[str, int] = {}
        self.cancelled: threading.Event | None = None

    def check(self) -> None:
        if self.cancelled is not None and self.cancelled.is_set():
            raise AppException("Operation cancelled.", "OPERATION_CANCELLED", 503)
        self.access.validate(self.grant.session_token)
        if time.monotonic() >= self.deadline:
            raise AppException("Operation timed out.", "OPERATION_TIMEOUT", 503)

    def require(self) -> Provider:
        self.check()
        if self.provider is None:
            raise AppException(
                "Live provider is disabled or unconfigured.", "PROVIDER_UNAVAILABLE", 503
            )
        return self.provider

    def _admit(self, kind: str, prompt: str, output: int = 0) -> Provider:
        provider = self.require()
        count = estimated_tokens(prompt)
        if count > 12000:
            raise AppException("Provider input exceeds the limit.", "INPUT_LIMIT", 422)
        if self.calls.get(kind, 0) >= 1:
            raise AppException("Provider replay is prohibited.", "CALL_LIMIT", 503)
        self.calls[kind] = 1
        self.limits.reserve(self.grant.workspace_id, "input", count)
        if output:
            self.limits.reserve(self.grant.workspace_id, "output", output)
        return provider

    def prepare_embeddings(self, texts: list[str]) -> None:
        provider = self.require()
        if not self.upload or self._prepared_embedding_tokens is not None:
            raise ValueError("Upload embeddings must be admitted exactly once")
        if not texts or any(estimated_tokens(text) > 12000 for text in texts):
            raise AppException("Embedding input exceeds limit.", "INPUT_LIMIT", 422)
        estimate = sum(estimated_tokens(text) for text in texts)
        if isinstance(provider, EmbeddingBudget):
            provider.check_embedding_budget(estimate, math.ceil(len(texts) / 16))
        self.limits.reserve(self.grant.workspace_id, "embedding", estimate)
        self._prepared_embedding_tokens = estimate

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        provider = self.require()
        if not texts or len(texts) > 16:
            raise AppException("Invalid embedding batch.", "INPUT_LIMIT", 422)
        if not self.upload:
            if self.calls.get("embedding", 0):
                raise AppException("Embedding replay is prohibited.", "CALL_LIMIT", 503)
            self.calls["embedding"] = 1
        estimate = sum(estimated_tokens(text) for text in texts)
        if any(estimated_tokens(text) > 12000 for text in texts):
            raise AppException("Embedding input exceeds limit.", "INPUT_LIMIT", 422)
        if self.upload:
            if self._prepared_embedding_tokens is None:
                raise AppException("Upload admission required.", "EMBEDDING_ADMISSION", 503)
            if estimate > self._prepared_embedding_tokens:
                raise AppException("Embedding admission exceeded.", "INPUT_LIMIT", 422)
            self._prepared_embedding_tokens -= estimate
        else:
            self.limits.reserve(self.grant.workspace_id, "embedding", estimate)
        result = provider.embeddings(texts)
        self.check()
        if len(result) != len(texts) or any(
            len(v) != 1024 or not all(math.isfinite(x) for x in v) for v in result
        ):
            raise AppException("Invalid provider vectors.", "PROVIDER_RESPONSE_INVALID", 503)
        return result

    def reformulate(self, prompt: str) -> str:
        result = self._admit("reformulate", prompt, 512).reformulate(prompt)
        self.check()
        return result

    def generate(self, prompt: str) -> Iterator[str]:
        provider = self._admit("generate", prompt, 1024)
        size = 0
        stream = provider.generate(prompt)
        try:
            for token in stream:
                self.check()
                size += len(token.encode("utf-8"))
                if size > 65536:
                    raise AppException("Provider output exceeds limit.", "OUTPUT_LIMIT", 503)
                yield token
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
