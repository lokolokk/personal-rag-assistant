"""Exclusive, reversible sandbox for offline tests and local inspection runs.

This context changes process globals; concurrent evaluations must inject their own
services instead. It never resets or deletes the workstation's persisted stores.
"""

import importlib
import ipaddress
import os
import socket
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

from app.core.config import get_settings


class OfflineNetworkError(BaseException):
    """Fail fast even when a provider service catches Exception for fallback."""


def _blocked_http(*args: Any, **kwargs: Any) -> Any:
    _ = (args, kwargs)
    raise OfflineNetworkError("Network HTTP transport is disabled for offline evaluation")


async def _blocked_async_http(*args: Any, **kwargs: Any) -> Any:
    return _blocked_http(*args, **kwargs)


@contextmanager
def offline_network() -> Iterator[None]:
    """Block HTTP network transports and non-loopback sockets; allow in-process ASGI.

    Loopback sockets support Windows asyncio's internal wakeup mechanism. HTTP to
    localhost is still blocked, so tests cannot accidentally use a running backend.
    """
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto
    original_getaddrinfo = socket.getaddrinfo

    def check_address(sock: socket.socket, address: Any) -> None:
        if sock.family not in (socket.AF_INET, socket.AF_INET6):
            return
        try:
            is_loopback = ipaddress.ip_address(address[0]).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            raise OfflineNetworkError("External sockets are disabled for offline evaluation")

    def connect(sock: socket.socket, address: Any) -> None:
        check_address(sock, address)
        original_connect(sock, address)

    def connect_ex(sock: socket.socket, address: Any) -> int:
        check_address(sock, address)
        return original_connect_ex(sock, address)

    def sendto(sock: socket.socket, data: bytes, *args: Any) -> int:
        check_address(sock, args[-1])
        return original_sendto(sock, data, *args)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host is not None and host != "localhost":
            try:
                ipaddress.ip_address(host)
            except ValueError:
                raise OfflineNetworkError(
                    "External DNS is disabled for offline evaluation"
                ) from None
        return original_getaddrinfo(host, *args, **kwargs)

    with ExitStack() as stack:
        http_modules = [httpx]
        if importlib.util.find_spec("httpx2") is not None:
            http_modules.append(importlib.import_module("httpx2"))
        for module in http_modules:
            stack.enter_context(patch.object(module.HTTPTransport, "handle_request", _blocked_http))
            stack.enter_context(
                patch.object(module.AsyncHTTPTransport, "handle_async_request", _blocked_async_http)
            )
        stack.enter_context(patch.object(socket.socket, "connect", connect))
        stack.enter_context(patch.object(socket.socket, "connect_ex", connect_ex))
        stack.enter_context(patch.object(socket.socket, "sendto", sendto))
        stack.enter_context(patch.object(socket, "getaddrinfo", getaddrinfo))
        yield


@contextmanager
def isolated_services(root: Path) -> Iterator[None]:
    """Use temporary paths, deterministic embeddings and mock chat, restoring globals on exit."""
    root = root.resolve()
    overrides = {
        "BASE_DIR": str(root),
        "STORAGE_DIR": str(root / "storage"),
        "UPLOAD_DIR": str(root / "storage/uploads"),
        "CHROMA_PERSIST_DIR": str(root / "storage/chroma"),
        "OPENAI_API_KEY": "",
        "OPENAI_API_BASE": "",
        "ANONYMIZED_TELEMETRY": "False",
    }
    singletons = {
        "retrieval.vector_store": ["_vector_store_instance"],
        "retrieval.bm25_store": ["_bm25_store_instance"],
        "retrieval.hybrid": ["_hybrid_retriever_instance"],
        "retrieval.memory": ["_GLOBAL_MEMORY_MANAGER"],
        "retrieval.conversation_registry": ["_GLOBAL_CONVERSATION_REGISTRY"],
        "retrieval.reformulator": ["_GLOBAL_REFORMULATOR"],
        "ingestion.registry": ["_document_registry_instance"],
        "generation.grounding": ["_GLOBAL_GRADER", "_GLOBAL_GENERATOR"],
        "generation.llm_client": ["_GLOBAL_LLM_CLIENT"],
        "generation.graph": [
            "_COMPILED_GRAPH",
            "_COMPILED_STREAMLINED_GRAPH",
            "_COMPILED_LEGACY_GRAPH",
        ],
    }
    with ExitStack() as stack:
        stack.enter_context(offline_network())
        stack.enter_context(patch.dict(os.environ, overrides))
        get_settings.cache_clear()
        try:
            for suffix, attributes in singletons.items():
                module = importlib.import_module(f"app.services.{suffix}")
                for attribute in attributes:
                    stack.enter_context(patch.object(module, attribute, None))

            from app.services.generation.grounding import (
                FALLBACK_RESPONSE,
                GroundedGenerator,
                RetrievalGrader,
                set_grounded_generator,
                set_retrieval_grader,
            )
            from app.services.retrieval.reformulator import (
                QueryReformulator,
                set_query_reformulator,
            )

            set_query_reformulator(
                QueryReformulator(
                    llm_client=lambda p: p.split("User Follow-up:\n")[-1].split(
                        "\n\nStandalone Search Query:"
                    )[0]
                )
            )
            set_retrieval_grader(RetrievalGrader(api_key=None))
            set_grounded_generator(
                GroundedGenerator(api_key=None, generating_fn=lambda _p, _c: FALLBACK_RESPONSE)
            )
            yield
        finally:
            get_settings.cache_clear()
