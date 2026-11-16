"""Invitation-only application. Authentication precedes request-body processing."""

import asyncio
import logging
import queue
import threading
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import ExitStack, asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.v1.chat import _format_sse
from app.core.exceptions import AppException
from app.schemas.access import (
    LiveChatRequest,
    LiveConversationCreate,
    LiveConversationUpdate,
    RedemptionRequest,
    RedemptionResponse,
    SessionResponse,
)
from app.schemas.chat import (
    ChatMessage,
    ChatMessageRequest,
    ConversationCreateRequest,
    ConversationCreateResponse,
    ConversationDeleteResponse,
    ConversationSummaryItem,
)
from app.schemas.document import DocumentDeleteResponse, DocumentItem, FullPassage
from app.services.access import AccessDenied, AccessStore, SessionGrant, digest
from app.services.generation.graph import compile_workspace_graph
from app.services.ingestion.workspace_ingestion import delete_document, ingest
from app.services.limits import PUBLIC_LIMITS, UsageLimits
from app.services.live_provider import BoundProvider, Provider
from app.services.workspaces import Workspace, WorkspaceManager

logger = logging.getLogger(__name__)


def error_response(error: AppException) -> JSONResponse:
    return JSONResponse(
        {
            "error": {"code": error.code, "message": error.message, "details": None},
            "detail": error.message,
        },
        status_code=error.status_code,
        headers={
            "Cache-Control": "no-store",
            **({"Retry-After": "60"} if error.status_code == 429 else {}),
        },
    )


class LiveBoundary:
    def __init__(
        self, app: ASGIApp, access: AccessStore, limits: UsageLimits, blocked: Callable[[], bool]
    ) -> None:
        self.app, self.access, self.limits = app, access, limits
        self.blocked = blocked

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        try:
            if path.startswith("/api/") and self.blocked():
                raise AppException(
                    "Live storage initialization failed.", "STORAGE_UNAVAILABLE", 503
                )
            if scope.get("query_string"):
                raise AppException("Query parameters are not accepted.", "VALIDATION_ERROR", 422)
            if path.startswith("/api/") and path != "/api/v1/access/redeem":
                headers = dict(scope.get("headers", []))
                raw = headers.get(b"authorization", b"").decode("latin-1")
                if not raw.startswith("Bearer "):
                    raise AccessDenied()
                grant = self.access.validate(raw[7:])
                scope.setdefault("state", {})["grant"] = grant
            elif path == "/api/v1/access/redeem":
                # Socket address is trustworthy; arbitrary forwarded headers are not.
                peer = scope.get("client")
                self.limits.reserve(
                    "ip:" + digest(str(peer[0] if peer else "unknown")), "redeem", 1
                )
        except AppException as error:
            await error_response(error)(scope, receive, send)
            return
        maximum = 11 * 1024 * 1024 if path == "/api/v1/documents/upload" else 65536
        count = 0

        async def bounded_receive() -> Message:
            nonlocal count
            message = await receive()
            count += len(message.get("body", b""))
            if count > maximum:
                raise AppException("Request body exceeds limit.", "FILE_TOO_LARGE", 413)
            return message

        async def protected_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [(b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff")]
                )
                message["headers"] = headers
            await send(message)

        await self.app(scope, bounded_receive, protected_send)


def create_live_application(
    root: Path, provider: Provider | None = None, origins: list[str] | None = None
) -> FastAPI:
    for origin in origins or []:
        parsed = urlsplit(origin)
        local_http = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}
        if (
            (parsed.scheme != "https" and not local_http)
            or not parsed.netloc
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username
        ):
            raise ValueError(
                "Live CORS requires exact HTTPS origins (localhost allowed for testing)"
            )
    access = AccessStore(root / "control.sqlite3")
    limits = UsageLimits(access)
    manager = WorkspaceManager(root, access, provider)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.ready = False
        try:
            for workspace in access.workspace_ids():
                manager.get(workspace)
            application.state.ready = True
        except Exception:
            application.state.initialization_failed = True
            logger.exception("Live workspace initialization failed")
        yield
        application.state.ready = False

    app = FastAPI(
        title="Invited Personal RAG",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.ready = False
    app.state.initialization_failed = False
    app.state.manager = manager
    app.state.access = access
    app.add_middleware(
        LiveBoundary,
        access=access,
        limits=limits,
        blocked=lambda: bool(app.state.initialization_failed),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or [],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "ngrok-skip-browser-warning"],
        expose_headers=["Retry-After"],
        max_age=600,
    )

    @app.exception_handler(AppException)
    async def domain_error(_: Request, error: AppException) -> JSONResponse:
        return error_response(error)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, _error: RequestValidationError) -> JSONResponse:
        return error_response(AppException("Invalid request.", "VALIDATION_ERROR", 422))

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, error: Exception) -> JSONResponse:
        logger.error("Live operation failed: %s", type(error).__name__)
        return error_response(AppException("Operation unavailable.", "STORAGE_UNAVAILABLE", 503))

    def grant(request: Request) -> SessionGrant:
        value: SessionGrant = request.state.grant
        return access.validate(value.session_token)

    def workspace(request: Request) -> Workspace:
        return manager.get(grant(request).workspace_id)

    def session_payload(value: SessionGrant) -> dict[str, Any]:
        return {
            "workspaceId": value.workspace_id,
            "expiresAt": datetime.fromtimestamp(value.expires_at, UTC).isoformat(),
            "limits": PUBLIC_LIMITS,
        }

    def conversation(bundle: Workspace, identifier: str) -> ConversationSummaryItem:
        item = bundle.conversations.get(identifier)
        if item is None:
            raise AppException("Conversation does not exist.", "CONVERSATION_NOT_FOUND", 404)
        return item

    def validate_aliases(payload: ConversationCreateRequest | ChatMessageRequest) -> None:
        if (
            payload.documentIds is not None
            and "document_ids" in payload.model_fields_set
            and payload.document_ids != payload.documentIds
        ):
            raise AppException("Conflicting document scopes.", "VALIDATION_ERROR", 422)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(response: Response) -> dict[str, str]:
        response.status_code = 200 if app.state.ready else 503
        return {"status": "ready" if app.state.ready else "not_ready"}

    @app.post("/api/v1/access/redeem", response_model=RedemptionResponse)
    async def redeem(payload: RedemptionRequest) -> dict[str, Any]:
        value = access.redeem(payload.invitation)
        manager.get(value.workspace_id)
        return {**session_payload(value), "sessionToken": value.session_token}

    @app.get("/api/v1/access/session", response_model=SessionResponse)
    async def session(request: Request) -> dict[str, Any]:
        return session_payload(grant(request))

    @app.delete("/api/v1/access/session", status_code=204)
    async def end_session(request: Request) -> Response:
        access.end_session(grant(request).session_token)
        return Response(status_code=204)

    @app.get("/api/v1/documents", response_model=list[DocumentItem])
    async def documents(request: Request) -> list[DocumentItem]:
        bundle = workspace(request)
        with bundle.lock:
            return bundle.documents.get_all()

    @app.get("/api/v1/documents/{identifier}", response_model=DocumentItem)
    async def document(identifier: str, request: Request) -> DocumentItem:
        bundle = workspace(request)
        with bundle.lock:
            bundle.validate_documents([identifier])
            item = bundle.documents.get(identifier)
            if item is None:
                raise AppException("Document does not exist.", "DOCUMENT_NOT_FOUND", 404)
            return item

    @app.get("/api/v1/chunks/{identifier}", response_model=FullPassage)
    async def full_passage(identifier: str, request: Request) -> FullPassage:
        bundle = workspace(request)
        with bundle.lock:
            return bundle.full_passage(identifier)

    @app.post("/api/v1/documents/upload", response_model=DocumentItem, status_code=201)
    async def upload(request: Request, file: UploadFile) -> DocumentItem:
        value = grant(request)
        bundle = manager.get(value.workspace_id)
        bound = BoundProvider(provider, access, limits, value, upload=True)
        bound.require()
        content = await file.read(10 * 1024 * 1024 + 1)
        if len(content) > 10 * 1024 * 1024:
            raise AppException("PDF exceeds 10 MiB.", "FILE_TOO_LARGE", 413)
        stack = ExitStack()
        stack.enter_context(limits.operation(value.workspace_id, upload=True))

        def work() -> DocumentItem:
            try:
                return ingest(bundle, manager, bound, content, file.filename or "document.pdf")
            finally:
                stack.close()

        task = asyncio.create_task(asyncio.to_thread(work))
        return await asyncio.shield(task)

    @app.delete("/api/v1/documents/{identifier}", response_model=DocumentDeleteResponse)
    async def remove_document(identifier: str, request: Request) -> DocumentDeleteResponse:
        value = grant(request)
        bundle = workspace(request)
        with limits.operation(value.workspace_id):
            delete_document(bundle, identifier)
        return DocumentDeleteResponse(id=identifier)

    @app.get("/api/v1/conversations", response_model=list[ConversationSummaryItem])
    async def conversations(request: Request) -> list[ConversationSummaryItem]:
        bundle = workspace(request)
        with bundle.lock:
            return bundle.conversations.get_all()

    @app.post("/api/v1/conversations", response_model=ConversationCreateResponse, status_code=201)
    async def create_conversation(
        request: Request, payload: LiveConversationCreate | None = None
    ) -> ConversationCreateResponse:
        value = grant(request)
        bundle = workspace(request)
        payload = payload or LiveConversationCreate()
        validate_aliases(payload)
        with limits.operation(value.workspace_id), bundle.lock:
            ids = bundle.validate_documents(payload.get_document_ids())
            if len(bundle.conversations.get_all()) >= 50:
                raise AppException("Conversation limit reached.", "QUOTA_EXCEEDED", 429)
            identifier = "conv_" + uuid.uuid4().hex
            if payload.title and len(payload.title) > 200:
                raise AppException("Title exceeds limit.", "VALIDATION_ERROR", 422)
            memory = bundle.memory.get_or_create_session(identifier, ids)
            bundle.conversations.add_or_update(
                ConversationSummaryItem(
                    id=identifier, title=payload.title or "New Chat", document_ids=ids
                )
            )
            return ConversationCreateResponse(
                conversation_id=identifier,
                conversationId=identifier,
                created_at=memory.created_at,
                createdAt=memory.created_at.isoformat(),
            )

    @app.get("/api/v1/conversations/{identifier}/messages", response_model=list[ChatMessage])
    async def history(identifier: str, request: Request) -> list[ChatMessage]:
        bundle = workspace(request)
        with bundle.lock:
            return bundle.citation_history(conversation(bundle, identifier).messages)

    @app.patch("/api/v1/conversations/{identifier}", response_model=ConversationSummaryItem)
    async def rename(
        identifier: str, request: Request, payload: LiveConversationUpdate
    ) -> ConversationSummaryItem:
        value = grant(request)
        bundle = workspace(request)
        with limits.operation(value.workspace_id), bundle.lock:
            conversation(bundle, identifier)
            if len(payload.title) > 200:
                raise AppException("Title exceeds limit.", "VALIDATION_ERROR", 422)
            result = bundle.conversations.update_title(identifier, payload.title)
            if result is None:
                raise AppException("Conversation does not exist.", "CONVERSATION_NOT_FOUND", 404)
            return result

    @app.delete("/api/v1/conversations/{identifier}", response_model=ConversationDeleteResponse)
    async def remove_conversation(identifier: str, request: Request) -> ConversationDeleteResponse:
        value = grant(request)
        bundle = workspace(request)
        with limits.operation(value.workspace_id), bundle.lock:
            conversation(bundle, identifier)
            bundle.conversations.delete(identifier)
            bundle.memory.delete_session(identifier)
        return ConversationDeleteResponse(id=identifier)

    @app.post("/api/v1/conversations/{identifier}/messages")
    async def chat(
        identifier: str, request: Request, payload: LiveChatRequest
    ) -> StreamingResponse:
        value = grant(request)
        bundle = workspace(request)
        validate_aliases(payload)
        bound = BoundProvider(provider, access, limits, value)
        with bundle.lock:
            item = conversation(bundle, identifier)
            ids = (
                payload.get_document_ids()
                if ({"documentIds", "document_ids"} & payload.model_fields_set)
                else item.document_ids
            )
            bundle.validate_documents(ids)
            if len(payload.get_query()) > 8000 or len(item.messages) >= 200:
                raise AppException("Chat limit reached.", "INPUT_LIMIT", 422)
            bound.require()
        stack = ExitStack()
        stack.enter_context(limits.operation(value.workspace_id))
        events: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=256)
        cancelled = threading.Event()
        bound.cancelled = cancelled

        def emit(event: str, data: Any) -> None:
            while not cancelled.is_set():
                try:
                    events.put((event, data), timeout=0.1)
                    return
                except queue.Full:
                    bound.check()

        def work() -> None:
            try:
                with bundle.lock:
                    conversation(bundle, identifier)
                    titles = ", ".join(
                        doc.filename
                        for doc in bundle.documents.get_all()
                        if not ids or doc.id in ids
                    )
                    bundle.memory.set_active_documents(identifier, ids)
                    state = bundle.memory.build_initial_state(
                        identifier, payload.get_query(), titles
                    )
                    bundle.conversations.record_turn(
                        identifier, "user", payload.get_query(), document_ids=ids
                    )
                    bundle.memory.add_message(identifier, "user", payload.get_query())
                emit("status", {"status": "reformulating" if state["messages"] else "searching"})
                graph = compile_workspace_graph(bundle, bound)
                reply: list[str] = []
                citations: list[dict[str, Any]] = []
                for mode, event in graph.stream(state, stream_mode=["updates", "custom"]):
                    bound.check()
                    if mode == "custom" and isinstance(event, dict) and event.get("token"):
                        token = str(event["token"])
                        reply.append(token)
                        emit("token", {"token": token})
                    elif mode == "updates" and isinstance(event, dict):
                        for node, update in event.items():
                            if node == "reformulate_query":
                                emit("status", {"status": "searching"})
                            if node == "scoped_hybrid_retriever":
                                emit("status", {"status": "streaming"})
                            if node in {"generate_grounded_response", "inability_fallback"}:
                                citations = update.get("citations", [])
                bound.check()
                with bundle.lock:
                    bundle.memory.add_message(identifier, "assistant", "".join(reply), citations)
                    bundle.conversations.record_turn(
                        identifier, "assistant", "".join(reply), citations, auto_title=False
                    )
                emit("citations", citations)
                emit("done", {"finishReason": "stop"})
            except Exception as error:
                code = error.code if isinstance(error, AppException) else "STREAM_ERROR"
                emit(
                    "error",
                    {"error": {"code": code, "message": "Chat interrupted or unavailable."}},
                )
                emit("done", {"finishReason": "error"})
            finally:
                stack.close()

        async def stream() -> AsyncIterator[str]:
            task = asyncio.create_task(asyncio.to_thread(work))
            elapsed = 0.0
            try:
                while not task.done() or not events.empty():
                    if await request.is_disconnected():
                        return
                    bound.check()
                    try:
                        event, data = events.get_nowait()
                        yield _format_sse(event, data)
                    except queue.Empty:
                        await asyncio.sleep(0.05)
                        elapsed += 0.05
                        if elapsed >= 15:
                            elapsed = 0
                            yield ": heartbeat\n\n"
            except AppException as error:
                yield _format_sse(
                    "error", {"error": {"code": error.code, "message": error.message}}
                )
                yield _format_sse("done", {"finishReason": "error"})
            finally:
                cancelled.set()
                # The worker retains its operation slot until the provider/thread exits.
                task.add_done_callback(
                    lambda result: result.exception() if not result.cancelled() else None
                )

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    return app
