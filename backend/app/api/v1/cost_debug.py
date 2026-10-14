"""Local opt-in GET interface. This mode has no ingestion/chat routes or lifespan."""

import os
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response

from app.services.evaluation.cost_station import RUN_IDS, RunView, read_run

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def local_evidence_access(request: Request) -> None:
    if os.environ.get("RAG_COST_DEBUG") != "1":
        raise HTTPException(404, "Cost debugging is disabled")
    origin = request.headers.get("origin")
    try:
        host = urlsplit("http://" + request.headers.get("host", "")).hostname
        origin_url = urlsplit(origin) if origin else None
    except ValueError:
        raise HTTPException(403, "Local evidence access required") from None
    if (
        request.client is None
        or request.client.host not in {"127.0.0.1", "::1"}
        or host not in LOCAL_HOSTS
        or any(
            h.lower() == "forwarded" or h.lower().startswith("x-forwarded-")
            for h in request.headers
        )
        or (
            origin_url is not None
            and (origin_url.scheme != "http" or origin_url.hostname not in LOCAL_HOSTS)
        )
        or request.headers.get("sec-fetch-site") == "cross-site"
    ):
        raise HTTPException(403, "Local evidence access required")
    if request.query_params:
        raise HTTPException(422, "This endpoint accepts no query parameters or filesystem paths")


router = APIRouter(dependencies=[Depends(local_evidence_access)])


@router.get("/api/v1/cost-debug/{run_id}", response_model=RunView)
def get_run(run_id: str, response: Response) -> RunView:
    if run_id not in RUN_IDS:
        raise HTTPException(404, "Run is not approved for this station")
    response.headers["Cache-Control"] = "no-store"
    try:
        return read_run(run_id)
    except (ValueError, OSError):
        raise HTTPException(409, "Evidence unavailable or binding validation failed") from None


def create_debug_application() -> FastAPI:
    """Provider/settings-free factory used by the existing entrypoint's explicit opt-in."""
    application = FastAPI(
        title="Read-only RAG Cost Debugging", docs_url=None, redoc_url=None, openapi_url=None
    )
    application.include_router(router)
    return application
