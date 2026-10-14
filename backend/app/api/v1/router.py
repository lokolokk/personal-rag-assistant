"""Aggregated API v1 router mounting all modular resource routers."""

from fastapi import APIRouter

from app.api.v1 import chat, documents

api_router = APIRouter()

api_router.include_router(
    documents.router,
    prefix="/documents",
    tags=["documents"],
)

api_router.include_router(
    chat.router,
    prefix="/conversations",
    tags=["conversations"],
)
