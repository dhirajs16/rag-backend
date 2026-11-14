"""Top-level API router."""

from fastapi import APIRouter

from rag_backend.api.v1.documents import router as documents_router
from rag_backend.api.v1.health import router as health_router
from rag_backend.api.v1.retrieval import router as retrieval_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(documents_router)
api_router.include_router(retrieval_router)
