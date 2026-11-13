"""Application health endpoint."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["Health"])


class HealthResponse(BaseModel):
    """Response returned when the application is healthy."""

    status: Literal["ok"]


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Report whether the HTTP application is running."""
    return HealthResponse(status="ok")

