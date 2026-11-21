"""Typed HTTP adapter for GET /health."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from ..dependencies import get_inference_service
from ..inference import InferenceService
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> HealthResponse:
    """Return status from the explicitly injected application service."""
    return service.health()
