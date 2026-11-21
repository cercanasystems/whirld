"""Typed HTTP adapter for GET /models."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from ..dependencies import get_inference_service
from ..inference import InferenceService
from ..schemas import ModelsResponse

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
async def list_models(
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> ModelsResponse:
    """Return status from the explicitly injected application service."""
    return await service.models()
