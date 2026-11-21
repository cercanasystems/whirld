"""Typed HTTP adapter for POST /embed."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form
from fastapi.responses import Response

from ..dependencies import get_inference_service, require_multipart
from ..inference import InferenceService
from ..schemas import EmbedRequest

router = APIRouter()


@router.post("/embed", dependencies=[Depends(require_multipart)])
async def embed_endpoint(
    form: Annotated[EmbedRequest, Form(media_type="multipart/form-data")],
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> Response:
    """Validate the multipart request and delegate to the injected service."""
    return await service.embed(form)
