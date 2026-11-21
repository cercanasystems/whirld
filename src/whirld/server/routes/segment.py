"""Typed HTTP adapter for POST /segment."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form
from fastapi.responses import Response

from ..dependencies import get_inference_service, require_multipart
from ..inference import InferenceService
from ..schemas import SegmentRequest

router = APIRouter()


@router.post("/segment", dependencies=[Depends(require_multipart)])
async def segment_endpoint(
    form: Annotated[SegmentRequest, Form(media_type="multipart/form-data")],
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> Response:
    """Validate the multipart request and delegate to the injected service."""
    return await service.segment(form)
