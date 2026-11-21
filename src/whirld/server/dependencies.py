"""FastAPI composition boundary for the application-scoped inference service."""

from __future__ import annotations

from typing import cast

from fastapi import HTTPException, Request

from .inference import InferenceService


async def get_inference_service(request: Request) -> InferenceService:
    """Resolve the application's service for explicit FastAPI dependency injection."""
    return cast(InferenceService, request.app.state.inference_service)


async def require_multipart(request: Request) -> None:
    """Reject deferred STAC JSON and unsupported content types predictably."""
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        raise HTTPException(400, "STAC URL input is not available; upload a GeoTIFF.")
    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(422, "Expected multipart/form-data with a 'file' field.")
