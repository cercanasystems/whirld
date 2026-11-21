"""Bound blocking raster, inference, and serialization work off the event loop."""

from __future__ import annotations

from collections.abc import Callable

from anyio import to_thread
from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import Response


async def require_multipart(request: Request) -> None:
    """Reject deferred STAC JSON and unsupported content types predictably."""
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        raise HTTPException(400, "STAC URL input is not available; upload a GeoTIFF.")
    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(422, "Expected multipart/form-data with a 'file' field.")


async def run_inference(
    request: Request, upload: UploadFile, operation: Callable[[bytes], Response]
) -> Response:
    """Serialize model work per application while leaving HTTP handling responsive.

    The semaphore covers upload materialization, raster decoding, model access,
    inference, and response serialization. A single active operation avoids races
    in shared backend state; waiting requests retain their spooled upload rather
    than allocating additional raster arrays. Cancellation waits for worker exit
    before releasing the slot.
    """
    async with request.app.state.inference_slots:
        data = await upload.read()
        return await to_thread.run_sync(operation, data)
