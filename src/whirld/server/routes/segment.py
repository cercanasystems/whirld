"""``POST /segment`` — per-pixel segmentation of an uploaded scene (PRD §6.4).

Accepts ``multipart/form-data`` with a ``file`` field plus ``model`` (and optional
``head`` / ``threshold`` / ``sensor``). Returns the mask as a single-band GeoTIFF
(``image/tiff``). STAC/JSON bodies are rejected (deferred).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from ... import api
from ...core.session import ModelSession
from ...io.output import mask_to_geotiff_bytes
from ...io.raster import read_raster_from_bytes

router = APIRouter()


@router.post("/segment")
async def segment_endpoint(request: Request) -> Response:
    """Segment an uploaded raster and return a mask GeoTIFF.

    Args:
        request: ``multipart/form-data`` with ``file`` (GeoTIFF), ``model``
            (required), and optional ``head`` / ``threshold`` / ``sensor``.

    Returns:
        ``image/tiff`` — a single-band mask GeoTIFF.

    Raises:
        HTTPException: STAC/JSON input (400) or missing fields / wrong type (422).
        WhirldError: Pipeline errors, mapped to HTTP status by the app handler.
    """
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        raise HTTPException(
            status_code=400,
            detail=(
                "STAC URL input is not available in this build. "
                "Upload a GeoTIFF via multipart/form-data with a 'file' field."
            ),
        )
    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(
            status_code=422,
            detail="Expected multipart/form-data with a 'file' field.",
        )

    form = await request.form()
    upload = form.get("file")
    model = form.get("model")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=422, detail="Missing 'file' upload.")
    if not model:
        raise HTTPException(status_code=422, detail="Missing 'model' field.")

    head = form.get("head") or None
    sensor = form.get("sensor") or None
    threshold = _opt_float(form.get("threshold"), "threshold")

    data = await upload.read()
    raster = read_raster_from_bytes(data, label=getattr(upload, "filename", "upload"))

    session: ModelSession = request.app.state.session
    result = api.segment_raster(
        raster,
        model=str(model),
        session=session,
        head=str(head) if head else None,
        threshold=threshold if threshold is not None else 0.5,
        sensor=sensor,
    )
    tiff = mask_to_geotiff_bytes(result.mask, result.transform, result.crs)
    return Response(content=tiff, media_type="image/tiff")


def _opt_float(value: object, field: str) -> float | None:
    """Parse an optional float form field (422 on bad value)."""
    if value is None or value == "":
        return None
    try:
        return float(str(value))
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Field '{field}' must be a number."
        ) from exc
