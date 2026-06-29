"""``POST /classify`` — zero-shot classification of an uploaded chip set (PRD §6.5).

Accepts ``multipart/form-data`` with a ``file`` field plus ``model`` and ``query``
(and optional ``top_k`` / ``threshold`` / ``sensor``). Returns a GeoJSON
FeatureCollection of per-chip scores. STAC/JSON bodies are rejected (deferred).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ... import api
from ...core.session import ModelSession
from ...io.raster import read_raster_from_bytes

router = APIRouter()


@router.post("/classify")
async def classify_endpoint(request: Request) -> JSONResponse:
    """Score an uploaded raster against a text query.

    Args:
        request: ``multipart/form-data`` with ``file`` (GeoTIFF), ``model`` and
            ``query`` (required), and optional ``top_k`` / ``threshold`` /
            ``sensor``.

    Returns:
        A GeoJSON FeatureCollection as JSON.

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
    queries = [q for q in form.getlist("query") if q and str(q).strip()]
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=422, detail="Missing 'file' upload.")
    if not model:
        raise HTTPException(status_code=422, detail="Missing 'model' field.")
    if not queries:
        raise HTTPException(status_code=422, detail="Missing 'query' field.")

    top_k = _opt_int(form.get("top_k"), "top_k")
    threshold = _opt_float(form.get("threshold"), "threshold")
    sensor = form.get("sensor") or None

    data = await upload.read()
    raster = read_raster_from_bytes(data, label=getattr(upload, "filename", "upload"))

    session: ModelSession = request.app.state.session
    result = api.classify_raster(
        raster,
        model=str(model),
        session=session,
        query=[str(q) for q in queries],
        top_k=top_k if top_k is not None else 5,
        threshold=threshold if threshold is not None else 0.0,
        sensor=sensor,
    )
    return JSONResponse(content=result.feature_collection)


def _opt_int(value: object, field: str) -> int | None:
    """Parse an optional integer form field (422 on bad value)."""
    if value is None or value == "":
        return None
    try:
        return int(str(value))
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Field '{field}' must be an integer."
        ) from exc


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
