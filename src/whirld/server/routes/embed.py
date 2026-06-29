"""``POST /embed`` — embed an uploaded GeoTIFF (PRD section 6.3).

This build accepts ``multipart/form-data`` with a ``file`` field (the server
cannot read a client's local path, and STAC URL input is deferred). The default
response is the raw ``.npy`` bytes as ``application/octet-stream`` with chip
metadata in the ``X-Whirld-Chips-Meta`` header (base64-encoded JSON);
``format=json`` returns a JSON body with the embeddings inlined.
"""

from __future__ import annotations

import base64
import io
import json

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from ... import api
from ...core.session import ModelSession
from ...io.raster import read_raster_from_bytes

router = APIRouter()


@router.post("/embed")
async def embed_endpoint(request: Request) -> Response:
    """Embed an uploaded raster and return embeddings.

    Args:
        request: The incoming request. Must be ``multipart/form-data`` carrying a
            ``file`` (GeoTIFF) plus form fields: ``model`` (required),
            ``chip_size``, ``overlap``, ``sensor``, ``format`` (``npy``/``json``).

    Returns:
        ``application/octet-stream`` ``.npy`` bytes (default) with an
        ``X-Whirld-Chips-Meta`` header, or a JSON body when ``format=json``.

    Raises:
        HTTPException: STAC/JSON input (400, deferred), wrong content type or
            missing fields (422).
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

    chip_size = _opt_int(form.get("chip_size"), "chip_size")
    overlap = _opt_int(form.get("overlap"), "overlap") or 0
    sensor = form.get("sensor") or None
    fmt = form.get("format") or "npy"

    data = await upload.read()
    raster = read_raster_from_bytes(data, label=getattr(upload, "filename", "upload"))

    session: ModelSession = request.app.state.session
    result = api.embed_raster(
        raster,
        model=str(model),
        session=session,
        fmt=str(fmt),
        chip_size=chip_size,
        overlap=overlap,
        sensor=sensor,
    )

    if fmt == "json":
        payload = dict(result.meta)
        payload["embeddings"] = result.embeddings.astype(np.float32).tolist()
        return JSONResponse(content=payload)

    buffer = io.BytesIO()
    np.save(buffer, result.embeddings.astype(np.float32))
    meta_b64 = base64.b64encode(json.dumps(result.meta).encode("utf-8")).decode("ascii")
    return Response(
        content=buffer.getvalue(),
        media_type="application/octet-stream",
        headers={"X-Whirld-Chips-Meta": meta_b64},
    )


def _opt_int(value: object, field: str) -> int | None:
    """Parse an optional integer form field.

    Args:
        value: The raw form value (str or ``None``).
        field: Field name for error messages.

    Returns:
        The parsed integer, or ``None`` if not provided.

    Raises:
        HTTPException: The value is present but not a valid integer (422).
    """
    if value is None or value == "":
        return None
    try:
        return int(str(value))
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Field '{field}' must be an integer."
        ) from exc
