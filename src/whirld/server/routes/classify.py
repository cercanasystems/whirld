"""POST /classify: validated multipart input and bounded classification work."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import JSONResponse, Response

from ... import api
from ...core.session import ModelSession
from ...io.raster import read_raster_from_bytes
from ..inference import require_multipart, run_inference
from ..schemas import ClassifyRequest

router = APIRouter()


@router.post("/classify", dependencies=[Depends(require_multipart)])
async def classify_endpoint(
    request: Request,
    form: Annotated[ClassifyRequest, Form(media_type="multipart/form-data")],
) -> Response:
    """Score an uploaded raster and return GeoJSON with validated query options."""
    session: ModelSession = request.app.state.session

    def compute(data: bytes) -> Response:
        """Decode and classify the upload in the bounded inference worker."""
        raster = read_raster_from_bytes(data, label=form.file.filename or "upload")
        result = api.classify_raster(
            raster,
            model=form.model,
            session=session,
            query=form.query,
            top_k=form.top_k,
            threshold=form.threshold,
            sensor=form.sensor,
        )
        return JSONResponse(content=result.feature_collection)

    return await run_inference(request, form.file, compute)
