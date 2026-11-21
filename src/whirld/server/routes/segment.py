"""POST /segment: validated multipart input and bounded segmentation work."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import Response

from ... import api
from ...core.session import ModelSession
from ...io.output import mask_to_geotiff_bytes
from ...io.raster import read_raster_from_bytes
from ..inference import require_multipart, run_inference
from ..schemas import SegmentRequest

router = APIRouter()


@router.post("/segment", dependencies=[Depends(require_multipart)])
async def segment_endpoint(
    request: Request,
    form: Annotated[SegmentRequest, Form(media_type="multipart/form-data")],
) -> Response:
    """Segment a GeoTIFF with validated options and return a mask GeoTIFF."""
    session: ModelSession = request.app.state.session

    def compute(data: bytes) -> Response:
        """Decode, segment, and encode the response in the inference worker."""
        raster = read_raster_from_bytes(data, label=form.file.filename or "upload")
        result = api.segment_raster(
            raster,
            model=form.model,
            session=session,
            head=form.head,
            threshold=form.threshold,
            sensor=form.sensor,
        )
        tiff = mask_to_geotiff_bytes(result.mask, result.transform, result.crs)
        return Response(content=tiff, media_type="image/tiff")

    return await run_inference(request, form.file, compute)
