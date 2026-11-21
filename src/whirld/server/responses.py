"""Serialize inference results without storing scene-sized data in HTTP headers."""

from __future__ import annotations

import io
import json
from typing import Literal
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from fastapi.responses import JSONResponse, Response

from ..api import ClassifyResult, EmbedResult, SegmentResult
from ..io.output import mask_to_geotiff_bytes


def embedding_response(
    result: EmbedResult, format: Literal["npz", "npy", "json"]
) -> Response:
    """Encode embeddings and metadata as the selected documented response format."""
    if format == "json":
        payload = dict(result.meta)
        payload["embeddings"] = result.embeddings.astype(np.float32).tolist()
        return JSONResponse(content=payload)
    buffer = io.BytesIO()
    np.save(buffer, result.embeddings.astype(np.float32), allow_pickle=False)
    if format == "npy":
        return Response(
            content=buffer.getvalue(), media_type="application/octet-stream"
        )
    archive = io.BytesIO()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        bundle.writestr("embeddings.npy", buffer.getvalue())
        bundle.writestr("metadata.json", json.dumps(result.meta))
    return Response(
        content=archive.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="embeddings.npz"'},
    )


def classification_response(result: ClassifyResult) -> Response:
    """Encode classification geometry and scores in a GeoJSON response body."""
    return JSONResponse(content=result.feature_collection)


def segmentation_response(result: SegmentResult) -> Response:
    """Encode a segmentation mask with spatial metadata and its nodata sentinel."""
    return Response(
        content=mask_to_geotiff_bytes(result.mask, result.transform, result.crs),
        media_type="image/tiff",
    )
