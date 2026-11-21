"""POST /embed: a NumPy archive with complete metadata, raw npy, or JSON."""

from __future__ import annotations

import io
import json
from typing import Annotated
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import JSONResponse, Response

from ... import api
from ...core.session import ModelSession
from ...io.raster import read_raster_from_bytes
from ..inference import require_multipart, run_inference
from ..schemas import EmbedRequest

router = APIRouter()


@router.post("/embed", dependencies=[Depends(require_multipart)])
async def embed_endpoint(
    request: Request,
    form: Annotated[EmbedRequest, Form(media_type="multipart/form-data")],
) -> Response:
    """Return an npz bundle by default; explicit npy omits chip metadata.

    The default archive contains embeddings.npy and metadata.json. JSON includes
    embeddings inline. Large chip metadata never travels in response headers.
    Raster decoding, inference, and serialization run in a bounded worker.
    """
    session: ModelSession = request.app.state.session

    def compute(data: bytes) -> Response:
        """Build the complete response while holding the inference slot."""
        raster = read_raster_from_bytes(data, label=form.file.filename or "upload")
        result = api.embed_raster(
            raster,
            model=form.model,
            session=session,
            fmt="npy",
            chip_size=form.chip_size,
            overlap=form.overlap,
            sensor=form.sensor,
            batch_size=form.batch_size,
        )
        if form.format == "json":
            payload = dict(result.meta)
            payload["embeddings"] = result.embeddings.astype(np.float32).tolist()
            return JSONResponse(content=payload)
        buffer = io.BytesIO()
        np.save(buffer, result.embeddings.astype(np.float32), allow_pickle=False)
        if form.format == "npy":
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

    return await run_inference(request, form.file, compute)
