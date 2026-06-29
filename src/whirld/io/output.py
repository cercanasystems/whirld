"""Serialization of inference results.

Embeddings are written as a NumPy ``.npy`` array with a JSON metadata sidecar
(PRD section 10.1); classification results as a GeoJSON FeatureCollection (PRD
section 10.3); segmentation masks as a single-band GeoTIFF (PRD section 10.2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .._version import __version__
from ..core.chips import ChipSet
from ..logging_setup import get_logger

_log = get_logger("io.output")


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 ``Z``-suffixed string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_meta(
    *,
    model: str,
    model_version: str,
    crs: str,
    embed_dim: int,
    chip_size_px: int,
    resolution_m: float,
    chipset: ChipSet,
) -> dict[str, Any]:
    """Build the embeddings metadata sidecar dictionary (PRD section 10.1).

    Args:
        model: Model identifier.
        model_version: Model version.
        crs: Source CRS string.
        embed_dim: Embedding dimensionality.
        chip_size_px: Chip edge length in pixels.
        resolution_m: Target resolution in meters.
        chipset: Per-chip geometry.

    Returns:
        A JSON-serializable metadata dictionary.
    """
    return {
        "model": model,
        "model_version": model_version,
        "whirld_version": __version__,
        "timestamp": _utc_now_iso(),
        "crs": crs,
        "embed_dim": embed_dim,
        "chip_size_px": chip_size_px,
        "resolution_m": resolution_m,
        "chips": [chip.model_dump() for chip in chipset.chips],
    }


def write_embeddings(
    embeddings: np.ndarray,
    meta: dict[str, Any],
    output_path: str | Path,
    *,
    fmt: str = "npy",
) -> tuple[Path, Path]:
    """Write embeddings and their metadata sidecar.

    Args:
        embeddings: Array of shape ``(n_chips, embed_dim)``.
        meta: Metadata dictionary from :func:`build_meta`.
        output_path: Destination path for the embeddings file.
        fmt: ``npy`` (binary array + sidecar) or ``json`` (array inline).

    Returns:
        A tuple ``(data_path, meta_path)`` of the files written. For ``json``
        format both elements are the same path (metadata is inlined).

    Raises:
        ValueError: ``fmt`` is not ``npy`` or ``json``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "npy":
        np.save(output_path, embeddings.astype(np.float32))
        # np.save appends .npy if missing; normalize for the caller.
        if output_path.suffix != ".npy":
            output_path = output_path.with_suffix(output_path.suffix + ".npy")
        meta_path = _meta_path_for(output_path)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _log.info("Wrote %s and %s", output_path.name, meta_path.name)
        return output_path, meta_path

    if fmt == "json":
        payload = dict(meta)
        payload["embeddings"] = embeddings.astype(np.float32).tolist()
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _log.info("Wrote %s", output_path.name)
        return output_path, output_path

    raise ValueError(f"Unsupported output format '{fmt}'. Use 'npy' or 'json'.")


def _meta_path_for(data_path: Path) -> Path:
    """Return the metadata sidecar path for an embeddings file.

    ``foo_embeddings.npy`` -> ``foo_embeddings_meta.json``.

    Args:
        data_path: The embeddings ``.npy`` path.

    Returns:
        The sidecar JSON path.
    """
    stem = data_path.stem
    return data_path.with_name(f"{stem}_meta.json")


def _bbox_polygon(bbox: list[float]) -> list[list[list[float]]]:
    """Return a closed GeoJSON Polygon ring from a ``[minx, miny, maxx, maxy]`` bbox.

    Args:
        bbox: Bounding box ``[minx, miny, maxx, maxy]``.

    Returns:
        Polygon coordinates (a single closed ring).
    """
    minx, miny, maxx, maxy = bbox
    return [
        [
            [minx, miny],
            [maxx, miny],
            [maxx, maxy],
            [minx, maxy],
            [minx, miny],
        ]
    ]


def build_feature_collection(
    scores: np.ndarray,
    chipset: ChipSet,
    *,
    queries: list[str],
    model: str,
    model_version: str,
    crs: str,
) -> dict[str, Any]:
    """Build a GeoJSON FeatureCollection of per-chip classification scores.

    One Feature per chip: a footprint polygon (in the input CRS) plus the score for
    the primary query, a per-query ``scores`` map, the model, and chip index (PRD
    section 10.3). Coordinates are in the input CRS, named in a top-level ``crs``
    field for downstream tools.

    Args:
        scores: Per-chip, per-query probabilities, shape ``(n_chips, n_queries)``.
        chipset: Per-chip geometry.
        queries: The text queries that produced the scores; the first is primary.
        model: Model identifier.
        model_version: Resolved model version.
        crs: Source CRS string (coordinates are in this CRS).

    Returns:
        A JSON-serializable GeoJSON FeatureCollection dictionary.
    """
    matrix = np.atleast_2d(scores)
    primary = queries[0]
    features = []
    for chip, row in zip(chipset.chips, matrix.tolist(), strict=True):
        per_query = {q: float(s) for q, s in zip(queries, row, strict=True)}
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": _bbox_polygon(chip.bbox),
                },
                "properties": {
                    "score": per_query[primary],
                    "query": primary,
                    "scores": per_query,
                    "model": model,
                    "chip_index": chip.index,
                    "row": chip.row,
                    "col": chip.col,
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "crs": crs,
        "model": model,
        "model_version": model_version,
        "whirld_version": __version__,
        "query": primary,
        "queries": queries,
        "features": features,
    }


def write_geojson(feature_collection: dict[str, Any], output_path: str | Path) -> Path:
    """Write a GeoJSON FeatureCollection to disk.

    Args:
        feature_collection: The FeatureCollection dict from
            :func:`build_feature_collection`.
        output_path: Destination path.

    Returns:
        The path written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(feature_collection, indent=2), encoding="utf-8")
    _log.info("Wrote %s", output_path.name)
    return output_path


def _mask_profile(mask: np.ndarray, transform: Any, crs: str, nodata: int) -> dict:
    """Build the rasterio profile for a single-band mask GeoTIFF (LZW, nodata)."""
    return {
        "driver": "GTiff",
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "LZW",
    }


def write_mask_geotiff(
    mask: np.ndarray,
    transform: Any,
    crs: str,
    output_path: str | Path,
    *,
    nodata: int = 0,
) -> Path:
    """Write a segmentation mask as a single-band uint8 GeoTIFF (PRD section 10.2).

    CRS/transform are carried from the translated raster so the mask aligns with
    the input; ``COMPRESS=LZW`` and a ``nodata`` value are set.

    Args:
        mask: 2-D class-index mask ``(height, width)``.
        transform: Affine transform of the (translated) raster.
        crs: Coordinate reference system string.
        output_path: Destination path.
        nodata: Nodata value written to the TIFF metadata.

    Returns:
        The path written.
    """
    import rasterio  # noqa: PLC0415

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = _mask_profile(mask, transform, crs, nodata)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)
    _log.info("Wrote %s", output_path.name)
    return output_path


def mask_to_geotiff_bytes(
    mask: np.ndarray,
    transform: Any,
    crs: str,
    *,
    nodata: int = 0,
) -> bytes:
    """Serialize a mask to in-memory GeoTIFF bytes (for the REST ``image/tiff``).

    Args:
        mask: 2-D class-index mask ``(height, width)``.
        transform: Affine transform of the (translated) raster.
        crs: Coordinate reference system string.
        nodata: Nodata value.

    Returns:
        The GeoTIFF file contents as bytes.
    """
    import rasterio  # noqa: PLC0415

    profile = _mask_profile(mask, transform, crs, nodata)
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(mask.astype(np.uint8), 1)
        return memfile.read()
