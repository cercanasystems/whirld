"""Raster reading helpers built on rasterio.

Whirld deliberately relies on rasterio's bundled GDAL (PRD section 15.5), so no
system GDAL is required. This module loads a GeoTIFF into an in-memory
:class:`RasterSource` carrying the pixel data plus the metadata the band-contract
pipeline and sensor detection need: band descriptions, TIFF tags, transform, CRS,
and ground resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioError

from ..errors import InvalidInputError
from ..logging_setup import get_logger

_log = get_logger("io.raster")


@dataclass
class RasterSource:
    """An in-memory raster with the metadata Whirld needs to translate it.

    Attributes:
        data: Pixel array of shape ``(bands, height, width)``.
        crs: Coordinate reference system as a string (e.g. ``EPSG:32630``).
        transform: Affine geotransform mapping pixel to world coordinates.
        band_descriptions: Per-band description strings (``None`` where absent).
        tags: Dataset-level TIFF tags (e.g. ``TIFFTAG_IMAGEDESCRIPTION``).
        nodata: Source nodata value, or ``None``.
    """

    data: np.ndarray
    crs: str
    transform: rasterio.Affine
    band_descriptions: list[str | None]
    tags: dict[str, str] = field(default_factory=dict)
    nodata: float | None = None

    @property
    def band_count(self) -> int:
        """Number of bands in the raster."""
        return int(self.data.shape[0])

    @property
    def height(self) -> int:
        """Raster height in pixels."""
        return int(self.data.shape[1])

    @property
    def width(self) -> int:
        """Raster width in pixels."""
        return int(self.data.shape[2])

    @property
    def resolution_m(self) -> tuple[float, float]:
        """Pixel size ``(x, y)`` in CRS units (meters for projected CRSs)."""
        return (abs(self.transform.a), abs(self.transform.e))


def read_raster(path: str | Path, crs: str | None = None) -> RasterSource:
    """Read a GeoTIFF into a :class:`RasterSource`.

    Args:
        path: Path to a local GeoTIFF (COG supported).
        crs: CRS to assign when the file declares none (e.g. ``EPSG:32630``).

    Returns:
        The loaded raster with metadata.

    Raises:
        InvalidInputError: The file is missing, unreadable, or has no CRS and no
            ``crs`` override was supplied.
    """
    path = Path(path)
    if not path.exists():
        raise InvalidInputError(
            f"Input file not found: {path}\n" f"       Check the path and try again."
        )
    try:
        with rasterio.open(path) as dataset:
            return _extract(dataset, label=path.name, crs_override=crs)
    except InvalidInputError:
        raise
    except RasterioError as exc:
        raise InvalidInputError(
            f"Could not read '{path.name}' as a raster.\n"
            f"       {exc}\n"
            f"       Ensure the file is a valid GeoTIFF."
        ) from exc


def read_raster_from_bytes(
    data: bytes, label: str = "upload", crs: str | None = None
) -> RasterSource:
    """Read a GeoTIFF from in-memory bytes (e.g. an HTTP file upload).

    Uses rasterio's :class:`~rasterio.io.MemoryFile` so no temporary file is
    written. Validation and metadata extraction match :func:`read_raster`.

    Args:
        data: Raw GeoTIFF bytes.
        label: Human-readable name used in error/log messages.
        crs: CRS to assign when the upload declares none.

    Returns:
        The loaded raster with metadata.

    Raises:
        InvalidInputError: The bytes are not a readable GeoTIFF, or have no CRS and
            no ``crs`` override was supplied.
    """
    if not data:
        raise InvalidInputError(
            "Uploaded file is empty.\n       Provide a valid GeoTIFF."
        )
    try:
        with rasterio.MemoryFile(data) as memfile, memfile.open() as dataset:
            return _extract(dataset, label=label, crs_override=crs)
    except InvalidInputError:
        raise
    except RasterioError as exc:
        raise InvalidInputError(
            f"Could not read '{label}' as a raster.\n"
            f"       {exc}\n"
            f"       Ensure the upload is a valid GeoTIFF."
        ) from exc


def _extract(
    dataset: rasterio.DatasetReader, label: str, crs_override: str | None = None
) -> RasterSource:
    """Build a :class:`RasterSource` from an open rasterio dataset.

    Args:
        dataset: An open rasterio dataset.
        label: Human-readable name for error/log messages.
        crs_override: CRS to assign when the dataset declares none.

    Returns:
        The populated raster source.

    Raises:
        InvalidInputError: The dataset has no CRS and no override was supplied.
    """
    if dataset.crs is not None:
        crs = dataset.crs.to_string()
    elif crs_override:
        crs = crs_override
        _log.debug("Input '%s' has no CRS; assigning override %s.", label, crs)
    else:
        raise InvalidInputError(
            f"Input '{label}' has no coordinate reference system.\n"
            f"       Whirld requires a CRS. Re-tag the file or supply "
            f"--crs to assign one (e.g. --crs EPSG:32630)."
        )
    data = dataset.read()
    _log.debug(
        "Read %s: %d bands, %dx%d, crs=%s",
        label,
        data.shape[0],
        data.shape[2],
        data.shape[1],
        crs,
    )
    return RasterSource(
        data=data,
        crs=crs,
        transform=dataset.transform,
        band_descriptions=list(dataset.descriptions),
        tags=dict(dataset.tags()),
        nodata=dataset.nodata,
    )
