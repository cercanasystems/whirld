"""The band-contract translation pipeline — Whirld's core contribution.

Given a raster and a model's band contract, this module produces the exact
normalized array the model expects, regardless of which supported sensor the
input came from. The steps (PRD section 7.2):

1. Validate the sensor is in the model's contract.
2. Select bands **by spectral alias** (blue, green, red, ...), never by index.
3. Resample to the contract's target resolution (bilinear for continuous data).
4. Apply the DN-to-reflectance scale factor where declared.
5. Normalize with the model's per-band mean/std.

Selecting by alias is what lets a Sentinel-2 scene and a Landsat scene both
produce the correct ordered input for Clay: Whirld looks for "blue", not "band 1".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from ..errors import UnsupportedSensorError
from ..io.raster import RasterSource
from ..logging_setup import get_logger
from .registry import ModelEntry

_log = get_logger("core.contract")


@dataclass
class TranslatedRaster:
    """A raster after band-contract translation, ready for chipping.

    Attributes:
        data: Normalized array of shape ``(bands, height, width)``, float32, with
            bands in the model's canonical alias order.
        crs: Coordinate reference system string (unchanged from the input).
        transform: Affine transform at the target resolution.
        target_resolution_m: The contract's target resolution in meters.
        aliases: Canonical spectral alias order of ``data``'s bands.
        sensor: The sensor key the translation was performed for.
    """

    data: np.ndarray
    crs: str
    transform: rasterio.Affine
    target_resolution_m: float
    aliases: list[str]
    sensor: str


def translate(
    raster: RasterSource,
    entry: ModelEntry,
    sensor: str,
) -> TranslatedRaster:
    """Run the band-contract translation pipeline.

    Args:
        raster: The loaded input raster.
        entry: The model whose band contract drives translation.
        sensor: The detected/declared sensor key.

    Returns:
        A :class:`TranslatedRaster` ready for chipping and inference.

    Raises:
        UnsupportedSensorError: ``sensor`` is not in the model's contract.
    """
    contract = entry.band_contract
    if sensor not in contract.sensors:
        supported = ", ".join(sorted(contract.sensors))
        raise UnsupportedSensorError(
            f"Sensor '{sensor}' is not supported by {entry.name}.\n"
            f"       {entry.name} supports: {supported}"
        )

    sensor_contract = contract.sensors[sensor]
    selected = _select_bands_by_alias(raster, sensor_contract.bands)
    resampled, transform = _resample(
        selected,
        raster.transform,
        raster.crs,
        raster.resolution_m[0],
        contract.target_resolution_m,
    )

    out = resampled.astype(np.float32, copy=True)
    norm = contract.normalization
    if norm.scale is not None:
        out *= float(norm.scale)

    mean = np.asarray(norm.mean, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(norm.std, dtype=np.float32).reshape(-1, 1, 1)
    if mean.shape[0] != out.shape[0]:
        raise UnsupportedSensorError(
            f"Band contract for '{entry.name}' has {mean.shape[0]} normalization "
            f"values but sensor '{sensor}' resolves to {out.shape[0]} bands."
        )
    out = (out - mean) / std

    _log.debug(
        "Translated to %d bands at %.1f m, shape %s",
        out.shape[0],
        contract.target_resolution_m,
        out.shape,
    )
    return TranslatedRaster(
        data=out,
        crs=raster.crs,
        transform=transform,
        target_resolution_m=float(contract.target_resolution_m),
        aliases=list(sensor_contract.aliases),
        sensor=sensor,
    )


def _select_bands_by_alias(raster: RasterSource, native_bands: list[str]) -> np.ndarray:
    """Reorder/select raster bands to match the sensor's declared band order.

    Bands are matched by description string when available (selection by name,
    per PRD section 7.2 step 3); otherwise a positional fallback assumes the
    raster bands already follow the declared order.

    Args:
        raster: The loaded raster.
        native_bands: Native band identifiers in the contract's canonical order.

    Returns:
        Array of shape ``(len(native_bands), height, width)``.
    """
    desc_to_index: dict[str, int] = {}
    for idx, desc in enumerate(raster.band_descriptions):
        if desc:
            desc_to_index[desc.strip().lower()] = idx

    selected = []
    for position, band_name in enumerate(native_bands):
        key = band_name.strip().lower()
        if key in desc_to_index:
            selected.append(raster.data[desc_to_index[key]])
        elif position < raster.band_count:
            _log.debug(
                "Band '%s' not found by description; using positional index %d.",
                band_name,
                position,
            )
            selected.append(raster.data[position])
        else:
            raise UnsupportedSensorError(
                f"Required band '{band_name}' is not present in the input "
                f"(only {raster.band_count} bands available)."
            )
    return np.stack(selected, axis=0)


def _resample(
    data: np.ndarray,
    transform: rasterio.Affine,
    crs: str,
    source_res_m: float,
    target_res_m: float,
) -> tuple[np.ndarray, rasterio.Affine]:
    """Resample a band stack to the target resolution (bilinear).

    Args:
        data: Band stack ``(bands, height, width)``.
        transform: Source affine transform.
        crs: Coordinate reference system string.
        source_res_m: Source pixel size in meters.
        target_res_m: Desired pixel size in meters.

    Returns:
        A tuple of the resampled array and its new affine transform. When source
        and target resolution match (within 0.5%), the input is returned as-is.
    """
    if abs(source_res_m - target_res_m) <= 0.005 * target_res_m:
        return data, transform

    scale = source_res_m / target_res_m
    bands, height, width = data.shape
    dst_height = max(1, int(round(height * scale)))
    dst_width = max(1, int(round(width * scale)))
    dst_transform = rasterio.Affine(
        target_res_m,
        transform.b,
        transform.c,
        transform.d,
        -target_res_m,
        transform.f,
    )
    destination = np.zeros((bands, dst_height, dst_width), dtype=np.float32)
    reproject(
        source=data.astype(np.float32),
        destination=destination,
        src_transform=transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=crs,
        resampling=Resampling.bilinear,
    )
    _log.debug(
        "Resampled %.1f m -> %.1f m: %dx%d -> %dx%d",
        source_res_m,
        target_res_m,
        width,
        height,
        dst_width,
        dst_height,
    )
    return destination, dst_transform
