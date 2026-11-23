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

import math
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from ..errors import InvalidInputError, UnsupportedSensorError
from ..io.raster import RasterMetadata, RasterSource, validate_metric_grid
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
        valid_mask: Two-dimensional validity where every required band has an
            observation, before normalization fills invalid values. None means
            all pixels are valid (compatibility for manually constructed rasters).
    """

    data: np.ndarray
    crs: str
    transform: rasterio.Affine
    target_resolution_m: float
    aliases: list[str]
    sensor: str
    valid_mask: np.ndarray | None = None


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
    indices = _select_band_indices(raster, sensor_contract.bands)
    selected = raster.data[indices].astype(np.float32, copy=True)
    if raster.nodata is not None:
        selected[selected == raster.nodata] = np.nan
    selected[~np.isfinite(selected)] = np.nan
    if raster.band_radiometry is not None:
        if len(raster.band_radiometry) != raster.band_count:
            raise InvalidInputError("STAC radiometry must align with raster bands.")
        for position, index in enumerate(indices):
            radiometry = raster.band_radiometry[index]
            if radiometry.has_transform:
                canonical_scale = sensor_contract.resolve_reflectance_scale(sensor)
                # STAC describes physical reflectance, whereas the existing
                # normalization contract starts with canonical DN (even Clay).
                valid = np.isfinite(selected[position])
                with np.errstate(over="ignore", invalid="ignore"):
                    converted = (
                        (
                            selected[position].astype(np.float64) * radiometry.scale
                            + radiometry.offset
                        )
                        / canonical_scale
                    ).astype(np.float32)
                if not np.isfinite(converted[valid]).all():
                    raise InvalidInputError(
                        "STAC radiometry overflows canonical float32 input values."
                    )
                selected[position] = converted
    resampled, transform = _resample(
        selected,
        raster.transform,
        raster.crs,
        contract.target_resolution_m,
        nodata=np.nan,
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
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = (out - mean) / std
    # Missing observations stay missing throughout conversion/interpolation;
    # only now replace them with the model's normalized padding value.
    valid = np.isfinite(resampled)
    if not np.isfinite(out[valid]).all():
        raise InvalidInputError("Model normalization produced nonfinite values.")
    out[~valid] = contract.nodata_fill

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
        valid_mask=valid.all(axis=0),
    )


def _select_band_indices(raster: RasterMetadata, native_bands: list[str]) -> list[int]:
    """Reorder/select raster bands to match the sensor's declared band order.

    Bands are matched by description string when available (selection by name,
    per PRD section 7.2 step 3); otherwise a positional fallback assumes the
    raster bands already follow the declared order.

    Args:
        raster: The loaded raster.
        native_bands: Native band identifiers in the contract's canonical order.

    Returns:
        Zero-based indices aligning data and per-band metadata to the contract.
    """
    desc_to_index: dict[str, int] = {}
    for idx, desc in enumerate(raster.band_descriptions):
        if desc and desc.strip():
            key = desc.strip().lower()
            if key in desc_to_index:
                raise UnsupportedSensorError(
                    f"Ambiguous band description '{desc}': band names must be unique."
                )
            desc_to_index[key] = idx

    selected = []
    for position, band_name in enumerate(native_bands):
        key = band_name.strip().lower()
        if key in desc_to_index:
            selected.append(desc_to_index[key])
        elif not desc_to_index and position < raster.band_count:
            _log.debug(
                "Band '%s' not found by description; using positional index %d.",
                band_name,
                position,
            )
            selected.append(position)
        else:
            raise UnsupportedSensorError(
                f"Required band '{band_name}' is not present in the input's "
                "declared band descriptions. For unlabeled rasters, "
                "all bands must follow the sensor contract's declared order."
            )
    return selected


def _resample(
    data: np.ndarray,
    transform: rasterio.Affine,
    crs: str,
    target_res_m: float,
    nodata: float | None = None,
) -> tuple[np.ndarray, rasterio.Affine]:
    """Resample a band stack to the target resolution (bilinear).

    Args:
        data: Band stack ``(bands, height, width)``.
        transform: Source affine transform.
        crs: Coordinate reference system string.
        target_res_m: Desired pixel size in meters.
        nodata: Missing-value sentinel excluded independently for each band.

    Returns:
        A tuple of the resampled array and its new affine transform. When source
        and target resolution match (within 0.5%), the input is returned as-is.
    """
    source_x, source_y = validate_metric_grid(crs, transform)
    if not math.isfinite(target_res_m) or target_res_m <= 0:
        raise InvalidInputError("Target resolution must be finite and positive.")
    if all(
        abs(size - target_res_m) <= 0.005 * target_res_m
        for size in (source_x, source_y)
    ):
        return data, transform

    bands, height, width = data.shape
    dst_height = max(1, math.ceil(height * source_y / target_res_m))
    dst_width = max(1, math.ceil(width * source_x / target_res_m))
    dst_transform = rasterio.Affine(
        target_res_m,
        transform.b,
        transform.c,
        transform.d,
        -target_res_m,
        transform.f,
    )
    destination = np.full(
        (bands, dst_height, dst_width),
        nodata if nodata is not None else 0.0,
        dtype=np.float32,
    )
    # Warp bands independently: GDAL's multiband unified nodata behavior can
    # otherwise treat a band's missing sample as valid when another band is valid.
    for index in range(bands):
        reproject(
            source=data[index].astype(np.float32),
            destination=destination[index],
            src_transform=transform,
            src_crs=crs,
            dst_transform=dst_transform,
            dst_crs=crs,
            resampling=Resampling.bilinear,
            src_nodata=nodata,
            dst_nodata=nodata,
        )
    _log.debug(
        "Resampled %.1fx%.1f m -> %.1f m: %dx%d -> %dx%d",
        source_x,
        source_y,
        target_res_m,
        width,
        height,
        dst_width,
        dst_height,
    )
    return destination, dst_transform
