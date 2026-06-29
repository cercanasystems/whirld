"""Generate tiny synthetic GeoTIFF fixtures for tests — no network, no big files.

Run directly to (re)write fixtures into this directory::

    python tests/fixtures/make_fixtures.py

The fixtures are deliberately small (a few hundred pixels) so the whole test
suite is fast and self-contained.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

# Sentinel-2 L2A six-band order matching the clay-v1 (reference) contract.
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
# Sentinel-2 L2A ten-band order matching the clay-v1.5 (real) contract.
S2_BANDS_10 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
# HLS six-band order matching the Prithvi (segmentation) contract.
HLS_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
_FIXTURE_DIR = Path(__file__).resolve().parent


def make_sentinel2(
    path: str | Path | None = None,
    *,
    width: int = 300,
    height: int = 300,
    resolution_m: float = 10.0,
    tag_sensor: bool = True,
    crs: str | None = "EPSG:32630",
    seed: int = 7,
) -> Path:
    """Write a small synthetic Sentinel-2 L2A GeoTIFF.

    The raster carries six bands described with their native S2 identifiers and a
    UTM CRS, plus (optionally) an ``IMAGEDESCRIPTION`` tag naming the sensor — so
    both tag-based and band-description-based detection can be exercised.

    Args:
        path: Output path; defaults to ``s2_small.tif`` beside this file.
        width: Raster width in pixels.
        height: Raster height in pixels.
        resolution_m: Pixel size in meters.
        tag_sensor: Whether to write a sensor-naming TIFF tag.
        crs: CRS to assign, or ``None`` to write a CRS-less file (for --crs tests).
        seed: RNG seed for reproducible pixel values.

    Returns:
        The path written.
    """
    path = Path(path) if path else _FIXTURE_DIR / "s2_small.tif"
    rng = np.random.default_rng(seed)
    # Reflectance-like digital numbers (0..4000) as uint16, the S2 L2A norm.
    data = rng.integers(0, 4000, size=(len(S2_BANDS), height, width), dtype=np.uint16)
    transform = from_origin(320000.0, 5822560.0, resolution_m, resolution_m)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(S2_BANDS),
        "dtype": "uint16",
        "transform": transform,
    }
    if crs is not None:
        profile["crs"] = crs
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for idx, band in enumerate(S2_BANDS, start=1):
            dst.set_band_description(idx, band)
        if tag_sensor:
            dst.update_tags(TIFFTAG_IMAGEDESCRIPTION="Sentinel-2 L2A scene")
    return path


def make_sentinel2_10band(
    path: str | Path | None = None,
    *,
    width: int = 256,
    height: int = 256,
    resolution_m: float = 10.0,
    tag_sensor: bool = True,
    seed: int = 11,
) -> Path:
    """Write a small synthetic 10-band Sentinel-2 L2A GeoTIFF (clay-v1.5 input).

    Carries the ten native S2 band identifiers Clay expects (incl. red-edge and
    narrow NIR) so band-description detection resolves ``sentinel-2-l2a``.

    Args:
        path: Output path; defaults to ``s2_10band.tif`` beside this file.
        width: Raster width in pixels.
        height: Raster height in pixels.
        resolution_m: Pixel size in meters.
        tag_sensor: Whether to write a sensor-naming TIFF tag.
        seed: RNG seed for reproducible pixel values.

    Returns:
        The path written.
    """
    path = Path(path) if path else _FIXTURE_DIR / "s2_10band.tif"
    rng = np.random.default_rng(seed)
    data = rng.integers(
        0, 4000, size=(len(S2_BANDS_10), height, width), dtype=np.uint16
    )
    transform = from_origin(320000.0, 5822560.0, resolution_m, resolution_m)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(S2_BANDS_10),
        "dtype": "uint16",
        "crs": "EPSG:32630",
        "transform": transform,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for idx, band in enumerate(S2_BANDS_10, start=1):
            dst.set_band_description(idx, band)
        if tag_sensor:
            dst.update_tags(TIFFTAG_IMAGEDESCRIPTION="Sentinel-2 L2A scene")
    return path


def make_hls(
    path: str | Path | None = None,
    *,
    width: int = 320,
    height: int = 320,
    resolution_m: float = 30.0,
    tag_sensor: bool = True,
    seed: int = 13,
) -> Path:
    """Write a small synthetic HLS GeoTIFF (Prithvi segmentation input).

    Six HLS bands at 30 m with the band identifiers Prithvi's contract expects, so
    band-description detection resolves the ``hls`` sensor.

    Args:
        path: Output path; defaults to ``hls_small.tif`` beside this file.
        width: Raster width in pixels.
        height: Raster height in pixels.
        resolution_m: Pixel size in meters.
        tag_sensor: Whether to write a sensor-naming TIFF tag.
        seed: RNG seed for reproducible pixel values.

    Returns:
        The path written.
    """
    path = Path(path) if path else _FIXTURE_DIR / "hls_small.tif"
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 4000, size=(len(HLS_BANDS), height, width), dtype=np.uint16)
    transform = from_origin(600000.0, 4000000.0, resolution_m, resolution_m)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(HLS_BANDS),
        "dtype": "uint16",
        "crs": "EPSG:32613",
        "transform": transform,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for idx, band in enumerate(HLS_BANDS, start=1):
            dst.set_band_description(idx, band)
        if tag_sensor:
            dst.update_tags(TIFFTAG_IMAGEDESCRIPTION="HLS scene")
    return path


def make_no_crs(path: str | Path | None = None) -> Path:
    """Write a small GeoTIFF with no CRS (for input-validation tests).

    Args:
        path: Output path; defaults to ``no_crs.tif`` beside this file.

    Returns:
        The path written.
    """
    path = Path(path) if path else _FIXTURE_DIR / "no_crs.tif"
    data = np.zeros((6, 16, 16), dtype=np.uint16)
    profile = {
        "driver": "GTiff",
        "height": 16,
        "width": 16,
        "count": 6,
        "dtype": "uint16",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def make_stac_item(
    directory: str | Path,
    *,
    width: int = 300,
    height: int = 300,
    resolution_m: float = 10.0,
    seed: int = 7,
    common_names: bool = False,
) -> Path:
    """Write a local STAC item whose assets are single-band COGs (``file://`` hrefs).

    Mirrors :func:`make_sentinel2` (same six S2 bands, CRS, footprint, and seed) but
    splits each band into its own cloud-optimized GeoTIFF and emits a STAC item JSON
    referencing them. Fully offline: the reader resolves ``file://`` hrefs and reads
    them through rasterio. Use ``common_names`` to key assets by spectral common name
    (``red``/``nir``/…) with ``eo:bands`` instead of by native band id (``B04``).

    Args:
        directory: Directory to write the item + per-band COGs into.
        width: Raster width in pixels.
        height: Raster height in pixels.
        resolution_m: Pixel size in meters.
        seed: RNG seed for reproducible pixel values.
        common_names: Key assets by spectral common name + ``eo:bands`` if true.

    Returns:
        Path to the written ``item.json``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    transform = from_origin(320000.0, 5822560.0, resolution_m, resolution_m)
    common = ["blue", "green", "red", "nir", "swir16", "swir22"]

    assets: dict[str, dict] = {}
    for band, alias in zip(S2_BANDS, common, strict=True):
        band_data = rng.integers(0, 4000, size=(1, height, width), dtype=np.uint16)
        asset_path = directory / f"{band}.tif"
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint16",
            "crs": "EPSG:32630",
            "transform": transform,
            "tiled": True,
            "blockxsize": 128,
            "blockysize": 128,
        }
        with rasterio.open(asset_path, "w", **profile) as dst:
            dst.write(band_data)
            dst.set_band_description(1, band)
        key = alias if common_names else band
        asset = {"href": asset_path.as_uri(), "type": "image/tiff"}
        if common_names:
            asset["eo:bands"] = [{"name": band, "common_name": alias}]
        assets[key] = asset

    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": "S2-test-item",
        "collection": "sentinel-2-l2a",
        "properties": {
            "datetime": "2024-06-01T10:00:00Z",
            "platform": "sentinel-2a",
            "constellation": "sentinel-2",
        },
        "geometry": None,
        "assets": assets,
    }
    item_path = directory / "item.json"
    item_path.write_text(json.dumps(item, indent=2), encoding="utf-8")
    return item_path


if __name__ == "__main__":
    written = make_sentinel2()
    print(f"Wrote {written}")
