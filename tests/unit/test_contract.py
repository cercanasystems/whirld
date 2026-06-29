"""Unit tests for the band-contract translation pipeline (PRD section 7.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.make_fixtures import make_sentinel2
from whirld.core.contract import translate
from whirld.core.registry import Registry
from whirld.errors import UnsupportedSensorError
from whirld.io.raster import read_raster


def test_translate_shapes_and_band_order(whirld_home: Path, s2_tif: Path) -> None:
    """Translation yields 6 float32 bands at the target resolution."""
    entry = Registry().get("clay-v1")
    raster = read_raster(s2_tif)
    out = translate(raster, entry, "sentinel-2-l2a")
    assert out.data.shape[0] == 6
    assert out.data.dtype == np.float32
    assert out.aliases == ["blue", "green", "red", "nir", "swir16", "swir22"]
    assert out.target_resolution_m == 10.0


def test_normalization_applied(whirld_home: Path, s2_tif: Path) -> None:
    """Output reflects scale + z-score normalization, not raw DNs."""
    entry = Registry().get("clay-v1")
    raster = read_raster(s2_tif)
    out = translate(raster, entry, "sentinel-2-l2a")
    # Raw DNs were 0..4000; after scale (1e-4) and z-score the values are small.
    assert np.abs(out.data).max() < 50
    assert out.data.std() > 0


def test_select_by_alias_is_order_independent(
    whirld_home: Path, tmp_path: Path
) -> None:
    """Selecting by band description gives the canonical order even if bands
    are physically stored in a shuffled order."""
    import rasterio
    from rasterio.transform import from_origin

    # Write S2 bands in a deliberately shuffled physical order.
    shuffled = ["B12", "B02", "B11", "B03", "B08", "B04"]
    rng = np.random.default_rng(1)
    data = rng.integers(0, 4000, size=(6, 64, 64), dtype=np.uint16)
    path = tmp_path / "shuffled.tif"
    profile = {
        "driver": "GTiff",
        "height": 64,
        "width": 64,
        "count": 6,
        "dtype": "uint16",
        "crs": "EPSG:32630",
        "transform": from_origin(320000.0, 5822560.0, 10.0, 10.0),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for idx, band in enumerate(shuffled, start=1):
            dst.set_band_description(idx, band)

    entry = Registry().get("clay-v1")
    raster = read_raster(path)
    out = translate(raster, entry, "sentinel-2-l2a")
    # Band 0 must be 'blue' == B02, i.e. physical index 1 in the shuffled file,
    # scaled and normalized.
    norm = entry.band_contract.normalization
    expected_blue = (data[1].astype(np.float32) * norm.scale - norm.mean[0]) / norm.std[
        0
    ]
    assert np.allclose(out.data[0], expected_blue, atol=1e-4)


def test_unsupported_sensor_raises(whirld_home: Path, s2_tif: Path) -> None:
    """Translating for a sensor outside the contract raises (exit code 4)."""
    entry = Registry().get("clay-v1")
    raster = read_raster(s2_tif)
    with pytest.raises(UnsupportedSensorError):
        translate(raster, entry, "spot-6")


def test_resample_changes_grid(whirld_home: Path, tmp_path: Path) -> None:
    """A 30 m Landsat-style input is resampled toward the 10 m target."""
    path = make_sentinel2(
        tmp_path / "coarse.tif",
        width=60,
        height=60,
        resolution_m=30.0,
        tag_sensor=False,
    )
    entry = Registry().get("clay-v1")
    raster = read_raster(path)
    # Treat it as landsat (30 m native) so resampling to 10 m triggers.
    out = translate(raster, entry, "landsat-8-l2")
    assert out.target_resolution_m == 10.0
    # 60 px @ 30 m -> ~180 px @ 10 m.
    assert out.data.shape[1] > 100
