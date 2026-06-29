"""Unit tests for sensor detection precedence (PRD section 7.1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.make_fixtures import make_sentinel2
from whirld.core.registry import Registry
from whirld.core.sensor import detect_sensor
from whirld.errors import UnsupportedSensorError
from whirld.io.raster import RasterSource, read_raster


def _entry(whirld_home: Path):
    """Return the clay-v1 registry entry under the isolated home."""
    return Registry().get("clay-v1")


def test_detect_from_band_descriptions(whirld_home: Path, tmp_path: Path) -> None:
    """A scene with S2 band descriptions but no tag detects via band names."""
    path = make_sentinel2(tmp_path / "notag.tif", tag_sensor=False)
    raster = read_raster(path)
    assert detect_sensor(raster, _entry(whirld_home)) == "sentinel-2-l2a"


def test_detect_from_tags(whirld_home: Path, tmp_path: Path) -> None:
    """A scene with an IMAGEDESCRIPTION naming the sensor detects via tags."""
    path = make_sentinel2(tmp_path / "tagged.tif", tag_sensor=True)
    raster = read_raster(path)
    assert detect_sensor(raster, _entry(whirld_home)) == "sentinel-2-l2a"


def test_override_wins(whirld_home: Path, s2_tif: Path) -> None:
    """An explicit, supported override is honored."""
    raster = read_raster(s2_tif)
    assert (
        detect_sensor(raster, _entry(whirld_home), override="landsat-8-l2")
        == "landsat-8-l2"
    )


def test_override_unsupported_raises(whirld_home: Path, s2_tif: Path) -> None:
    """An unsupported override raises UnsupportedSensorError (exit code 4)."""
    raster = read_raster(s2_tif)
    with pytest.raises(UnsupportedSensorError) as excinfo:
        detect_sensor(raster, _entry(whirld_home), override="spot-6")
    assert excinfo.value.exit_code == 4
    assert "supports" in excinfo.value.message


def test_undetectable_raises(whirld_home: Path) -> None:
    """A raster with no usable metadata raises with an actionable message."""
    # No band descriptions, no tags, ambiguous 100 m resolution.
    raster = RasterSource(
        data=np.zeros((6, 8, 8), dtype="uint16"),
        crs="EPSG:32630",
        transform=__import__("rasterio").transform.from_origin(0, 0, 100, 100),
        band_descriptions=[None] * 6,
        tags={},
    )
    with pytest.raises(UnsupportedSensorError) as excinfo:
        detect_sensor(raster, _entry(whirld_home))
    assert "--sensor" in excinfo.value.message
