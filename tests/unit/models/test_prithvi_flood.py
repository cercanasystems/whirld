"""Pinned flood model contracts remain distinct from HLS and Sentinel-2 L2A."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

from whirld.core.contract import translate
from whirld.core.registry import Registry
from whirld.core.sensor import detect_sensor
from whirld.errors import UnsupportedSensorError
from whirld.io.raster import RasterSource

BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
MEAN = [0.1412956, 0.13795798, 0.12353792, 0.30902815, 0.2044958, 0.11912015]
STD = [0.07406382, 0.07370365, 0.08692279, 0.11798815, 0.09772074, 0.07659938]


def test_flood_registry_pins_official_artifacts_and_sensor(whirld_home: Path) -> None:
    """Separate checkpoint/config identity and L1C-only semantics are explicit."""
    entry = Registry().get("prithvi-flood")
    assert entry.category == "segmentation"
    assert entry.backend == "prithvi"
    assert entry.config_file == "config.yaml"
    assert entry.trust == "first-party"
    assert entry.source.repo == (
        "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
    )
    assert entry.source.revision == "91ce9d38086a80b078a192b374df758b8855b732"
    assert entry.source.files == [
        "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt",
        "config.yaml",
    ]
    assert entry.distribution.sha256 == (
        "76eed77d8bd543ae441308b80e8408502243a871cb8a338ddc8220ed96dfc270"
    )
    assert entry.distribution.size_bytes == 1276843350
    assert set(entry.band_contract.sensors) == {"sentinel-2-l1c"}
    contract = entry.band_contract.sensors["sentinel-2-l1c"]
    assert contract.bands == BANDS
    assert contract.canonical_reflectance_scale == 0.0001
    assert entry.band_contract.chip_size_px == 512
    assert entry.band_contract.target_resolution_m == 10
    assert entry.band_contract.normalization.mean == MEAN
    assert entry.band_contract.normalization.std == STD
    assert entry.output.classes == 2


def test_flood_raw_dn_translation_matches_published_sen1floods_stats(
    whirld_home: Path,
) -> None:
    """Reordered DN pixels use task-specific reflectance standardization and masks."""
    entry = Registry().get("prithvi-flood")
    raw = np.arange(6 * 32 * 48, dtype=np.float32).reshape(6, 32, 48) + 100
    raw[:, 2, 3] = 0
    raster = RasterSource(
        data=raw[::-1].copy(),
        crs="EPSG:32646",
        transform=from_origin(573000, 2961000, 10, 10),
        band_descriptions=[band for band in BANDS[::-1]],
        tags={},
        nodata=0,
    )
    actual = translate(raster, entry, "sentinel-2-l1c")
    expected = (
        raw * 0.0001 - np.asarray(MEAN, dtype=np.float32)[:, None, None]
    ) / np.asarray(STD, dtype=np.float32)[:, None, None]
    expected[:, 2, 3] = 0
    np.testing.assert_allclose(actual.data, expected, rtol=1e-6, atol=1e-6)
    assert actual.valid_mask is not None
    assert not actual.valid_mask[2, 3]
    assert actual.valid_mask.sum() == 32 * 48 - 1


@pytest.mark.parametrize("sensor", ["hls", "sentinel-2-l2a"])
def test_flood_does_not_claim_other_processing_levels(
    whirld_home: Path,
    sensor: str,
) -> None:
    """Unsupported processing levels are rejected rather than sharing L1C stats."""
    entry = Registry().get("prithvi-flood")
    raster = RasterSource(
        data=np.ones((6, 16, 16), dtype=np.float32),
        crs="EPSG:32646",
        transform=from_origin(573000, 2961000, 10, 10),
        band_descriptions=[band for band in BANDS],
        tags={},
        nodata=None,
    )
    with pytest.raises(UnsupportedSensorError):
        detect_sensor(raster, entry, sensor)
