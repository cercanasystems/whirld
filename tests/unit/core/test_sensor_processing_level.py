"""Processing levels cannot be inferred from shared Sentinel-2 bands or grids."""

from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

from whirld.core.registry import Registry
from whirld.core.sensor import detect_sensor
from whirld.errors import UnsupportedSensorError
from whirld.io.raster import RasterSource


@pytest.mark.parametrize(
    "tag", ["sentinel-2-l2a", "Sentinel2 L2A", "S2A_MSIL2A", "", "sentinel-2"]
)
def test_flood_needs_explicit_l1c_level(whirld_home: Path, tag: str) -> None:
    """Shared family hints or L2A tags cannot silently claim a supported L1C source."""
    entry = Registry().get("prithvi-flood")
    source = RasterSource(
        data=np.ones((6, 4, 4), dtype=np.float32),
        crs="EPSG:32630",
        transform=from_origin(500000, 4500000, 10, 10),
        band_descriptions=entry.band_contract.sensors["sentinel-2-l1c"].bands,
        tags={"TIFFTAG_IMAGEDESCRIPTION": tag},
    )
    with pytest.raises(UnsupportedSensorError, match="L1C"):
        detect_sensor(source, entry)
    assert detect_sensor(source, entry, override="sentinel-2-l1c") == "sentinel-2-l1c"


@pytest.mark.parametrize("tag", ["sentinel-2-l1c", "Sentinel2 L1C", "S2A_MSIL1C"])
def test_flood_accepts_declared_l1c(whirld_home: Path, tag: str) -> None:
    """An unambiguous processing-level tag permits existing spectral detection."""
    entry = Registry().get("prithvi-flood")
    source = RasterSource(
        data=np.ones((6, 4, 4), dtype=np.float32),
        crs="EPSG:32630",
        transform=from_origin(500000, 4500000, 10, 10),
        band_descriptions=entry.band_contract.sensors["sentinel-2-l1c"].bands,
        tags={"TIFFTAG_IMAGEDESCRIPTION": tag},
    )
    assert detect_sensor(source, entry) == "sentinel-2-l1c"
