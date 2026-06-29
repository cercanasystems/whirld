"""Unit tests for chipping and georeferenced chip metadata."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin

from whirld.core.chips import chip_raster
from whirld.core.contract import TranslatedRaster
from whirld.errors import WhirldError


def _translated(
    bands: int = 6, height: int = 300, width: int = 300
) -> TranslatedRaster:
    """Build a synthetic translated raster for chipping tests."""
    return TranslatedRaster(
        data=np.ones((bands, height, width), dtype=np.float32),
        crs="EPSG:32630",
        transform=from_origin(320000.0, 5822560.0, 10.0, 10.0),
        target_resolution_m=10.0,
        aliases=["blue", "green", "red", "nir", "swir16", "swir22"],
        sensor="sentinel-2-l2a",
    )


def test_chip_counts_and_shape() -> None:
    """A 300x300 raster chips into a 2x2 grid of 256px tiles."""
    arr, chipset = chip_raster(_translated(), 256)
    assert arr.shape == (4, 6, 256, 256)
    assert chipset.n_rows == 2 and chipset.n_cols == 2
    assert len(chipset.chips) == 4


def test_edge_padding_with_nodata() -> None:
    """Partial edge tiles are padded with the nodata fill value."""
    arr, _ = chip_raster(_translated(height=300, width=300), 256, nodata_fill=-1.0)
    # The last chip (bottom-right) is mostly padding.
    last = arr[-1]
    assert (last == -1.0).any()
    assert (last == 1.0).any()


def test_chip_bbox_in_crs() -> None:
    """Chip 0's bbox starts at the raster origin in CRS coordinates."""
    _, chipset = chip_raster(_translated(), 256)
    minx, miny, maxx, maxy = chipset.chips[0].bbox
    assert minx == 320000.0
    assert maxy == 5822560.0
    assert maxx == 320000.0 + 256 * 10.0


def test_overlap() -> None:
    """Overlap increases the number of tiles along each axis."""
    _, no_overlap = chip_raster(_translated(), 256, overlap=0)
    _, with_overlap = chip_raster(_translated(), 256, overlap=128)
    assert len(with_overlap.chips) > len(no_overlap.chips)


def test_invalid_chip_size_raises() -> None:
    """Non-positive chip size raises a WhirldError."""
    with pytest.raises(WhirldError):
        chip_raster(_translated(), 0)


def test_invalid_overlap_raises() -> None:
    """Overlap >= chip size raises a WhirldError."""
    with pytest.raises(WhirldError):
        chip_raster(_translated(), 256, overlap=256)
