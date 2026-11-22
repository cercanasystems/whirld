"""Opt-in direct URL validation against pinned public HLS reflectance imagery.

Run with ``WHIRLD_TEST_COG=1``. This public six-band 512px GeoTIFF uses row
strips, not COG tiles. It verifies HTTP range-compatible direct TIFF input,
explicit source metadata and normalized windows without downloading checkpoints.
Hermetic tests in test_cog_input.py separately exercise actual tiled COGs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import rasterio

from whirld.core.options import CogOptions
from whirld.core.registry import Registry
from whirld.io.cog import CogWindowReader

pytestmark = pytest.mark.skipif(
    os.environ.get("WHIRLD_TEST_COG") != "1",
    reason="set WHIRLD_TEST_COG=1 to read the pinned public HLS GeoTIFF",
)

_URL = (
    "https://huggingface.co/ibm-nasa-geospatial/"
    "Prithvi-EO-2.0-300M-BurnScars/resolve/"
    "a3f2c410e45b8ac7417976614528a872f024d831/examples/"
    "subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif"
)
_BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")


def test_pinned_public_direct_tiff_normalization(whirld_home: Path) -> None:
    """Compare every streamed chip with independently normalized public pixels.

    Args:
        whirld_home: Isolated registry home; no checkpoints or user cache are used.

    Raises:
        AssertionError: Source layout, pixel units or normalized windows disagree.
    """
    entry = Registry().get("prithvi-burn-scar")
    contract = entry.band_contract
    normalization = contract.normalization
    with (
        rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True),
        rasterio.open(f"/vsicurl/{_URL}") as source,
    ):
        assert source.shape == (512, 512)
        assert source.count == 6
        assert all(width == 512 for _, width in source.block_shapes)
        assert source.transform.a == 30
        assert source.transform.e == -30
        assert source.crs is not None
        crs = source.crs.to_string()
        transform = source.transform
        stored = source.read(masked=True).astype(np.float64)

    # These values already represent reflectance: scale=1 is essential. Apply
    # only the published model mean/std, independently of Whirld translation.
    assert 0 < np.ma.median(stored) < 1
    mean = np.asarray(normalization.mean, dtype=np.float64)[:, None, None]
    std = np.asarray(normalization.std, dtype=np.float64)[:, None, None]
    expected = ((stored - mean) / std).filled(contract.nodata_fill)
    expected[~np.isfinite(expected)] = contract.nodata_fill

    with CogWindowReader(
        _URL,
        entry=entry,
        sensor="hls",
        cog=CogOptions(bands=_BANDS, scale=1.0),
        chip_size=128,
        batch_size=3,
    ) as reader:
        assert reader.crs == crs
        assert reader.chip_count == 16
        count = 0
        for batch in reader.batches():
            assert len(batch.chips) <= 3
            for pixels, chip in zip(batch.pixels, batch.chips, strict=True):
                row, col = chip.row * 128, chip.col * 128
                np.testing.assert_allclose(
                    pixels,
                    expected[:, row : row + 128, col : col + 128],
                    rtol=2e-5,
                    atol=2e-5,
                )
                west, north = transform * (col, row)
                east, south = transform * (col + 128, row + 128)
                assert chip.bbox == [west, south, east, north]
                count += 1
        assert count == 16
