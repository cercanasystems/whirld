"""Unit tests for mask reassembly and the segmentation GeoTIFF writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from whirld.core.chips import chip_raster, reassemble_mask
from whirld.core.contract import TranslatedRaster
from whirld.io.output import mask_to_geotiff_bytes, write_mask_geotiff


def _translated(height: int = 300, width: int = 300) -> TranslatedRaster:
    return TranslatedRaster(
        data=np.ones((6, height, width), dtype=np.float32),
        crs="EPSG:32613",
        transform=from_origin(600000.0, 4000000.0, 30.0, 30.0),
        target_resolution_m=30.0,
        aliases=["blue", "green", "red", "nir_narrow", "swir1", "swir2"],
        sensor="hls",
    )


def test_reassemble_round_trips_and_crops() -> None:
    """Per-chip masks stitch back to the original extent (padding cropped)."""
    translated = _translated(300, 300)  # 2x2 grid of 256 tiles, padded
    _, chipset = chip_raster(translated, 256)
    # Each chip's mask is all-ones; reassembled non-padding region must be all ones.
    masks = np.ones((len(chipset.chips), 256, 256), dtype=np.uint8)
    out = reassemble_mask(masks, chipset, 300, 300)
    assert out.shape == (300, 300)
    assert out.dtype == np.uint8
    assert (out == 1).all()  # padded tile areas were cropped, not written


def test_reassemble_places_chips_by_position() -> None:
    """Each chip's class value lands in its grid cell."""
    translated = _translated(512, 512)  # 2x2 grid of 256 tiles, exact
    _, chipset = chip_raster(translated, 256)
    masks = np.stack(
        [np.full((256, 256), i, dtype=np.uint8) for i in range(len(chipset.chips))]
    )
    out = reassemble_mask(masks, chipset, 512, 512)
    # chip 0 = (row0,col0), chip 3 = (row1,col1) in row-major order.
    assert out[0, 0] == 0
    assert out[300, 300] == 3


def test_write_mask_geotiff_roundtrip(tmp_path: Path) -> None:
    """The mask writes as a single-band uint8 LZW GeoTIFF and reads back."""
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 1
    transform = from_origin(600000.0, 4000000.0, 30.0, 30.0)
    out = write_mask_geotiff(mask, transform, "EPSG:32613", tmp_path / "m.tif")
    with rasterio.open(out) as ds:
        assert ds.count == 1
        assert ds.dtypes[0] == "uint8"
        assert ds.crs.to_string() == "EPSG:32613"
        assert ds.nodata == 0
        read = ds.read(1)
    assert np.array_equal(read, mask)


def test_mask_to_geotiff_bytes() -> None:
    """In-memory GeoTIFF bytes are a valid readable raster."""
    mask = np.ones((8, 8), dtype=np.uint8)
    transform = from_origin(0.0, 0.0, 30.0, 30.0)
    data = mask_to_geotiff_bytes(mask, transform, "EPSG:32613")
    with rasterio.MemoryFile(data) as mem, mem.open() as ds:
        assert ds.count == 1 and ds.dtypes[0] == "uint8"
        assert np.array_equal(ds.read(1), mask)
