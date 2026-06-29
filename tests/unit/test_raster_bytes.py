"""Unit tests for reading a raster from in-memory bytes."""

from __future__ import annotations

from pathlib import Path

import pytest

from whirld.errors import InvalidInputError
from whirld.io.raster import read_raster, read_raster_from_bytes


def test_read_from_bytes_matches_path(s2_tif: Path) -> None:
    """Reading bytes yields the same shape/CRS/bands as reading the path."""
    from_path = read_raster(s2_tif)
    from_bytes = read_raster_from_bytes(s2_tif.read_bytes(), label="s2.tif")
    assert from_bytes.data.shape == from_path.data.shape
    assert from_bytes.crs == from_path.crs
    assert from_bytes.band_descriptions == from_path.band_descriptions


def test_empty_bytes_rejected() -> None:
    """Empty upload bytes raise InvalidInputError."""
    with pytest.raises(InvalidInputError):
        read_raster_from_bytes(b"")


def test_garbage_bytes_rejected() -> None:
    """Non-raster bytes raise InvalidInputError with guidance."""
    with pytest.raises(InvalidInputError):
        read_raster_from_bytes(b"this is not a geotiff", label="bad.tif")
