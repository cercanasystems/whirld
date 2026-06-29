"""Shared pytest fixtures.

Every test runs against an isolated ``WHIRLD_HOME`` in a temp directory so the
real ``~/.whirld`` is never touched and tests never collide.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from tests.fixtures.make_fixtures import (
    make_hls,
    make_no_crs,
    make_sentinel2,
    make_sentinel2_10band,
    make_stac_item,
)

# Starlette's TestClient emits this at import time (before per-test filters apply).
# It is a third-party, harmless deprecation; silence it for clean test output.
warnings.filterwarnings(
    "ignore",
    message=r"Using `httpx` with `starlette\.testclient`.*",
)


@pytest.fixture
def whirld_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``WHIRLD_HOME`` at an isolated temp directory.

    Args:
        tmp_path: Pytest's per-test temp directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The isolated home directory path.
    """
    home = tmp_path / "whirld_home"
    monkeypatch.setenv("WHIRLD_HOME", str(home))
    return home


@pytest.fixture
def s2_tif(tmp_path: Path) -> Path:
    """Write a small synthetic Sentinel-2 GeoTIFF and return its path.

    Args:
        tmp_path: Pytest's per-test temp directory.

    Returns:
        Path to the generated GeoTIFF.
    """
    return make_sentinel2(tmp_path / "s2_small.tif", width=300, height=300)


@pytest.fixture
def s2_10band_tif(tmp_path: Path) -> Path:
    """Write a small synthetic 10-band Sentinel-2 GeoTIFF (clay-v1.5 input)."""
    return make_sentinel2_10band(tmp_path / "s2_10band.tif")


@pytest.fixture
def hls_tif(tmp_path: Path) -> Path:
    """Write a small synthetic 6-band HLS GeoTIFF (Prithvi segmentation input)."""
    return make_hls(tmp_path / "hls_small.tif")


@pytest.fixture
def stac_item(tmp_path: Path) -> Path:
    """Write a local STAC item (file:// per-band COGs) keyed by native band id."""
    return make_stac_item(tmp_path / "stac")


@pytest.fixture
def stac_item_common(tmp_path: Path) -> Path:
    """Write a local STAC item keyed by spectral common name + eo:bands."""
    return make_stac_item(tmp_path / "stac_cn", common_names=True)


@pytest.fixture
def no_crs_tif(tmp_path: Path) -> Path:
    """Write a small GeoTIFF lacking a CRS and return its path."""
    return make_no_crs(tmp_path / "no_crs.tif")


@pytest.fixture
def s2_no_crs_tif(tmp_path: Path) -> Path:
    """Write a CRS-less but otherwise-valid S2 GeoTIFF (for --crs override tests)."""
    return make_sentinel2(tmp_path / "s2_no_crs.tif", crs=None)


@pytest.fixture
def pulled_clay(whirld_home: Path) -> str:
    """Pull the clay-v1 reference model into the isolated home.

    Args:
        whirld_home: The isolated home fixture (ensures env is set first).

    Returns:
        The model name, for convenience.
    """
    from whirld.core.fetch import pull

    pull("clay-v1")
    return "clay-v1"
