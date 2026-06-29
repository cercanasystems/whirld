"""Unit tests for Clay metadata encoding + the orchestrator's metadata helpers."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from whirld.models._clay_metadata import (
    latlon_vector,
    normalize_latlon,
    normalize_timestamp,
    time_vector,
)


def test_normalize_timestamp_matches_clay_formula() -> None:
    """normalize_timestamp reproduces Clay's exact week/hour sin-cos encoding."""
    date = datetime(2024, 6, 1, 10, 0, 0)
    week = date.isocalendar().week * 2 * math.pi / 52
    hour = 10 * 2 * math.pi / 24
    (sw, cw), (sh, ch) = normalize_timestamp(date)
    assert (sw, cw) == pytest.approx((math.sin(week), math.cos(week)))
    assert (sh, ch) == pytest.approx((math.sin(hour), math.cos(hour)))


def test_normalize_latlon_matches_clay_formula() -> None:
    """normalize_latlon reproduces Clay's exact radians sin-cos encoding."""
    (sla, cla), (slo, clo) = normalize_latlon(51.5, -0.1)
    assert (sla, cla) == pytest.approx(
        (math.sin(math.radians(51.5)), math.cos(math.radians(51.5)))
    )
    assert (slo, clo) == pytest.approx(
        (math.sin(math.radians(-0.1)), math.cos(math.radians(-0.1)))
    )


def test_vectors_are_4_element() -> None:
    """time/latlon vectors are 4-element [sin, cos, sin, cos]."""
    assert len(time_vector(datetime(2024, 1, 1, 0, 0))) == 4
    assert len(latlon_vector((10.0, 20.0))) == 4


def test_vectors_zero_when_unknown() -> None:
    """Missing metadata yields zero vectors (faithful fallback)."""
    assert time_vector(None) == [0.0, 0.0, 0.0, 0.0]
    assert latlon_vector(None) == [0.0, 0.0, 0.0, 0.0]


def test_chip_latlons_reprojects_utm(whirld_home: Path, s2_tif: Path) -> None:
    """_chip_latlons turns UTM chip centroids into plausible WGS84 lat/lon."""
    from whirld.api import _chip_latlons
    from whirld.core.chips import chip_raster
    from whirld.core.contract import translate
    from whirld.core.registry import Registry
    from whirld.io.raster import read_raster

    entry = Registry().get("clay-v1")
    raster = read_raster(s2_tif)
    translated = translate(raster, entry, "sentinel-2-l2a")
    _, chipset = chip_raster(translated, 256, nodata_fill=0.0)

    latlons = _chip_latlons(chipset, translated.crs)
    assert latlons is not None and len(latlons) == len(chipset.chips)
    lat, lon = latlons[0]
    # UTM 30N, easting 320000 (west of the 500000 central meridian) → ~52.5N, ~-5.6E.
    assert 50 < lat < 54
    assert -8 < lon < 0


def test_resolve_datetime_precedence() -> None:
    """_resolve_datetime honors override > TIFF tag > None."""
    from whirld.api import _resolve_datetime

    # ISO override wins.
    dt = _resolve_datetime(
        "2024-06-01T10:00:00Z", {"TIFFTAG_DATETIME": "2000:01:01 00:00:00"}
    )
    assert dt.year == 2024 and dt.hour == 10

    # TIFF tag used when no override.
    dt = _resolve_datetime(None, {"TIFFTAG_DATETIME": "2021:07:15 08:30:00"})
    assert dt.year == 2021 and dt.month == 7 and dt.hour == 8

    # Nothing → None.
    assert _resolve_datetime(None, {}) is None
    # Unparseable override → falls through to None (no tag).
    assert _resolve_datetime("not-a-date", {}) is None


def test_clay_datacube_metadata(whirld_home: Path) -> None:
    """ClayTorchBackend._build_datacube fills real time/latlon, zeros when absent."""
    pytest.importorskip("torch")
    import torch

    from whirld.models.clay_torch import ClayTorchBackend

    backend = ClayTorchBackend(
        name="clay-v1.5",
        version="1.5.0",
        device="cpu",
        embed_dim=1024,
        patch_size=8,
        encoder=object(),  # not used by _build_datacube
    )
    chips = np.zeros((2, 10, 32, 32), dtype=np.float32)
    waves = [0.49, 0.56, 0.66, 0.70, 0.74, 0.78, 0.84, 0.86, 1.61, 2.19]

    with_meta = backend._build_datacube(
        chips,
        waves,
        10.0,
        datetime(2024, 6, 1, 10, 0),
        [(51.5, -0.1), (52.0, 0.2)],
        torch,
    )
    assert with_meta["time"].shape == (2, 4)
    assert with_meta["latlon"].shape == (2, 4)
    assert torch.any(with_meta["time"] != 0)
    assert torch.any(with_meta["latlon"] != 0)
    # Per-chip lat/lon differ.
    assert not torch.allclose(with_meta["latlon"][0], with_meta["latlon"][1])

    without = backend._build_datacube(chips, waves, 10.0, None, None, torch)
    assert torch.all(without["time"] == 0)
    assert torch.all(without["latlon"] == 0)
