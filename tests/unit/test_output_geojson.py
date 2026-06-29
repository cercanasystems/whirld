"""Unit tests for the GeoJSON classification output writer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from whirld.core.chips import ChipMeta, ChipSet
from whirld.io.output import build_feature_collection, write_geojson


def _chipset() -> ChipSet:
    """A 2-chip chipset with known bounding boxes."""
    return ChipSet(
        chips=[
            ChipMeta(index=0, row=0, col=0, bbox=[0.0, 0.0, 10.0, 10.0]),
            ChipMeta(index=1, row=0, col=1, bbox=[10.0, 0.0, 20.0, 10.0]),
        ],
        chip_size_px=256,
        n_rows=1,
        n_cols=2,
    )


def test_build_feature_collection() -> None:
    """A Feature per chip carries a polygon plus per-query scores."""
    scores = np.array([[0.8], [0.2]], dtype=np.float32)
    fc = build_feature_collection(
        scores,
        _chipset(),
        queries=["solar farm"],
        model="remoteclip",
        model_version="1.0.0",
        crs="EPSG:32630",
    )
    assert fc["type"] == "FeatureCollection"
    assert fc["crs"] == "EPSG:32630"
    assert fc["query"] == "solar farm"
    assert fc["queries"] == ["solar farm"]
    assert len(fc["features"]) == 2

    feat = fc["features"][0]
    assert feat["geometry"]["type"] == "Polygon"
    # Closed ring: first == last point, 5 vertices.
    ring = feat["geometry"]["coordinates"][0]
    assert len(ring) == 5 and ring[0] == ring[-1]
    assert feat["properties"]["score"] == pytest.approx(0.8, abs=1e-6)
    assert feat["properties"]["query"] == "solar farm"
    assert feat["properties"]["scores"]["solar farm"] == pytest.approx(0.8, abs=1e-6)
    assert feat["properties"]["model"] == "remoteclip"
    assert feat["properties"]["chip_index"] == 0


def test_build_feature_collection_multi_query() -> None:
    """Per-query scores are recorded for each query; primary drives `score`."""
    scores = np.array([[0.7, 0.3], [0.1, 0.9]], dtype=np.float32)
    fc = build_feature_collection(
        scores,
        _chipset(),
        queries=["solar farm", "forest"],
        model="remoteclip",
        model_version="1.0.0",
        crs="EPSG:32630",
    )
    props = fc["features"][1]["properties"]
    assert props["score"] == pytest.approx(0.1, abs=1e-6)  # primary = solar farm
    assert props["scores"]["forest"] == pytest.approx(0.9, abs=1e-6)


def test_write_geojson_roundtrip(tmp_path: Path) -> None:
    """write_geojson produces a parseable GeoJSON file."""
    fc = build_feature_collection(
        np.array([[0.5], [0.1]], dtype=np.float32),
        _chipset(),
        queries=["airport"],
        model="remoteclip",
        model_version="1.0.0",
        crs="EPSG:32630",
    )
    out = tmp_path / "scores.geojson"
    path = write_geojson(fc, out)
    assert path == out
    loaded = json.loads(out.read_text())
    assert loaded["type"] == "FeatureCollection"
    assert len(loaded["features"]) == 2
