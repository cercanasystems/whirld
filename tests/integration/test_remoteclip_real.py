"""Gated integration test for REAL RemoteCLIP weights.

Skipped unless a real checkpoint is available, so the normal suite never needs the
605 MB download. To run it, point ``WHIRLD_TEST_REMOTECLIP_CKPT`` at a downloaded
``RemoteCLIP-ViT-B-32.pt``::

    WHIRLD_TEST_REMOTECLIP_CKPT=/path/to/RemoteCLIP-ViT-B-32.pt \
        pytest tests/integration/test_remoteclip_real.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("open_clip")

_CKPT_ENV = "WHIRLD_TEST_REMOTECLIP_CKPT"
_ckpt = os.environ.get(_CKPT_ENV)

pytestmark = pytest.mark.skipif(
    not _ckpt or not Path(_ckpt).exists(),
    reason=f"set {_CKPT_ENV} to a real RemoteCLIP checkpoint to run real tests",
)


def test_real_remoteclip_classify(whirld_home: Path, s2_tif: Path) -> None:
    """Real RemoteCLIP weights load and score a scene into valid GeoJSON."""
    from whirld.core.chips import chip_raster
    from whirld.core.contract import translate
    from whirld.core.registry import Registry
    from whirld.core.sensor import detect_sensor
    from whirld.io.output import build_feature_collection
    from whirld.io.raster import read_raster
    from whirld.models.remoteclip import RemoteCLIPBackend

    entry = Registry().get("remoteclip")
    raster = read_raster(s2_tif)
    sensor = detect_sensor(raster, entry)
    translated = translate(raster, entry, sensor)
    chips, chipset = chip_raster(
        translated,
        entry.band_contract.chip_size_px,
        nodata_fill=entry.band_contract.nodata_fill,
    )

    backend = RemoteCLIPBackend(
        name="remoteclip",
        version="1.0.0",
        device="cpu",
        arch=entry.model_name,
        ckpt_path=Path(_ckpt),
    )
    queries = [
        "a satellite photo of a solar farm",
        "a satellite photo of a forest",
    ]
    scores = backend.classify(chips, queries)
    assert scores.shape == (len(chipset.chips), 2)
    assert np.isfinite(scores).all()
    # Real softmax probabilities: in [0, 1] and sum to ~1 across the two queries.
    assert ((scores >= 0) & (scores <= 1)).all()
    assert np.allclose(scores.sum(axis=1), 1.0, atol=1e-4)

    fc = build_feature_collection(
        scores,
        chipset,
        queries=queries,
        model="remoteclip",
        model_version="1.0.0",
        crs=translated.crs,
    )
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == len(chipset.chips)
    assert set(fc["features"][0]["properties"]["scores"]) == set(queries)
