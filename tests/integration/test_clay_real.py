"""Gated integration test for REAL Clay v1.5 weights.

Skipped unless a real checkpoint is available, so the normal suite never needs the
5 GB download. To run it, point ``WHIRLD_TEST_CLAY_CKPT`` at a downloaded
``clay-v1.5.ckpt`` (e.g. the path printed by ``whirld pull clay-v1.5`` or the
Hugging Face cache), then::

    WHIRLD_TEST_CLAY_CKPT=/path/to/clay-v1.5.ckpt \
        pytest tests/integration/test_clay_real.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_CKPT_ENV = "WHIRLD_TEST_CLAY_CKPT"
_ckpt = os.environ.get(_CKPT_ENV)

pytestmark = pytest.mark.skipif(
    not _ckpt or not Path(_ckpt).exists(),
    reason=f"set {_CKPT_ENV} to a real clay-v1.5.ckpt to run real-weights tests",
)


def test_real_clay_embed_1024(whirld_home: Path, s2_10band_tif: Path) -> None:
    """Real Clay v1.5 weights load and embed a 10-band scene to (n, 1024)."""
    from whirld.core.chips import chip_raster
    from whirld.core.contract import translate
    from whirld.core.registry import Registry
    from whirld.core.sensor import detect_sensor
    from whirld.io.raster import read_raster
    from whirld.models.base import InferenceContext
    from whirld.models.clay_torch import ClayTorchBackend

    entry = Registry().get("clay-v1.5")
    raster = read_raster(s2_10band_tif)
    sensor = detect_sensor(raster, entry)
    assert sensor == "sentinel-2-l2a"

    translated = translate(raster, entry, sensor)
    chips, chipset = chip_raster(
        translated,
        entry.band_contract.chip_size_px,
        nodata_fill=entry.band_contract.nodata_fill,
    )

    backend = ClayTorchBackend(
        name="clay-v1.5",
        version="1.5.0",
        device="cpu",
        embed_dim=entry.output.embed_dim,
        patch_size=entry.band_contract.patch_size,
        ckpt_path=Path(_ckpt),
    )
    ctx = InferenceContext(
        sensor=sensor,
        gsd_m=translated.target_resolution_m,
        wavelengths=entry.band_contract.sensors[sensor].wavelengths,
    )
    out = backend.embed(chips, ctx)

    assert out.shape == (len(chipset.chips), 1024)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_real_clay_metadata_conditions_embedding(
    whirld_home: Path, s2_10band_tif: Path
) -> None:
    """Supplying time/lat-lon metadata changes the real Clay embedding.

    Proves the metadata is actually consumed by the encoder (not a no-op), which
    is the whole point of the Pass-5 fidelity work.
    """
    from datetime import datetime

    from whirld.core.chips import chip_raster
    from whirld.core.contract import translate
    from whirld.core.registry import Registry
    from whirld.core.sensor import detect_sensor
    from whirld.io.raster import read_raster
    from whirld.models.base import InferenceContext
    from whirld.models.clay_torch import ClayTorchBackend

    entry = Registry().get("clay-v1.5")
    raster = read_raster(s2_10band_tif)
    sensor = detect_sensor(raster, entry)
    translated = translate(raster, entry, sensor)
    chips, _ = chip_raster(
        translated,
        entry.band_contract.chip_size_px,
        nodata_fill=entry.band_contract.nodata_fill,
    )
    backend = ClayTorchBackend(
        name="clay-v1.5",
        version="1.5.0",
        device="cpu",
        embed_dim=1024,
        patch_size=8,
        ckpt_path=Path(_ckpt),
    )
    waves = entry.band_contract.sensors[sensor].wavelengths

    zeros_ctx = InferenceContext(sensor=sensor, gsd_m=10.0, wavelengths=waves)
    meta_ctx = InferenceContext(
        sensor=sensor,
        gsd_m=10.0,
        wavelengths=waves,
        latlons=[(51.5, -0.1)] * len(chips),
        acquisition_datetime=datetime(2024, 6, 1, 10, 0),
    )
    no_meta = backend.embed(chips, zeros_ctx)
    with_meta = backend.embed(chips, meta_ctx)
    assert not np.allclose(no_meta, with_meta)
