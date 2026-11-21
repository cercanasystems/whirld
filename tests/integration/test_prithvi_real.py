"""Gated integration test for REAL Prithvi burn-scar weights.

Skipped unless a real checkpoint + config are available, so the normal suite never
needs the 1.3 GB download or TerraTorch. To run it, set both::

    WHIRLD_TEST_PRITHVI_CKPT=/path/to/Prithvi_EO_V2_300M_BurnScars.pt \
    WHIRLD_TEST_PRITHVI_CONFIG=/path/to/burn_scars_config.yaml \
        pytest tests/integration/test_prithvi_real.py
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("terratorch")

_CKPT = os.environ.get("WHIRLD_TEST_PRITHVI_CKPT")
_CONFIG = os.environ.get("WHIRLD_TEST_PRITHVI_CONFIG")

pytestmark = pytest.mark.skipif(
    not _CKPT or not _CONFIG or not Path(_CKPT).exists() or not Path(_CONFIG).exists(),
    reason="set WHIRLD_TEST_PRITHVI_CKPT + _CONFIG to run real Prithvi tests",
)


def test_real_prithvi_segment(
    whirld_home: Path, hls_tif: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real installed weights segment offline, without a cached base backbone."""
    from huggingface_hub import constants as hf_constants

    cache = tmp_path / "empty-hf-cache"
    cache.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(cache))

    def deny_network(*args: object, **kwargs: object) -> None:
        """Fail if initialization or inference tries to contact a remote host."""
        pytest.fail("Real Prithvi inference attempted network access")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    from whirld.core.chips import chip_raster, reassemble_mask
    from whirld.core.contract import translate
    from whirld.core.registry import Registry
    from whirld.core.sensor import detect_sensor
    from whirld.io.raster import read_raster
    from whirld.models.prithvi import PrithviBackend

    entry = Registry().get("prithvi-burn-scar")
    raster = read_raster(hls_tif)
    sensor = detect_sensor(raster, entry)
    assert sensor == "hls"

    translated = translate(raster, entry, sensor)
    chips, chipset = chip_raster(
        translated,
        entry.band_contract.chip_size_px,
        nodata_fill=entry.band_contract.nodata_fill,
    )

    backend = PrithviBackend(
        name="prithvi-burn-scar",
        version="2.0.0",
        device="cpu",
        classes=entry.output.classes,
        config_path=Path(_CONFIG),
        ckpt_path=Path(_CKPT),
        trusted_checkpoint=True,  # Explicit opt-in to a supplied curated checkpoint.
    )
    masks = backend.segment(chips)
    assert masks.dtype == np.uint8
    assert set(np.unique(masks)).issubset({0, 1})

    mask = reassemble_mask(masks, chipset, *translated.data.shape[1:])
    assert mask.shape == translated.data.shape[1:]
    assert mask.dtype == np.uint8
