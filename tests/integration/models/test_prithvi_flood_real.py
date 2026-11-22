"""Opt-in real flood checkpoint comparison against task-specific raw inference.

Set WHIRLD_TEST_FLOOD_HOME to an existing verified installation and
WHIRLD_TEST_FLOOD_SAMPLE to the projected six-band example from docs/flood-model.md.
No checkpoint or sample is downloaded by the test.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

_HOME = os.environ.get("WHIRLD_TEST_FLOOD_HOME")
_SAMPLE = os.environ.get("WHIRLD_TEST_FLOOD_SAMPLE")
pytestmark = pytest.mark.skipif(
    not _HOME or not _SAMPLE,
    reason="set WHIRLD_TEST_FLOOD_HOME and WHIRLD_TEST_FLOOD_SAMPLE",
)


def test_real_flood_matches_metadata_free_training_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent DN standardization and raw no-metadata logits match backend masks."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("terratorch")
    import yaml

    from whirld.core.chips import chip_raster
    from whirld.core.contract import translate
    from whirld.core.fetch import ModelStore
    from whirld.io.raster import read_raster
    from whirld.models.prithvi import PrithviBackend

    assert _HOME is not None and _SAMPLE is not None
    monkeypatch.setenv("WHIRLD_HOME", _HOME)
    store = ModelStore()
    entry = store.registry.get("prithvi-flood")
    manifest = store.load_manifest("prithvi-flood")
    assert entry.config_file is not None
    config = yaml.safe_load(
        store.paths.model_file(entry.name, entry.config_file).read_text()
    )
    assert config["data"]["init_args"]["use_metadata"] is False

    def deny_network(*args: object, **kwargs: object) -> None:
        """Already installed model/sample must run without backbone bootstrapping."""
        pytest.fail("installed flood inference attempted network access")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    raw = read_raster(Path(_SAMPLE))
    translated = translate(raw, entry, "sentinel-2-l1c")
    actual, layout = chip_raster(translated, 512, nodata_fill=0)
    means = np.array(
        [0.1412956, 0.13795798, 0.12353792, 0.30902815, 0.2044958, 0.11912015],
        dtype=np.float32,
    )[:, None, None]
    stds = np.array(
        [0.07406382, 0.07370365, 0.08692279, 0.11798815, 0.09772074, 0.07659938],
        dtype=np.float32,
    )[:, None, None]
    expected = np.zeros_like(actual)
    independent = (raw.data.astype(np.float32) * 0.0001 - means) / stds
    assert translated.valid_mask is not None
    independent[:, ~translated.valid_mask] = 0
    for chip in layout.chips:
        row, col = chip.row * 512, chip.col * 512
        cut = independent[:, row : row + 512, col : col + 512]
        expected[chip.index, :, : cut.shape[1], : cut.shape[2]] = cut
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        backend = PrithviBackend.load(entry, manifest, "cpu", paths=store.paths)
        model = backend._ensure_model()
        with torch.no_grad():
            output = model(torch.from_numpy(expected).unsqueeze(2))
            logits = cast(Any, getattr(output, "output", output))
            reference = logits.argmax(dim=1).numpy().astype(np.uint8)
        masks = backend.segment(actual)
        np.testing.assert_array_equal(masks, reference)
        assert masks.shape == (len(layout.chips), 512, 512)
        assert set(np.unique(masks)).issubset({0, 1})
    finally:
        torch.set_num_threads(previous_threads)
