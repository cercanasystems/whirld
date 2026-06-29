"""Unit tests for the Prithvi segmentation backend using a fake model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from whirld.core.registry import Registry  # noqa: E402
from whirld.errors import WhirldError  # noqa: E402
from whirld.models.prithvi import PrithviBackend  # noqa: E402


class _FakeSegModel:
    """Returns logits (B, 2, H, W); class-1 logit = per-chip mean (content-driven)."""

    def __call__(self, x):  # type: ignore[no-untyped-def]  # x: (B, C, T, H, W)
        b, _c, _t, h, w = x.shape
        base = x.mean(dim=(1, 2, 3, 4))  # (B,)
        logits = torch.zeros(b, 2, h, w)
        logits[:, 1, :, :] = base.view(b, 1, 1)
        return logits

    def to(self, _device):  # type: ignore[no-untyped-def]
        return self

    def eval(self):  # type: ignore[no-untyped-def]
        return self


def _backend() -> PrithviBackend:
    return PrithviBackend(
        name="prithvi-burn-scar",
        version="2.0.0",
        device="cpu",
        classes=2,
        model=_FakeSegModel(),
    )


def _chips(n: int = 3, tile: int = 64, fill: float = 1.0) -> np.ndarray:
    return np.full((n, 6, tile, tile), fill, dtype=np.float32)


def test_segment_shape_and_dtype() -> None:
    """Segment returns (n, tile, tile) uint8 masks with class values in {0,1}."""
    out = _backend().segment(_chips(3, 64))
    assert out.shape == (3, 64, 64)
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 1})


def test_segment_varies_with_content() -> None:
    """Positive-mean chips → class 1; negative-mean chips → class 0 (argmax)."""
    pos = _chips(1, 32, fill=1.0)
    neg = _chips(1, 32, fill=-1.0)
    out = _backend().segment(np.concatenate([pos, neg]))
    assert out[0].mean() == 1.0  # all burn
    assert out[1].mean() == 0.0  # all unburned


def test_segment_threshold() -> None:
    """A high threshold suppresses weak positives (vs argmax at 0.5)."""
    chips = _chips(1, 16, fill=0.1)  # weak class-1 logit
    argmaxed = _backend().segment(chips, threshold=0.5)
    strict = _backend().segment(chips, threshold=0.99)
    assert argmaxed[0].mean() == 1.0  # softmax([0,0.1]) → class1 wins at argmax
    assert strict[0].mean() == 0.0  # but below the 0.99 probability bar


def test_segment_empty_batch() -> None:
    """An empty batch returns a (0, tile, tile) array."""
    out = _backend().segment(np.empty((0, 6, 32, 32), dtype=np.float32))
    assert out.shape == (0, 32, 32)


def test_segment_rejects_bad_ndim() -> None:
    """A non-4D input is rejected."""
    with pytest.raises(ValueError):
        _backend().segment(np.ones((6, 32, 32), dtype=np.float32))


def test_embed_declines() -> None:
    """Prithvi declines embedding (segmentation model)."""
    with pytest.raises(WhirldError):
        _backend().embed(_chips(1, 16))


def test_registry_entry(whirld_home: Path) -> None:
    """The prithvi-burn-scar registry entry carries the expected values."""
    entry = Registry().get("prithvi-burn-scar")
    assert entry.category == "segmentation"
    assert entry.config_file == "burn_scars_config.yaml"
    assert entry.output.type == "mask"
    assert entry.output.format == "geotiff"
    assert entry.output.classes == 2
    sensor = entry.band_contract.sensors["hls"]
    assert sensor.aliases == ["blue", "green", "red", "nir_narrow", "swir1", "swir2"]
    assert entry.band_contract.chip_size_px == 512
    assert entry.source.files[0].endswith(".pt")
    assert entry.source.repo == "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
