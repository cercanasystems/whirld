"""Unit tests for the real Clay v1.5 torch backend.

These exercise the backend's plumbing (datacube assembly, batching, CLS-token
extraction, error handling) with a **tiny** randomly-initialized encoder — no
5 GB checkpoint required. The real-weights path is covered by the gated test in
``tests/integration/test_clay_real.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from whirld.core.registry import Registry  # noqa: E402
from whirld.errors import WhirldError  # noqa: E402
from whirld.models._vendor.clay_v15 import Encoder  # noqa: E402
from whirld.models.base import InferenceContext  # noqa: E402
from whirld.models.clay_torch import ClayTorchBackend  # noqa: E402

_TINY_DIM = 64
_WAVES = [0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]


def _tiny_encoder() -> Encoder:
    """A tiny Clay-architecture encoder for fast, weightless tests."""
    return Encoder(
        mask_ratio=0.0,
        patch_size=8,
        shuffle=False,
        dim=_TINY_DIM,
        depth=2,
        heads=4,
        dim_head=16,
        mlp_ratio=4,
    ).eval()


def _backend() -> ClayTorchBackend:
    """A backend wrapping the tiny encoder (no checkpoint load)."""
    return ClayTorchBackend(
        name="clay-v1.5",
        version="1.5.0",
        device="cpu",
        embed_dim=_TINY_DIM,
        patch_size=8,
        encoder=_tiny_encoder(),
    )


def _chips(n: int = 3, bands: int = 10, size: int = 32) -> np.ndarray:
    """A small normalized chip batch."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, bands, size, size)).astype(np.float32)


def _ctx() -> InferenceContext:
    """Inference context with the real S2 wavelengths and 10 m GSD."""
    return InferenceContext(sensor="sentinel-2-l2a", gsd_m=10.0, wavelengths=_WAVES)


def test_embed_shape_and_finite() -> None:
    """Embedding yields (n, dim) float32, finite, via the CLS token."""
    out = _backend().embed(_chips(n=3), _ctx())
    assert out.shape == (3, _TINY_DIM)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_embed_batches_match_single() -> None:
    """Batching over chips produces the same result as the whole set at once."""
    backend = _backend()
    chips = _chips(n=5)
    backend._batch_size = 2
    batched = backend.embed(chips, _ctx())
    backend._batch_size = 100
    whole = backend.embed(chips, _ctx())
    assert np.allclose(batched, whole, atol=1e-5)


def test_embed_empty_batch() -> None:
    """An empty batch returns a well-formed (0, dim) array."""
    out = _backend().embed(np.empty((0, 10, 32, 32), dtype=np.float32), _ctx())
    assert out.shape == (0, _TINY_DIM)


def test_embed_requires_wavelengths() -> None:
    """Missing wavelength metadata raises a clear WhirldError."""
    ctx = InferenceContext(sensor="sentinel-2-l2a", gsd_m=10.0, wavelengths=None)
    with pytest.raises(WhirldError):
        _backend().embed(_chips(), ctx)


def test_embed_rejects_bad_ndim() -> None:
    """A non-4D input is rejected."""
    with pytest.raises(ValueError):
        _backend().embed(np.ones((10, 32, 32), dtype=np.float32), _ctx())


def test_registry_entry_real_values(whirld_home: Path) -> None:
    """The clay-v1.5 registry entry carries the real (not PRD) values."""
    entry = Registry().get("clay-v1.5")
    assert entry.output.embed_dim == 1024
    assert entry.band_contract.patch_size == 8
    sensor = entry.band_contract.sensors["sentinel-2-l2a"]
    assert len(sensor.bands) == 10
    assert sensor.wavelengths == _WAVES
    assert entry.band_contract.normalization.scale == 1.0
    assert entry.source.type == "huggingface"
    assert entry.source.repo == "made-with-clay/Clay"
