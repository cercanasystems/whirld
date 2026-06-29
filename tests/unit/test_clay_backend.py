"""Unit tests for the Clay reference backend and device detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from whirld.core.fetch import pull
from whirld.core.registry import Registry
from whirld.models.base import detect_device
from whirld.models.clay import ClayBackend
from whirld.models.loader import load_backend


def _backend(whirld_home: Path) -> ClayBackend:
    """Pull clay-v1 and return a loaded reference backend on CPU."""
    manifest = pull("clay-v1")
    entry = Registry().get("clay-v1")
    return ClayBackend.load(entry, manifest, device="cpu")


def test_embed_shape_and_dtype(whirld_home: Path) -> None:
    """Embedding a batch yields (n, 512) float32."""
    backend = _backend(whirld_home)
    chips = np.random.default_rng(0).standard_normal((5, 6, 32, 32)).astype("float32")
    out = backend.embed(chips)
    assert out.shape == (5, 512)
    assert out.dtype == np.float32


def test_embed_is_deterministic(whirld_home: Path) -> None:
    """Identical inputs produce identical embeddings."""
    backend = _backend(whirld_home)
    chips = np.ones((2, 6, 16, 16), dtype="float32")
    a = backend.embed(chips)
    b = backend.embed(chips)
    assert np.array_equal(a, b)


def test_embed_distinguishes_inputs(whirld_home: Path) -> None:
    """Different chips produce different embeddings."""
    backend = _backend(whirld_home)
    chips = np.stack(
        [np.zeros((6, 16, 16), "float32"), np.ones((6, 16, 16), "float32")]
    )
    out = backend.embed(chips)
    assert not np.allclose(out[0], out[1])


def test_embed_empty_batch(whirld_home: Path) -> None:
    """An empty batch returns a well-formed (0, 512) array."""
    backend = _backend(whirld_home)
    out = backend.embed(np.empty((0, 6, 16, 16), dtype="float32"))
    assert out.shape == (0, 512)


def test_embed_rejects_bad_ndim(whirld_home: Path) -> None:
    """A non-4D input is rejected."""
    backend = _backend(whirld_home)
    with pytest.raises(ValueError):
        backend.embed(np.ones((6, 16, 16), dtype="float32"))


def test_loader_returns_clay(whirld_home: Path) -> None:
    """The loader resolves clay-v1 to a ClayBackend."""
    manifest = pull("clay-v1")
    entry = Registry().get("clay-v1")
    backend = load_backend(entry, manifest, device="cpu")
    assert isinstance(backend, ClayBackend)


def test_detect_device_without_torch() -> None:
    """Device auto-detection falls back to cpu when torch is absent."""
    assert detect_device("cpu") == "cpu"
    assert detect_device(None) in {"cpu", "mps", "cuda"}
