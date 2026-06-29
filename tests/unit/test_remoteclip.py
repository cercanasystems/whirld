"""Unit tests for the RemoteCLIP backend using a fake model (no 605 MB download)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from whirld.core.registry import Registry  # noqa: E402
from whirld.errors import WhirldError  # noqa: E402
from whirld.models.remoteclip import RemoteCLIPBackend  # noqa: E402

_WAVES = [0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]


class _FakeModel:
    """A stand-in CLIP model with distinct per-prompt text directions + logit_scale.

    Image features = per-channel mean+std (non-degenerate even for zero chips);
    text features = distinct one-hot directions per prompt, so the softmax is
    meaningful and varies with both image content and the prompt.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.logit_scale = torch.tensor(0.0)  # exp() = 1.0

    def encode_image(self, pixels):  # type: ignore[no-untyped-def]
        m = pixels.mean(dim=(2, 3))
        s = pixels.std(dim=(2, 3)) + 1e-3
        feat = torch.cat([m, s], dim=1)
        return torch.nn.functional.pad(feat, (0, self.dim - feat.shape[1]))

    def encode_text(self, tokens):  # type: ignore[no-untyped-def]
        n = tokens.shape[0]
        eye = torch.zeros(n, self.dim)
        for i in range(n):
            eye[i, i % self.dim] = 1.0
        return eye

    def to(self, _device):  # type: ignore[no-untyped-def]
        return self

    def eval(self):  # type: ignore[no-untyped-def]
        return self


def _fake_tokenizer(texts):  # type: ignore[no-untyped-def]
    return torch.zeros(len(texts), 77, dtype=torch.long)


def _backend() -> RemoteCLIPBackend:
    return RemoteCLIPBackend(
        name="remoteclip",
        version="1.0.0",
        device="cpu",
        arch="ViT-B-32",
        model=_FakeModel(),
        tokenizer=_fake_tokenizer,
    )


def _chips(n: int = 4, size: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3, size, size)).astype(np.float32)


def test_classify_returns_probabilities() -> None:
    """Scores are (n, n_queries) probabilities in [0, 1]."""
    out = _backend().classify(_chips(4), ["solar farm"])
    assert out.shape == (4, 1)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert ((out >= 0) & (out <= 1)).all()


def test_classify_multi_query_softmax() -> None:
    """With multiple queries, each chip's per-query probabilities sum to ~1."""
    out = _backend().classify(_chips(5), ["solar farm", "forest", "airport"])
    assert out.shape == (5, 3)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)


def test_classify_is_deterministic() -> None:
    """Identical inputs give identical scores."""
    backend = _backend()
    chips = _chips(3)
    assert np.array_equal(
        backend.classify(chips, ["x"]), backend.classify(chips, ["x"])
    )


def test_classify_varies_with_content() -> None:
    """Different chips produce different scores."""
    # Two distinct content patterns: one biased toward band 0, one toward band 1.
    a = np.zeros((3, 16, 16), "float32")
    a[0] = 5.0
    b = np.zeros((3, 16, 16), "float32")
    b[1] = 5.0
    out = _backend().classify(np.stack([a, b]), ["x"])
    assert out[0, 0] != out[1, 0]


def test_classify_empty_batch() -> None:
    """An empty batch returns a (0, n_queries) array."""
    out = _backend().classify(np.empty((0, 3, 16, 16), dtype="float32"), ["x"])
    assert out.shape == (0, 1)


def test_classify_empty_query_rejected() -> None:
    """An all-blank query list raises WhirldError."""
    with pytest.raises(WhirldError):
        _backend().classify(_chips(), ["   "])


def test_classify_rejects_bad_ndim() -> None:
    """A non-4D input is rejected."""
    with pytest.raises(ValueError):
        _backend().classify(np.ones((3, 16, 16), dtype="float32"), ["x"])


def test_registry_entry(whirld_home: Path) -> None:
    """The remoteclip registry entry carries the expected real values."""
    entry = Registry().get("remoteclip")
    assert entry.category == "classification"
    assert entry.model_name == "ViT-B-32"
    assert entry.output.format == "geojson"
    sensor = entry.band_contract.sensors["sentinel-2-l2a"]
    assert sensor.aliases == ["red", "green", "blue"]
    assert entry.band_contract.chip_size_px == 224
    assert entry.source.repo == "chendelong/RemoteCLIP"
