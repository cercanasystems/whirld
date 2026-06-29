"""Integration tests for the classify pipeline (Python API + CLI).

Uses a fake backend injected via monkeypatch so the full pipeline (registry →
raster → sensor → RGB band contract → 224 chips → score → GeoJSON) runs without
the 605 MB RemoteCLIP download.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import whirld  # noqa: E402
from whirld.core.fetch import Manifest  # noqa: E402
from whirld.models.remoteclip import RemoteCLIPBackend  # noqa: E402


class _FakeModel:
    """Distinct per-prompt text directions + logit_scale; non-degenerate image feats."""

    dim = 8

    def __init__(self) -> None:
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


@pytest.fixture
def fake_remoteclip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch api.load_manifest + load_backend to use a fake RemoteCLIP backend."""

    def fake_manifest(name, paths=None):  # type: ignore[no-untyped-def]
        return Manifest(
            name=name,
            version="1.0.0",
            source_type="huggingface",
            sha256="0" * 64,
            weights_file="RemoteCLIP-ViT-B-32.pt",
        )

    def fake_load_backend(entry, manifest, device=None):  # type: ignore[no-untyped-def]
        return RemoteCLIPBackend(
            name=entry.name,
            version=manifest.version,
            device="cpu",
            arch=entry.model_name or "ViT-B-32",
            model=_FakeModel(),
            tokenizer=_fake_tokenizer,
        )

    monkeypatch.setattr("whirld.api.load_manifest", fake_manifest)
    monkeypatch.setattr("whirld.api.load_backend", fake_load_backend)


def test_classify_api(whirld_home: Path, s2_tif: Path, fake_remoteclip: None) -> None:
    """classify() returns a GeoJSON FeatureCollection of per-chip scores."""
    result = whirld.classify(
        s2_tif, model="remoteclip", query="solar farm", top_k=10, threshold=-1.0
    )
    fc = result.feature_collection
    assert fc["type"] == "FeatureCollection"
    assert fc["query"] == "solar farm"
    assert result.sensor == "sentinel-2-l2a"
    assert len(fc["features"]) == len(result.scores)
    # Features are sorted by descending score.
    scores = [f["properties"]["score"] for f in fc["features"]]
    assert scores == sorted(scores, reverse=True)
    assert fc["features"][0]["geometry"]["type"] == "Polygon"
    # Scores are probabilities in [0, 1].
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_classify_multi_query(
    whirld_home: Path, s2_tif: Path, fake_remoteclip: None
) -> None:
    """Multiple --query values produce per-query softmax scores summing to ~1."""
    result = whirld.classify(
        s2_tif,
        model="remoteclip",
        query=["solar farm", "forest", "airport"],
        top_k=10,
        threshold=-1.0,
    )
    assert result.queries == ["solar farm", "forest", "airport"]
    feat = result.feature_collection["features"][0]
    per_query = feat["properties"]["scores"]
    assert set(per_query) == {"solar farm", "forest", "airport"}
    assert abs(sum(per_query.values()) - 1.0) < 1e-4


def test_classify_top_k(whirld_home: Path, s2_tif: Path, fake_remoteclip: None) -> None:
    """top_k truncates the feature list."""
    result = whirld.classify(
        s2_tif, model="remoteclip", query="x", top_k=2, threshold=-1.0
    )
    assert len(result.feature_collection["features"]) == 2


def test_classify_writes_geojson(
    whirld_home: Path, s2_tif: Path, tmp_path: Path, fake_remoteclip: None
) -> None:
    """An --output path writes a parseable GeoJSON file."""
    out = tmp_path / "matches.geojson"
    result = whirld.classify(s2_tif, model="remoteclip", query="x", output=out)
    assert result.output_path == out
    loaded = json.loads(out.read_text())
    assert loaded["type"] == "FeatureCollection"


def test_classify_usage_record(
    whirld_home: Path, s2_tif: Path, fake_remoteclip: None
) -> None:
    """A classify usage record is written (command=classify, no PII)."""
    whirld.classify(s2_tif, model="remoteclip", query="solar farm")
    usage = whirld_home / "logs" / "usage.jsonl"
    last = json.loads(usage.read_text().strip().splitlines()[-1])
    assert last["command"] == "classify"
    assert last["model"] == "remoteclip"
    assert last["error"] is None


def test_classify_cli(
    whirld_home: Path, s2_tif: Path, tmp_path: Path, fake_remoteclip: None
) -> None:
    """`whirld classify` writes GeoJSON and reports the match count."""
    from typer.testing import CliRunner

    from whirld.cli import app

    out = tmp_path / "out.geojson"
    result = CliRunner().invoke(
        app,
        [
            "classify",
            str(s2_tif),
            "--model",
            "remoteclip",
            "--query",
            "solar farm",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Classified" in result.output
    assert out.exists()
