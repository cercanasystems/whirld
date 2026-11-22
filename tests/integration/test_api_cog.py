"""Public API direct COG execution through real HTTP and preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

import whirld
from tests.integration.io.test_cog_input import BANDS, CogScene, CogServer
from whirld.core.fetch import Manifest, ModelStore
from whirld.core.registry import ModelEntry
from whirld.errors import InvalidInputError
from whirld.models.base import InferenceContext, ModelBackend


class DeterministicBackend(ModelBackend):
    """Small test backend retaining real input/normalization/output orchestration."""

    classes = 2

    def embed(
        self, chips: np.ndarray, context: InferenceContext | None = None
    ) -> np.ndarray:
        """Reject embedding calls; the reference Clay backend handles those tests."""
        raise NotImplementedError

    def classify(
        self,
        chips: np.ndarray,
        queries: list[str],
        context: InferenceContext | None = None,
    ) -> np.ndarray:
        """Return normalized deterministic scores for prepared three-band chips."""
        assert chips.shape[1] == 3
        return np.full((len(chips), len(queries)), 1 / len(queries), dtype=np.float32)

    def segment(
        self,
        chips: np.ndarray,
        head: str | None = None,
        threshold: float = 0.5,
        context: InferenceContext | None = None,
    ) -> np.ndarray:
        """Produce a binary pixel mask from the real normalized six-band input."""
        assert chips.shape[1] == 6
        return (chips[:, 0] > 0).astype(np.uint8)


def patch_nonembedding_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace checkpoint seams, preserving actual COG readers and model contracts."""

    def manifest(store: ModelStore, name: str) -> Manifest:
        """Return a valid identity without downloading optional model weights."""
        entry = store.registry.get(name)
        return Manifest(
            name=name,
            version=entry.version,
            source_type="huggingface",
            sha256="0" * 64,
            weights_file="fixture.pt",
        )

    def load(
        entry: ModelEntry,
        manifest: Manifest,
        device: str | None = None,
        *,
        store: ModelStore | None = None,
    ) -> ModelBackend:
        """Build the deterministic inference owner for this model identity."""
        return DeterministicBackend(entry.name, manifest.version, "cpu")

    monkeypatch.setattr(ModelStore, "load_manifest", manifest)
    monkeypatch.setattr("whirld.api.load_backend", load)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_direct_eager_stream_and_local_embeddings_agree(
    whirld_home: Path,
    tmp_path: Path,
    batch_size: int,
) -> None:
    """Direct URL reads preserve local pixels, chip geometry, metadata and outputs."""
    whirld.pull("clay-v1")
    scene = CogScene(tmp_path / "scene")
    scene.create()
    local = whirld.embed(
        scene.path,
        model="clay-v1",
        sensor="sentinel-2-l2a",
        chip_size=128,
        overlap=17,
        device="cpu",
        write=False,
    )
    with CogServer(scene) as server:
        eager = whirld.embed(
            server.url + "?signature=fixture",
            model="clay-v1",
            sensor="sentinel-2-l2a",
            chip_size=128,
            overlap=17,
            device="cpu",
            write=False,
        )
        disk = whirld.embed_to_disk(
            server.url + "?signature=fixture",
            model="clay-v1",
            sensor="sentinel-2-l2a",
            chip_size=128,
            overlap=17,
            batch_size=batch_size,
            device="cpu",
            output=tmp_path / "stream.npy",
        )
    np.testing.assert_allclose(eager.embeddings, local.embeddings, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        np.load(disk.output_path), eager.embeddings, rtol=1e-5, atol=1e-6
    )
    assert eager.chips == local.chips
    metadata = json.loads(disk.meta_path.read_text())
    assert {k: v for k, v in metadata.items() if k != "timestamp"} == {
        k: v for k, v in eager.meta.items() if k != "timestamp"
    }
    usage = json.loads((whirld_home / "logs/usage.jsonl").read_text().splitlines()[-1])
    assert usage["input_type"] == "cog"
    assert usage["error"] is None


def test_explicit_metadata_selects_extensionless_unnamed_cog(
    whirld_home: Path,
    tmp_path: Path,
) -> None:
    """Source-position names and reflectance overrides work without URL extensions."""
    whirld.pull("clay-v1")
    scene = CogScene(tmp_path / "unnamed", named=False)
    scene.create()
    with CogServer(scene, transformed=True) as server:
        actual = whirld.embed(
            server.origin + "/download?signature=fixture",
            model="clay-v1",
            sensor="sentinel-2-l2a",
            device="cpu",
            write=False,
            cog=whirld.CogOptions(bands=BANDS, scale=0.0001, offset=-0.1),
        )
        expected = whirld.embed(
            server.origin + "/item.json",
            model="clay-v1",
            sensor="sentinel-2-l2a",
            device="cpu",
            write=False,
        )
    np.testing.assert_allclose(
        actual.embeddings, expected.embeddings, rtol=1e-5, atol=1e-6
    )


def test_direct_classification_and_segmentation_use_shared_input(
    whirld_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both nonembedding APIs accept direct URLs while retaining spatial outputs."""
    patch_nonembedding_backends(monkeypatch)
    scene = CogScene(tmp_path / "nonembedding")
    scene.create()
    with CogServer(scene) as server:
        classified = whirld.classify(
            server.url,
            model="remoteclip",
            query=["water", "forest"],
            sensor="sentinel-2-l2a",
            device="cpu",
            top_k=10,
        )
        segmented = whirld.segment(
            server.url,
            model="prithvi-burn-scar",
            sensor="hls",
            device="cpu",
            cog=whirld.CogOptions(
                bands=tuple("B8A" if b == "B08" else b for b in BANDS)
            ),
            output=tmp_path / "mask.tif",
        )
    assert len(classified.feature_collection["features"]) == 4
    np.testing.assert_allclose(classified.scores.sum(axis=1), 1)
    assert segmented.mask.shape == (100, 100)
    assert set(np.unique(segmented.mask)).issubset({0, 1, 255})
    with rasterio.open(segmented.output_path) as mask:
        assert mask.crs.to_string() == "EPSG:32630"
        assert mask.transform.a == 30
    usage = [
        json.loads(line)
        for line in (whirld_home / "logs/usage.jsonl").read_text().splitlines()
    ]
    assert [record["input_type"] for record in usage] == ["cog", "cog"]


def test_signed_url_never_enters_default_filename_or_error(
    whirld_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Output naming and failed direct reads do not disclose URL query credentials."""
    whirld.pull("clay-v1")
    scene = CogScene(tmp_path / "signed")
    scene.create()
    monkeypatch.chdir(tmp_path)
    secret = "signed-query-must-not-leak"
    with CogServer(scene) as server:
        result = whirld.embed(
            server.url + "?token=" + secret,
            model="clay-v1",
            sensor="sentinel-2-l2a",
            device="cpu",
        )
        assert result.output_path is not None
        assert result.output_path.name == "scene_embeddings.npy"
        with pytest.raises(InvalidInputError) as error:
            whirld.embed(
                server.origin + "/missing.tif?token=" + secret,
                model="clay-v1",
                sensor="sentinel-2-l2a",
                device="cpu",
                write=False,
            )
    assert secret not in str(error.value)
    assert secret not in caplog.text
    for log in (whirld_home / "logs").glob("*"):
        assert secret not in log.read_text()


@pytest.mark.parametrize(
    "value", ["local.tif", "item.json", "https://example.invalid/item.json"]
)
def test_cog_options_are_not_silently_ignored(value: str) -> None:
    """Explicit direct metadata rejects incompatible input kinds before loading."""
    with pytest.raises(InvalidInputError):
        whirld.embed(value, model="clay-v1", cog=whirld.CogOptions(scale=1.0))


def test_direct_input_requires_explicit_sensor() -> None:
    """An unreachable direct URL fails sensor validation before network/model I/O."""
    with pytest.raises(InvalidInputError, match="sensor"):
        whirld.embed_to_disk("https://example.invalid/scene.tif", model="clay-v1")
