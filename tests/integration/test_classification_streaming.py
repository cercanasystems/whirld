"""Streaming classification through real HTTP windows, model contracts and GeoJSON."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import whirld
from tests.integration.io.test_cog_input import CogScene, CogServer
from whirld.cli import app
from whirld.core.fetch import Manifest, ModelStore
from whirld.core.registry import ModelEntry
from whirld.errors import InvalidInputError
from whirld.models.base import InferenceContext, ModelBackend


class PixelClassifier(ModelBackend):
    """A batch-independent score function sensitive to real prepared pixels."""

    def __init__(self) -> None:
        """Retain only small call observations for these finite integration tests."""
        super().__init__("remoteclip", "1.0.0", "cpu")
        self.calls: list[int] = []

    def embed(
        self, chips: np.ndarray, context: InferenceContext | None = None
    ) -> np.ndarray:
        """Reject accidental calls to the embedding paradigm."""
        raise AssertionError("classification must not call embed")

    def classify(
        self,
        chips: np.ndarray,
        queries: list[str],
        context: InferenceContext | None = None,
    ) -> np.ndarray:
        """Produce finite deterministic probabilities including single-query scores."""
        self.calls.append(len(chips))
        assert chips.shape[1] == 3
        logits = chips.mean(axis=(1, 2, 3), dtype=np.float64)
        first = (1 / (1 + np.exp(-logits))).astype(np.float32)
        if len(queries) == 1:
            return first[:, None]
        return np.column_stack((first, 1 - first))


@pytest.fixture
def classifier(monkeypatch: pytest.MonkeyPatch) -> PixelClassifier:
    """Patch only installed model seams, keeping real source and output execution."""
    backend = PixelClassifier()

    def manifest(store: ModelStore, name: str) -> Manifest:
        """Resolve registry identity without optional checkpoint downloads."""
        return Manifest(
            name=name,
            version=store.registry.get(name).version,
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
        """Reuse the observed backend across eager and stream invocations."""
        return backend

    monkeypatch.setattr(ModelStore, "load_manifest", manifest)
    monkeypatch.setattr("whirld.api.load_backend", load)
    monkeypatch.setattr("whirld.core.classification_stream.load_backend", load)
    return backend


@pytest.mark.parametrize("queries", [["forest"], ["forest", "water"]])
@pytest.mark.parametrize("top_k,threshold", [(2, 0.0), (None, 0.0), (None, 1.0)])
def test_stream_matches_eager_ranking_pixels_and_geometry(
    whirld_home: Path,
    tmp_path: Path,
    classifier: PixelClassifier,
    queries: list[str],
    top_k: int | None,
    threshold: float,
) -> None:
    """Rank/filter output equals eager for direct and STAC inputs, including edges."""
    scene = CogScene(tmp_path / "scene", size=512)
    scene.create()
    with CogServer(scene, token="fixture-token") as server:
        for url in (server.url + "?signature=fixture", server.origin + "/item.json"):
            eager = whirld.classify(
                url,
                model="remoteclip",
                query=queries,
                top_k=top_k or 10000,
                threshold=threshold,
                sensor="sentinel-2-l2a",
                stac_token="fixture-token",
                device="cpu",
            )
            classifier.calls.clear()
            result = whirld.classify_to_disk(
                url,
                model="remoteclip",
                query=queries,
                top_k=top_k,
                threshold=threshold,
                sensor="sentinel-2-l2a",
                stac_token="fixture-token",
                device="cpu",
                batch_size=2,
                output=tmp_path / "stream.geojson",
            )
            assert json.loads(result.output_path.read_text()) == json.loads(
                json.dumps(eager.feature_collection)
            )
            assert result.chip_count == len(eager.chips) == 9
            assert result.feature_count == len(eager.feature_collection["features"])
            assert classifier.calls == [2, 2, 2, 2, 1]


def test_stream_cli_all_and_top_k(
    whirld_home: Path,
    tmp_path: Path,
    classifier: PixelClassifier,
) -> None:
    """CLI flags reach the public streaming path and preserve eager stdout defaults."""
    scene = CogScene(tmp_path / "cli")
    scene.create()
    output = tmp_path / "matches.geojson"
    with CogServer(scene) as server:
        arguments = [
            "classify",
            server.url,
            "--model",
            "remoteclip",
            "--query",
            "forest",
            "--sensor",
            "sentinel-2-l2a",
            "--device",
            "cpu",
        ]
        streamed = CliRunner().invoke(
            app,
            [
                *arguments,
                "--stream",
                "--all",
                "--batch-size",
                "2",
                "--output",
                str(output),
            ],
        )
        assert streamed.exit_code == 0, streamed.output
        assert "Streamed 4 chips" in streamed.output
        assert len(json.loads(output.read_text())["features"]) == 4
        eager = CliRunner().invoke(app, arguments)
        assert eager.exit_code == 0, eager.output
        assert json.loads(eager.stdout)["type"] == "FeatureCollection"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--stream"],
        ["--all"],
        ["--batch-size", "2"],
        ["--cache-mb", "4"],
        ["--stream", "--all", "--top-k", "2", "--output", "never.geojson"],
        ["--stream", "--batch-size", "0", "--output", "never.geojson"],
    ],
)
def test_cli_incompatible_flags_fail_before_io(arguments: list[str]) -> None:
    """Invalid flag combinations cannot start source or model acquisition."""
    result = CliRunner().invoke(
        app,
        [
            "classify",
            "https://example.invalid/scene.tif",
            "--model",
            "remoteclip",
            "--query",
            "forest",
            "--sensor",
            "sentinel-2-l2a",
            *arguments,
        ],
    )
    assert result.exit_code == 7, result.output


@pytest.mark.parametrize(
    "options",
    [{"batch_size": 0}, {"cache_mb": 0}, {"top_k": 0}, {"threshold": float("nan")}],
)
def test_invalid_stream_options_fail_before_installation(
    options: dict[str, object], tmp_path: Path
) -> None:
    """Public disk API checks its domains even when no model is installed."""
    with pytest.raises(InvalidInputError):
        whirld.classify_to_disk(
            "https://example.invalid/scene.tif",
            model="remoteclip",
            query="forest",
            sensor="sentinel-2-l2a",
            output=tmp_path / "never.geojson",
            **options,
        )


def test_classification_rss_plateaus_as_scene_and_output_grow(tmp_path: Path) -> None:
    """Fresh workers classify actual remote windows and externally sort every result."""
    worker = """
import json,resource,sys
from pathlib import Path
import numpy as np
import whirld
from whirld.core.fetch import Manifest,ModelStore
from whirld.models.base import ModelBackend
import whirld.core.classification_stream as stream
class Classifier(ModelBackend):
 def embed(self,chips,context=None): raise AssertionError()
 def classify(self,chips,queries,context=None):
  return np.full((len(chips),len(queries)),0.5,dtype=np.float32)
ModelStore.load_manifest=lambda self,name: Manifest(
 name=name,version='1.0.0',source_type='huggingface',sha256='0'*64,
 weights_file='fixture.pt')
stream.load_backend=lambda entry,manifest,device=None,**kw: Classifier(
 entry.name,manifest.version,'cpu')
r=whirld.classify_to_disk(sys.argv[1],model='remoteclip',query=['forest','water'],sensor='sentinel-2-l2a',output=sys.argv[2],top_k=None,batch_size=4,cache_mb=8,device='cpu')
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
sys.stdout.write(json.dumps({'count':r.feature_count,
 'rss':int(peak if sys.platform=='darwin' else peak*1024),
 'bytes':r.output_path.stat().st_size}))
"""
    observations = []
    for size in (512, 2048, 4096):
        scene = CogScene(tmp_path / str(size), size=size)
        scene.create()
        with CogServer(scene) as server:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    worker,
                    server.url,
                    str(scene.directory / "all.geojson"),
                ],
                env={
                    **os.environ,
                    "WHIRLD_HOME": str(scene.directory / "home"),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        value = json.loads(result.stdout)
        assert value["count"] == ((size + 223) // 224) ** 2
        observations.append(value)
    assert observations[2]["bytes"] > 3 * observations[1]["bytes"]
    assert observations[1]["rss"] - observations[0]["rss"] < 64 * 1024**2, observations
    assert observations[2]["rss"] - observations[1]["rss"] < 32 * 1024**2, observations


def test_fractional_bbox_and_explicit_radiometry_match_eager(
    whirld_home: Path,
    tmp_path: Path,
    classifier: PixelClassifier,
) -> None:
    """Fractional crops preserve normalized values, nodata and edge footprints."""
    from rasterio.warp import transform_bounds

    scene = CogScene(tmp_path / "cropped", size=512)
    scene.create()
    bbox = transform_bounds("EPSG:32630", "EPSG:4326", 500015, 4496005, 504005, 4499985)
    options = whirld.CogOptions(scale=0.0001, offset=-0.1)
    with CogServer(scene) as server:
        eager = whirld.classify(
            server.url,
            model="remoteclip",
            query=["water", "land"],
            top_k=100,
            sensor="sentinel-2-l2a",
            device="cpu",
            bbox=bbox,
            cog=options,
        )
        streamed = whirld.classify_to_disk(
            server.url,
            model="remoteclip",
            query=["water", "land"],
            top_k=None,
            sensor="sentinel-2-l2a",
            device="cpu",
            bbox=bbox,
            cog=options,
            batch_size=3,
            output=tmp_path / "cropped.geojson",
        )
    assert json.loads(streamed.output_path.read_text()) == json.loads(
        json.dumps(eager.feature_collection)
    )
    assert streamed.chip_count == len(eager.chips) == 4


@pytest.mark.parametrize(
    "location,sensor,query,bbox",
    [
        ("local.tif", None, "water", (-1.0, 50.0, 0.0, 51.0)),
        ("https://example.invalid/scene.tif", None, "water", None),
        ("https://example.invalid/scene.tif", "sentinel-2-l2a", " ", None),
    ],
)
def test_input_and_query_domains_fail_before_io(
    location: str,
    sensor: str | None,
    query: str,
    bbox: tuple[float, float, float, float] | None,
    tmp_path: Path,
) -> None:
    """Reject local crops, missing remote sensor identity and empty queries early."""
    with pytest.raises(InvalidInputError):
        whirld.classify_to_disk(
            location,
            model="remoteclip",
            query=query,
            sensor=sensor,
            bbox=bbox,
            output=tmp_path / "never.geojson",
        )
