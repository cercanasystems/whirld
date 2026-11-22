"""Public streaming masks preserve eager pixels, source validity and output grids."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.shutil import copy as copy_raster
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from typer.testing import CliRunner

import whirld
from tests.integration.io.test_cog_input import CogScene, CogServer
from whirld.cli import app
from whirld.core.fetch import Manifest, ModelStore
from whirld.core.registry import ModelEntry
from whirld.errors import InvalidInputError
from whirld.models.base import InferenceContext, ModelBackend

BANDS = ("B12", "B11", "B8A", "B04", "B03", "B02")


class SegmentationScene(CogScene):
    """Own an actual six-band COG with non-square extent and validity holes."""

    def __init__(
        self, directory: Path, resolution: int = 10, width: int = 700, height: int = 600
    ) -> None:
        """Bind a native-grid fixture so partial-edge chips require cropped writes."""
        super().__init__(directory, height)
        self.width, self.height, self.resolution = width, height, resolution

    def create(self) -> Path:
        """Generate source blocks and internal masks without whole-scene arrays."""
        self.directory.mkdir(parents=True, exist_ok=True)
        raw = self.directory / "raw.tif"
        with rasterio.Env(GDAL_CACHEMAX=8 * 1024**2, GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(
                raw,
                "w",
                driver="GTiff",
                width=self.width,
                height=self.height,
                count=6,
                dtype="uint16",
                crs="EPSG:32630",
                nodata=0,
                transform=from_origin(
                    500000, 4500000, self.resolution, self.resolution
                ),
                tiled=True,
                blockxsize=128,
                blockysize=128,
            ) as source:
                rng = np.random.default_rng(381)
                source.descriptions = BANDS
                for _, window in source.block_windows(1):
                    pixels = rng.integers(
                        100,
                        9000,
                        size=(6, int(window.height), int(window.width)),
                        dtype=np.uint16,
                    )
                    mask = np.full(pixels.shape[1:], 255, dtype=np.uint8)
                    if window.row_off == 0 and window.col_off == 0:
                        pixels[:, 2:5, 3:8] = 0
                        mask[10:15, 12:18] = 0
                    source.write(pixels, window=window)
                    source.write_mask(mask, window=window)
            copy_raster(
                raw,
                self.path,
                driver="COG",
                BLOCKSIZE=128,
                COMPRESS="NONE",
                OVERVIEWS="NONE",
            )
        raw.unlink()
        return self.path

    def document(
        self, url: str, *, transformed: bool = False, per_band: bool = False
    ) -> bytes:
        """Name narrow-NIR correctly at the fixture's external STAC boundary."""
        item = json.loads(
            super().document(url, transformed=transformed, per_band=per_band)
        )
        item["assets"]["image"]["eo:bands"] = [{"name": name} for name in BANDS]
        return json.dumps(item).encode()


class PixelSegmenter(ModelBackend):
    """Batch-independent segmentation sensitive to actual normalized source pixels."""

    classes = 2

    def __init__(self) -> None:
        """Retain small call observations for finite functional checks."""
        super().__init__("prithvi-burn-scar", "2.0.0", "cpu")
        self.calls: list[int] = []
        self.interrupt = False

    def embed(
        self, chips: np.ndarray, context: InferenceContext | None = None
    ) -> np.ndarray:
        """Reject accidental use of embedding inference."""
        raise AssertionError("segment must not call embed")

    def segment(
        self,
        chips: np.ndarray,
        head: str | None = None,
        threshold: float = 0.5,
        context: InferenceContext | None = None,
    ) -> np.ndarray:
        """Use the first model-order band; no dependence on batch composition."""
        self.calls.append(len(chips))
        if self.interrupt:
            raise KeyboardInterrupt()
        assert chips.shape[1] == 6
        return (chips[:, 0] > threshold).astype(np.uint8)


@pytest.fixture
def segmenter(monkeypatch: pytest.MonkeyPatch) -> PixelSegmenter:
    """Replace model-install seams while keeping source/translation/output real."""
    backend = PixelSegmenter()

    def manifest(store: ModelStore, name: str) -> Manifest:
        """Supply registry-consistent identity without optional checkpoint downloads."""
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
        """Reuse the observed inference owner for the functional comparison."""
        return backend

    monkeypatch.setattr(ModelStore, "load_manifest", manifest)
    monkeypatch.setattr("whirld.api.load_backend", load)
    monkeypatch.setattr("whirld.core.segmentation_stream.load_backend", load)
    return backend


@pytest.mark.parametrize(
    "model,sensor,resolution",
    [("prithvi-burn-scar", "hls", 30), ("prithvi-flood", "sentinel-2-l1c", 10)],
)
@pytest.mark.parametrize("threshold", [0.2, 0.5, 0.8])
def test_streamed_masks_match_eager_source_validity_and_grid(
    whirld_home: Path,
    tmp_path: Path,
    segmenter: PixelSegmenter,
    model: str,
    sensor: str,
    resolution: int,
    threshold: float,
) -> None:
    """Both tasks retain exact mask bytes, all-band validity, class0 and edge extent."""
    scene = SegmentationScene(tmp_path / "source", resolution)
    scene.create()
    with CogServer(scene, token="fixture-token") as server:
        for url in (server.url + "?signature=fixture", server.origin + "/item.json"):
            eager = whirld.segment(
                url,
                model=model,
                sensor=sensor,
                device="cpu",
                threshold=threshold,
                stac_token="fixture-token",
                output=tmp_path / "eager.tif",
            )
            segmenter.calls.clear()
            result = whirld.segment_to_disk(
                url,
                model=model,
                sensor=sensor,
                device="cpu",
                threshold=threshold,
                stac_token="fixture-token",
                batch_size=3,
                output=tmp_path / "stream.tif",
            )
            assert segmenter.calls == [3, 1]
            assert (
                result.chip_count == 4 and result.width == 700 and result.height == 600
            )
            with rasterio.open(result.output_path) as actual:
                np.testing.assert_array_equal(actual.read(1), eager.mask)
                assert actual.transform == eager.transform and actual.crs == eager.crs
                assert actual.nodata == 255 and actual.is_tiled
            assert set(np.unique(eager.mask)) == {0, 1, 255}
            assert result.valid_pixel_count == int((eager.mask != 255).sum())
            assert result.positive_pixel_count == int(
                ((eager.mask != 255) & (eager.mask > 0)).sum()
            )


def test_fractional_crop_matches_eager(
    whirld_home: Path, tmp_path: Path, segmenter: PixelSegmenter
) -> None:
    """Crop affine/extent and partially masked observations survive streamed writes."""
    scene = SegmentationScene(tmp_path / "crop", resolution=30)
    scene.create()
    bbox = transform_bounds("EPSG:32630", "EPSG:4326", 500015, 4483015, 519005, 4499985)
    with CogServer(scene) as server:
        eager = whirld.segment(
            server.url,
            model="prithvi-burn-scar",
            sensor="hls",
            bbox=bbox,
            device="cpu",
            output=tmp_path / "eager.tif",
        )
        result = whirld.segment_to_disk(
            server.url,
            model="prithvi-burn-scar",
            sensor="hls",
            bbox=bbox,
            device="cpu",
            batch_size=1,
            output=tmp_path / "cropped.tif",
        )
    with rasterio.open(result.output_path) as actual:
        np.testing.assert_array_equal(actual.read(1), eager.mask)
        assert actual.transform == eager.transform
        assert actual.shape == eager.mask.shape


def test_cli_stream_alias_and_old_output_survives_interrupt(
    whirld_home: Path, tmp_path: Path, segmenter: PixelSegmenter
) -> None:
    """CLI task aliases stream successfully; interrupted inference preserves output."""
    scene = SegmentationScene(tmp_path / "cli")
    scene.create()
    output = tmp_path / "mask.tif"
    with CogServer(scene) as server:
        result = CliRunner().invoke(
            app,
            [
                "segment",
                server.url,
                "--model",
                "prithvi-eo-2",
                "--head",
                "flood",
                "--sensor",
                "sentinel-2-l1c",
                "--device",
                "cpu",
                "--stream",
                "--batch-size",
                "2",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Streamed 4 chips" in result.output
        previous = output.read_bytes()
        segmenter.interrupt = True
        with pytest.raises(KeyboardInterrupt):
            whirld.segment_to_disk(
                server.url,
                model="prithvi-flood",
                sensor="sentinel-2-l1c",
                device="cpu",
                output=output,
            )
    assert output.read_bytes() == previous
    assert not list(tmp_path.glob(".whirld-segment-*"))


@pytest.mark.parametrize(
    "arguments",
    [
        ["--stream"],
        ["--batch-size", "2"],
        ["--cache-mb", "8"],
        ["--stream", "--batch-size", "0", "--output", "never.tif"],
    ],
)
def test_invalid_cli_flags_fail_before_io(arguments: list[str]) -> None:
    """Streaming output/memory domains fail without source or model access."""
    result = CliRunner().invoke(
        app,
        [
            "segment",
            "https://example.invalid/scene.tif",
            "--model",
            "prithvi-flood",
            "--sensor",
            "sentinel-2-l1c",
            *arguments,
        ],
    )
    assert result.exit_code == 7, result.output


@pytest.mark.parametrize(
    "options", [{"batch_size": 0}, {"cache_mb": 0}, {"threshold": float("nan")}]
)
def test_invalid_stream_options_fail_early(
    tmp_path: Path, options: dict[str, object]
) -> None:
    """New disk API rejects invalid memory/threshold settings before installation."""
    with pytest.raises(InvalidInputError):
        whirld.segment_to_disk(
            "https://example.invalid/scene.tif",
            model="prithvi-flood",
            sensor="sentinel-2-l1c",
            output=tmp_path / "never.tif",
            **options,
        )


def test_http_segmentation_rss_plateaus_as_outputs_grow(tmp_path: Path) -> None:
    """Fresh workers write larger masks at fixed native tile/batch/cache."""
    worker = """
import json,resource,sys
import numpy as np
import whirld
from whirld.core.fetch import Manifest,ModelStore
from whirld.models.base import ModelBackend
import whirld.core.segmentation_stream as stream
class Segmenter(ModelBackend):
 classes=2
 def embed(self,chips,context=None): raise AssertionError()
 def segment(self,chips,head=None,threshold=0.5,context=None):
  return (chips[:,0]>threshold).astype(np.uint8)
ModelStore.load_manifest=lambda self,name: Manifest(name=name,version='2.0.0',
 source_type='huggingface',sha256='0'*64,weights_file='fixture.pt')
stream.load_backend=lambda entry,manifest,device=None,**kw: Segmenter(
 entry.name,manifest.version,'cpu')
r=whirld.segment_to_disk(sys.argv[1],model='prithvi-burn-scar',sensor='hls',
 output=sys.argv[2],batch_size=2,cache_mb=8,device='cpu')
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
sys.stdout.write(json.dumps({'pixels':r.width*r.height,
 'rss':int(peak if sys.platform=='darwin' else peak*1024),
 'valid':r.valid_pixel_count}))
"""
    observations = []
    for size in (512, 2048, 4096):
        scene = SegmentationScene(
            tmp_path / str(size), resolution=30, width=size, height=size
        )
        scene.create()
        with CogServer(scene) as server:
            process = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    worker,
                    server.url,
                    str(scene.directory / "mask.tif"),
                ],
                env={
                    **os.environ,
                    "WHIRLD_HOME": str(scene.directory / "home"),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
        result = json.loads(process.stdout)
        assert result["pixels"] == size * size and result["valid"] < result["pixels"]
        observations.append(result)
    assert observations[1]["rss"] - observations[0]["rss"] < 64 * 1024**2, observations
    assert observations[2]["rss"] - observations[1]["rss"] < 32 * 1024**2, observations
