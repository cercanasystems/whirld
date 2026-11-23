"""Local raster disk APIs preserve eager metadata, pixels, scoring and masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import whirld
from whirld.core.fetch import Manifest, ModelStore
from whirld.core.registry import ModelEntry
from whirld.errors import InvalidInputError, UnsupportedSensorError
from whirld.models.base import InferenceContext, ModelBackend

BANDS = ("B12", "B11", "B8A", "B08", "B04", "B03", "B02")


class LocalScene:
    """Own a block-generated local GeoTIFF, including external-mask variants."""

    def __init__(
        self,
        path: Path,
        *,
        width: int = 345,
        height: int = 271,
        resolution: tuple[float, float] = (10.0, 10.0),
        descriptions: tuple[str | None, ...] = BANDS,
        external_mask: bool = False,
        with_crs: bool = True,
        transformed: bool = False,
        nonfinite: bool = False,
    ) -> None:
        """Bind fixture geometry, names and metadata without allocating scene arrays."""
        self.path, self.width, self.height = path, width, height
        self.resolution, self.descriptions = resolution, descriptions
        self.external_mask, self.with_crs = external_mask, with_crs
        self.transformed = transformed
        self.nonfinite = nonfinite

    def create(self) -> Path:
        """Write bounded TIFF blocks with nodata and dataset-mask holes.

        Returns:
            The actual local path; external variants also create a .msk sidecar.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        count = len(self.descriptions)
        with rasterio.Env(
            GDAL_CACHEMAX=8 * 1024**2,
            GDAL_TIFF_INTERNAL_MASK=not self.external_mask,
        ):
            with rasterio.open(
                self.path,
                "w",
                driver="GTiff",
                width=self.width,
                height=self.height,
                count=count,
                dtype="float32" if self.nonfinite else "uint16",
                crs="EPSG:32630" if self.with_crs else None,
                transform=from_origin(500000, 4500000, *self.resolution),
                nodata=0,
                tiled=True,
                blockxsize=128,
                blockysize=128,
            ) as source:
                source.descriptions = self.descriptions
                source.update_tags(TIFFTAG_DATETIME="2024:06:01 12:00:00")
                if self.transformed:
                    source.scales = tuple(0.0001 for _ in range(count))
                    source.offsets = tuple(-0.1 for _ in range(count))
                rng = np.random.default_rng(513)
                for _, window in source.block_windows(1):
                    pixels = rng.integers(
                        300,
                        9000,
                        size=(count, int(window.height), int(window.width)),
                        dtype=np.uint16,
                    )
                    if self.nonfinite:
                        pixels = pixels.astype(np.float32)
                    mask = np.full(pixels.shape[1:], 255, dtype=np.uint8)
                    if window.row_off == 0 and window.col_off == 0:
                        pixels[:, 2:5, 3:8] = 0
                        mask[10:15, 12:18] = 0
                        if self.nonfinite:
                            pixels[0, 20, 20] = float("inf")
                            pixels[-1, 30, 30] = float("-inf")
                    source.write(pixels, window=window)
                    source.write_mask(mask, window=window)
        return self.path


class LocalBackend(ModelBackend):
    """Batch-independent fake inference retaining actual local preprocessing."""

    classes = 2

    def __init__(self) -> None:
        """Keep small call observations and an interrupt switch for finite tests."""
        super().__init__("local-fixture", "1.0.0", "cpu")
        self.calls: list[int] = []
        self.interrupt = False
        self.fail = False

    def embed(
        self, chips: np.ndarray, context: InferenceContext | None = None
    ) -> np.ndarray:
        """Support only failure-path tests; parity tests use real reference Clay."""
        self._observe(chips)
        return np.ones((len(chips), 512), dtype=np.float32)

    def classify(
        self,
        chips: np.ndarray,
        queries: list[str],
        context: InferenceContext | None = None,
    ) -> np.ndarray:
        """Derive finite probabilities from image values independently of batching."""
        self._observe(chips)
        first = (
            1 / (1 + np.exp(-chips.mean(axis=(1, 2, 3), dtype=np.float64)))
        ).astype(np.float32)
        return (
            first[:, None] if len(queries) == 1 else np.column_stack((first, 1 - first))
        )

    def segment(
        self,
        chips: np.ndarray,
        head: str | None = None,
        threshold: float = 0.5,
        context: InferenceContext | None = None,
    ) -> np.ndarray:
        """Produce model-order pixel classes while retaining source/mask sensitivity."""
        self._observe(chips)
        return (chips[:, 0] > threshold).astype(np.uint8)

    def _observe(self, chips: np.ndarray) -> None:
        """Record bounded calls or simulate an interruption during inference."""
        self.calls.append(len(chips))
        if self.interrupt:
            raise KeyboardInterrupt()
        if self.fail:
            raise RuntimeError("simulated model failure")


def patch_local_backends(monkeypatch: pytest.MonkeyPatch) -> LocalBackend:
    """Patch only model-manifest/backend boundaries for local class/segment tests."""
    backend = LocalBackend()

    def manifest(store: ModelStore, name: str) -> Manifest:
        """Return registry-consistent metadata without retrieving optional weights."""
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
        """Reuse the small observed backend across eager and streamed calls."""
        return backend

    monkeypatch.setattr(ModelStore, "load_manifest", manifest)
    for module in (
        "whirld.api",
        "whirld.core.embedding_stream",
        "whirld.core.classification_stream",
        "whirld.core.segmentation_stream",
    ):
        monkeypatch.setattr(module + ".load_backend", load)
    return backend


@pytest.mark.parametrize("as_uri", [False, True])
@pytest.mark.parametrize("resolution", [(10.0, 10.0), (10.04, 9.98), (15.0, 20.0)])
def test_local_embedding_matches_eager_grid_order_and_units(
    tmp_path: Path,
    whirld_home: Path,
    as_uri: bool,
    resolution: tuple[float, float],
) -> None:
    """Local native/tolerance/resampled grids and nondefault TIFF transforms agree."""
    path = LocalScene(
        tmp_path / "local scene%20#?.tif",
        resolution=resolution,
        external_mask=True,
        transformed=True,
    ).create()
    whirld.pull("clay-v1")
    eager = whirld.embed(
        path, model="clay-v1", chip_size=128, overlap=17, device="cpu", write=False
    )
    streamed = whirld.embed_to_disk(
        path.as_uri() if as_uri else path,
        model="clay-v1",
        chip_size=128,
        overlap=17,
        batch_size=2,
        device="cpu",
        output=None if as_uri else tmp_path / "local.npy",
    )
    np.testing.assert_allclose(
        np.load(streamed.output_path), eager.embeddings, rtol=1e-5, atol=1e-6
    )
    metadata = json.loads(streamed.meta_path.read_text())
    assert metadata["chips"] == [chip.model_dump(mode="json") for chip in eager.chips]
    assert metadata["crs"] == eager.meta["crs"]
    assert streamed.sensor == eager.sensor == "sentinel-2-l2a"
    assert streamed.chip_count == len(eager.chips)
    if as_uri:
        assert streamed.output_path == path.with_name(f"{path.stem}_embeddings.npy")


def test_uppercase_localhost_uri_uses_local_default_output(
    tmp_path: Path, whirld_home: Path
) -> None:
    """URI routing and default output naming agree on case-insensitive localhost."""
    path = LocalScene(tmp_path / "scene #?.tif", width=64, height=64).create()
    whirld.pull("clay-v1")
    result = whirld.embed_to_disk(
        path.as_uri().replace("file://", "file://LOCALHOST", 1),
        model="clay-v1",
        chip_size=64,
        device="cpu",
    )
    assert result.output_path == path.with_name(f"{path.stem}_embeddings.npy")
    assert result.chip_count == 1 and result.output_path.is_file()


@pytest.mark.parametrize("as_uri", [False, True])
@pytest.mark.parametrize("external_mask", [False, True])
def test_local_classification_matches_eager(
    tmp_path: Path,
    whirld_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    as_uri: bool,
    external_mask: bool,
) -> None:
    """Actual local named-band selection and mask handling preserve global scores."""
    path = LocalScene(
        tmp_path / "classification.tif",
        width=700,
        height=600,
        external_mask=external_mask,
        transformed=True,
    ).create()
    backend = patch_local_backends(monkeypatch)
    eager = whirld.classify(
        path, model="remoteclip", query=["forest", "water"], top_k=100, device="cpu"
    )
    backend.calls.clear()
    result = whirld.classify_to_disk(
        path.as_uri() if as_uri else path,
        model="remoteclip",
        query=["forest", "water"],
        top_k=None,
        batch_size=3,
        device="cpu",
        output=tmp_path / "local.geojson",
    )
    assert json.loads(result.output_path.read_text()) == json.loads(
        json.dumps(eager.feature_collection)
    )
    assert result.chip_count == result.feature_count == len(eager.chips) == 12
    assert backend.calls == [3, 3, 3, 3]


@pytest.mark.parametrize("as_uri", [False, True])
@pytest.mark.parametrize("external_mask", [False, True])
def test_local_segmentation_preserves_class_zero_and_sidecar_masks(
    tmp_path: Path,
    whirld_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    as_uri: bool,
    external_mask: bool,
) -> None:
    """Non-square masks preserve exact affine, nodata and partial native tile writes."""
    path = LocalScene(
        tmp_path / "mask-input.tif",
        width=700,
        height=600,
        resolution=(30.04, 29.98),
        external_mask=external_mask,
        transformed=True,
    ).create()
    backend = patch_local_backends(monkeypatch)
    eager = whirld.segment(
        path,
        model="prithvi-burn-scar",
        sensor="hls",
        device="cpu",
        output=tmp_path / "eager.tif",
    )
    backend.calls.clear()
    result = whirld.segment_to_disk(
        path.as_uri() if as_uri else path,
        model="prithvi-burn-scar",
        sensor="hls",
        batch_size=3,
        device="cpu",
        output=tmp_path / "stream.tif",
    )
    with rasterio.open(result.output_path) as actual:
        np.testing.assert_array_equal(actual.read(1), eager.mask)
        assert actual.shape == (600, 700)
        assert actual.transform == eager.transform and actual.crs == eager.crs
        assert actual.nodata == 255
    assert backend.calls == [3, 1]
    assert set(np.unique(eager.mask)) == {0, 1, 255}
    assert result.valid_pixel_count == int((eager.mask != 255).sum())
    assert result.positive_pixel_count == int(
        ((eager.mask != 255) & (eager.mask > 0)).sum()
    )


def test_unlabeled_local_positional_bands_and_crs_assignment(
    tmp_path: Path,
    whirld_home: Path,
) -> None:
    """Entirely unnamed local bands retain legacy positional and explicit CRS rules."""
    path = LocalScene(
        tmp_path / "unnamed.tif", descriptions=(None,) * 6, with_crs=False
    ).create()
    whirld.pull("clay-v1")
    eager = whirld.embed(
        path,
        model="clay-v1",
        sensor="sentinel-2-l2a",
        crs="EPSG:32630",
        chip_size=128,
        device="cpu",
        write=False,
    )
    result = whirld.embed_to_disk(
        path,
        model="clay-v1",
        sensor="sentinel-2-l2a",
        crs="EPSG:32630",
        chip_size=128,
        device="cpu",
        output=tmp_path / "unnamed.npy",
    )
    np.testing.assert_allclose(
        np.load(result.output_path), eager.embeddings, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize(
    "descriptions",
    [
        ("B02", None, None, None, None, None),
        ("B02", "b02", "B04", "B08", "B11", "B12"),
        ("B02", "B03", "B04", "B08", "B11", "B01"),
    ],
)
def test_partial_duplicate_or_missing_local_names_are_rejected(
    tmp_path: Path,
    whirld_home: Path,
    descriptions: tuple[str | None, ...],
) -> None:
    """Streaming must not broaden eager local spectral identity fallback rules."""
    path = LocalScene(tmp_path / "badnames.tif", descriptions=descriptions).create()
    whirld.pull("clay-v1")
    with pytest.raises(UnsupportedSensorError):
        whirld.embed_to_disk(
            path,
            model="clay-v1",
            sensor="sentinel-2-l2a",
            output=tmp_path / "never.npy",
            device="cpu",
        )


@pytest.mark.parametrize("operation", ["embed", "classify", "segment"])
def test_local_bbox_is_explicitly_rejected_before_model_io(
    tmp_path: Path, operation: str
) -> None:
    """The first local-stream milestone does not silently change eager bbox behavior."""
    path = tmp_path / "not-opened.tif"
    output = tmp_path / "never"
    bbox = (-1.0, 50.0, 0.0, 51.0)
    with pytest.raises(InvalidInputError, match="bbox"):
        if operation == "embed":
            whirld.embed_to_disk(path, model="missing-model", output=output, bbox=bbox)
        elif operation == "classify":
            whirld.classify_to_disk(
                path, query="forest", model="missing-model", output=output, bbox=bbox
            )
        else:
            whirld.segment_to_disk(
                path, model="missing-model", output=output, bbox=bbox
            )
    assert not output.exists()


@pytest.mark.parametrize("operation", ["embed", "classify", "segment"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_local_streaming_failure_preserves_previous_result(
    tmp_path: Path,
    whirld_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    interrupt: bool,
) -> None:
    """Staged writers retain prior local results after model failures or interrupts."""
    path = LocalScene(
        tmp_path / "interrupt.tif",
        resolution=(30.0, 30.0) if operation == "segment" else (10.0, 10.0),
    ).create()
    backend = patch_local_backends(monkeypatch)
    backend.interrupt, backend.fail = interrupt, not interrupt
    output = tmp_path / "previous"
    output.write_bytes(b"previous output")
    error = KeyboardInterrupt if interrupt else RuntimeError
    with pytest.raises(error):
        if operation == "embed":
            whirld.embed_to_disk(
                path, model="clay-v1", chip_size=128, device="cpu", output=output
            )
        elif operation == "classify":
            whirld.classify_to_disk(
                path, model="remoteclip", query="forest", device="cpu", output=output
            )
        else:
            whirld.segment_to_disk(
                path,
                model="prithvi-burn-scar",
                sensor="hls",
                device="cpu",
                output=output,
            )
    assert output.read_bytes() == b"previous output"
    assert not list(tmp_path.glob(".whirld-*"))


def test_local_vrt_is_rejected_without_following_external_source(
    tmp_path: Path,
    whirld_home: Path,
) -> None:
    """A local path cannot use another GDAL driver to dereference external pixels."""
    path = tmp_path / "disguised.tif"
    path.write_text(
        '<VRTDataset rasterXSize="32" rasterYSize="32">'
        '<VRTRasterBand dataType="Byte" band="1"><SimpleSource>'
        "<SourceFilename>/never-open-external.tif</SourceFilename>"
        "</SimpleSource></VRTRasterBand></VRTDataset>"
    )
    whirld.pull("clay-v1")
    with pytest.raises(InvalidInputError) as error:
        whirld.embed_to_disk(
            path,
            model="clay-v1",
            sensor="sentinel-2-l2a",
            device="cpu",
            output=tmp_path / "never.npy",
        )
    assert "never-open-external" not in str(error.value)


@pytest.mark.parametrize("external_mask", [False, True])
def test_local_resampled_float_nonfinite_matches_eager_validity(
    tmp_path: Path,
    whirld_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    external_mask: bool,
) -> None:
    """Pre-warp finite filtering preserves local pixel classes and mask footprint."""
    path = LocalScene(
        tmp_path / "float.tif",
        resolution=(45.0, 60.0),
        nonfinite=True,
        external_mask=external_mask,
    ).create()
    patch_local_backends(monkeypatch)
    eager = whirld.segment(
        path,
        model="prithvi-burn-scar",
        sensor="hls",
        device="cpu",
        output=tmp_path / "eager-float.tif",
    )
    streamed = whirld.segment_to_disk(
        path,
        model="prithvi-burn-scar",
        sensor="hls",
        device="cpu",
        output=tmp_path / "stream-float.tif",
        batch_size=2,
    )
    with rasterio.open(streamed.output_path) as actual:
        np.testing.assert_array_equal(actual.read(1), eager.mask)
        assert actual.transform == eager.transform
    assert streamed.valid_pixel_count == int((eager.mask != 255).sum())
