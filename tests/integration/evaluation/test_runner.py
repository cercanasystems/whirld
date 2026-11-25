"""Evaluation retains actual reflectance preparation, inference I/O and scores."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from whirld.core.fetch import Manifest, ModelStore
from whirld.core.registry import ModelEntry
from whirld.evaluation.assets import EvaluationAssetCache
from whirld.evaluation.errors import EvaluationError
from whirld.evaluation.runner import EvaluationRunner, SuiteDocument, SuiteLoader
from whirld.evaluation.schemas import (
    AssetPin,
    ConfusionCounts,
    EvaluationCase,
    EvaluationOptions,
    EvaluationReport,
    EvaluationSuite,
    ModelIdentity,
    SpatialBounds,
    SplitPin,
)
from whirld.models.base import InferenceContext, ModelBackend


class FixtureBackend(ModelBackend):
    """Observe actual normalized HLS batches while replacing only model weights."""

    classes = 2

    def __init__(self) -> None:
        """Bind observations and explicit failure switches for resource checks."""
        super().__init__("prithvi-burn-scar", "2.0.0", "cpu")
        self.extrema: list[tuple[float, float]] = []
        self.failure: Literal["error", "interrupt"] | None = None
        self.failure_after = 0

    def embed(
        self, chips: np.ndarray, context: InferenceContext | None = None
    ) -> np.ndarray:
        """Fail if evaluation dispatches to an embedding interface."""
        raise AssertionError("unexpected embedding request")

    def segment(
        self,
        chips: np.ndarray,
        head: str | None = None,
        threshold: float = 0.5,
        context: InferenceContext | None = None,
    ) -> np.ndarray:
        """Classify the high-reflectance half after real normalization.

        Raises:
            RuntimeError: An explicitly requested inference failure.
            KeyboardInterrupt: An explicitly requested interruption.
        """
        self.extrema.append((float(chips[:, 0].min()), float(chips[:, 0].max())))
        if self.failure == "error" and len(self.extrema) > self.failure_after:
            raise RuntimeError("simulated inference failure")
        if self.failure == "interrupt" and len(self.extrema) > self.failure_after:
            raise KeyboardInterrupt()
        return (chips[:, 0] > 0).astype(np.uint8)


class FixtureSuite:
    """Own four real paired rasters and a content-pinned offline evaluation cache."""

    CASES: tuple[tuple[Literal["validation", "test"], str, int], ...] = (
        ("validation", "T10TFM.2018110.v1", 10),
        ("validation", "T11SPD.2020184.v1", 11),
        ("test", "T13REQ.2018156.v1", 13),
        ("test", "T14SPF.2019345.v1", 14),
    )

    def __init__(self, root: Path) -> None:
        """Write small compressed, georeferenced fixtures without network access."""
        self.root = root
        self.cache = EvaluationAssetCache(root / "cache", offline=True)
        self.cache.directory.mkdir(parents=True)
        self.case_files: dict[str, tuple[Path, Path]] = {}
        cases = tuple(self._case(split, name, zone) for split, name, zone in self.CASES)
        names: dict[Literal["train", "validation", "test"], str] = {
            "train": "T18TDR.2018180.v1\n",
            "validation": "T10TFM.2018110.v1\nT11SPD.2020184.v1\n",
            "test": "T13REQ.2018156.v1\nT14SPF.2019345.v1\n",
        }
        template = SuiteLoader.load().suite
        self.suite = EvaluationSuite.model_validate(
            template.model_dump()
            | {
                "id": "fixture-burn-scars",
                "cases": cases,
                "split_pins": tuple(
                    SplitPin(
                        split=split,
                        asset=self._cache_bytes(text.encode(), f"{split}.txt"),
                    )
                    for split, text in names.items()
                ),
            }
        )
        self.path = root / "suite.json"
        self.document = self.write(self.suite)

    def _cache_bytes(self, content: bytes, name: str) -> AssetPin:
        """Store one independently pinned object at its digest filename."""
        pin = AssetPin(
            url=f"https://example.invalid/pinned/{name}",
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        )
        self.cache.path(pin).write_bytes(content)
        return pin

    def _case(
        self, split: Literal["validation", "test"], name: str, zone: int
    ) -> EvaluationCase:
        """Generate quadrant truth and an image with distinct spectral predictions."""
        basename = f"subsetted_512x512_HLS.S30.{name}.4"
        image_path = self.root / f"{basename}_merged.tif"
        label_path = self.root / f"{basename}.mask.tif"
        pixels = np.zeros((6, 512, 512), dtype=np.float64)
        pixels[:, :, :256] = 0.5
        pixels[:, 0, 0] = -1000
        labels = np.zeros((512, 512), dtype=np.int16)
        labels[:256] = 1
        labels[1, 1] = -1
        profile = {
            "driver": "GTiff",
            "width": 512,
            "height": 512,
            "transform": from_origin(500000, 4500000, 30, 30),
            "crs": f"EPSG:326{zone}",
            "nodata": -1000,
            "tiled": True,
            "blockxsize": 128,
            "blockysize": 128,
            "compress": "deflate",
        }
        with rasterio.open(image_path, "w", count=6, dtype="float64", **profile) as out:
            out.write(pixels)
        with rasterio.open(label_path, "w", count=1, dtype="int16", **profile) as out:
            out.write(labels, 1)
        with rasterio.open(image_path) as image:
            west, south, east, north = transform_bounds(
                image.crs, "EPSG:4326", *image.bounds
            )
        image_pin = self._cache_bytes(image_path.read_bytes(), image_path.name)
        label = self._cache_bytes(label_path.read_bytes(), label_path.name)
        self.case_files[name] = (self.cache.path(image_pin), self.cache.path(label))
        return EvaluationCase(
            id=name,
            source_name=name,
            split=split,
            spatial_group=name.split(".")[0],
            image=image_pin,
            label=label,
            bounds=SpatialBounds(west=west, south=south, east=east, north=north),
        )

    def write(self, suite: EvaluationSuite) -> SuiteDocument:
        """Serialize and load a manifest through the real public loader boundary."""
        self.path.write_text(suite.model_dump_json(indent=2) + "\n")
        return SuiteLoader.load(self.path)

    def options(
        self, split: Literal["validation", "test", "all"] = "all"
    ) -> EvaluationOptions:
        """Return deterministic CPU/offline options with a fresh output destination."""
        return EvaluationOptions(
            output_dir=self.root / "output",
            cache_dir=self.cache.directory,
            split=split,
            threads=1,
            offline=True,
        )


def patch_weights(
    monkeypatch: pytest.MonkeyPatch, document: SuiteDocument
) -> FixtureBackend:
    """Replace checkpoint/manifest seams while keeping preprocessing and disk APIs."""
    backend = FixtureBackend()

    def manifest(store: ModelStore, name: str) -> Manifest:
        """Return registry-consistent installed metadata without optional downloads."""
        return Manifest(
            name=name,
            version=store.registry.get(name).version,
            source_type="huggingface",
            sha256=document.suite.model_sha256,
            weights_file="fixture.pt",
        )

    def load(
        entry: ModelEntry,
        manifest: Manifest,
        device: str | None = None,
        *,
        store: ModelStore | None = None,
    ) -> ModelBackend:
        """Return a fake backend with independent batches and real input sensitivity."""
        import torch

        # The actual LightningCLI similarly resets all host generators on load.
        random.seed(2)
        np.random.seed(2)
        torch.manual_seed(2)
        os.environ["PL_GLOBAL_SEED"] = "2"
        os.environ["PL_SEED_WORKERS"] = "0"
        return backend

    def identity(self: EvaluationRunner) -> tuple[ModelEntry, Manifest, ModelIdentity]:
        """Stand in only for large checkpoint hashing at the verification boundary."""
        entry = self._store.registry.get(document.suite.model)
        installed = manifest(self._store, entry.name)
        return (
            entry,
            installed,
            ModelIdentity(
                name=entry.name,
                version=entry.version,
                revision=document.suite.model_revision,
                checkpoint_sha256=document.suite.model_sha256,
                config_sha256=document.suite.model_config_sha256,
                registry_sha256=entry.fingerprint(),
            ),
        )

    monkeypatch.setattr(ModelStore, "load_manifest", manifest)
    monkeypatch.setattr(EvaluationRunner, "_model_identity", identity)
    monkeypatch.setattr("whirld.core.segmentation_stream.load_backend", load)
    return backend


def expected_counts(multiplier: int = 1) -> ConfusionCounts:
    """Return hand-derived quadrant confusion counts, ignored labels and coverage."""
    return ConfusionCounts(
        tp=65534 * multiplier,
        fp=65536 * multiplier,
        tn=65536 * multiplier,
        fn=65536 * multiplier,
        label_ignored=multiplier,
        prediction_invalid_positive=multiplier,
    )


@pytest.mark.parametrize("split", ["validation", "test", "all"])
def test_offline_suite_runs_real_public_preprocessing_and_scores(
    tmp_path: Path,
    whirld_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    split: Literal["validation", "test", "all"],
) -> None:
    """Correct reflectance units preserve zeros, masks, counts and split provenance."""
    import torch

    fixture = FixtureSuite(tmp_path)
    backend = patch_weights(monkeypatch, fixture.document)
    threads = torch.get_num_threads()
    state = torch.random.get_rng_state().clone()
    python_state = random.getstate()
    numpy_state = np.random.get_state(legacy=True)
    seed_environment = {
        key: os.environ.get(key) for key in ("PL_GLOBAL_SEED", "PL_SEED_WORKERS")
    }
    report = EvaluationRunner(fixture.document, fixture.options(split)).run()
    count = 4 if split == "all" else 2
    assert len(report.cases) == len(backend.extrema) == count
    assert all(low < 0 and high > 1 for low, high in backend.extrema)
    assert torch.get_num_threads() == threads
    assert torch.equal(torch.random.get_rng_state(), state)
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(legacy=True), numpy_state)
    assert {key: os.environ.get(key) for key in seed_environment} == seed_environment
    assert report.manifest_sha256 == fixture.document.sha256
    assert report.suite == fixture.suite
    assert report.model.checkpoint_sha256 == fixture.suite.model_sha256
    assert report.model.config_sha256 == fixture.suite.model_config_sha256
    assert report.options.threshold == 0.5 and report.runtime.device == "cpu"
    output = fixture.options().output_dir
    persisted = EvaluationReport.model_validate_json(
        (output / "report.json").read_text()
    )
    assert persisted == report
    assert len(list(output.glob("*.prediction.tif"))) == count
    assert len(list(output.glob("*.stac.json"))) == 4
    for result in report.cases:
        assert result.counts == expected_counts()
        assert result.counts.total_pixels == 512**2
        assert result.metrics.iou == pytest.approx(65534 / (65534 + 65536 + 65536))
        assert result.metrics.coverage == pytest.approx(262142 / 262143)
        source, _ = fixture.case_files[result.id]
        with (
            rasterio.open(output / result.prediction_file) as prediction,
            rasterio.open(source) as original,
        ):
            assert prediction.shape == original.shape == (512, 512)
            assert prediction.crs == original.crs
            assert prediction.transform == original.transform
            mask = prediction.read(1)
            assert mask[0, 0] == 255
            assert mask[300, 10] == 1 and mask[10, 300] == 0
        assert hashlib.sha256(
            (output / result.prediction_file).read_bytes()
        ).hexdigest() == (result.prediction_sha256)
        metadata = json.loads((output / f"{result.id}.stac.json").read_text())
        assert (
            metadata["assets"]["reflectance"]["raster:bands"]
            == [{"scale": 1.0, "offset": 0.0, "nodata": -1000.0}] * 6
        )
    assert {group.split for group in report.splits} == (
        {"validation", "test"} if split == "all" else {split}
    )
    for group in report.splits:
        assert group.case_count == 2 and group.counts == expected_counts(2)
        assert group.metrics.coverage == pytest.approx(262142 / 262143)


def test_existing_output_directory_is_preserved(
    tmp_path: Path, whirld_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refusing an existing run cannot erase previous reports or begin inference."""
    fixture = FixtureSuite(tmp_path)
    backend = patch_weights(monkeypatch, fixture.document)
    output = fixture.options().output_dir
    output.mkdir()
    previous = output / "report.json"
    previous.write_text("previous result")
    with pytest.raises(FileExistsError):
        EvaluationRunner(fixture.document, fixture.options()).run()
    assert previous.read_text() == "previous result"
    assert list(output.iterdir()) == [previous] and backend.extrema == []


@pytest.mark.parametrize("failure", ["error", "interrupt"])
@pytest.mark.parametrize("successful_cases", [0, 1])
def test_inference_failure_cleans_owned_output_and_restores_runtime(
    tmp_path: Path,
    whirld_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Literal["error", "interrupt"],
    successful_cases: int,
) -> None:
    """Model failures and interrupts cannot leave a partially credible report."""
    import torch

    fixture = FixtureSuite(tmp_path)
    backend = patch_weights(monkeypatch, fixture.document)
    backend.failure = failure
    backend.failure_after = successful_cases
    cache_files = set(fixture.cache.directory.iterdir())
    threads = torch.get_num_threads()
    state = torch.random.get_rng_state().clone()
    python_state = random.getstate()
    numpy_state = np.random.get_state(legacy=True)
    seed_environment = {
        key: os.environ.get(key) for key in ("PL_GLOBAL_SEED", "PL_SEED_WORKERS")
    }
    exception = RuntimeError if failure == "error" else KeyboardInterrupt
    with pytest.raises(exception):
        EvaluationRunner(fixture.document, fixture.options()).run()
    assert not fixture.options().output_dir.exists()
    assert set(fixture.cache.directory.iterdir()) == cache_files
    assert len(backend.extrema) == successful_cases + 1
    assert torch.get_num_threads() == threads
    assert torch.equal(torch.random.get_rng_state(), state)
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(legacy=True), numpy_state)
    assert {key: os.environ.get(key) for key in seed_environment} == seed_environment


def test_unhashed_label_mask_cannot_hide_scoring_errors(
    tmp_path: Path, whirld_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An external mask changes score inputs despite unchanged label TIFF hashes."""
    fixture = FixtureSuite(tmp_path)
    backend = patch_weights(monkeypatch, fixture.document)
    case = fixture.suite.cases[0]
    label = fixture.case_files[case.id][1]
    mask = np.full((512, 512), 255, dtype=np.uint8)
    mask[:256, 256:] = 0
    mask[256:, :256] = 0
    with (
        rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False),
        rasterio.open(label, "r+", driver="GTiff") as source,
    ):
        source.write_mask(mask)
    assert label.with_name(label.name + ".msk").is_file()
    assert hashlib.sha256(label.read_bytes()).hexdigest() == case.label.sha256
    with pytest.raises(EvaluationError, match="sidecar"):
        EvaluationRunner(fixture.document, fixture.options()).run()
    assert backend.extrema == [] and not fixture.options().output_dir.exists()


def test_report_publication_failure_removes_completed_case_outputs(
    tmp_path: Path, whirld_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A final disk failure cannot leave completed masks suggesting a finished run."""
    fixture = FixtureSuite(tmp_path)
    backend = patch_weights(monkeypatch, fixture.document)
    replace = Path.replace

    def fail_report(self: Path, target: Path) -> Path:
        """Fail only the report commit after every independent mask was published."""
        if self.name == ".report.json.part":
            assert len(list(self.parent.glob("*.prediction.tif"))) == 4
            raise OSError("simulated report publication failure")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_report)
    with pytest.raises(OSError, match="publication"):
        EvaluationRunner(fixture.document, fixture.options()).run()
    assert len(backend.extrema) == 4
    assert not fixture.options().output_dir.exists()
    assert len(list(fixture.cache.directory.iterdir())) == 11


@pytest.mark.parametrize("payload", [b"{", b'{"unknown":true}', b" " * (1024**2 + 1)])
def test_suite_loader_rejects_invalid_or_oversized_manifest(
    tmp_path: Path, payload: bytes
) -> None:
    """Malformed external manifests fail through a domain error before inference."""
    path = tmp_path / "invalid.json"
    path.write_bytes(payload)
    with pytest.raises(EvaluationError, match="manifest"):
        SuiteLoader.load(path)


def test_unselected_case_footprint_is_verified_before_inference(
    tmp_path: Path, whirld_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing a test footprint fails even during a validation-only invocation."""
    fixture = FixtureSuite(tmp_path)
    original = fixture.suite.cases[-1]
    moved = original.model_copy(
        update={
            "bounds": original.bounds.model_copy(
                update={"west": original.bounds.west + 0.01}
            )
        }
    )
    suite = EvaluationSuite.model_validate(
        fixture.suite.model_dump() | {"cases": (*fixture.suite.cases[:-1], moved)}
    )
    document = fixture.write(suite)
    backend = patch_weights(monkeypatch, document)
    with pytest.raises(EvaluationError, match="footprint"):
        EvaluationRunner(document, fixture.options("validation")).run()
    assert backend.extrema == [] and not fixture.options().output_dir.exists()


@pytest.mark.parametrize(
    "violation",
    ["overlap", "train-group", "validation-group", "duplicate", "missing"],
)
def test_official_split_violations_fail_before_any_inference(
    tmp_path: Path,
    whirld_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    violation: str,
) -> None:
    """Full-list audits detect exposure, duplicate identities and wrong membership."""
    fixture = FixtureSuite(tmp_path)
    split = "train"
    if violation == "overlap":
        content = "T10TFM.2018110.v1\n"
    elif violation == "train-group":
        content = "T10TFM.2019001.v1\n"
    elif violation == "validation-group":
        split = "validation"
        content = "T10TFM.2018110.v1\nT11SPD.2020184.v1\nT13REQ.2020001.v1\n"
    elif violation == "duplicate":
        content = "T18TDR.2018180.v1\nT18TDR.2018180.v1\n"
    else:
        split = "validation"
        content = "T11SPD.2020184.v1\n"
    replacement = fixture._cache_bytes(content.encode(), f"{split}.txt")
    pins = tuple(
        pin.model_copy(update={"asset": replacement}) if pin.split == split else pin
        for pin in fixture.suite.split_pins
    )
    suite = EvaluationSuite.model_validate(
        fixture.suite.model_dump() | {"split_pins": pins}
    )
    document = fixture.write(suite)
    backend = patch_weights(monkeypatch, document)
    with pytest.raises(EvaluationError, match="[Ss]plit|identities"):
        EvaluationRunner(document, fixture.options("validation")).run()
    assert backend.extrema == [] and not fixture.options().output_dir.exists()
