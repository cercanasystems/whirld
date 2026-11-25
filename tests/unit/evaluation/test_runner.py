"""Manifest, header and installed-artifact verification before evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from whirld.core.fetch import Manifest, ModelStore
from whirld.core.registry import ModelEntry, Registry
from whirld.evaluation import runner
from whirld.evaluation.assets import EvaluationAssetCache
from whirld.evaluation.errors import EvaluationError
from whirld.evaluation.runner import EvaluationRunner, HlsCasePreparer, SuiteLoader
from whirld.evaluation.schemas import (
    AssetPin,
    EvaluationCase,
    EvaluationOptions,
    EvaluationSuite,
    SpatialBounds,
    SplitPin,
    SuiteDocument,
)


def test_packaged_manifest_hash_identifies_exact_original_bytes() -> None:
    """Default suite identity includes the pinned JSON bytes, not a reserialization."""
    raw = (
        files("whirld.evaluation")
        .joinpath("data/hls-burn-scars-mini.json")
        .read_bytes()
    )
    document = SuiteLoader.load()
    assert document.sha256 == hashlib.sha256(raw).hexdigest()
    assert document.suite == EvaluationSuite.model_validate_json(raw)
    assert len(document.suite.cases) == 4
    assert {case.split for case in document.suite.cases} == {"validation", "test"}


def test_custom_manifest_keeps_its_own_serialized_identity(tmp_path: Path) -> None:
    """Compatible user manifests are validated without substituting packaged pins."""
    suite = SuiteLoader.load().suite
    custom = suite.model_dump(mode="json") | {"id": "custom-mini-baseline"}
    raw = json.dumps(custom, indent=4).encode() + b"\n"
    path = tmp_path / "custom.json"
    path.write_bytes(raw)
    document = SuiteLoader.load(path)
    assert document.suite.id == "custom-mini-baseline"
    assert document.sha256 == hashlib.sha256(raw).hexdigest()
    assert document.suite.cases == suite.cases


@pytest.mark.parametrize("payload", [b"{", b"\xff", b"[]", b" " * (1024**2 + 1)])
def test_malformed_or_oversized_manifests_fail_before_model_io(
    tmp_path: Path, payload: bytes
) -> None:
    """Invalid JSON, encodings, roots or oversized inputs produce a domain error."""
    path = tmp_path / "invalid.json"
    path.write_bytes(payload)
    with pytest.raises(EvaluationError, match="manifest"):
        SuiteLoader.load(path)


def test_missing_extra_fields_and_absent_manifest_paths_fail(tmp_path: Path) -> None:
    """A manifest cannot silently omit provenance or accept undeclared settings."""
    raw = SuiteLoader.load().suite.model_dump(mode="json")
    path = tmp_path / "incompatible.json"
    for payload in (
        raw | {"unknown": True},
        {k: v for k, v in raw.items() if k != "cases"},
    ):
        path.write_text(json.dumps(payload))
        with pytest.raises(EvaluationError, match="manifest"):
            SuiteLoader.load(path)
    for missing in (tmp_path / "missing.json", tmp_path):
        with pytest.raises(EvaluationError, match="manifest"):
            SuiteLoader.load(missing)


class InstalledFixture:
    """Own tiny checkpoint/config bytes behind a real typed installation contract."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch manifest/registry retrieval while retaining actual file hashing."""
        checkpoint = b"independent known checkpoint bytes\n"
        configuration = b"independent known configuration bytes\n"
        template = SuiteLoader.load()
        entry = Registry().get(template.suite.model)
        self.entry = ModelEntry.model_validate(
            entry.model_dump()
            | {
                "distribution": {
                    "sha256": hashlib.sha256(checkpoint).hexdigest(),
                    "size_bytes": len(checkpoint),
                }
            }
        )
        self.manifest = Manifest(
            name=self.entry.name,
            version=self.entry.version,
            source_type=self.entry.source.type,
            sha256=self.entry.distribution.sha256,
            weights_file=self.entry.source.files[0],
            entry_sha256=self.entry.fingerprint(),
        )
        self.store = ModelStore()
        self.checkpoint = self.store.paths.model_file(
            self.entry.name, self.manifest.weights_file
        )
        assert self.entry.config_file is not None
        self.configuration = self.store.paths.model_file(
            self.entry.name, self.entry.config_file
        )
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(checkpoint)
        self.configuration.write_bytes(configuration)
        self.document = SuiteDocument(
            suite=EvaluationSuite.model_validate(
                template.suite.model_dump()
                | {
                    "model_sha256": hashlib.sha256(checkpoint).hexdigest(),
                    "model_config_sha256": hashlib.sha256(configuration).hexdigest(),
                }
            ),
            sha256=template.sha256,
        )
        self.manifest_requests: list[str] = []

        def get(registry: Registry, name: str) -> ModelEntry:
            """Expose the test's independent validated registry entry."""
            assert name == self.entry.name
            return self.entry

        def manifest(store: ModelStore, name: str) -> Manifest:
            """Expose editable local provenance through the normal store method."""
            self.manifest_requests.append(name)
            return self.manifest

        def store_factory() -> ModelStore:
            """Give the evaluator the isolated real filesystem path owner."""
            return self.store

        monkeypatch.setattr(Registry, "get", get)
        monkeypatch.setattr(ModelStore, "load_manifest", manifest)
        monkeypatch.setattr(runner, "ModelStore", store_factory)
        self.runner = EvaluationRunner(
            self.document,
            EvaluationOptions(output_dir=root / "out", cache_dir=root / "cache"),
        )


def test_model_identity_hashes_both_artifacts_and_records_registry_fingerprint(
    tmp_path: Path, whirld_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid local manifests alone cannot replace checkpoint/config byte checks."""
    fixture = InstalledFixture(tmp_path, monkeypatch)
    observed: list[Path] = []
    actual_hash = runner.sha256_file

    def record_hash(path: Path) -> str:
        """Observe every real SHA-256 request without substituting its result."""
        observed.append(path)
        return actual_hash(path)

    monkeypatch.setattr(runner, "sha256_file", record_hash)
    entry, manifest, identity = fixture.runner._model_identity()
    assert entry is fixture.entry and manifest is fixture.manifest
    assert fixture.manifest_requests == [entry.name]
    assert observed == [fixture.checkpoint, fixture.configuration]
    assert (
        identity.checkpoint_sha256
        == hashlib.sha256(fixture.checkpoint.read_bytes()).hexdigest()
    )
    assert (
        identity.config_sha256
        == hashlib.sha256(fixture.configuration.read_bytes()).hexdigest()
    )
    assert identity.registry_sha256 == entry.fingerprint()
    assert identity.revision == fixture.document.suite.model_revision


@pytest.mark.parametrize(
    "violation",
    [
        "checkpoint-bytes",
        "config-bytes",
        "manifest-hash",
        "manifest-fingerprint",
        "manifest-version",
        "registry-hash",
        "registry-revision",
        "no-config",
        "suite-config-pin",
    ],
)
def test_installed_identity_mismatches_fail_without_loading_a_backend(
    tmp_path: Path,
    whirld_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    violation: str,
) -> None:
    """Verification rejects modified bytes, stale provenance and incompatible pins."""
    fixture = InstalledFixture(tmp_path, monkeypatch)
    if violation == "checkpoint-bytes":
        fixture.checkpoint.write_bytes(b"changed checkpoint")
    elif violation == "config-bytes":
        fixture.configuration.write_bytes(b"changed configuration")
    elif violation == "manifest-hash":
        fixture.manifest = fixture.manifest.model_copy(update={"sha256": "0" * 64})
    elif violation == "manifest-fingerprint":
        fixture.manifest = fixture.manifest.model_copy(
            update={"entry_sha256": "0" * 64}
        )
    elif violation == "manifest-version":
        fixture.manifest = fixture.manifest.model_copy(update={"version": "999.0.0"})
    elif violation == "registry-hash":
        fixture.entry = fixture.entry.model_copy(
            update={
                "distribution": fixture.entry.distribution.model_copy(
                    update={"sha256": "0" * 64}
                )
            }
        )
    elif violation == "registry-revision":
        fixture.entry = fixture.entry.model_copy(
            update={
                "source": fixture.entry.source.model_copy(update={"revision": "0" * 40})
            }
        )
    elif violation == "no-config":
        fixture.entry = fixture.entry.model_copy(update={"config_file": None})
    else:
        changed = fixture.document.suite.model_copy(
            update={"model_config_sha256": "0" * 64}
        )
        fixture.runner = EvaluationRunner(
            fixture.document.model_copy(update={"suite": changed}),
            fixture.runner.options,
        )
    with pytest.raises(EvaluationError, match="contract|SHA-256"):
        fixture.runner._model_identity()


@pytest.mark.parametrize("missing", ["torch", "terratorch"])
def test_missing_optional_runtime_has_actionable_domain_error(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    """Base-only users receive the required extra instead of a metadata traceback."""

    def version(name: str) -> str:
        """Report installed base libraries but omit one optional model runtime."""
        if name == missing:
            raise importlib.metadata.PackageNotFoundError(name)
        return "1.0.0"

    monkeypatch.setattr(runner.importlib.metadata, "version", version)
    with pytest.raises(EvaluationError, match="prithvi extra"):
        EvaluationRunner._runtime()


def test_runtime_provenance_uses_actual_distribution_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reports identify every numerical/model dependency without importing a model."""
    observed: list[str] = []

    def version(name: str) -> str:
        """Return distinguishable test distribution versions."""
        observed.append(name)
        return f"{name}-verified"

    monkeypatch.setattr(runner.importlib.metadata, "version", version)
    identity = EvaluationRunner._runtime()
    assert observed == ["numpy", "rasterio", "torch", "terratorch"]
    assert identity.numpy == "numpy-verified" and identity.torch == "torch-verified"
    assert identity.terratorch == "terratorch-verified"
    assert identity.device == "cpu" and identity.seed == 2
    assert identity.python and identity.platform and identity.whirld


@pytest.mark.parametrize(
    "violation", ["malformed-list", "wrong-group", "image-filename", "label-filename"]
)
def test_official_source_names_and_asset_filenames_are_verified(
    tmp_path: Path, whirld_home: Path, violation: str
) -> None:
    """Hash-verified lists still require valid identities and case-to-asset binding."""
    template = SuiteLoader.load()
    suite = template.suite
    cache = tmp_path / "cache"
    cache.mkdir()
    pins: list[SplitPin] = []
    for split in ("train", "validation", "test"):
        names = (
            ["T18TDR.2018180.v1"]
            if split == "train"
            else [case.source_name for case in suite.cases if case.split == split]
        )
        content = ("\n".join(names) + "\n").encode()
        if split == "train" and violation == "malformed-list":
            content = b"arbitrary-split-name\n"
        pin = AssetPin(
            url=f"https://example.invalid/{split}.txt",
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        )
        (cache / pin.sha256).write_bytes(content)
        pins.append(SplitPin.model_validate({"split": split, "asset": pin}))
    first = suite.cases[0]
    if violation == "wrong-group":
        first = first.model_copy(update={"spatial_group": "T12AAA"})
    elif violation in {"image-filename", "label-filename"}:
        field = "image" if violation == "image-filename" else "label"
        asset = getattr(first, field).model_copy(
            update={"url": "https://example.invalid/unrelated.tif"}
        )
        first = first.model_copy(update={field: asset})
    changed = EvaluationSuite.model_validate(
        suite.model_dump()
        | {"split_pins": tuple(pins), "cases": (first, *suite.cases[1:])}
    )
    evaluator = EvaluationRunner(
        template.model_copy(update={"suite": changed}),
        EvaluationOptions(output_dir=tmp_path / "out", cache_dir=cache, offline=True),
    )
    with pytest.raises(EvaluationError, match="split|identities"):
        evaluator._verify_splits()
    assert not evaluator.options.output_dir.exists()


def hls_pair(root: Path, violation: str = "none") -> tuple[EvaluationCase, Path, Path]:
    """Generate independent native HLS headers, optionally violating one contract."""
    image, label = root / "image.tif", root / "label.tif"
    transform = from_origin(500000, 4500000, 30, 30)
    if violation == "resolution":
        transform = from_origin(500000, 4500000, 10, 10)
    height, width = (511, 512) if violation == "shape" else (512, 512)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "transform": transform,
        "crs": "EPSG:32630",
        "nodata": -1000,
    }
    with rasterio.open(
        image,
        "w",
        **(profile | {"nodata": None if violation == "image-nodata" else -1000}),
        count=5 if violation == "image-bands" else 6,
        dtype="int16" if violation == "image-dtype" else "float32",
    ) as source:
        source.write(
            np.full((source.count, height, width), 0.2, dtype=source.dtypes[0])
        )
    with rasterio.open(
        label,
        "w",
        **(
            profile
            | {
                "nodata": None if violation == "label-nodata" else -1000,
                "crs": "EPSG:32631" if violation == "label-crs" else "EPSG:32630",
            }
        ),
        count=2 if violation == "label-bands" else 1,
        dtype="float32" if violation == "label-dtype" else "int16",
    ) as target:
        target.write(np.zeros((target.count, height, width), dtype=target.dtypes[0]))
    with rasterio.open(image) as source:
        west, south, east, north = transform_bounds(
            source.crs, "EPSG:4326", *source.bounds
        )
    case = (
        SuiteLoader.load()
        .suite.cases[0]
        .model_copy(
            update={
                "bounds": SpatialBounds(west=west, south=south, east=east, north=north)
            }
        )
    )
    return case, image, label


@pytest.mark.parametrize(
    "violation",
    [
        "image-bands",
        "image-dtype",
        "image-nodata",
        "shape",
        "label-bands",
        "label-dtype",
        "label-nodata",
        "label-crs",
        "resolution",
    ],
)
def test_preparer_rejects_incompatible_native_headers(
    tmp_path: Path, violation: str
) -> None:
    """Unit metadata preparation cannot silently adapt a different raster contract."""
    case, image, label = hls_pair(tmp_path, violation)
    with pytest.raises(EvaluationError, match="contract|30 m"):
        HlsCasePreparer.prepare(case, image, label, tmp_path)
    assert not list(tmp_path.glob("*.stac.json"))


def test_preparer_retains_reflectance_units_and_source_geometry(tmp_path: Path) -> None:
    """The metadata adapter declares scale one and leaves pinned raster bytes alone."""
    case, image, label = hls_pair(tmp_path)
    before = (image.read_bytes(), label.read_bytes())
    prepared = HlsCasePreparer.prepare(case, image, label, tmp_path)
    payload = json.loads(prepared.item.read_text())
    asset = payload["assets"]["reflectance"]
    assert asset["href"] == image.resolve().as_uri()
    assert (
        asset["raster:bands"] == [{"scale": 1.0, "offset": 0.0, "nodata": -1000.0}] * 6
    )
    assert [band["name"] for band in asset["eo:bands"]] == [
        "B02",
        "B03",
        "B04",
        "B8A",
        "B11",
        "B12",
    ]
    assert prepared.case == case and prepared.image == image and prepared.label == label
    assert before == (image.read_bytes(), label.read_bytes())


def test_non_utf8_official_split_has_a_domain_error(
    tmp_path: Path, whirld_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compatible custom manifests cannot turn malformed split text into a traceback."""
    malformed = tmp_path / "invalid-split.txt"
    malformed.write_bytes(b"\xff\xfe")

    def fetch(cache: EvaluationAssetCache, pin: AssetPin) -> Path:
        """Return verified fixture bytes at the isolated transport boundary."""
        return malformed

    monkeypatch.setattr(EvaluationAssetCache, "fetch", fetch)
    evaluation = EvaluationRunner(
        SuiteLoader.load(),
        EvaluationOptions(output_dir=tmp_path / "run", cache_dir=tmp_path / "cache"),
    )
    with pytest.raises(EvaluationError, match="UTF-8"):
        evaluation._verify_splits()
