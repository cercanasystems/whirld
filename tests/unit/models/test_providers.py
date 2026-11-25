"""Catalog conformance, metadata boundaries and trusted lazy construction."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from whirld import config
from whirld.core.fetch import Manifest, ModelStore
from whirld.core.registry import ModelEntry
from whirld.errors import RegistryError, UnsupportedOperationError, WhirldError
from whirld.models.base import ModelBackend
from whirld.models.loader import load_backend
from whirld.models.providers import Operation, ProviderCatalog


def _document(name: str) -> dict[str, Any]:
    """Return mutable external-boundary data from a validated bundled document."""
    return ModelEntry.from_registry_file(
        config.bundled_registry_dir() / "models" / f"{name}.yaml"
    ).model_dump(mode="json", exclude_none=True)


@pytest.mark.parametrize(
    ("name", "provider_id", "operation", "extra"),
    [
        ("clay-v1", "clay-reference", "embed", None),
        ("clay-v1.5", "clay", "embed", "hf"),
        ("remoteclip-rn50", "remoteclip", "classify", "remoteclip"),
        ("remoteclip", "remoteclip", "classify", "remoteclip"),
        ("remoteclip-vit-l14", "remoteclip", "classify", "remoteclip"),
        ("prithvi-burn-scar", "prithvi", "segment", "prithvi"),
        ("prithvi-flood", "prithvi", "segment", "prithvi"),
    ],
)
def test_bundled_provider_conformance(
    name: str, provider_id: str, operation: Operation, extra: str | None
) -> None:
    """All bundled entries share capabilities and runtime policy with their owner."""
    entry = ModelEntry.from_registry_data(_document(name))
    definition = ProviderCatalog.get(entry.backend)
    assert definition.id == provider_id
    assert definition.operations == (operation,)
    assert definition.optional_extra == extra
    entry.require_operation(operation)
    for unsupported in cast(tuple[Operation, ...], ("embed", "classify", "segment")):
        if unsupported == operation:
            continue
        with pytest.raises(UnsupportedOperationError) as raised:
            entry.require_operation(unsupported)
        assert raised.value.model == name
        assert raised.value.operation == unsupported
        assert raised.value.supported_operations == (operation,)


def test_schema_and_catalog_have_identical_closed_backend_ids() -> None:
    """New provider code and accepted document identifiers must change together."""
    schema = json.loads(
        (config.bundled_registry_dir() / "schema/model.schema.json").read_text()
    )
    assert set(schema["properties"]["backend"]["enum"]) == {
        definition.id for definition in ProviderCatalog.all()
    }


@pytest.mark.parametrize("backend_id", [None, "", "os.system", "whirld.models.clay"])
def test_unknown_ids_are_rejected_before_module_import(backend_id: str | None) -> None:
    """An untrusted registry selector cannot become a Python import target."""
    with pytest.raises(WhirldError, match="Unsupported backend"):
        ProviderCatalog.get(backend_id)


def test_catalog_definitions_and_dependencies_are_immutable() -> None:
    """One caller cannot change the process-wide capability or repair policy."""
    definition = ProviderCatalog.get("remoteclip")
    with pytest.raises(ValidationError):
        definition.operations = ("embed",)
    with pytest.raises(ValidationError):
        definition.dependencies[0].module = "untrusted"


def test_catalog_and_registry_import_without_numerical_or_optional_runtime() -> None:
    """Base imports remain available even if every numerical runtime is broken."""
    source = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'numpy', 'rasterio', 'torch', 'open_clip', 'terratorch', 'torchgeo'
        }:
            raise AssertionError('Unexpected heavy import: ' + fullname)
sys.meta_path.insert(0, Block())
from whirld.models.providers import ProviderCatalog
from whirld.core.registry import ModelEntry
assert len(ProviderCatalog.all()) == 4
"""
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("name", "field", "value"),
    [
        ("clay-v1", "output.embed_dim", None),
        ("clay-v1", "output.embed_dim", 0),
        ("clay-v1", "source.seed", None),
        ("clay-v1", "source.seed", -1),
        ("clay-v1", "output.shape", ["n", 12]),
        ("clay-v1", "output.dtype", "float64"),
        ("clay-v1.5", "output.embed_dim", 512),
        ("clay-v1.5", "band_contract.patch_size", None),
        ("clay-v1.5", "band_contract.patch_size", 16),
        ("clay-v1.5", "band_contract.sensors.sentinel-2-l2a.wavelengths", None),
        ("clay-v1.5", "band_contract.sensors.sentinel-2-l2a.wavelengths", [0.5]),
        ("prithvi-flood", "output.classes", None),
        ("prithvi-flood", "output.classes", 0),
        ("prithvi-flood", "output.classes", 256),
        ("prithvi-flood", "config_file", "undeclared.yaml"),
        ("prithvi-flood", "trust", "community"),
        ("prithvi-flood", "source.files", ["model.safetensors", "config.yaml"]),
    ],
)
def test_provider_metadata_rejected_at_external_boundary(
    name: str, field: str, value: object
) -> None:
    """Malformed provider declarations fail before artifacts or runtime are loaded."""
    document = _document(name)
    keys = field.split(".")
    current = document
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value
    with pytest.raises(RegistryError):
        ModelEntry.from_registry_data(document)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_clay_requires_finite_positive_wavelengths(bad: float) -> None:
    """Both typed construction and document parsing enforce spectral metadata."""
    document = _document("clay-v1.5")
    document["band_contract"]["sensors"]["sentinel-2-l2a"]["wavelengths"][0] = bad
    with pytest.raises(ValueError, match="wavelengths"):
        ModelEntry.model_validate(document)
    with pytest.raises(RegistryError):
        ModelEntry.from_registry_data(document)


def test_reference_dimensions_remain_flexible_and_prithvi_is_not_binary_only() -> None:
    """Provider validation preserves legitimately configurable model families."""
    reference = _document("clay-v1")
    reference["output"]["embed_dim"] = 3
    reference["output"]["shape"] = ["n_chips", 3]
    assert ModelEntry.from_registry_data(reference).output.embed_dim == 3
    segmenter = _document("prithvi-flood")
    segmenter["output"]["classes"] = 7
    assert ModelEntry.from_registry_data(segmenter).output.classes == 7


@pytest.mark.parametrize("classes", [1, 255])
def test_prithvi_class_count_boundaries_preserve_nodata(classes: int) -> None:
    """The first and last representable class counts remain legitimate metadata."""
    document = _document("prithvi-flood")
    document["output"]["classes"] = classes
    assert ModelEntry.from_registry_data(document).output.classes == classes


def test_prithvi_rejects_nodata_collision_in_direct_metadata_validation() -> None:
    """Typed metadata cannot allocate the reserved uint8 nodata label as a class."""
    document = _document("prithvi-flood")
    document["output"]["classes"] = 256
    with pytest.raises(ValueError, match="255.*nodata"):
        ModelEntry.model_validate(document)


@pytest.mark.parametrize(
    ("name", "sensor", "channels"),
    [
        ("remoteclip-rn50", "sentinel-2-l2a", 3),
        ("prithvi-flood", "sentinel-2-l1c", 6),
    ],
)
def test_fixed_channel_providers_reject_consistently_reduced_sensor_contracts(
    name: str, sensor: str, channels: int
) -> None:
    """Aligned one-channel metadata still cannot rebuild a three/six-channel model."""
    document = _document(name)
    contract = document["band_contract"]["sensors"][sensor]
    contract["bands"] = contract["bands"][:1]
    contract["aliases"] = contract["aliases"][:1]
    normalization = document["band_contract"]["normalization"]
    normalization["mean"] = normalization["mean"][:1]
    normalization["std"] = normalization["std"][:1]
    with pytest.raises(RegistryError, match=f"requires {channels} bands"):
        ModelEntry.from_registry_data(document)
    with pytest.raises(ValueError, match=f"requires {channels} bands"):
        ModelEntry.model_validate(document)


@pytest.mark.parametrize(
    "name", ["clay-v1", "clay-v1.5", "remoteclip-rn50", "prithvi-flood"]
)
def test_every_provider_rejects_normalization_count_mismatch(name: str) -> None:
    """Per-band statistics must cover every declared input channel before loading."""
    document = _document(name)
    normalization = document["band_contract"]["normalization"]
    normalization["mean"] = normalization["mean"][:1]
    normalization["std"] = normalization["std"][:1]
    with pytest.raises(RegistryError, match="normalization"):
        ModelEntry.from_registry_data(document)


def test_provider_alignment_checks_every_sensor_and_preserves_variable_clay_bands() -> (
    None
):
    """Clay supports varying declared channel counts when spectral metadata aligns."""
    document = _document("clay-v1.5")
    contract = document["band_contract"]["sensors"]["sentinel-2-l2a"]
    for field in ("bands", "aliases", "wavelengths"):
        contract[field] = contract[field][:4]
    normalization = document["band_contract"]["normalization"]
    for field in ("mean", "std"):
        normalization[field] = normalization[field][:4]
    assert ModelEntry.from_registry_data(document).backend == "clay"
    second = dict(contract)
    second["bands"] = second["bands"][:3]
    second["aliases"] = second["aliases"][:3]
    second["wavelengths"] = second["wavelengths"][:3]
    document["band_contract"]["sensors"]["additional-sensor"] = second
    with pytest.raises(RegistryError, match="additional-sensor.*normalization"):
        ModelEntry.from_registry_data(document)


def test_reference_zero_seed_remains_valid() -> None:
    """Zero is a legitimate deterministic NumPy seed at the supported boundary."""
    document = _document("clay-v1")
    document["source"]["seed"] = 0
    assert ModelEntry.from_registry_data(document).source.seed == 0


class _LeaseStore(ModelStore):
    """Record the factory's artifact lease independently of adapter construction."""

    leased = False

    @contextmanager
    def installation_lease(
        self, entry: ModelEntry, manifest: Manifest
    ) -> Iterator[None]:
        """Expose an active lease only inside the loader's context manager."""
        self.leased = True
        try:
            yield
        finally:
            self.leased = False


@pytest.mark.parametrize(
    ("name", "module", "class_name"),
    [
        ("clay-v1", "clay", "ClayBackend"),
        ("clay-v1.5", "clay_torch", "ClayTorchBackend"),
        ("remoteclip-rn50", "remoteclip", "RemoteCLIPBackend"),
        ("prithvi-flood", "prithvi", "PrithviBackend"),
    ],
)
def test_factory_uses_trusted_provider_constructor_inside_artifact_lease(
    whirld_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    module: str,
    class_name: str,
) -> None:
    """Every catalog implementation preserves selected device and cache ownership."""
    entry = ModelEntry.from_registry_data(_document(name))
    manifest = Manifest(
        name=name,
        version=entry.version,
        source_type=entry.source.type,
        sha256=entry.distribution.sha256,
        weights_file="weights",
    )
    store = _LeaseStore()
    sentinel = cast(ModelBackend, object())
    constructed = False

    def load(
        selected: ModelEntry,
        cached: Manifest,
        device: str | None,
        *,
        paths: config.WhirldPaths,
    ) -> ModelBackend:
        """Assert the provider delegates complete typed state under the same lease."""
        nonlocal constructed
        assert store.leased
        assert selected is entry and cached is manifest
        assert device == "cpu" and paths is store.paths
        constructed = True
        return sentinel

    adapter = getattr(importlib.import_module(f"whirld.models.{module}"), class_name)
    monkeypatch.setattr(adapter, "load", load)
    assert load_backend(entry, manifest, "cpu", store=store) is sentinel
    assert constructed and not store.leased


def test_invalid_metadata_fails_before_artifact_lease(
    whirld_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Programmatic model_copy cannot bypass the factory's metadata boundary."""
    entry = ModelEntry.from_registry_data(_document("clay-v1.5"))
    entry = entry.model_copy(
        update={"output": entry.output.model_copy(update={"embed_dim": 512})}
    )
    manifest = Manifest(
        name=entry.name,
        version=entry.version,
        source_type="huggingface",
        sha256=entry.distribution.sha256,
        weights_file="weights",
    )

    def no_lease(*args: object, **kwargs: object) -> None:
        """Fail immediately if malformed metadata reaches filesystem acquisition."""
        raise AssertionError("Metadata must be rejected before leasing")

    monkeypatch.setattr(ModelStore, "installation_lease", no_lease)
    with pytest.raises(RegistryError, match="embed_dim|shape"):
        load_backend(entry, manifest)
