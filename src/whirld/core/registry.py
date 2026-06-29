"""Registry fetch, parse, validation, and local caching.

The registry is the stable, versioned source of truth for every model Whirld can
run (PRD section 8). Each model is described by a YAML file conforming to
``registry_data/schema/model.schema.json``. Whirld ships a bundled copy of the
registry inside the package and seeds it into ``~/.whirld/registry`` on first use,
so the tool works fully offline out of the box.

This module exposes:

* Pydantic models (:class:`ModelEntry` and friends) — the typed, validated
  in-memory representation passed between layers (no raw dicts).
* :class:`Registry` — loads, validates, and queries registry YAMLs from the
  local cache, seeding from the bundled copy when needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .. import config
from ..errors import ModelNotFoundError, RegistryError
from ..logging_setup import get_logger

_log = get_logger("core.registry")


class SensorContract(BaseModel):
    """Band mapping for a single sensor within a model's band contract.

    Attributes:
        bands: Native band identifiers as they appear in source products.
        aliases: Spectral aliases (blue, green, red, nir, ...) aligned 1:1 with
            ``bands``. Selection is by alias, never by band index (PRD section 7.2).
        native_resolution_m: Native ground sample distance in meters.
    """

    model_config = ConfigDict(extra="allow")

    bands: list[str] = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)
    native_resolution_m: float = Field(gt=0)
    wavelengths: list[float] | None = None

    @field_validator("aliases")
    @classmethod
    def _aliases_match_bands(cls, value: list[str], info: Any) -> list[str]:
        """Ensure aliases and bands are the same length."""
        bands = info.data.get("bands")
        if bands is not None and len(bands) != len(value):
            raise ValueError(
                f"aliases ({len(value)}) and bands ({len(bands)}) must align 1:1"
            )
        return value


class Normalization(BaseModel):
    """Per-band normalization parameters.

    Attributes:
        type: Normalization strategy; only ``per_band_zscore`` in MVP.
        mean: Per-band means in the model's canonical alias order.
        std: Per-band standard deviations in the canonical alias order.
        scale: Optional DN-to-reflectance multiplier applied before normalizing.
    """

    type: str
    mean: list[float] = Field(min_length=1)
    std: list[float] = Field(min_length=1)
    scale: float | None = None

    @field_validator("std")
    @classmethod
    def _std_matches_mean(cls, value: list[float], info: Any) -> list[float]:
        """Ensure ``std`` and ``mean`` have equal length."""
        mean = info.data.get("mean")
        if mean is not None and len(mean) != len(value):
            raise ValueError("normalization mean and std must have equal length")
        return value


class BandContract(BaseModel):
    """The full sensor-to-model translation contract (PRD section 7).

    Attributes:
        sensors: Per-sensor band/alias mappings keyed by sensor identifier.
        target_resolution_m: Resolution all inputs are resampled to.
        chip_size_px: Tile size in pixels.
        normalization: Per-band normalization parameters.
        nodata_fill: Fill value for nodata and edge padding.
    """

    model_config = ConfigDict(extra="allow")

    sensors: dict[str, SensorContract] = Field(min_length=1)
    target_resolution_m: float = Field(gt=0)
    chip_size_px: int = Field(gt=0)
    patch_size: int | None = None
    normalization: Normalization
    nodata_fill: float = 0.0


class Source(BaseModel):
    """Where a model's weights come from.

    Attributes:
        type: ``huggingface`` (real weights) or ``reference`` (offline backend).
        repo: HF Hub repo id (huggingface sources).
        revision: Pinned commit SHA (huggingface sources).
        files: Weight file names within the repo.
        seed: Deterministic seed (reference sources).
    """

    model_config = ConfigDict(extra="allow")

    type: str
    repo: str | None = None
    revision: str | None = None
    files: list[str] = Field(default_factory=list)
    seed: int | None = None


class Distribution(BaseModel):
    """Checksum and size for the model artifact.

    Attributes:
        sha256: Lowercase hex sha256 of the canonical weights artifact.
        size_bytes: Artifact size in bytes.
    """

    model_config = ConfigDict(extra="allow")

    sha256: str
    size_bytes: int = Field(ge=0)


class OutputSpec(BaseModel):
    """Declared output type for a model.

    Attributes:
        type: ``embedding``, ``mask``, or ``scores``.
        format: ``npy``, ``geotiff``, or ``geojson``.
        embed_dim: Embedding dimensionality (embedding models only).
        dtype: Output numpy dtype string.
        shape: Symbolic output shape.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    format: str
    embed_dim: int | None = None
    classes: int | None = None
    dtype: str | None = None
    shape: list[Any] | None = None


class ModelEntry(BaseModel):
    """A fully validated registry entry for one model.

    This is the typed object every other layer consumes; raw YAML/dicts never
    cross module boundaries.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    display_name: str
    version: str
    category: str
    # Backend implementation that runs this model. A model that reuses an existing
    # backend is pure registry data; only a new architecture needs new code.
    # The JSON schema requires it (gating new submissions); kept optional here so a
    # registry cache seeded by an older version still parses (the loader gives a
    # clear error if it is actually needed and missing).
    backend: str | None = None
    # Provenance tier governing security/auto-merge. Defaults to the safe
    # ``community`` (must ship safetensors); curated entries set ``first-party``.
    trust: str = "community"
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    # Optional architecture identifier for backends that build a named model
    # (e.g. open_clip's ``ViT-B-32`` for RemoteCLIP).
    model_name: str | None = None
    # Optional auxiliary config file downloaded alongside the weights (e.g. a
    # TerraTorch YAML for Prithvi).
    config_file: str | None = None
    source: Source
    distribution: Distribution
    band_contract: BandContract
    output: OutputSpec
    license: str
    license_url: str | None = None
    citation: str | None = None
    authors: list[str] = Field(default_factory=list)
    source_url: str | None = None
    hardware: dict[str, Any] = Field(default_factory=dict)
    whirld_min_version: str | None = None

    def supported_sensors(self) -> list[str]:
        """Return the sorted list of sensor identifiers this model supports."""
        return sorted(self.band_contract.sensors)


class Registry:
    """Loads and queries the local model registry.

    On construction the registry directory is seeded from the package-bundled
    registry if it does not already exist, guaranteeing offline availability.

    Args:
        paths: Path bundle to use; defaults to the current environment's paths.
    """

    def __init__(self, paths: config.WhirldPaths | None = None) -> None:
        self._paths = paths or config.get_paths()
        self._seed_if_missing()

    @property
    def models_dir(self) -> Path:
        """Directory containing registry model YAMLs."""
        return self._paths.registry_models_dir

    def _seed_if_missing(self) -> None:
        """Copy the bundled registry into the cache when absent.

        Idempotent: only seeds when the registry models directory has no YAMLs.
        """
        if self.models_dir.exists() and any(self.models_dir.glob("*.yaml")):
            return
        import shutil

        bundled = config.bundled_registry_dir()
        _log.debug("Seeding registry from bundled copy at %s", bundled)
        self._paths.registry_models_dir.mkdir(parents=True, exist_ok=True)
        self._paths.registry_schema_dir.mkdir(parents=True, exist_ok=True)
        for yaml_file in (bundled / "models").glob("*.yaml"):
            shutil.copy2(yaml_file, self.models_dir / yaml_file.name)
        for schema_file in (bundled / "schema").glob("*.json"):
            shutil.copy2(
                schema_file, self._paths.registry_schema_dir / schema_file.name
            )

    def available(self) -> list[str]:
        """Return the sorted names of all models present in the registry."""
        return sorted(p.stem for p in self.models_dir.glob("*.yaml"))

    def get(self, name: str) -> ModelEntry:
        """Load and validate a single registry entry by name.

        Args:
            name: Model identifier (e.g. ``clay-v1``).

        Returns:
            The validated :class:`ModelEntry`.

        Raises:
            ModelNotFoundError: No YAML for ``name`` exists in the registry.
            RegistryError: The YAML is malformed or fails validation.
        """
        path = self.models_dir / f"{name}.yaml"
        if not path.exists():
            available = ", ".join(self.available()) or "(none)"
            raise ModelNotFoundError(
                f"Model '{name}' is not in the registry.\n"
                f"       Available models: {available}\n"
                f"       Run 'whirld update' to refresh the registry."
            )
        return self._parse(path)

    def _parse(self, path: Path) -> ModelEntry:
        """Parse and validate a registry YAML file into a :class:`ModelEntry`.

        Args:
            path: Path to the YAML file.

        Returns:
            The validated entry.

        Raises:
            RegistryError: On YAML or schema validation failure.
        """
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise RegistryError(
                f"Registry file '{path.name}' is not valid YAML.\n"
                f"       {exc}\n"
                f"       Re-run 'whirld update' to restore a clean copy."
            ) from exc
        if not isinstance(raw, dict):
            raise RegistryError(
                f"Registry file '{path.name}' must be a YAML mapping, "
                f"got {type(raw).__name__}."
            )
        try:
            return ModelEntry.model_validate(raw)
        except Exception as exc:  # pydantic.ValidationError and friends
            raise RegistryError(
                f"Registry entry '{path.stem}' failed validation:\n"
                f"       {exc}\n"
                f"       The entry does not conform to model.schema.json."
            ) from exc
