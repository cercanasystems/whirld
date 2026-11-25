"""Closed, lightweight provider contracts shared by registry and runtime layers.

Only this module selects concrete backend implementations. Registry documents
describe model data; they never supply Python import paths or executable selectors.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from ..errors import SecurityError, UnsupportedOperationError, WhirldError

if TYPE_CHECKING:
    from ..config import WhirldPaths
    from ..core.fetch import Manifest
    from ..core.registry import ModelEntry
    from .base import ModelBackend

Operation = Literal["embed", "classify", "segment"]


class RuntimeDependency(BaseModel):
    """Installed distribution and importable module needed by a provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    distribution: str
    module: str


class ProviderDefinition(BaseModel, ABC):
    """Immutable provider policy; subclasses own metadata and trusted loading."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    operations: tuple[Operation, ...]
    optional_extra: str | None = None
    dependencies: tuple[RuntimeDependency, ...]
    source_type: str
    category: str
    output_type: str
    output_format: str
    output_dtype: str | None = None
    input_channels: int | None = None

    def require_operation(self, model_name: str, operation: Operation) -> None:
        """Reject unsupported requests before loading or processing model inputs.

        Args:
            model_name: Registry identifier reported in the domain error.
            operation: Requested public inference operation.

        Raises:
            UnsupportedOperationError: This provider cannot perform the operation.
        """
        if operation not in self.operations:
            raise UnsupportedOperationError(
                model_name, operation, supported_operations=self.operations
            )

    def validate_entry(self, entry: ModelEntry) -> None:
        """Validate shared source/output requirements then provider metadata.

        Args:
            entry: Typed registry entry being constructed or loaded.

        Raises:
            ValueError: Source, category, output or provider metadata disagree.
            SecurityError: An architecture selector violates the local allowlist.
        """
        actual = (
            entry.source.type,
            entry.category,
            entry.output.type,
            entry.output.format,
        )
        expected = (
            self.source_type,
            self.category,
            self.output_type,
            self.output_format,
        )
        if actual != expected:
            raise ValueError(f"Incompatible source/category/output for {self.id}")
        if self.source_type == "huggingface" and not (
            entry.source.repo and entry.source.revision and entry.source.files
        ):
            raise ValueError(
                "Hugging Face sources require repo, pinned revision, and files"
            )
        if entry.output.dtype is not None and entry.output.dtype != self.output_dtype:
            raise ValueError(f"{self.id} output dtype must be {self.output_dtype}")
        normalization_channels = len(entry.band_contract.normalization.mean)
        for sensor, contract in entry.band_contract.sensors.items():
            channels = len(contract.bands)
            if self.input_channels is not None and channels != self.input_channels:
                raise ValueError(
                    f"{self.id} sensor '{sensor}' requires {self.input_channels} bands"
                )
            if channels != normalization_channels:
                raise ValueError(
                    f"{self.id} sensor '{sensor}' bands must align 1:1 "
                    "with normalization mean/std"
                )
        self._validate_metadata(entry)

    @abstractmethod
    def _validate_metadata(self, entry: ModelEntry) -> None:
        """Validate provider-specific architecture and required metadata."""

    @abstractmethod
    def load(
        self,
        entry: ModelEntry,
        manifest: Manifest,
        device: str | None,
        *,
        paths: WhirldPaths,
    ) -> ModelBackend:
        """Lazily construct a backend using trusted code and leased artifact paths.

        Args:
            entry: Validated model metadata selecting this provider.
            manifest: Verified installation snapshot, protected by the caller's lease.
            device: Explicit device or None for the adapter's normal selection.
            paths: Artifact paths belonging to the same leased installation.

        Returns:
            A ready backend implementing this definition's operations.

        Raises:
            WhirldError: Dependencies, artifacts or device cannot initialize.
        """

    @staticmethod
    def _embedding_dimension(entry: ModelEntry) -> int:
        """Require a positive declared embedding width and an aligned output shape."""
        dimension = entry.output.embed_dim
        if dimension is None or dimension <= 0:
            raise ValueError("Embedding output requires a positive embed_dim")
        if entry.output.shape is not None and (
            len(entry.output.shape) != 2 or entry.output.shape[1] != dimension
        ):
            raise ValueError("Embedding output shape must align with embed_dim")
        return dimension


class ReferenceProvider(ProviderDefinition):
    """Deterministic NumPy embeddings with configurable positive output width."""

    def _validate_metadata(self, entry: ModelEntry) -> None:
        """Require reproducible seed and dimension without imposing a fixed width."""
        self._embedding_dimension(entry)
        if entry.source.seed is None or entry.source.seed < 0:
            raise ValueError(
                "Reference source requires a nonnegative deterministic seed"
            )

    def load(
        self,
        entry: ModelEntry,
        manifest: Manifest,
        device: str | None,
        *,
        paths: WhirldPaths,
    ) -> ModelBackend:
        """Import and load the reference adapter only when explicitly selected."""
        from .clay import ClayBackend

        return ClayBackend.load(entry, manifest, device, paths=paths)


class ClayProvider(ProviderDefinition):
    """Published Clay v1.5 large encoder and its spectral metadata contract."""

    def _validate_metadata(self, entry: ModelEntry) -> None:
        """Reject incompatible encoder widths, patch sizes and spectral metadata."""
        if self._embedding_dimension(entry) != 1024:
            raise ValueError("Clay v1.5 large requires embed_dim=1024")
        if entry.band_contract.patch_size != 8:
            raise ValueError("Clay v1.5 large requires patch_size=8")
        for sensor, contract in entry.band_contract.sensors.items():
            waves = contract.wavelengths
            if (
                waves is None
                or len(waves) != len(contract.bands)
                or any(not math.isfinite(value) or value <= 0 for value in waves)
            ):
                raise ValueError(
                    f"Clay sensor '{sensor}' requires finite positive wavelengths "
                    "aligned 1:1 with bands"
                )

    def load(
        self,
        entry: ModelEntry,
        manifest: Manifest,
        device: str | None,
        *,
        paths: WhirldPaths,
    ) -> ModelBackend:
        """Import and load the published Clay adapter only when selected."""
        from .clay_torch import ClayTorchBackend

        return ClayTorchBackend.load(entry, manifest, device, paths=paths)


class RemoteCLIPProvider(ProviderDefinition):
    """Local OpenCLIP architecture allowlist for prompt-relative classification."""

    @staticmethod
    def validate_architecture(value: str | None) -> str:
        """Return a supported local architecture or reject external selectors.

        Args:
            value: Architecture name supplied by registry metadata or direct code.

        Raises:
            SecurityError: The name is absent or could select unsupported code/data.
        """
        if value not in {"RN50", "ViT-B-32", "ViT-L-14"}:
            raise SecurityError(
                "RemoteCLIP model_name must be a supported local architecture: "
                "RN50, ViT-B-32, or ViT-L-14. External source selectors and "
                "architectures with external tokenizer/model dependencies "
                "are not allowed."
            )
        assert value is not None
        return value

    def _validate_metadata(self, entry: ModelEntry) -> None:
        """Require the closed local architecture policy for every registry entry."""
        self.validate_architecture(entry.model_name)

    def load(
        self,
        entry: ModelEntry,
        manifest: Manifest,
        device: str | None,
        *,
        paths: WhirldPaths,
    ) -> ModelBackend:
        """Import and load RemoteCLIP without accepting registry import targets."""
        from .remoteclip import RemoteCLIPBackend

        return RemoteCLIPBackend.load(entry, manifest, device, paths=paths)


class PrithviProvider(ProviderDefinition):
    """Trusted TerraTorch configuration and mask metadata requirements."""

    # The uint8 GeoTIFF contract reserves label 255 for source-invalid pixels.
    MAX_CLASSES: ClassVar[int] = 255

    def _validate_metadata(self, entry: ModelEntry) -> None:
        """Require trusted artifacts and classes representable alongside nodata."""
        if entry.trust != "first-party":
            raise ValueError(
                "Prithvi's executable configuration requires first-party trust"
            )
        if not entry.config_file or entry.config_file not in entry.source.files:
            raise ValueError("Prithvi config_file must be a declared source artifact")
        if entry.source.files[0].endswith(".safetensors"):
            raise ValueError(
                "Prithvi requires a trusted Lightning checkpoint, not safetensors"
            )
        if (
            entry.output.classes is None
            or not 1 <= entry.output.classes <= self.MAX_CLASSES
        ):
            raise ValueError(
                f"Prithvi mask output requires 1..{self.MAX_CLASSES} classes; "
                "uint8 label 255 is reserved for nodata"
            )

    def load(
        self,
        entry: ModelEntry,
        manifest: Manifest,
        device: str | None,
        *,
        paths: WhirldPaths,
    ) -> ModelBackend:
        """Import and load the trusted TerraTorch adapter only when selected."""
        from .prithvi import PrithviBackend

        return PrithviBackend.load(entry, manifest, device, paths=paths)


class ProviderCatalog:
    """Closed immutable catalog; registry values never determine import paths."""

    _COMMON: ClassVar[tuple[RuntimeDependency, ...]] = (
        RuntimeDependency(distribution="numpy", module="numpy"),
        RuntimeDependency(distribution="rasterio", module="rasterio"),
    )
    _TENSOR: ClassVar[tuple[RuntimeDependency, ...]] = (
        RuntimeDependency(distribution="torch", module="torch"),
        RuntimeDependency(distribution="huggingface_hub", module="huggingface_hub"),
    )
    _PROVIDERS: ClassVar[tuple[ProviderDefinition, ...]] = (
        ReferenceProvider(
            id="clay-reference",
            operations=("embed",),
            dependencies=_COMMON,
            source_type="reference",
            category="embedding",
            output_type="embedding",
            output_format="npy",
            output_dtype="float32",
        ),
        ClayProvider(
            id="clay",
            operations=("embed",),
            optional_extra="hf",
            dependencies=_COMMON
            + _TENSOR
            + (
                RuntimeDependency(distribution="einops", module="einops"),
                RuntimeDependency(distribution="safetensors", module="safetensors"),
            ),
            source_type="huggingface",
            category="embedding",
            output_type="embedding",
            output_format="npy",
            output_dtype="float32",
        ),
        RemoteCLIPProvider(
            id="remoteclip",
            operations=("classify",),
            optional_extra="remoteclip",
            dependencies=_COMMON
            + _TENSOR
            + (
                RuntimeDependency(distribution="open_clip_torch", module="open_clip"),
                RuntimeDependency(distribution="safetensors", module="safetensors"),
            ),
            source_type="huggingface",
            category="classification",
            output_type="scores",
            output_format="geojson",
            output_dtype="float32",
            input_channels=3,
        ),
        PrithviProvider(
            id="prithvi",
            operations=("segment",),
            optional_extra="prithvi",
            dependencies=_COMMON
            + _TENSOR
            + (
                RuntimeDependency(distribution="terratorch", module="terratorch"),
                RuntimeDependency(distribution="torchgeo", module="torchgeo"),
            ),
            source_type="huggingface",
            category="segmentation",
            output_type="mask",
            output_format="geotiff",
            output_dtype="uint8",
            input_channels=6,
        ),
    )

    @classmethod
    def all(cls) -> tuple[ProviderDefinition, ...]:
        """Return immutable definitions for documentation and conformance checks."""
        return cls._PROVIDERS

    @classmethod
    def get(cls, backend_id: str | None) -> ProviderDefinition:
        """Resolve a trusted backend id without importing a concrete adapter.

        Args:
            backend_id: Registry backend identifier; never an import selector.

        Raises:
            WhirldError: The backend is missing or unsupported in this build.
        """
        for definition in cls._PROVIDERS:
            if definition.id == backend_id:
                return definition
        raise WhirldError(f"Unsupported backend: {backend_id!r}")
