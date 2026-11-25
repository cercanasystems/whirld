"""Abstract model-backend interface and device selection.

Every model Whirld runs is wrapped in a :class:`ModelBackend`. The interface is
shared by offline and learned providers. The closed provider catalog declares
which operations each implementation supports. Optional operations remain on this
interface and consistently raise UnsupportedOperationError when unavailable.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from torch import Tensor

import numpy as np

from ..core.options import InferenceOptions
from ..errors import BackendContractError, UnsupportedOperationError
from ..logging_setup import get_logger

_log = get_logger("models.base")


@dataclass(frozen=True)
class InferenceContext:
    """Per-request spectral/geometric/acquisition metadata for backends that need it.

    The reference backend ignores this; metadata-conditioned models (e.g. Clay)
    use it to build their datacube. Spectral fields are derived from the detected
    sensor's band contract (so they match the chips' band order); ``latlons`` and
    ``acquisition_datetime`` are derived from the raster geometry and tags/flags.

    Attributes:
        sensor: Detected sensor key.
        gsd_m: Ground sample distance (target resolution) in meters.
        wavelengths: Per-band center wavelengths (micrometres), aligned to the
            chips' band order, or ``None`` if the contract declares none.
        latlons: Per-chip ``(lat, lon)`` centroids in degrees (EPSG:4326), aligned
            to the chip order, or ``None`` if unavailable.
        acquisition_datetime: Scene acquisition time (UTC), or ``None`` if unknown.
        options: Immutable per-call execution policy; omitted settings inherit
            the backend's defaults without mutating the resident model.
    """

    sensor: str
    gsd_m: float
    wavelengths: list[float] | None = None
    latlons: list[tuple[float, float]] | None = None
    acquisition_datetime: datetime | None = None
    options: InferenceOptions = field(default_factory=InferenceOptions)


def detect_device(requested: str | None = None) -> str:
    """Select an inference device following PRD section 12.1 precedence.

    Order: an explicit request wins; otherwise CUDA, then Apple MPS, then CPU.
    PyTorch is imported lazily so the reference backend never forces a torch
    import; when torch is unavailable the result is ``cpu``.

    Args:
        requested: One of ``cuda``, ``mps``, ``cpu``, or ``None`` for auto.

    Returns:
        The selected device string.
    """
    if requested is not None and requested != "auto":
        return requested
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        _log.debug("torch not installed; defaulting device to cpu.")
    return "cpu"


class BackendContract(BaseModel):
    """Immutable declared metadata and validation owned by an inference backend.

    Validators reject malformed arrays before model work and malformed model
    results before casting or returning them. Tensor validation imports PyTorch
    only when a learned provider actually produces a tensor.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    model_name: str
    embed_dim: Annotated[int, Field(strict=True, gt=0)] | None = None
    classes: (
        Annotated[int, Field(strict=True, ge=1, le=int(np.iinfo(np.uint8).max))] | None
    ) = None
    channels: Annotated[int, Field(strict=True, gt=0)] | None = None
    patch_size: Annotated[int, Field(strict=True, gt=0)] | None = None

    @classmethod
    def create(
        cls,
        model_name: str,
        *,
        embed_dim: int | None = None,
        classes: int | None = None,
        channels: int | None = None,
        patch_size: int | None = None,
    ) -> BackendContract:
        """Bind validated metadata, translating schema failures to domain errors."""
        try:
            return cls(
                model_name=model_name,
                embed_dim=embed_dim,
                classes=classes,
                channels=channels,
                patch_size=patch_size,
            )
        except ValidationError as exc:
            raise BackendContractError(
                f"Invalid backend metadata for '{model_name}': {exc}"
            ) from exc

    def fail(self, detail: str) -> None:
        """Raise an actionable domain error identifying the responsible model."""
        raise BackendContractError(f"Model '{self.model_name}' contract: {detail}")

    def array(
        self,
        value: object,
        shape: tuple[int | None, ...],
        dtype: np.dtype,
        label: str,
    ) -> np.ndarray:
        """Require exact array rank, dimensions, dtype and finite numeric values."""
        if not isinstance(value, np.ndarray):
            self.fail(f"{label} must be a NumPy array.")
        assert isinstance(value, np.ndarray)
        if len(value.shape) != len(shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(value.shape, shape, strict=False)
        ):
            self.fail(f"{label} expected shape {shape}, got {value.shape}.")
        if value.dtype != dtype:
            self.fail(f"{label} expected dtype {dtype}, got {value.dtype}.")
        # Bounded temporary storage even for a full scene supplied by direct users.
        for start in range(0, value.size, 65536):
            if not np.isfinite(value.flat[start : start + 65536]).all():
                self.fail(f"{label} contains non-finite values.")
        return value

    def chips(self, chips: np.ndarray) -> None:
        """Validate normalized NCHW float32 input, allowing an empty batch."""
        self.array(
            chips, (None, self.channels, None, None), np.dtype("float32"), "chips"
        )
        if any(size <= 0 for size in chips.shape[1:]):
            self.fail("chips require positive channel and spatial dimensions.")
        if self.patch_size is not None and any(
            size % self.patch_size for size in chips.shape[2:]
        ):
            self.fail(f"chip height and width must be multiples of {self.patch_size}.")

    def embeddings(self, value: np.ndarray, count: int) -> np.ndarray:
        """Validate embedding output against the declared embedding dimension."""
        return self.array(
            value, (count, self.embed_dim), np.dtype("float32"), "embeddings"
        )

    def probabilities(self, value: np.ndarray, count: int, queries: int) -> np.ndarray:
        """Validate prompt-relative scores; a single background column is omitted."""
        self.array(
            value, (count, queries), np.dtype("float32"), "classification scores"
        )
        if np.any(value < 0) or np.any(value > 1):
            self.fail("classification scores must lie in [0, 1].")
        if queries > 1 and not np.allclose(value.sum(axis=1), 1, rtol=1e-5, atol=1e-6):
            self.fail("classification score rows must sum to one.")
        return value

    def masks(self, value: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        """Require uint8 class indices within the declared class vocabulary."""
        self.array(value, shape, np.dtype("uint8"), "segmentation masks")
        if self.classes is not None and np.any(value >= self.classes):
            self.fail(
                f"segmentation masks contain indices outside {self.classes} classes."
            )
        return value

    def tensor(
        self,
        value: object,
        shape: tuple[int | None, ...],
        label: str,
        *,
        nonzero_rows: bool = False,
    ) -> Tensor:
        """Check learned floating tensors before arithmetic can hide bad outputs."""
        import torch  # noqa: PLC0415

        if not isinstance(value, torch.Tensor):
            self.fail(f"{label} must be a PyTorch tensor.")
        assert isinstance(value, torch.Tensor)
        if value.ndim != len(shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(value.shape, shape, strict=False)
        ):
            self.fail(f"{label} expected shape {shape}, got {tuple(value.shape)}.")
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            self.fail(f"{label} must contain finite floating-point values.")
        if nonzero_rows and (
            value.shape[-1] == 0
            or not bool(torch.isfinite(value.norm(dim=-1)).all())
            or bool((value.norm(dim=-1) == 0).any())
        ):
            self.fail(f"{label} must have nonzero feature vectors.")
        return value


class ModelBackend(abc.ABC):
    """Abstract base class for all model backends.

    Args:
        name: Model identifier.
        version: Model version.
        device: Resolved inference device string.
    """

    provider_id: ClassVar[str | None] = None

    @property
    def supported_operations(self) -> tuple[str, ...]:
        """Return capabilities owned by the closed catalog; custom fakes opt out."""
        if self.provider_id is None:
            return ()
        from .providers import ProviderCatalog  # noqa: PLC0415

        return ProviderCatalog.get(self.provider_id).operations

    def __init__(
        self, name: str, version: str, device: str, *, batch_size: int | None = None
    ) -> None:
        self.name = name
        self.version = version
        self.device = device
        self._default_options = InferenceOptions.from_arguments(batch_size=batch_size)

    def batches(
        self, count: int, context: InferenceContext | None = None
    ) -> Iterator[slice]:
        """Partition chips under this call's policy and the backend defaults.

        The backend owns its defaults; request overrides remain in the context
        and cannot leak into another call, including concurrent or failed calls.
        """
        options = context.options if context is not None else self._default_options
        return options.batches(count, fallback=self._default_options)

    @abc.abstractmethod
    def embed(
        self, chips: np.ndarray, context: InferenceContext | None = None
    ) -> np.ndarray:
        """Embed a batch of chips.

        Args:
            chips: Array of shape ``(n_chips, bands, height, width)``, float32,
                already band-contract-translated and normalized.
            context: Optional per-request spectral/geometric metadata. Backends
                that don't need it (e.g. the reference backend) ignore it.

        Returns:
            Array of shape ``(n_chips, embed_dim)``, float32.

        Raises:
            UnsupportedOperationError: The backend does not support embedding.
        """
        raise UnsupportedOperationError(
            self.name, "embed", supported_operations=self.supported_operations
        )

    def segment(
        self,
        chips: np.ndarray,
        head: str | None = None,
        threshold: float = 0.5,
        context: InferenceContext | None = None,
    ) -> np.ndarray:
        """Run dense per-pixel prediction over a batch of chips.

        Args:
            chips: Array ``(n_chips, bands, height, width)``, float32, already
                band-contract-normalized.
            head: Task head name (informational; the model already encodes it).
            threshold: Binary mask threshold on the positive class probability;
                ``0.5`` is equivalent to argmax.
            context: Optional per-request metadata (ignored by most backends).

        Returns:
            Per-chip class-index masks, shape ``(n_chips, height, width)``, uint8.

        Raises:
            UnsupportedOperationError: The backend does not support segmentation.
        """
        raise UnsupportedOperationError(
            self.name, "segment", supported_operations=self.supported_operations
        )

    def classify(
        self,
        chips: np.ndarray,
        queries: list[str],
        context: InferenceContext | None = None,
    ) -> np.ndarray:
        """Score each chip against one or more text queries (zero-shot).

        Args:
            chips: Array ``(n_chips, bands, height, width)``, float32, already
                band-contract-normalized.
            queries: One or more free-text queries.
            context: Optional per-request metadata (ignored by most backends).

        Returns:
            Per-chip, per-query probabilities, shape ``(n_chips, len(queries))``,
            float32, each in ``[0, 1]``.

        Raises:
            UnsupportedOperationError: The backend does not support classification.
        """
        raise UnsupportedOperationError(
            self.name, "classify", supported_operations=self.supported_operations
        )
