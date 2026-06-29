"""Abstract model-backend interface and device selection.

Every model Whirld runs is wrapped in a :class:`ModelBackend`. The interface is
deliberately narrow so that the offline reference backend and a future real
PyTorch backend are interchangeable: the translation pipeline and CLI never know
which one they are talking to.

Capabilities are expressed as optional methods. The walking-skeleton scope only
implements :meth:`ModelBackend.embed`; ``segment`` and ``classify`` raise
``NotImplementedError`` until their backends land.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime

import numpy as np

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
    """

    sensor: str
    gsd_m: float
    wavelengths: list[float] | None = None
    latlons: list[tuple[float, float]] | None = None
    acquisition_datetime: datetime | None = None


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


class ModelBackend(abc.ABC):
    """Abstract base class for all model backends.

    Args:
        name: Model identifier.
        version: Model version.
        device: Resolved inference device string.
    """

    def __init__(self, name: str, version: str, device: str) -> None:
        self.name = name
        self.version = version
        self.device = device

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
            NotImplementedError: The backend does not support embedding.
        """
        raise NotImplementedError

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
            NotImplementedError: The backend does not support segmentation.
        """
        raise NotImplementedError(
            f"{self.name} does not support segmentation in this build."
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
            NotImplementedError: The backend does not support classification.
        """
        raise NotImplementedError(
            f"{self.name} does not support classification in this build."
        )
