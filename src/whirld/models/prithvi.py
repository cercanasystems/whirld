"""Prithvi EO 2.0 backend — per-pixel segmentation via TerraTorch.

Prithvi's fine-tuned segmentation heads (burn-scar, flood) are TerraTorch
``SemanticSegmentationTask`` checkpoints. Each head is a separate checkpoint plus a
YAML config that defines the architecture; TerraTorch rebuilds the model from the
config and loads the weights. Whirld owns normalization (via the band contract), so
this backend feeds **already-normalized** HLS tiles straight to the raw ``nn.Module``
and turns the logits into a class-index mask.

Requires the ``prithvi`` extra (``terratorch``). Unlike Clay, no vendoring is needed
— TerraTorch installs cleanly on Python 3.13.

**Single-image (T=1).** The published flood/burn-scar heads are single-scene binary
segmentation; the PRD's before/after temporal stacking does not apply.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config
from ..core.fetch import Manifest
from ..core.registry import ModelEntry
from ..errors import ModelNotInstalledError, WhirldError
from ..logging_setup import get_logger
from .base import InferenceContext, ModelBackend, detect_device

_log = get_logger("models.prithvi")

_DEFAULT_BATCH = 4


def _build_inference_model(config_path: Path, ckpt_path: Path):
    """Lazily build a TerraTorch inference model, with a clear missing-extra error.

    Args:
        config_path: Path to the TerraTorch YAML config.
        ckpt_path: Path to the ``.pt`` checkpoint.

    Returns:
        The underlying ``nn.Module`` (TerraTorch ``SemanticSegmentationTask.model``).

    Raises:
        WhirldError: ``terratorch`` is not installed.
    """
    try:
        from terratorch.cli_tools import LightningInferenceModel  # noqa: PLC0415
    except ImportError as exc:
        raise WhirldError(
            "Prithvi segmentation requires the optional 'prithvi' extra "
            "(terratorch).\n       Install it:  pip install 'whirld[prithvi]'"
        ) from exc

    lightning_model = LightningInferenceModel.from_config(
        str(config_path), str(ckpt_path)
    )
    return lightning_model.model


class PrithviBackend(ModelBackend):
    """TerraTorch-backed Prithvi segmentation backend.

    Args:
        name: Model identifier.
        version: Model version.
        device: Resolved inference device string.
        classes: Number of output classes (2 for the binary heads).
        config_path: Path to the TerraTorch YAML config (``None`` if a model is
            injected directly, e.g. in tests).
        ckpt_path: Path to the ``.pt`` checkpoint (``None`` if injected).
        model: A prebuilt segmentation module to use instead of loading (tests).
        batch_size: Inference batch size over chips.
    """

    def __init__(
        self,
        name: str,
        version: str,
        device: str,
        *,
        classes: int,
        config_path: Path | None = None,
        ckpt_path: Path | None = None,
        model: object | None = None,
        batch_size: int = _DEFAULT_BATCH,
    ) -> None:
        super().__init__(name=name, version=version, device=device)
        self.classes = int(classes)
        self._config_path = config_path
        self._ckpt_path = ckpt_path
        self._model = model
        self._batch_size = batch_size

    @classmethod
    def load(
        cls,
        entry: ModelEntry,
        manifest: Manifest,
        device: str | None = None,
    ) -> PrithviBackend:
        """Build the backend and load the real TerraTorch weights.

        Args:
            entry: The validated registry entry (provides ``config_file``/classes).
            manifest: The local manifest from ``whirld pull``.
            device: Requested device, or ``None`` for auto-detection.

        Returns:
            A ready :class:`PrithviBackend` with the model loaded.

        Raises:
            ModelNotInstalledError: The checkpoint or config file is missing.
            WhirldError: The entry lacks ``config_file`` or the extra is absent.
        """
        resolved = detect_device(device)
        if not entry.config_file:
            raise WhirldError(
                f"Registry entry '{entry.name}' must declare config_file "
                f"(the TerraTorch YAML)."
            )
        model_dir = config.get_paths().model_dir(entry.name)
        ckpt = model_dir / manifest.weights_file
        cfg = model_dir / entry.config_file
        for path, kind in ((ckpt, "checkpoint"), (cfg, "config")):
            if not path.exists():
                raise ModelNotInstalledError(
                    f"{kind.capitalize()} for '{entry.name}' is missing at {path}.\n"
                    f"       Re-pull it:  whirld pull {entry.name}"
                )
        backend = cls(
            name=manifest.name,
            version=manifest.version,
            device=resolved,
            classes=entry.output.classes or 2,
            config_path=cfg,
            ckpt_path=ckpt,
        )
        backend._ensure_model()
        return backend

    def _ensure_model(self) -> object:
        """Build the TerraTorch model on first use (or return the cached one)."""
        if self._model is not None:
            return self._model
        _log.info(
            "Loading Prithvi (%s) weights on device '%s'.", self.name, self.device
        )
        model = _build_inference_model(self._config_path, self._ckpt_path)
        model.to(self.device).eval()
        self._model = model
        return model

    def embed(
        self, chips: np.ndarray, context: InferenceContext | None = None
    ) -> np.ndarray:
        """Prithvi is a segmentation model; embedding is not supported.

        Raises:
            WhirldError: Always — use ``segment`` instead.
        """
        raise WhirldError(
            f"Model '{self.name}' does not produce embeddings; use 'segment'."
        )

    def segment(
        self,
        chips: np.ndarray,
        head: str | None = None,
        threshold: float = 0.5,
        context: InferenceContext | None = None,
    ) -> np.ndarray:
        """Produce per-chip class-index masks with the real Prithvi head.

        Args:
            chips: Array ``(n_chips, 6, tile, tile)``, float32, already
                band-contract-normalized HLS reflectance.
            head: Unused (the loaded checkpoint already encodes the task).
            threshold: Positive-class probability threshold; ``0.5`` ≡ argmax.
            context: Unused.

        Returns:
            Per-chip masks ``(n_chips, tile, tile)``, uint8 class indices.

        Raises:
            ValueError: ``chips`` is not 4-D.
        """
        if chips.ndim != 4:
            raise ValueError(
                f"Expected chips of shape (n, bands, h, w), got {chips.shape}."
            )
        if chips.shape[0] == 0:
            return np.empty((0, chips.shape[2], chips.shape[3]), dtype=np.uint8)

        import torch  # noqa: PLC0415

        model = self._ensure_model()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, chips.shape[0], self._batch_size):
                batch = chips[start : start + self._batch_size]
                # Prithvi expects (B, C, T, H, W); the heads are single-frame (T=1).
                pixels = (
                    torch.from_numpy(np.ascontiguousarray(batch))
                    .float()
                    .unsqueeze(2)
                    .to(self.device)
                )
                logits = _as_logits(model(pixels))
                if abs(threshold - 0.5) < 1e-9 or logits.shape[1] != 2:
                    mask = logits.argmax(dim=1)
                else:
                    prob = logits.softmax(dim=1)[:, 1]
                    mask = (prob >= threshold).to(torch.uint8)
                outputs.append(mask.to(torch.uint8).cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.uint8)


def _as_logits(output: object):
    """Extract the logits tensor from a possibly-wrapped TerraTorch output.

    TerraTorch tasks may return a tensor or an object with a ``.output`` attribute.

    Args:
        output: The raw model output.

    Returns:
        The logits tensor ``(B, classes, H, W)``.
    """
    return getattr(output, "output", output)
