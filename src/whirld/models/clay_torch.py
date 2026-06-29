"""Real Clay v1.5 backend — runs the actual encoder weights via PyTorch.

This is the production counterpart to the offline reference backend in
:mod:`whirld.models.clay`. It loads the genuine Clay v1.5 "large" encoder
(vendored under :mod:`whirld.models._vendor.clay_v15`) from the downloaded
checkpoint and produces real 1024-dim embeddings (the encoder CLS token).

Requires the ``hf`` extra (``torch`` + ``einops``).

**Metadata.** Clay's encoder consumes a "datacube": normalized pixels plus spectral
metadata (wavelengths, GSD) and acquisition metadata (time, lat/lon). All are
supplied faithfully — wavelengths/GSD from the band contract, per-chip lat/lon
reprojected from the raster geometry, and ``time`` from the scene acquisition
datetime (``--datetime`` or ``TIFFTAG_DATETIME``). When the datetime is unknown,
``time`` falls back to zeros (Clay's neutral value); lat/lon is always derived.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from .. import config
from ..core.fetch import Manifest
from ..core.registry import ModelEntry
from ..errors import ModelNotInstalledError, WhirldError
from ..logging_setup import get_logger
from .base import InferenceContext, ModelBackend, detect_device

_log = get_logger("models.clay_torch")

_ENCODER_PREFIX = "model.encoder."
_DEFAULT_BATCH = 8


def _import_torch():
    """Lazily import torch, raising an actionable error if the extra is absent.

    Returns:
        The imported ``torch`` module.

    Raises:
        WhirldError: ``torch`` is not installed.
    """
    try:
        import torch  # noqa: PLC0415

        return torch
    except ImportError as exc:
        raise WhirldError(
            "Running clay-v1.5 requires the optional 'hf' extra (torch + einops).\n"
            "       Install it:  pip install 'whirld[hf]'"
        ) from exc


class ClayTorchBackend(ModelBackend):
    """Torch-backed Clay v1.5 encoder backend.

    Args:
        name: Model identifier.
        version: Model version.
        device: Resolved inference device string.
        embed_dim: Embedding dimensionality (1024 for Clay v1.5 large).
        patch_size: ViT patch size in pixels.
        ckpt_path: Path to the downloaded ``.ckpt`` (``None`` if an encoder is
            injected directly, e.g. in tests).
        encoder: A prebuilt encoder to use instead of loading from ``ckpt_path``
            (used by tests with a tiny architecture).
        batch_size: Inference batch size over chips.
    """

    def __init__(
        self,
        name: str,
        version: str,
        device: str,
        *,
        embed_dim: int,
        patch_size: int,
        ckpt_path: Path | None = None,
        encoder: object | None = None,
        batch_size: int = _DEFAULT_BATCH,
    ) -> None:
        super().__init__(name=name, version=version, device=device)
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self._ckpt_path = ckpt_path
        self._encoder = encoder
        self._batch_size = batch_size

    @classmethod
    def load(
        cls,
        entry: ModelEntry,
        manifest: Manifest,
        device: str | None = None,
    ) -> ClayTorchBackend:
        """Build the backend and load the real encoder weights.

        Args:
            entry: The validated registry entry.
            manifest: The local manifest from ``whirld pull``.
            device: Requested device, or ``None`` for auto-detection.

        Returns:
            A ready :class:`ClayTorchBackend` with weights loaded.

        Raises:
            ModelNotInstalledError: The checkpoint file is missing.
            WhirldError: The ``hf`` extra is not installed.
        """
        resolved = detect_device(device)
        ckpt = config.get_paths().model_dir(entry.name) / manifest.weights_file
        if not ckpt.exists():
            raise ModelNotInstalledError(
                f"Checkpoint for '{entry.name}' is missing at {ckpt}.\n"
                f"       Re-pull it:  whirld pull {entry.name}"
            )
        backend = cls(
            name=manifest.name,
            version=manifest.version,
            device=resolved,
            embed_dim=entry.output.embed_dim or 1024,
            patch_size=entry.band_contract.patch_size or 8,
            ckpt_path=ckpt,
        )
        backend._ensure_encoder()
        return backend

    def _ensure_encoder(self) -> object:
        """Build the encoder and load weights on first use (or return the cached one).

        Returns:
            The ready encoder module (in eval mode, on ``self.device``).
        """
        if self._encoder is not None:
            return self._encoder

        torch = _import_torch()
        from ._vendor.clay_v15 import clay_v15_large_encoder  # noqa: PLC0415

        _log.info("Loading clay-v1.5 encoder weights on device '%s'.", self.device)
        encoder = clay_v15_large_encoder()
        checkpoint = torch.load(
            self._ckpt_path, map_location="cpu", weights_only=False, mmap=True
        )
        state_dict = checkpoint["state_dict"]
        encoder_state = {
            key[len(_ENCODER_PREFIX) :]: value
            for key, value in state_dict.items()
            if key.startswith(_ENCODER_PREFIX)
        }
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        _log.info(
            "clay-v1.5 weights: %d tensors, %d missing, %d unexpected.",
            len(encoder_state),
            len(missing),
            len(unexpected),
        )
        if unexpected:
            _log.warning("Unexpected keys when loading clay-v1.5: %s", unexpected)
        encoder.to(self.device).eval()
        self._encoder = encoder
        return encoder

    def embed(
        self, chips: np.ndarray, context: InferenceContext | None = None
    ) -> np.ndarray:
        """Embed chips with the real Clay encoder (CLS token).

        Args:
            chips: Array ``(n_chips, bands, height, width)``, float32, already
                band-contract-normalized.
            context: Spectral/geometric metadata (wavelengths + GSD). Required —
                Clay's encoder cannot run without per-band wavelengths.

        Returns:
            Array ``(n_chips, embed_dim)``, float32.

        Raises:
            ValueError: ``chips`` is not 4-D.
            WhirldError: Wavelength metadata is missing from the contract.
        """
        if chips.ndim != 4:
            raise ValueError(
                f"Expected chips of shape (n, bands, h, w), got {chips.shape}."
            )
        if chips.shape[0] == 0:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        wavelengths = context.wavelengths if context else None
        if not wavelengths:
            raise WhirldError(
                f"Model '{self.name}' needs per-band wavelengths, but the band "
                f"contract declares none for this sensor."
            )
        gsd = context.gsd_m if context else float(self.patch_size)

        latlons = context.latlons if context else None
        acq_dt = context.acquisition_datetime if context else None

        torch = _import_torch()
        encoder = self._ensure_encoder()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, chips.shape[0], self._batch_size):
                stop = start + self._batch_size
                batch = chips[start:stop]
                batch_latlons = latlons[start:stop] if latlons is not None else None
                datacube = self._build_datacube(
                    batch, wavelengths, gsd, acq_dt, batch_latlons, torch
                )
                encoded, *_ = encoder(datacube)
                outputs.append(encoded[:, 0, :].cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def _build_datacube(
        self,
        chips: np.ndarray,
        wavelengths: list[float],
        gsd: float,
        acquisition_datetime: datetime | None,
        latlons: list[tuple[float, float]] | None,
        torch: object,
    ) -> dict:
        """Assemble Clay's encoder input dict for a chip batch.

        ``time`` uses the scene's acquisition datetime (Clay's sin/cos encoding) for
        every chip in the batch; ``latlon`` uses each chip's centroid. Missing
        metadata falls back to zeros — faithful where present, never erroring.

        Args:
            chips: Normalized chip batch ``(b, bands, h, w)``.
            wavelengths: Per-band center wavelengths (micrometres).
            gsd: Ground sample distance in meters.
            acquisition_datetime: Scene acquisition time, or ``None`` → zeros.
            latlons: Per-chip ``(lat, lon)`` for this batch, or ``None`` → zeros.
            torch: The imported torch module.

        Returns:
            The datacube dict the encoder ``forward`` expects.
        """
        from ._clay_metadata import latlon_vector, time_vector  # noqa: PLC0415

        batch = chips.shape[0]
        pixels = torch.from_numpy(np.ascontiguousarray(chips)).float().to(self.device)
        time_row = time_vector(acquisition_datetime)
        time = torch.tensor([time_row] * batch, dtype=torch.float32, device=self.device)
        if latlons is not None:
            latlon_rows = [latlon_vector(ll) for ll in latlons]
        else:
            latlon_rows = [latlon_vector(None) for _ in range(batch)]
        latlon = torch.tensor(latlon_rows, dtype=torch.float32, device=self.device)
        return {
            "pixels": pixels,
            "time": time,
            "latlon": latlon,
            "gsd": torch.tensor(float(gsd), device=self.device),
            "waves": torch.tensor(wavelengths, dtype=torch.float32, device=self.device),
        }
