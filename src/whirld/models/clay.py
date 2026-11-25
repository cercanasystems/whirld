"""Clay v1 backend.

This build ships the **offline reference backend**: a deterministic, numpy-only
stand-in for Clay's encoder. It produces a reproducible 512-dim embedding per
chip by projecting per-band pooled statistics through a fixed seeded random
matrix. It imports no torch and downloads nothing, so the whole embed pipeline
runs offline and is exactly testable.

The learned Clay v1.5 encoder implements the same shared interface in
:mod:`whirld.models.clay_torch`. This reference provider remains a deterministic
offline option for testing the complete translation and embedding pipeline.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from .. import config
from ..core.fetch import Manifest
from ..core.registry import ModelEntry
from ..errors import BackendContractError
from ..logging_setup import get_logger
from .base import BackendContract, ModelBackend, detect_device

_log = get_logger("models.clay")


class ClayBackend(ModelBackend):
    """Deterministic offline reference implementation of Clay v1.

    Args:
        manifest: The local manifest produced by ``whirld pull`` (provides the
            embedding dimension and deterministic seed).
        device: Resolved inference device string.
    """

    provider_id: ClassVar[str] = "clay-reference"

    def __init__(self, manifest: Manifest, device: str) -> None:
        super().__init__(name=manifest.name, version=manifest.version, device=device)
        if manifest.embed_dim is None or manifest.seed is None:
            raise BackendContractError(
                "Clay reference backend requires embed_dim and seed in the manifest."
            )
        self._contract = BackendContract.create(self.name, embed_dim=manifest.embed_dim)
        self.embed_dim = manifest.embed_dim
        if isinstance(manifest.seed, bool) or manifest.seed < 0:
            raise BackendContractError(
                "Clay reference requires a nonnegative integer seed."
            )
        self._seed = manifest.seed
        self._projection: np.ndarray | None = None
        self._bias: np.ndarray | None = None

    @classmethod
    def load(
        cls,
        entry: ModelEntry,
        manifest: Manifest,
        device: str | None = None,
        *,
        paths: config.WhirldPaths | None = None,
    ) -> ClayBackend:
        """Construct a backend from a registry entry and local manifest.

        Args:
            entry: The validated registry entry (unused by the reference
                backend beyond interface symmetry; the real path would read
                architecture details from it).
            manifest: The local manifest from ``whirld pull``.
            device: Requested device, or ``None`` for auto-detection.
            paths: Owning store paths; unused by the in-memory reference backend.

        Returns:
            A ready :class:`ClayBackend`.
        """
        entry.require_operation("embed")
        resolved = detect_device(device)
        _log.info("Loading clay-v1 reference backend on device '%s'.", resolved)
        return cls(manifest=manifest, device=resolved)

    def _ensure_projection(self, n_features: int) -> tuple[np.ndarray, np.ndarray]:
        """Lazily build the deterministic projection matrix and bias.

        Args:
            n_features: Length of the per-chip feature vector.

        Returns:
            A tuple ``(projection, bias)`` with shapes ``(embed_dim, n_features)``
            and ``(embed_dim,)``.
        """
        if self._projection is None or self._projection.shape[1] != n_features:
            rng = np.random.default_rng(self._seed)
            self._projection = rng.standard_normal((self.embed_dim, n_features)).astype(
                np.float32
            ) / np.sqrt(n_features)
            self._bias = rng.standard_normal(self.embed_dim).astype(np.float32)
        assert self._bias is not None
        return self._projection, self._bias

    def embed(self, chips: np.ndarray, context: object = None) -> np.ndarray:
        """Embed a batch of chips deterministically.

        Reference behavior: pool each chip to per-band mean and std (a
        ``2 * bands`` feature vector), project through a fixed seeded matrix, and
        apply ``tanh``. Identical inputs always yield identical embeddings. The
        real torch-backed encoder lives in
        :class:`~whirld.models.clay_torch.ClayTorchBackend` (model ``clay-v1.5``).

        Args:
            chips: Array ``(n_chips, bands, height, width)``, float32.
            context: Ignored by the reference backend (spectral metadata).

        Returns:
            Array ``(n_chips, embed_dim)``, float32.
        """
        self._contract.chips(chips)
        if chips.shape[0] == 0:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        per_band_mean = chips.mean(axis=(2, 3))
        per_band_std = chips.std(axis=(2, 3))
        features = np.concatenate([per_band_mean, per_band_std], axis=1).astype(
            np.float32
        )

        projection, bias = self._ensure_projection(features.shape[1])
        embeddings = np.tanh(features @ projection.T + bias)
        return self._contract.embeddings(embeddings.astype(np.float32), chips.shape[0])
