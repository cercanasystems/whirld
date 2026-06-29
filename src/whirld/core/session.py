"""Warm model session — load model backends once and keep them resident.

The ``serve`` command (PRD section 5.8) loads requested models into memory at
startup and keeps every loaded model warm across requests. :class:`ModelSession`
provides that cache: it resolves a model's registry entry and local manifest,
instantiates its backend once, and returns the cached instance on subsequent
calls. It reuses the existing building blocks — :class:`~whirld.core.registry.Registry`,
:func:`~whirld.core.fetch.load_manifest`, and
:func:`~whirld.models.loader.load_backend` — so there is one loading path.

The same cache is the natural foundation for a future public ``whirld.Session``
context manager (PRD section 14); that public surface is deferred.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..logging_setup import get_logger
from ..models.base import ModelBackend, detect_device
from ..models.loader import load_backend
from .fetch import Manifest, load_manifest
from .registry import ModelEntry, Registry

_log = get_logger("core.session")


@dataclass(frozen=True)
class LoadedModel:
    """A resident model: its registry entry, local manifest, and live backend.

    Attributes:
        entry: The validated registry entry.
        manifest: The local manifest from ``whirld pull``.
        backend: The instantiated, ready-to-run backend.
    """

    entry: ModelEntry
    manifest: Manifest
    backend: ModelBackend


class ModelSession:
    """An in-memory cache of loaded model backends keyed by ``(model, device)``.

    Args:
        device: Default device for loads; ``None`` resolves via auto-detection.
        registry: Registry instance; defaults to a freshly seeded one.
    """

    def __init__(
        self,
        device: str | None = None,
        registry: Registry | None = None,
    ) -> None:
        self._device = detect_device(device)
        self._registry = registry or Registry()
        self._cache: dict[tuple[str, str], LoadedModel] = {}

    @property
    def device(self) -> str:
        """The session's resolved default device."""
        return self._device

    @property
    def loaded(self) -> list[str]:
        """Sorted names of models currently resident in the cache."""
        return sorted({name for (name, _device) in self._cache})

    def get(self, model: str, device: str | None = None) -> LoadedModel:
        """Return a resident model, loading and caching it on first request.

        Args:
            model: Model identifier (e.g. ``clay-v1``).
            device: Device override; defaults to the session device.

        Returns:
            The cached :class:`LoadedModel`.

        Raises:
            ModelNotFoundError: Model is not in the registry.
            ModelNotInstalledError: Model has not been pulled.
            WhirldError: No backend is registered for the model.
        """
        resolved = detect_device(device) if device is not None else self._device
        key = (model, resolved)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        _log.info("Loading model '%s' on device '%s' into session.", model, resolved)
        entry = self._registry.get(model)
        manifest = load_manifest(model)
        backend = load_backend(entry, manifest, resolved)
        loaded = LoadedModel(entry=entry, manifest=manifest, backend=backend)
        self._cache[key] = loaded
        return loaded

    def preload(self, models: list[str], device: str | None = None) -> None:
        """Eagerly load a list of models (e.g. at server startup).

        Args:
            models: Model identifiers to load now.
            device: Device override; defaults to the session device.
        """
        for model in models:
            self.get(model, device)

    def clear(self) -> None:
        """Drop all resident models from the cache."""
        self._cache.clear()
