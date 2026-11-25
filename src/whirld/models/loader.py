"""Backend factory: instantiate a model's backend from its registry entry.

Backend **selection is registry-driven**: each model declares a ``backend`` id in
its YAML, and the closed provider catalog maps that id to trusted construction.
Additional weights for an already supported architecture satisfying its provider
requirements are registry data. A new architecture or loading mechanism needs a
backend implementation and a trusted provider definition; the registry schema's
backend enum is checked against that catalog.

Kept separate from ``models/__init__.py`` so ``import whirld.models`` stays cheap;
each concrete backend (and its heavy imports) is loaded lazily, only when selected.
"""

from __future__ import annotations

from ..core.fetch import Manifest, ModelStore
from ..core.registry import ModelEntry
from ..errors import RegistryError
from .base import ModelBackend
from .providers import ProviderCatalog


def load_backend(
    entry: ModelEntry,
    manifest: Manifest,
    device: str | None = None,
    *,
    store: ModelStore | None = None,
) -> ModelBackend:
    """Instantiate the backend declared by a model's registry entry.

    Args:
        entry: The validated registry entry (its ``backend`` field selects the
            implementation).
        manifest: The local manifest from ``whirld pull``.
        device: Requested device, or ``None`` for auto-detection.
        store: Model cache/registry owning the installation; defaults to the
            current environment. Its lease and artifact paths are used together.

    Returns:
        A ready :class:`ModelBackend`.

    Raises:
        WhirldError: The entry declares no backend, or an unknown one.
        RegistryError: Programmatically altered provider metadata is invalid.
        SecurityError: An architecture selector violates the local allowlist.
    """
    definition = ProviderCatalog.get(entry.backend)
    try:
        definition.validate_entry(entry)
    except ValueError as exc:
        raise RegistryError(
            f"Invalid provider metadata for '{entry.name}': {exc}"
        ) from exc
    store = store or ModelStore()
    with store.installation_lease(entry, manifest):
        return definition.load(entry, manifest, device, paths=store.paths)
