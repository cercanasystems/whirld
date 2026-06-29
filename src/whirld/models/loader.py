"""Backend factory: instantiate a model's backend from its registry entry.

Backend **selection is registry-driven**: each model declares a ``backend`` id in
its YAML, and this factory maps that id to the implementing class. A model that
reuses an existing backend (another open_clip classifier, another HF-torch encoder,
another TerraTorch segmenter) is therefore **pure registry data** — no code change.
Only a genuinely new architecture / loading mechanism needs a new backend class
plus one branch here.

Kept separate from ``models/__init__.py`` so ``import whirld.models`` stays cheap;
each concrete backend (and its heavy imports) is loaded lazily, only when selected.
"""

from __future__ import annotations

from ..core.fetch import Manifest
from ..core.registry import ModelEntry
from ..errors import WhirldError
from .base import ModelBackend


def load_backend(
    entry: ModelEntry,
    manifest: Manifest,
    device: str | None = None,
) -> ModelBackend:
    """Instantiate the backend declared by a model's registry entry.

    Args:
        entry: The validated registry entry (its ``backend`` field selects the
            implementation).
        manifest: The local manifest from ``whirld pull``.
        device: Requested device, or ``None`` for auto-detection.

    Returns:
        A ready :class:`ModelBackend`.

    Raises:
        WhirldError: The entry declares no backend, or an unknown one.
    """
    backend = entry.backend
    if not backend:
        raise WhirldError(
            f"Registry entry '{entry.name}' declares no 'backend'.\n"
            f"       Refresh the registry (re-pull / 'whirld update')."
        )

    if backend == "clay-reference":
        from .clay import ClayBackend  # noqa: PLC0415

        return ClayBackend.load(entry, manifest, device)
    if backend == "clay":
        from .clay_torch import ClayTorchBackend  # noqa: PLC0415

        return ClayTorchBackend.load(entry, manifest, device)
    if backend == "remoteclip":
        from .remoteclip import RemoteCLIPBackend  # noqa: PLC0415

        return RemoteCLIPBackend.load(entry, manifest, device)
    if backend == "prithvi":
        from .prithvi import PrithviBackend  # noqa: PLC0415

        return PrithviBackend.load(entry, manifest, device)

    raise WhirldError(
        f"Model '{entry.name}' declares backend '{backend}', which is not "
        f"implemented in this build."
    )
