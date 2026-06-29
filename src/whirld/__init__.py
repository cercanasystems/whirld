"""Whirld — local-first geospatial foundation models (the Ollama for geospatial).

Public Python API (mirrors the CLI surface, PRD section 14)::

    import whirld

    whirld.pull("clay-v1")
    result = whirld.embed("scene.tif", model="clay-v1")
    result.embeddings  # np.ndarray, shape (n_chips, embed_dim)
    result.chips       # list of chip metadata with bounding boxes

    fc = whirld.classify("scene.tif", model="remoteclip", query="solar farm")
    fc.feature_collection  # GeoJSON FeatureCollection of per-chip scores

Imports are lazy: ``import whirld`` does not import numpy, rasterio, or torch.
Those are pulled in only when an operation actually runs, keeping startup fast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._version import __version__

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .api import (
        ClassifyResult,
        EmbedResult,
        SegmentResult,
        classify,
        embed,
        pull,
        segment,
    )

__all__ = [
    "__version__",
    "pull",
    "embed",
    "classify",
    "segment",
    "EmbedResult",
    "ClassifyResult",
    "SegmentResult",
]

_PUBLIC = {
    "pull",
    "embed",
    "classify",
    "segment",
    "EmbedResult",
    "ClassifyResult",
    "SegmentResult",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve public API symbols from :mod:`whirld.api`.

    Args:
        name: Attribute being accessed on the ``whirld`` package.

    Returns:
        The resolved attribute.

    Raises:
        AttributeError: The attribute is not part of the public API.
    """
    if name in _PUBLIC:
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module 'whirld' has no attribute '{name}'")
