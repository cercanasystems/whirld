"""Gated integration test for REAL STAC item input over the network.

Skipped unless ``WHIRLD_TEST_STAC_URL`` points at a real STAC item URL (e.g. a
Sentinel-2 L2A item from Earth Search). Exercises the real fetch + ``/vsicurl/``
range-read path end to end::

    WHIRLD_TEST_STAC_URL="https://earth-search.aws.element84.com/v1/collections/\
sentinel-2-l2a/items/<id>" \
        pytest tests/integration/test_stac_real.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import whirld

_URL_ENV = "WHIRLD_TEST_STAC_URL"
_url = os.environ.get(_URL_ENV)

pytestmark = pytest.mark.skipif(
    not _url,
    reason=f"set {_URL_ENV} to a real STAC item URL to run this test",
)


def test_embed_real_stac_item(whirld_home: Path) -> None:
    """Embed a real STAC item, windowed to a small bbox to bound the read."""
    whirld.pull("clay-v1")
    # A tiny window keeps the range read small regardless of scene size; the bbox
    # must intersect the item — callers should pass one that does for their item.
    bbox_env = os.environ.get("WHIRLD_TEST_STAC_BBOX")
    bbox = (
        tuple(float(x) for x in bbox_env.split(",")) if bbox_env else None  # type: ignore[assignment]
    )
    result = whirld.embed(
        str(_url),
        model="clay-v1",
        sensor="sentinel-2-l2a",
        bbox=bbox,
        write=False,
    )
    assert result.embeddings.shape[1] == 512
    assert result.embeddings.shape[0] >= 1
