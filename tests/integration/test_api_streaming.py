"""Streaming integration tests for the corresponding source interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import whirld
from whirld.config import get_paths
from whirld.errors import InvalidInputError, UnsupportedOperationError


@pytest.mark.parametrize("batch_size", [1, 4, 20])
def test_streaming_matches_eager_outputs(
    whirld_home: Path, stac_item: Path, tmp_path: Path, batch_size: int
) -> None:
    """Batch boundaries do not change embedding values or global chip geometry."""
    whirld.pull("clay-v1")
    eager = whirld.embed(
        stac_item,
        model="clay-v1",
        chip_size=128,
        overlap=17,
        device="cpu",
        write=False,
    )
    result = whirld.embed_to_disk(
        stac_item,
        model="clay-v1",
        output=tmp_path / "stream.npy",
        chip_size=128,
        overlap=17,
        batch_size=batch_size,
        device="cpu",
    )
    assert isinstance(result, whirld.DiskEmbedResult)
    assert not hasattr(result, "embeddings")
    assert not hasattr(result, "chips")
    assert result.chip_count == len(eager.chips)
    assert result.embed_dim == eager.embeddings.shape[1]
    np.testing.assert_allclose(
        np.load(result.output_path), eager.embeddings, atol=1e-6, rtol=1e-5
    )
    metadata = json.loads(result.meta_path.read_text())
    for key in eager.meta:
        if key != "timestamp":
            assert metadata[key] == eager.meta[key]
    usage = json.loads(get_paths().usage_log.read_text().splitlines()[-1])
    assert usage["input_type"] == "stac"
    assert usage["chip_count"] == result.chip_count
    assert usage["error"] is None


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 0},
        {"batch_size": True},
        {"cache_mb": -1},
        {"fmt": "json"},
        {"chip_size": 32, "overlap": 32},
        {"bbox": (10, 0, -10, 1)},
    ],
)
def test_streaming_options_fail_before_io(options: dict[str, object]) -> None:
    """Unsupported options reject an unreachable input before any source access."""
    with pytest.raises(InvalidInputError):
        whirld.embed_to_disk(
            "https://example.invalid/item.json", model="clay-v1", **options
        )


def test_streaming_rejects_local_bbox() -> None:
    """Unsupported local crops fail without loading a model or raster."""
    with pytest.raises(InvalidInputError, match="bbox"):
        whirld.embed_to_disk("scene.tif", model="clay-v1", bbox=(-1.0, 50.0, 0.0, 51.0))


def test_streaming_rejects_nonembedding_model(
    whirld_home: Path, stac_item: Path
) -> None:
    """Unsupported capabilities fail before acquiring weights or reading assets."""
    with pytest.raises(UnsupportedOperationError, match="embed"):
        whirld.embed_to_disk(stac_item, model="remoteclip", device="cpu")
