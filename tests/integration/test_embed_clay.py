"""Integration tests: full embed pipeline via the public Python API."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import whirld
from whirld.errors import InvalidInputError, ModelNotInstalledError


def test_embed_end_to_end(whirld_home: Path, s2_tif: Path) -> None:
    """pull -> embed writes a (n, 512) npy plus a metadata sidecar."""
    whirld.pull("clay-v1")
    result = whirld.embed(s2_tif, model="clay-v1")

    assert result.embeddings.ndim == 2
    assert result.embeddings.shape[1] == 512
    assert result.sensor == "sentinel-2-l2a"
    assert result.embeddings.shape[0] == len(result.chips)

    assert result.output_path is not None and result.output_path.exists()
    loaded = np.load(result.output_path)
    assert loaded.shape == result.embeddings.shape

    assert result.meta_path is not None and result.meta_path.exists()
    meta = json.loads(result.meta_path.read_text())
    assert meta["model"] == "clay-v1"
    assert meta["embed_dim"] == 512
    assert meta["crs"] == "EPSG:32630"
    assert len(meta["chips"]) == result.embeddings.shape[0]
    assert meta["chips"][0]["bbox"][0] == 320000.0


def test_embed_default_output_name(whirld_home: Path, s2_tif: Path) -> None:
    """The default output path is derived from the input stem."""
    whirld.pull("clay-v1")
    result = whirld.embed(s2_tif, model="clay-v1")
    assert result.output_path.name == "s2_small_embeddings.npy"
    assert result.meta_path.name == "s2_small_embeddings_meta.json"


def test_embed_json_format(whirld_home: Path, s2_tif: Path, tmp_path: Path) -> None:
    """JSON output inlines the embeddings array."""
    whirld.pull("clay-v1")
    out = tmp_path / "emb.json"
    result = whirld.embed(s2_tif, model="clay-v1", output=out, fmt="json")
    payload = json.loads(out.read_text())
    assert "embeddings" in payload
    assert len(payload["embeddings"]) == result.embeddings.shape[0]


def test_embed_requires_pull(whirld_home: Path, s2_tif: Path) -> None:
    """Embedding before pulling raises ModelNotInstalledError (exit code 3)."""
    with pytest.raises(ModelNotInstalledError) as excinfo:
        whirld.embed(s2_tif, model="clay-v1")
    assert excinfo.value.exit_code == 3


def test_embed_stac_url_is_wired(whirld_home: Path) -> None:
    """A STAC URL is now routed to the STAC reader (no longer a deferred error).

    An unreachable URL fails at fetch with a network error — not the old
    "not available in this build" message — proving the path is wired.
    """
    from whirld.errors import NetworkError

    whirld.pull("clay-v1")
    with pytest.raises(NetworkError):
        whirld.embed("https://invalid.invalid/item.json", model="clay-v1")


def test_embed_no_crs_rejected(whirld_home: Path, no_crs_tif: Path) -> None:
    """A CRS-less input raises InvalidInputError (exit code 7)."""
    whirld.pull("clay-v1")
    with pytest.raises(InvalidInputError) as excinfo:
        whirld.embed(no_crs_tif, model="clay-v1")
    assert excinfo.value.exit_code == 7


def test_usage_record_written(whirld_home: Path, s2_tif: Path) -> None:
    """A usage record with the right fields is appended after embed."""
    whirld.pull("clay-v1")
    whirld.embed(s2_tif, model="clay-v1")
    usage = whirld_home / "logs" / "usage.jsonl"
    assert usage.exists()
    last = json.loads(usage.read_text().strip().splitlines()[-1])
    assert last["command"] == "embed"
    assert last["model"] == "clay-v1"
    assert last["sensor_detected"] == "sentinel-2-l2a"
    assert last["error"] is None
    # No sensitive fields leak into the usage record.
    assert "input" not in last and "crs" not in last
