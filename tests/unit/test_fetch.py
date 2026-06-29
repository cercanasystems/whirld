"""Unit tests for the pull / fetch pipeline and checksum verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from whirld.core import fetch
from whirld.core.registry import Registry
from whirld.errors import (
    ChecksumMismatchError,
    ModelNotInstalledError,
    WhirldError,
)


def test_reference_blob_is_deterministic() -> None:
    """The canonical reference blob is byte-for-byte stable."""
    a = fetch.reference_blob_bytes("clay-v1", "1.0.0", 512, 1234)
    b = fetch.reference_blob_bytes("clay-v1", "1.0.0", 512, 1234)
    assert a == b
    assert b'"backend":"reference"' in a


def test_pull_materializes_and_verifies(whirld_home: Path) -> None:
    """Pull materializes the blob, verifies sha256, and writes a manifest."""
    manifest = fetch.pull("clay-v1")
    assert manifest.name == "clay-v1"
    assert manifest.embed_dim == 512
    assert manifest.seed == 1234
    assert fetch.is_installed("clay-v1")

    entry = Registry().get("clay-v1")
    assert manifest.sha256 == entry.distribution.sha256


def test_pull_is_idempotent_without_force(whirld_home: Path) -> None:
    """A second pull without --force returns the cached manifest."""
    first = fetch.pull("clay-v1")
    second = fetch.pull("clay-v1")
    assert first.sha256 == second.sha256


def test_load_manifest_missing_raises(whirld_home: Path) -> None:
    """Loading a manifest for an un-pulled model raises (exit code 3)."""
    with pytest.raises(ModelNotInstalledError) as excinfo:
        fetch.load_manifest("clay-v1")
    assert excinfo.value.exit_code == 3


def test_checksum_mismatch_deletes_and_raises(
    whirld_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tampered blob fails verification, is deleted, and raises (exit code 6)."""

    def _tampered(*_args: object, **_kwargs: object) -> bytes:
        return b"not the real blob"

    monkeypatch.setattr(fetch, "reference_blob_bytes", _tampered)
    with pytest.raises(ChecksumMismatchError) as excinfo:
        fetch.pull("clay-v1")
    assert excinfo.value.exit_code == 6
    weights = whirld_home / "models" / "clay-v1" / fetch._REFERENCE_WEIGHTS_FILENAME
    assert not weights.exists()


def test_quantize_is_deferred(whirld_home: Path) -> None:
    """Requesting a quantized variant raises a clear deferred-feature error."""
    with pytest.raises(WhirldError) as excinfo:
        fetch.pull("clay-v1", quantize="int8")
    assert "not available" in excinfo.value.message


def test_sha256_file(tmp_path: Path) -> None:
    """sha256_file matches a known digest for known bytes."""
    import hashlib

    f = tmp_path / "blob.bin"
    f.write_bytes(b"hello whirld")
    assert fetch.sha256_file(f) == hashlib.sha256(b"hello whirld").hexdigest()
