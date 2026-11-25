"""Pinned evaluation downloads verify bytes and preserve atomic cache state."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from urllib.error import URLError

import pytest

from whirld.evaluation import assets
from whirld.evaluation.assets import EvaluationAssetCache
from whirld.evaluation.errors import EvaluationError
from whirld.evaluation.schemas import AssetPin


class TrackedResponse(io.BytesIO):
    """Observe finite reads and simulate transport errors inside owned contexts."""

    def __init__(self, content: bytes, *, interrupt: bool = False) -> None:
        """Bind response bytes and an optional first-read interruption."""
        super().__init__(content)
        self.requests: list[int] = []
        self.interrupt = interrupt

    def read(self, size: int | None = -1) -> bytes:
        """Reject whole-response reads; optionally propagate KeyboardInterrupt."""
        assert isinstance(size, int) and 0 < size <= 1024 * 1024
        self.requests.append(size)
        if self.interrupt:
            raise KeyboardInterrupt()
        return super().read(size)


def asset(content: bytes) -> AssetPin:
    """Return a public identity for fixed fixture bytes without network access."""
    return AssetPin(
        url="https://example.invalid/pinned/image.tif",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


def test_cache_paths_depend_only_on_validated_content_identity(tmp_path: Path) -> None:
    """URL path components cannot escape the selected cache directory."""
    cache = EvaluationAssetCache(tmp_path / "cache")
    pin = asset(b"fixture")
    unusual = pin.model_copy(
        update={"url": "https://example.invalid/../../elsewhere?key=private"}
    )
    assert cache.path(pin) == cache.path(unusual) == cache.directory / pin.sha256
    assert not cache.directory.exists()


def test_verified_offline_cache_hit_performs_no_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing artifacts remain usable offline only after both checks pass."""
    cache = EvaluationAssetCache(tmp_path, offline=True)
    pin = asset(b"fixture")
    target = cache.path(pin)
    target.write_bytes(b"fixture")

    def unexpected(*args: object, **kwargs: object) -> None:
        """Fail if an offline, verified hit attempts network access."""
        raise AssertionError("unexpected network request")

    monkeypatch.setattr(assets, "urlopen", unexpected)
    assert cache.fetch(pin) == target
    assert target.read_bytes() == b"fixture"


@pytest.mark.parametrize("corrupt", [b"changed", b"short"])
def test_corrupt_cache_is_rejected_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: bytes
) -> None:
    """A named cache entry cannot bypass digest checks or trigger silent refresh."""
    cache = EvaluationAssetCache(tmp_path)
    pin = asset(b"fixture")
    target = cache.path(pin)
    target.write_bytes(corrupt)

    def unexpected(*args: object, **kwargs: object) -> None:
        """A corrupt hit must fail closed instead of downloading replacements."""
        raise AssertionError("unexpected network request")

    monkeypatch.setattr(assets, "urlopen", unexpected)
    with pytest.raises(EvaluationError, match="Cached"):
        cache.fetch(pin)
    assert target.read_bytes() == corrupt
    assert list(tmp_path.iterdir()) == [target]


def test_directory_at_cache_path_is_not_an_artifact(tmp_path: Path) -> None:
    """A directory with a digest filename fails with a domain error."""
    cache = EvaluationAssetCache(tmp_path)
    pin = asset(b"fixture")
    cache.path(pin).mkdir()
    with pytest.raises(EvaluationError, match="Cached"):
        cache.fetch(pin)


def test_offline_miss_does_not_create_cache(tmp_path: Path) -> None:
    """Offline operation refuses a missing object without creating partial state."""
    cache = EvaluationAssetCache(tmp_path / "absent", offline=True)
    with pytest.raises(EvaluationError, match="offline"):
        cache.fetch(asset(b"fixture"))
    assert not cache.directory.exists()


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("suffix", [".msk", ".aux.xml", ".ovr"])
def test_unhashed_sidecars_are_rejected_before_reuse_or_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
    suffix: str,
) -> None:
    """Digest-named TIFF bytes cannot acquire unverified mask or metadata inputs."""
    cache = EvaluationAssetCache(tmp_path)
    pin = asset(b"fixture")
    target = cache.path(pin)
    if existing:
        target.write_bytes(b"fixture")
    sidecar = target.with_name(target.name + suffix)
    sidecar.write_bytes(b"unverified sidecar")

    def unexpected(*args: object, **kwargs: object) -> None:
        """An orphan sidecar must fail before fetching the original source bytes."""
        raise AssertionError("unexpected network request")

    monkeypatch.setattr(assets, "urlopen", unexpected)
    with pytest.raises(EvaluationError):
        cache.fetch(pin)
    assert target.exists() == existing
    if existing:
        assert target.read_bytes() == b"fixture"
    assert sidecar.read_bytes() == b"unverified sidecar"


def test_other_digest_objects_do_not_invalidate_an_artifact(tmp_path: Path) -> None:
    """Sidecar isolation is scoped to the requested hash instead of the full cache."""
    cache = EvaluationAssetCache(tmp_path, offline=True)
    pin, other = asset(b"fixture"), asset(b"another fixture")
    target = cache.path(pin)
    target.write_bytes(b"fixture")
    unrelated = cache.path(other).with_suffix(".msk")
    unrelated.write_bytes(b"unrelated object")
    assert cache.fetch(pin) == target
    assert unrelated.read_bytes() == b"unrelated object"


def test_download_is_bounded_verified_and_atomically_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple finite reads produce one complete digest-named file and no staging."""
    payload = b"abcdef" * 300000
    pin = asset(payload)
    cache = EvaluationAssetCache(tmp_path / "cache")
    response = TrackedResponse(payload)
    observed: list[tuple[str, int]] = []

    def open_response(url: str, timeout: int) -> TrackedResponse:
        """Supply actual finite bytes through the same response lifetime as HTTP."""
        observed.append((url, timeout))
        assert not cache.path(pin).exists()
        return response

    monkeypatch.setattr(assets, "urlopen", open_response)
    result = cache.fetch(pin)
    assert result == cache.path(pin)
    assert result.read_bytes() == payload
    assert len(response.requests) >= 3 and response.closed
    assert observed == [(pin.url, 60)]
    assert list(cache.directory.iterdir()) == [result]


@pytest.mark.parametrize("content", [b"changed", b"short", b"fixture plus excess"])
def test_invalid_download_removes_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    """Same-size corruption, truncation and overrun cannot publish an artifact."""
    pin = asset(b"fixture")
    response = TrackedResponse(content)
    monkeypatch.setattr(assets, "urlopen", lambda *args, **kwargs: response)
    cache = EvaluationAssetCache(tmp_path)
    with pytest.raises(EvaluationError, match="download"):
        cache.fetch(pin)
    assert response.closed
    assert list(tmp_path.iterdir()) == []


def test_network_error_is_a_sanitized_domain_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transport failures do not leave cache files or echo a source URL."""
    pin = asset(b"fixture")

    def unavailable(*args: object, **kwargs: object) -> None:
        """Simulate a failed connection before any response body exists."""
        raise URLError("unavailable transport")

    monkeypatch.setattr(assets, "urlopen", unavailable)
    with pytest.raises(EvaluationError, match="download failed") as caught:
        EvaluationAssetCache(tmp_path).fetch(pin)
    assert pin.url not in str(caught.value)
    assert list(tmp_path.iterdir()) == []


def test_interrupted_download_closes_response_and_removes_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KeyboardInterrupt propagates without leaving a seemingly valid cache hit."""
    response = TrackedResponse(b"fixture", interrupt=True)
    monkeypatch.setattr(assets, "urlopen", lambda *args, **kwargs: response)
    with pytest.raises(KeyboardInterrupt):
        EvaluationAssetCache(tmp_path).fetch(asset(b"fixture"))
    assert response.closed
    assert list(tmp_path.iterdir()) == []


def test_atomic_publication_failure_removes_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disk publication errors preserve the invariant that no partial hit exists."""
    response = TrackedResponse(b"fixture")
    monkeypatch.setattr(assets, "urlopen", lambda *args, **kwargs: response)

    def fail_replace(source: Path, target: Path) -> None:
        """Verify complete staging exists, then simulate the final disk failure."""
        assert source.read_bytes() == b"fixture" and not target.exists()
        raise OSError("disk publication failure")

    monkeypatch.setattr(assets.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication"):
        EvaluationAssetCache(tmp_path).fetch(asset(b"fixture"))
    assert response.closed
    assert list(tmp_path.iterdir()) == []
