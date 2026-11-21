"""ModelStore dependency, isolation, and cross-instance locking contracts."""

from __future__ import annotations

import select
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from whirld.config import WhirldPaths
from whirld.core.fetch import Manifest, ModelStore
from whirld.core.registry import Registry
from whirld.errors import WhirldError
from whirld.models.loader import load_backend


def test_store_uses_injected_registry_paths(tmp_path: Path) -> None:
    """An injected registry determines the cache root without environment mutation."""
    paths = WhirldPaths(tmp_path / "explicit")
    registry = Registry(paths)
    store = ModelStore(registry=registry)
    assert store.paths == paths
    assert store.registry is registry
    manifest = store.pull("clay-v1")
    assert paths.model_manifest("clay-v1").is_file()
    assert store.load_manifest("clay-v1") == manifest
    with pytest.raises(WhirldError, match="same cache home"):
        ModelStore(WhirldPaths(tmp_path / "different"), registry)


def test_relative_store_root_is_bound_before_working_directory_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store retains its installation and registry after the caller changes cwd."""
    original = tmp_path / "original"
    elsewhere = tmp_path / "elsewhere"
    original.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(original)
    store = ModelStore(WhirldPaths(Path("cache")))
    manifest = store.pull("clay-v1")
    registry = store.registry
    monkeypatch.chdir(elsewhere)
    assert store.paths.home == original / "cache"
    assert store.is_installed("clay-v1")
    assert store.load_manifest("clay-v1") == manifest
    assert registry.get("clay-v1").name == "clay-v1"
    assert not (elsewhere / "cache").exists()


def test_selected_root_symlink_is_resolved_once(
    tmp_path: Path,
) -> None:
    """Retargeting a root-selection alias cannot redirect an existing store."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    alias = tmp_path / "selected"
    alias.symlink_to(first, target_is_directory=True)
    store = ModelStore(WhirldPaths(alias))
    store.pull("clay-v1")
    alias.unlink()
    alias.symlink_to(second, target_is_directory=True)
    assert store.paths.home == first
    assert store.is_installed("clay-v1")
    store.remove_model("clay-v1")
    assert not (first / "models" / "clay-v1").exists()
    assert not (second / "models").exists()


def test_nested_independent_stores_reenter_same_home(tmp_path: Path) -> None:
    """Nested same-thread leases across store instances cannot self-deadlock."""
    script = """
import sys
from pathlib import Path
from whirld.config import WhirldPaths
from whirld.core.fetch import ModelStore
first = ModelStore(WhirldPaths(Path(sys.argv[1])))
second = ModelStore(first.paths)
with first.transaction():
    manifest = first.pull('clay-v1')
    with second.transaction():
        assert second.load_manifest('clay-v1') == manifest
        with first.installation_lease(first.registry.get('clay-v1'), manifest):
            assert second.is_installed('clay-v1')
print('complete')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "shared")],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert result.stdout.strip() == "complete"


def test_independent_stores_serialize_same_home_threads(tmp_path: Path) -> None:
    """Two stores coordinate via the cache lock while retaining separate objects."""
    first = ModelStore(WhirldPaths(tmp_path / "shared"))
    second = ModelStore(first.paths)
    first.pull("clay-v1")
    started = threading.Event()
    completed = threading.Event()

    def remove() -> None:
        """Remove through another store after announcing the attempt."""
        started.set()
        second.remove_model("clay-v1")
        completed.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with first.transaction():
            future = executor.submit(remove)
            assert started.wait(2)
            assert not completed.wait(0.05)
            assert first.is_installed("clay-v1")
        future.result(timeout=2)
    assert completed.is_set()
    assert not first.is_installed("clay-v1")


def test_different_homes_do_not_share_a_mutex(tmp_path: Path) -> None:
    """A long transaction in one home cannot stall installations in another."""
    first = ModelStore(WhirldPaths(tmp_path / "first"))
    second = ModelStore(WhirldPaths(tmp_path / "second"))
    with ThreadPoolExecutor(max_workers=1) as executor:
        with first.transaction():
            future = executor.submit(second.pull, "clay-v1")
            assert future.result(timeout=5).name == "clay-v1"
    assert second.is_installed("clay-v1")
    assert not first.is_installed("clay-v1")


def test_store_lock_serializes_separate_processes(tmp_path: Path) -> None:
    """A second process cannot consume the same cache while its lease is held."""
    store = ModelStore(WhirldPaths(tmp_path / "shared"))
    script = """
import sys
from pathlib import Path
from whirld.config import WhirldPaths
from whirld.core.fetch import ModelStore
store = ModelStore(WhirldPaths(Path(sys.argv[1])))
print('attempt', flush=True)
with store.transaction():
    print('acquired', flush=True)
"""
    process = None
    try:
        with store.transaction():
            process = subprocess.Popen(
                [sys.executable, "-c", script, str(store.paths.home)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert process.stdout is not None
            assert process.stdout.readline().strip() == "attempt"
            assert not select.select([process.stdout], [], [], 0.05)[0]
        output, errors = process.communicate(timeout=5)
        assert process.returncode == 0, errors
        assert output.strip() == "acquired"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_lease_is_released_after_exception(tmp_path: Path) -> None:
    """Failed transactions release both process locks and thread-local nesting."""
    first = ModelStore(WhirldPaths(tmp_path / "shared"))
    with pytest.raises(RuntimeError, match="abort"):
        with first.transaction():
            with ModelStore(first.paths).transaction():
                raise RuntimeError("abort")
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(ModelStore(first.paths).pull, "clay-v1").result(5)


def test_real_backend_reads_injected_store_not_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backend artifact paths use the same explicit home as the installation lease."""
    from whirld.models.clay_torch import ClayTorchBackend

    explicit = WhirldPaths(tmp_path / "explicit")
    monkeypatch.setenv("WHIRLD_HOME", str(tmp_path / "environment"))
    store = ModelStore(explicit)
    entry = store.registry.get("clay-v1.5")
    directory = explicit.model_dir(entry.name)
    directory.mkdir(parents=True)
    filename = "placeholder.ckpt"
    explicit.model_file(entry.name, filename).write_bytes(b"placeholder")
    manifest = Manifest(
        name=entry.name,
        version=entry.version,
        source_type=entry.source.type,
        sha256=entry.distribution.sha256,
        weights_file=filename,
        required_files=[filename],
    )
    explicit.model_manifest(entry.name).write_text(manifest.model_dump_json())
    # Skip checkpoint deserialization, retaining the real backend's path checks.
    monkeypatch.setattr(ClayTorchBackend, "_ensure_encoder", lambda self: None)
    backend = load_backend(entry, manifest, "cpu", store=store)
    assert isinstance(backend, ClayTorchBackend)
    assert not (tmp_path / "environment" / "models" / entry.name).exists()
