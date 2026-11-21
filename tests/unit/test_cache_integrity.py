"""Regression tests for installation transactions and bundled-registry upgrades."""

from __future__ import annotations

import hashlib
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from whirld import config
from whirld.core import fetch
from whirld.core.registry import Registry
from whirld.errors import ChecksumMismatchError, ModelNotInstalledError, NetworkError


def test_failed_forced_pull_preserves_previous_installation(
    whirld_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checksum failure leaves both the old weights and manifest usable."""
    first = fetch.pull("clay-v1")
    paths = config.get_paths()
    before = paths.model_file("clay-v1", first.weights_file).read_bytes()
    monkeypatch.setattr(fetch, "reference_blob_bytes", lambda *_args: b"corrupt")
    with pytest.raises(ChecksumMismatchError):
        fetch.pull("clay-v1", force=True)
    assert fetch.load_manifest("clay-v1") == first
    assert paths.model_file("clay-v1", first.weights_file).read_bytes() == before
    assert fetch.pull("clay-v1") == first
    assert sorted(p.name for p in paths.models_dir.iterdir()) == ["clay-v1"]


@pytest.mark.parametrize("damage", ["missing", "truncated", "manifest"])
def test_damaged_installation_is_repaired_by_ordinary_pull(
    whirld_home: Path, damage: str
) -> None:
    """Broken cache contents are not reported as installed or skipped on pull."""
    first = fetch.pull("clay-v1")
    paths = config.get_paths()
    artifact = paths.model_file("clay-v1", first.weights_file)
    if damage == "missing":
        artifact.unlink()
    elif damage == "truncated":
        artifact.write_bytes(b"short")
    else:
        paths.model_manifest("clay-v1").write_text("{broken")
    assert not fetch.is_installed("clay-v1")
    with pytest.raises(ModelNotInstalledError):
        fetch.load_manifest("clay-v1")
    restored = fetch.pull("clay-v1")
    assert fetch.is_installed("clay-v1")
    assert fetch.sha256_file(artifact) == restored.sha256


def test_interrupted_directory_swap_recovers_previous_installation(
    whirld_home: Path,
) -> None:
    """A process dying between directory renames is repaired on next access."""
    first = fetch.pull("clay-v1")
    paths = config.get_paths()
    paths.model_dir("clay-v1").rename(paths.models_dir / ".clay-v1.previous")
    assert fetch.load_manifest("clay-v1") == first
    assert not (paths.models_dir / ".clay-v1.previous").exists()


@pytest.mark.parametrize("new_installation_present", [False, True])
def test_remove_all_cannot_resurrect_interrupted_backups(
    whirld_home: Path, new_installation_present: bool
) -> None:
    """Deleting all models finishes interrupted swaps and removes their backups."""
    fetch.pull("clay-v1")
    paths = config.get_paths()
    model_dir = paths.model_dir("clay-v1")
    backup = paths.models_dir / ".clay-v1.previous"
    if new_installation_present:
        shutil.copytree(model_dir, backup)
    else:
        model_dir.rename(backup)
    assert fetch.remove_all() == ["clay-v1"]
    assert not fetch.is_installed("clay-v1")
    assert not backup.exists()


def test_hf_auxiliary_download_failure_preserves_old_installation(
    whirld_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial downloads never replace a complete multi-artifact installation."""
    hf = pytest.importorskip("huggingface_hub")
    registry = Registry()
    path = registry.models_dir / "prithvi-burn-scar.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["distribution"]["sha256"] = hashlib.sha256(b"checkpoint").hexdigest()
    path.write_text(yaml.safe_dump(raw))

    def download(*, filename: str, local_dir: str, **_kwargs: object) -> str:
        """Materialize tiny deterministic artifacts instead of network downloads."""
        target = Path(local_dir) / filename
        target.write_bytes(b"checkpoint" if filename.endswith(".pt") else b"config")
        return str(target)

    monkeypatch.setattr(hf, "hf_hub_download", download)
    old = fetch.pull("prithvi-burn-scar")
    assert len(old.required_files) == 2

    def interrupted(*, filename: str, local_dir: str, **kwargs: object) -> str:
        """Fail after downloading weights but before the required config."""
        if filename.endswith(".yaml"):
            raise OSError("simulated connection loss")
        return download(filename=filename, local_dir=local_dir, **kwargs)

    monkeypatch.setattr(hf, "hf_hub_download", interrupted)
    with pytest.raises(NetworkError):
        fetch.pull("prithvi-burn-scar", force=True)
    assert fetch.load_manifest("prithvi-burn-scar") == old
    assert fetch.is_installed("prithvi-burn-scar")
    config.get_paths().model_file("prithvi-burn-scar", raw["config_file"]).unlink()
    assert not fetch.is_installed("prithvi-burn-scar")


def test_old_registry_cache_gains_new_bundled_models(whirld_home: Path) -> None:
    """A pre-journal cache with one model no longer blocks bundled additions."""
    models = whirld_home / "registry" / "models"
    models.mkdir(parents=True)
    bundled = config.bundled_registry_dir()
    shutil.copy2(bundled / "models" / "clay-v1.yaml", models / "clay-v1.yaml")
    custom = models / "custom.yaml"
    custom.write_text("user-owned content")
    registry = Registry()
    assert "clay-v1.5" in registry.available()
    assert registry.get("clay-v1.5").backend == "clay"
    assert custom.read_text() == "user-owned content"


def test_bundle_refresh_updates_unedited_entries_and_preserves_overrides(
    whirld_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Digest provenance distinguishes bundle upgrades from local user edits."""
    bundle = tmp_path / "bundle"
    shutil.copytree(config.bundled_registry_dir(), bundle)
    monkeypatch.setattr(config, "bundled_registry_dir", lambda: bundle)
    registry = Registry()
    override = registry.models_dir / "remoteclip.yaml"
    override.write_text(override.read_text() + "\n# user customization\n")
    prior_override = override.read_bytes()
    for name in ("clay-v1", "remoteclip"):
        source = bundle / "models" / f"{name}.yaml"
        source.write_text(
            source.read_text().replace("version: 1.0.0", "version: 1.0.1")
        )
    schema = bundle / "schema" / "model.schema.json"
    schema.write_text(schema.read_text() + "\n")
    Registry()
    assert registry.get("clay-v1").version == "1.0.1"
    assert override.read_bytes() == prior_override
    assert (
        whirld_home / "registry/schema/model.schema.json"
    ).read_bytes() == schema.read_bytes()
    Registry()
    assert override.read_bytes() == prior_override
    override.unlink()
    Registry()
    assert override.read_bytes() == (bundle / "models/remoteclip.yaml").read_bytes()


@pytest.mark.parametrize("changed", ["manifest", "registry"])
def test_backend_rejects_state_changed_after_resolution(
    whirld_home: Path, changed: str
) -> None:
    """Loading must not combine stale metadata with newly replaced artifacts."""
    from whirld.models.loader import load_backend

    manifest = fetch.pull("clay-v1")
    entry = Registry().get("clay-v1")
    if changed == "manifest":
        path = config.get_paths().model_manifest("clay-v1")
        path.write_text(manifest.model_copy(update={"seed": 999}).model_dump_json())
    else:
        path = Registry().models_dir / "clay-v1.yaml"
        path.write_text(path.read_text().replace("version: 1.0.0", "version: 1.0.1"))
    with pytest.raises(ModelNotInstalledError, match="changed while loading"):
        load_backend(entry, manifest, "cpu")


def test_registry_contract_change_requires_repull_before_loading(
    whirld_home: Path,
) -> None:
    """Even fresh resolution must not combine old weights with new configuration."""
    from whirld.models.loader import load_backend

    manifest = fetch.pull("clay-v1")
    path = Registry().models_dir / "clay-v1.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["band_contract"]["chip_size_px"] = 128
    path.write_text(yaml.safe_dump(raw))
    entry = Registry().get("clay-v1")
    with pytest.raises(ModelNotInstalledError, match="Re-pull"):
        load_backend(entry, manifest, "cpu")
    updated = fetch.pull("clay-v1")
    assert updated.entry_sha256 != manifest.entry_sha256
    assert load_backend(entry, updated, "cpu").name == "clay-v1"


def test_backend_artifact_consumption_holds_cache_lease(
    whirld_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent removal waits until the backend finishes reading its artifacts."""
    from whirld.models.clay import ClayBackend
    from whirld.models.loader import load_backend

    manifest = fetch.pull("clay-v1")
    entry = Registry().get("clay-v1")
    reading = threading.Event()
    release = threading.Event()
    removing = threading.Event()
    removed = threading.Event()
    real_load = ClayBackend.load

    def paused_load(*args: object, **kwargs: object) -> ClayBackend:
        """Pause at the artifact-consumption boundary while removal is attempted."""
        reading.set()
        assert release.wait(5)
        assert config.get_paths().model_file("clay-v1", manifest.weights_file).is_file()
        return real_load(*args, **kwargs)

    def remove() -> None:
        """Try to remove the model from another thread."""
        removing.set()
        fetch.remove_model("clay-v1")
        removed.set()

    monkeypatch.setattr(ClayBackend, "load", paused_load)
    with ThreadPoolExecutor(max_workers=2) as executor:
        loading = executor.submit(load_backend, entry, manifest, "cpu")
        assert reading.wait(5)
        deletion = executor.submit(remove)
        try:
            assert removing.wait(5)
            assert not removed.wait(0.05)
        finally:
            release.set()
        loading.result(timeout=5)
        deletion.result(timeout=5)
    assert not fetch.is_installed("clay-v1")
