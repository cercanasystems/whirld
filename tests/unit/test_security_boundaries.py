"""Regression tests for checkpoint, cache-path, and registry trust boundaries."""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from whirld import config
from whirld.core import fetch
from whirld.core.registry import ModelEntry, Registry
from whirld.errors import RegistryError, SecurityError
from whirld.models.checkpoints import load_checkpoint


class MarkerPayload:
    """A harmless pickle payload proving whether deserialization was executed."""

    def __init__(self, marker: Path) -> None:
        """Store the isolated marker path."""
        self.marker = marker

    def __reduce__(self) -> tuple[Any, tuple[Path, str]]:
        """Write a marker only if an unsafe pickle reader executes the payload."""
        return Path.write_text, (self.marker, "executed")


def test_renamed_pickle_never_falls_back_to_torch(tmp_path: Path) -> None:
    """A pickle disguised as safetensors cannot execute its reducer."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    marker = tmp_path / "executed"
    artifact = tmp_path / "weights.safetensors"
    torch.save({"state_dict": MarkerPayload(marker)}, artifact)
    for trusted in (False, True):
        with pytest.raises(SecurityError, match="Invalid safetensors"):
            load_checkpoint(artifact, trusted=trusted)
    assert not marker.exists()


def test_actual_safetensors_load_without_pickle(tmp_path: Path) -> None:
    """Safe tensor files are supported and preserve tensor values."""
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    artifact = tmp_path / "weights.safetensors"
    safetensors.save_file({"model.encoder.weight": torch.arange(3)}, str(artifact))
    result = load_checkpoint(artifact)
    assert torch.equal(result["model.encoder.weight"], torch.arange(3))


def test_clay_backend_loads_real_safe_tensor_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual Clay loading path supports community safetensors state dicts."""
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from whirld.models._vendor import clay_v15
    from whirld.models.clay_torch import ClayTorchBackend

    artifact = tmp_path / "encoder.safetensors"
    expected = torch.tensor([[1.0, 2.0]])
    safetensors.save_file({"model.encoder.weight": expected}, str(artifact))
    monkeypatch.setattr(
        clay_v15, "clay_v15_large_encoder", lambda: torch.nn.Linear(2, 1, bias=False)
    )
    backend = ClayTorchBackend(
        "safe", "1.0.0", "cpu", embed_dim=1, patch_size=1, ckpt_path=artifact
    )
    encoder = backend._ensure_encoder()
    assert torch.equal(encoder.weight, expected)


def test_prithvi_direct_loader_rejects_untrusted_checkpoint(tmp_path: Path) -> None:
    """Direct backend construction cannot bypass the executable-config boundary."""
    from whirld.models.prithvi import PrithviBackend

    backend = PrithviBackend(
        "unsafe",
        "1.0.0",
        "cpu",
        classes=2,
        ckpt_path=tmp_path / "model.pt",
        config_path=tmp_path / "config.yaml",
    )
    with pytest.raises(SecurityError, match="first-party"):
        backend._ensure_model()


def test_pickle_requires_explicit_trust_at_read(tmp_path: Path) -> None:
    """Direct checkpoint loading enforces trust independently of the pull gate."""
    torch = pytest.importorskip("torch")
    artifact = tmp_path / "weights.pt"
    torch.save({"weight": torch.arange(2)}, artifact)
    with pytest.raises(SecurityError, match="first-party"):
        load_checkpoint(artifact)
    assert torch.equal(
        load_checkpoint(artifact, trusted=True)["weight"], torch.arange(2)
    )


@pytest.mark.parametrize(
    "name", ["../../unrelated", "/tmp/unrelated", "../x", "x/y", "x\\y", ".", ".."]
)
def test_unsafe_names_rejected_across_cache_operations(
    whirld_home: Path, tmp_path: Path, name: str
) -> None:
    """Invalid names never reach cache reads, writes, or recursive deletion."""
    marker = tmp_path / "unrelated"
    marker.mkdir(exist_ok=True)
    for operation in (
        fetch.remove_model,
        fetch.load_manifest,
        fetch.is_installed,
        fetch.pull,
        Registry().get,
    ):
        with pytest.raises(SecurityError):
            operation(name)
    assert marker.is_dir()


def test_symlink_model_and_cache_roots_are_rejected(
    whirld_home: Path, tmp_path: Path
) -> None:
    """Both a model link and a substituted models root preserve outside content."""
    paths = config.get_paths()
    paths.ensure_dirs()
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "marker"
    marker.write_text("keep")
    link = paths.models_dir / "evil"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SecurityError):
        fetch.remove_model("evil")
    with pytest.raises(SecurityError):
        fetch.remove_all()
    link.unlink()
    paths.models_dir.rmdir()
    (whirld_home / "models").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SecurityError):
        fetch.is_installed("evil")
    assert marker.read_text() == "keep"


def test_symlink_artifact_and_manifest_paths_are_rejected(
    whirld_home: Path, tmp_path: Path
) -> None:
    """Editable manifests and artifact links cannot point outside the cache."""
    paths = config.get_paths()
    manifest = fetch.pull("clay-v1")
    artifact = paths.model_file("clay-v1", manifest.weights_file)
    artifact.unlink()
    artifact.symlink_to(tmp_path / "outside")
    with pytest.raises(SecurityError):
        fetch.load_manifest("clay-v1")
    with pytest.raises(SecurityError):
        fetch.Manifest.model_validate(
            {**manifest.model_dump(), "weights_file": "../../outside"}
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "../../escape"),
        ("category", "invalid"),
        ("backend", "os.system"),
        ("trust", "trusted-ish"),
        ("version", "unversioned"),
        ("source.type", "arbitrary"),
        ("source.files", ["../weights.pt"]),
        ("source.files", ["manifest.json"]),
        ("source.files", [".cache/huggingface/metadata"]),
        ("source.files", ["a", "a/b"]),
        ("source.files", ["weights.pt", "weights.pt"]),
        ("distribution.sha256", "bad"),
        ("output.format", "invalid"),
        ("band_contract.normalization.type", "unknown"),
        ("backend", "remoteclip"),
    ],
)
def test_schema_rejections_apply_to_runtime_and_checker(
    whirld_home: Path, tmp_path: Path, field: str, value: Any
) -> None:
    """Malformed fields fail before download and cannot receive AUTO-MERGE."""
    registry = Registry()
    raw = yaml.safe_load((registry.models_dir / "clay-v1.5.yaml").read_text())
    raw = copy.deepcopy(raw)
    target = raw
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises(RegistryError):
        ModelEntry.from_registry_data(raw)
    path = registry.models_dir / "clay-v1.5.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(RegistryError):
        registry.get("clay-v1.5")
    result = subprocess.run(
        [sys.executable, "scripts/check_registry.py", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "AUTO-MERGE" not in result.stdout


def test_valid_community_safetensors_passes_schema(whirld_home: Path) -> None:
    """Schema enforcement still accepts compatible safe community submissions."""
    raw = yaml.safe_load((Registry().models_dir / "clay-v1.5.yaml").read_text())
    raw["trust"] = "community"
    raw["source"]["files"] = ["weights.safetensors"]
    assert ModelEntry.from_registry_data(raw).trust == "community"
