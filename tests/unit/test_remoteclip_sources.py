"""RemoteCLIP architecture selection cannot bypass verified artifact loading."""

from pathlib import Path

import pytest
import yaml

from whirld.core.registry import ModelEntry, Registry
from whirld.errors import RegistryError, SecurityError
from whirld.models.remoteclip import RemoteCLIPBackend

_UNSUPPORTED = [
    "hf-hub:example/unpinned-model",
    "local-dir:/private/unverified-model",
    "https://example.invalid/model",
    "ViT-B-32-roberta-base",
    "ViT-B-32-xlm-roberta-base",
    "../unverified-model",
    "",
]


@pytest.mark.parametrize("architecture", _UNSUPPORTED)
def test_registry_rejects_remoteclip_external_sources(
    whirld_home: Path, architecture: str
) -> None:
    """A community safetensors entry cannot select an unverified external source."""
    raw = yaml.safe_load((Registry().models_dir / "remoteclip.yaml").read_text())
    raw["trust"] = "community"
    raw["source"]["files"] = ["weights.safetensors"]
    raw["model_name"] = architecture
    with pytest.raises(RegistryError, match="supported local architecture"):
        ModelEntry.from_registry_data(raw)


@pytest.mark.parametrize("architecture", _UNSUPPORTED)
def test_direct_backend_rejects_source_before_open_clip_dispatch(
    monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    """Source dispatch is forbidden even when registry validation is bypassed."""
    import whirld.models.remoteclip as remoteclip

    def forbidden_import() -> None:
        pytest.fail("open_clip import/config/network dispatch must not be reached")

    monkeypatch.setattr(remoteclip, "_import_open_clip", forbidden_import)
    with pytest.raises(SecurityError, match="supported local architecture"):
        RemoteCLIPBackend("model", "1", "cpu", arch=architecture)
    backend = RemoteCLIPBackend("model", "1", "cpu", arch="ViT-B-32")
    backend.arch = architecture
    with pytest.raises(SecurityError, match="supported local architecture"):
        backend._ensure_model()


@pytest.mark.parametrize("architecture", ["RN50", "ViT-B-32", "ViT-L-14"])
def test_supported_remoteclip_architectures_remain_valid(
    whirld_home: Path, architecture: str
) -> None:
    """The published local RemoteCLIP architecture names remain supported."""
    raw = yaml.safe_load((Registry().models_dir / "remoteclip.yaml").read_text())
    raw["model_name"] = architecture
    entry = ModelEntry.from_registry_data(raw)
    backend = RemoteCLIPBackend(entry.name, entry.version, "cpu", arch=entry.model_name)
    assert backend.arch == architecture
