"""Test deployment configuration without downloading a checkpoint or backbone."""

import socket
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from whirld.models.checkpoints import CheckpointReader
from whirld.models.prithvi import PrithviDeployment


def test_prithvi_disables_pretrained_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loader receives a local checkpoint and an offline inference config."""
    original = {
        "model": {
            "init_args": {
                "model_args": {
                    "backbone": "prithvi_eo_v2_300",
                    "backbone_pretrained": True,
                    "pretrained": True,
                }
            }
        },
        "trainer": {"logger": True, "callbacks": [{"class_path": "training.Callback"}]},
        "ckpt_path": "https://example.invalid/training.ckpt",
    }
    config = tmp_path / "pinned-training.yaml"
    config.write_text(yaml.safe_dump(original))
    checkpoint = tmp_path / "installed-task.pt"
    checkpoint.touch()
    cache = tmp_path / "empty-hf-cache"
    cache.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache))

    def deny_network(*args: object, **kwargs: object) -> None:
        """Any attempted network operation fails this hermetic test."""
        pytest.fail("Prithvi attempted a network request")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    marker = object()

    class FakeInference:
        """Inspect the third-party boundary without allocating a large model."""

        @staticmethod
        def from_config(config_path: str, checkpoint_path: str) -> SimpleNamespace:
            """Emulate the pretrained flag's download trigger."""
            data = yaml.safe_load(Path(config_path).read_text())
            arguments = data["model"]["init_args"]["model_args"]
            if arguments["backbone_pretrained"] or arguments["pretrained"]:
                socket.create_connection(("example.invalid", 443))
            assert arguments["backbone"] == "prithvi_eo_v2_300"
            assert checkpoint_path == str(checkpoint)
            assert "ckpt_path" not in data
            assert data["trainer"]["logger"] is False
            assert data["trainer"]["callbacks"] == []
            return SimpleNamespace(model=marker)

    module = ModuleType("terratorch.cli_tools")
    module.LightningInferenceModel = FakeInference
    monkeypatch.setitem(sys.modules, "terratorch.cli_tools", module)
    assert (
        PrithviDeployment(config, CheckpointReader(checkpoint, trusted=True)).load()
        is marker
    )
    assert yaml.safe_load(config.read_text()) == original
    assert not list(cache.iterdir())
