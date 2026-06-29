"""Integration tests for license surfacing (pull/info) and the CI checker script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from typer.testing import CliRunner

from whirld.cli import app
from whirld.core.fetch import Manifest
from whirld.core.registry import Registry

runner = CliRunner()


def test_pull_shows_oss_license(whirld_home: Path) -> None:
    """`whirld pull` displays the license (annotated OSS) before downloading."""
    result = runner.invoke(app, ["pull", "clay-v1"])  # MIT, reference (offline)
    assert result.exit_code == 0, result.output
    assert "License:" in result.output
    assert "(OSS)" in result.output


def test_pull_warns_non_oss(whirld_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-OSS license triggers a prominent terms warning at pull time."""
    entry = (
        Registry()
        .get("clay-v1")
        .model_copy(update={"license": "Custom Restricted License"})
    )

    class _FakeRegistry:
        def get(self, name):  # type: ignore[no-untyped-def]
            return entry

    def _fake_pull(name, **kwargs):  # type: ignore[no-untyped-def]
        return Manifest(
            name=name,
            version="1.0.0",
            source_type="reference",
            sha256="0" * 64,
            weights_file="w",
        )

    monkeypatch.setattr("whirld.cli.commands.pull.Registry", _FakeRegistry)
    monkeypatch.setattr("whirld.cli.commands.pull.api.pull", _fake_pull)

    result = runner.invoke(app, ["pull", "clay-v1"])
    assert result.exit_code == 0, result.output
    assert "(non-OSS)" in result.output
    assert "not a recognized open-source license" in result.output


def test_info_annotates_oss(whirld_home: Path) -> None:
    """`whirld info` annotates the license and shows the trust tier."""
    result = runner.invoke(app, ["info", "clay-v1.5"])
    assert result.exit_code == 0
    assert "(OSS)" in result.output
    assert "first-party" in result.output


def _load_checker():
    """Import the standalone scripts/check_registry.py module."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "check_registry.py"
    spec = importlib.util.spec_from_file_location("check_registry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_checker_review_for_bundled(whirld_home: Path) -> None:
    """The checker routes a first-party bundled entry to review (exit 1)."""
    checker = _load_checker()
    yaml_path = Path("src/whirld/registry_data/models/clay-v1.5.yaml").resolve()
    assert checker.main([str(yaml_path)]) == 1


def test_checker_automerge_for_clean_community(
    whirld_home: Path, tmp_path: Path
) -> None:
    """A clean community entry (OSS + trusted org + safetensors) auto-merges (0)."""
    checker = _load_checker()
    yaml_text = Path("src/whirld/registry_data/models/clay-v1.5.yaml").read_text()
    # Make it community + MIT + safetensors (keep trusted org made-with-clay).
    yaml_text = (
        yaml_text.replace("trust: first-party", "trust: community")
        .replace("license: Apache-2.0", "license: MIT")
        .replace("clay-v1.5.ckpt", "clay-v1.5.safetensors")
        .replace("name: clay-v1.5", "name: clay-community")
    )
    p = tmp_path / "clay-community.yaml"
    p.write_text(yaml_text)
    assert checker.main([str(p)]) == 0


def test_checker_invalid_entry(tmp_path: Path) -> None:
    """An invalid (schema-failing) entry exits 2."""
    checker = _load_checker()
    p = tmp_path / "bad.yaml"
    p.write_text("name: bad\n")  # missing required fields
    assert checker.main([str(p)]) == 2
