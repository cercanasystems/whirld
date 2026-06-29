"""Integration tests: the CLI surface via Typer's CliRunner."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from whirld.cli import app

runner = CliRunner()


def test_cli_pull(whirld_home: Path) -> None:
    """`whirld pull clay-v1` succeeds and reports verification."""
    result = runner.invoke(app, ["pull", "clay-v1"])
    assert result.exit_code == 0, result.output
    assert "Verifying sha256" in result.output


def test_cli_pull_unknown_model(whirld_home: Path) -> None:
    """Pulling an unknown model exits with code 2 and an actionable message."""
    result = runner.invoke(app, ["pull", "no-such-model"])
    assert result.exit_code == 2
    assert "not in the registry" in result.output


def test_cli_list(whirld_home: Path) -> None:
    """`whirld list` shows a pulled model."""
    runner.invoke(app, ["pull", "clay-v1"])
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "clay-v1" in result.output


def test_cli_info(whirld_home: Path) -> None:
    """`whirld info clay-v1` prints the band contract and sensors."""
    result = runner.invoke(app, ["info", "clay-v1"])
    assert result.exit_code == 0
    assert "sentinel-2-l2a" in result.output
    assert "Band contract" in result.output


def test_cli_info_json(whirld_home: Path) -> None:
    """`whirld info --json` emits machine-readable output."""
    result = runner.invoke(app, ["info", "clay-v1", "--json"])
    assert result.exit_code == 0
    assert '"name": "clay-v1"' in result.output


def test_cli_embed_end_to_end(whirld_home: Path, s2_tif: Path) -> None:
    """The five-minute-test analog: pull then embed writes outputs, exit 0."""
    assert runner.invoke(app, ["pull", "clay-v1"]).exit_code == 0
    result = runner.invoke(app, ["embed", str(s2_tif), "--model", "clay-v1"])
    assert result.exit_code == 0, result.output
    assert "Embedded" in result.output
    assert (s2_tif.parent / "s2_small_embeddings.npy").exists()
    assert (s2_tif.parent / "s2_small_embeddings_meta.json").exists()


def test_cli_embed_not_installed(whirld_home: Path, s2_tif: Path) -> None:
    """Embedding before pull exits with code 3."""
    result = runner.invoke(app, ["embed", str(s2_tif), "--model", "clay-v1"])
    assert result.exit_code == 3
    assert "not installed" in result.output


def test_cli_embed_unsupported_sensor(whirld_home: Path, s2_tif: Path) -> None:
    """An unsupported --sensor override exits with code 4."""
    runner.invoke(app, ["pull", "clay-v1"])
    result = runner.invoke(
        app, ["embed", str(s2_tif), "--model", "clay-v1", "--sensor", "spot-6"]
    )
    assert result.exit_code == 4
    assert "supports" in result.output


def test_cli_list_empty(whirld_home: Path) -> None:
    """`whirld list` with nothing installed prints guidance, exit 0."""
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "No models installed" in result.output
