"""Streaming integration tests for the corresponding source interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

import whirld
from whirld.cli import app


def test_cli_streaming_output(
    whirld_home: Path, stac_item: Path, tmp_path: Path
) -> None:
    """CLI --stream reports disk summary without relying on resident embeddings."""
    whirld.pull("clay-v1")
    output = tmp_path / "cli.npy"
    result = CliRunner().invoke(
        app,
        [
            "embed",
            str(stac_item),
            "--model",
            "clay-v1",
            "--stream",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Streamed 4 chips" in result.output
    assert np.load(output).shape == (4, 512)
    rejected = CliRunner().invoke(
        app,
        [
            "embed",
            str(stac_item),
            "--model",
            "clay-v1",
            "--stream",
            "--format",
            "json",
        ],
    )
    assert rejected.exit_code == 7
