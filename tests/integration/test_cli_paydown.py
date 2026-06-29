"""Integration tests for Pass-5 CLI/IO completeness: rm, --crs, --batch-size."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import whirld
from whirld.cli import app
from whirld.errors import InvalidInputError

runner = CliRunner()


def test_cli_rm(whirld_home: Path) -> None:
    """`whirld rm <model>` removes a pulled model."""
    assert runner.invoke(app, ["pull", "clay-v1"]).exit_code == 0
    result = runner.invoke(app, ["rm", "clay-v1"])
    assert result.exit_code == 0
    assert "Removed" in result.output
    # Now gone.
    assert runner.invoke(app, ["list"]).output.strip().startswith("No models installed")


def test_cli_rm_not_installed(whirld_home: Path) -> None:
    """Removing an un-installed model exits with code 3."""
    result = runner.invoke(app, ["rm", "clay-v1"])
    assert result.exit_code == 3
    assert "not installed" in result.output


def test_cli_rm_all(whirld_home: Path) -> None:
    """`whirld rm --all` clears all models but keeps the registry."""
    runner.invoke(app, ["pull", "clay-v1"])
    result = runner.invoke(app, ["rm", "--all"])
    assert result.exit_code == 0
    assert "Removed 1 model" in result.output
    # Registry still works (info resolves).
    assert runner.invoke(app, ["info", "clay-v1"]).exit_code == 0


def test_cli_rm_requires_target(whirld_home: Path) -> None:
    """`whirld rm` with neither model nor --all errors (general error)."""
    result = runner.invoke(app, ["rm"])
    assert result.exit_code == 1


def test_crs_override_succeeds(whirld_home: Path, s2_no_crs_tif: Path) -> None:
    """A CRS-less input embeds when --crs is supplied."""
    whirld.pull("clay-v1")
    result = whirld.embed(s2_no_crs_tif, model="clay-v1", crs="EPSG:32630")
    assert result.embeddings.shape[1] == 512
    assert result.meta["crs"] == "EPSG:32630"


def test_crs_missing_still_rejected(whirld_home: Path, s2_no_crs_tif: Path) -> None:
    """Without --crs, a CRS-less input is still rejected (exit code 7)."""
    whirld.pull("clay-v1")
    with pytest.raises(InvalidInputError) as excinfo:
        whirld.embed(s2_no_crs_tif, model="clay-v1")
    assert excinfo.value.exit_code == 7


def test_batch_size_matches_unbatched(whirld_home: Path, s2_tif: Path) -> None:
    """Embeddings are identical regardless of --batch-size (reference backend
    ignores it; this guards the plumbing doesn't corrupt results)."""
    whirld.pull("clay-v1")
    a = whirld.embed(s2_tif, model="clay-v1", write=False)
    b = whirld.embed(s2_tif, model="clay-v1", batch_size=1, write=False)
    assert np.array_equal(a.embeddings, b.embeddings)
