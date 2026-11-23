"""CLI local files and file URLs reach bounded public disk workflows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from typer.testing import CliRunner

import whirld
from tests.integration.test_local_streaming import LocalScene, patch_local_backends
from whirld.cli import app


@pytest.mark.parametrize("operation", ["embed", "classify", "segment"])
@pytest.mark.parametrize("as_uri", [False, True])
def test_cli_streams_local_raster_without_stac_wrapper(
    tmp_path: Path,
    whirld_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    as_uri: bool,
) -> None:
    """Actual source pixels and writers run for all CLI paradigms and local forms."""
    path = LocalScene(
        tmp_path / "scene with spaces.tif",
        width=700,
        height=600,
        resolution=(30.0, 30.0) if operation == "segment" else (10.0, 10.0),
        external_mask=True,
    ).create()
    if operation == "embed":
        whirld.pull("clay-v1")
        model, suffix, arguments = "clay-v1", ".npy", ["--chip-size", "128"]
    elif operation == "classify":
        patch_local_backends(monkeypatch)
        model, suffix, arguments = (
            "remoteclip",
            ".geojson",
            ["--query", "forest", "--query", "water", "--all"],
        )
    else:
        patch_local_backends(monkeypatch)
        model, suffix, arguments = (
            "prithvi-eo-2",
            ".tif",
            ["--head", "burn-scar", "--sensor", "hls"],
        )
    output = tmp_path / ("output" + suffix)
    result = CliRunner().invoke(
        app,
        [
            operation,
            path.as_uri() if as_uri else str(path),
            "--model",
            model,
            "--device",
            "cpu",
            "--stream",
            "--batch-size",
            "2",
            "--output",
            str(output),
            *arguments,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Streamed" in result.output
    assert output.is_file()
    if operation == "embed":
        embeddings = np.load(output, mmap_mode="r")
        assert embeddings.shape[1] == 512 and np.isfinite(embeddings).all()
    elif operation == "classify":
        collection = json.loads(output.read_text())
        assert len(collection["features"]) == 12
        assert collection["queries"] == ["forest", "water"]
    else:
        with rasterio.open(output) as source:
            assert source.shape == (600, 700)
            assert source.nodata == 255 and set(np.unique(source.read(1))) == {
                0,
                1,
                255,
            }


@pytest.mark.parametrize("operation", ["embed", "classify", "segment"])
def test_cli_local_stream_bbox_rejected_before_installation(
    whirld_home: Path, operation: str
) -> None:
    """Explicit rejection avoids silently presenting legacy eager bbox as cropping."""
    arguments = [
        operation,
        "missing-local.tif",
        "--model",
        "missing-model",
        "--stream",
        "--bbox=-1,50,0,51",
        "--output",
        "never-output",
    ]
    if operation == "classify":
        arguments.extend(["--query", "forest"])
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 7, result.output
    assert "bbox" in result.output
