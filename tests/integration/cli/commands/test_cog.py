"""CLI direct COG options exercise actual remote reads and public validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import whirld
from tests.integration.io.test_cog_input import BANDS, CogScene, CogServer
from tests.integration.test_api_cog import patch_nonembedding_backends
from whirld.cli import app


def test_cli_streams_extensionless_authenticated_cog(
    whirld_home: Path,
    tmp_path: Path,
) -> None:
    """Explicit names/units and the generic token alias reach bounded COG execution."""
    whirld.pull("clay-v1")
    scene = CogScene(tmp_path / "cli", named=False)
    scene.create()
    output = tmp_path / "stream.npy"
    with CogServer(scene, token="fixture-token") as server:
        result = CliRunner().invoke(
            app,
            [
                "embed",
                server.origin + "/download?signature=fixture",
                "--model",
                "clay-v1",
                "--sensor",
                "sentinel-2-l2a",
                "--cog",
                "--bands",
                ",".join(BANDS),
                "--scale",
                "0.0001",
                "--offset=-0.1",
                "--token",
                "fixture-token",
                "--stream",
                "--batch-size",
                "2",
                "--chip-size",
                "128",
                "--device",
                "cpu",
                "--output",
                str(output),
            ],
        )
    assert result.exit_code == 0, result.output
    assert "Streamed 9 chips" in result.output
    assert np.load(output).shape == (9, 512)


def test_cli_classify_and_segment_accept_direct_metadata(
    whirld_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both CLI surfaces forward explicit band order and scalar radiometry."""
    patch_nonembedding_backends(monkeypatch)
    scene = CogScene(tmp_path / "cli-other", named=False)
    scene.create()
    geojson = tmp_path / "scores.geojson"
    mask = tmp_path / "mask.tif"
    with CogServer(scene) as server:
        classification = CliRunner().invoke(
            app,
            [
                "classify",
                server.url,
                "--model",
                "remoteclip",
                "--query",
                "water",
                "--query",
                "forest",
                "--sensor",
                "sentinel-2-l2a",
                "--bands",
                ",".join(BANDS),
                "--scale",
                "0.0001",
                "--device",
                "cpu",
                "--output",
                str(geojson),
            ],
        )
        segmentation = CliRunner().invoke(
            app,
            [
                "segment",
                server.url,
                "--model",
                "prithvi-burn-scar",
                "--sensor",
                "hls",
                "--bands",
                ",".join("B8A" if band == "B08" else band for band in BANDS),
                "--scale",
                "0.0001",
                "--device",
                "cpu",
                "--output",
                str(mask),
            ],
        )
    assert classification.exit_code == 0, classification.output
    assert segmentation.exit_code == 0, segmentation.output
    assert len(json.loads(geojson.read_text())["features"]) == 4
    assert mask.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--scale", "nan"],
        ["--offset", "bad"],
        ["--bands", "B02,,B04"],
    ],
)
def test_cli_invalid_metadata_fails_before_model_io(arguments: list[str]) -> None:
    """Malformed unit/name overrides return the domain input-error exit code."""
    result = CliRunner().invoke(
        app,
        [
            "embed",
            "https://example.invalid/scene.tif",
            "--model",
            "clay-v1",
            "--sensor",
            "sentinel-2-l2a",
            *arguments,
        ],
    )
    assert result.exit_code == 7, result.output
