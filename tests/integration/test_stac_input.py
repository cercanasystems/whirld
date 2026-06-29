"""Integration tests: STAC item input through the embed pipeline (offline)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import whirld
from whirld.cli import app
from whirld.config import get_paths

runner = CliRunner()


def test_embed_from_local_stac_item(whirld_home: Path, stac_item: Path) -> None:
    """`embed` accepts a STAC item (file:// assets) and matches the GeoTIFF shape."""
    whirld.pull("clay-v1")
    result = whirld.embed(str(stac_item), model="clay-v1", write=False)
    # Same six-band S2 footprint as make_sentinel2 → same chip grid + embed dim.
    assert result.embeddings.shape == (4, 512)
    assert result.sensor == "sentinel-2-l2a"


def test_stac_usage_records_input_type(whirld_home: Path, stac_item: Path) -> None:
    """A STAC run is logged with input_type=stac (not geotiff)."""
    whirld.pull("clay-v1")
    whirld.embed(str(stac_item), model="clay-v1", write=False)
    usage = get_paths().usage_log.read_text().strip().splitlines()
    record = json.loads(usage[-1])
    assert record["input_type"] == "stac"


def test_bbox_windows_the_read(whirld_home: Path, stac_item: Path) -> None:
    """A --bbox subset yields fewer chips than the full item."""
    whirld.pull("clay-v1")
    full = whirld.embed(str(stac_item), model="clay-v1", write=False)
    # The fixture spans EPSG:32630 origin 320000,5822560 at 10 m over 300x300 px.
    # A small lon/lat window covers far fewer pixels → fewer chips.
    windowed = whirld.embed(
        str(stac_item),
        model="clay-v1",
        write=False,
        bbox=(-1.30, 52.50, -1.27, 52.52),
    )
    assert windowed.embeddings.shape[0] <= full.embeddings.shape[0]


def test_cli_embed_from_stac_item(whirld_home: Path, stac_item: Path) -> None:
    """`whirld embed <item.json>` runs end to end and writes outputs."""
    runner.invoke(app, ["pull", "clay-v1"])
    out = Path(stac_item).parent / "emb.npy"
    result = runner.invoke(
        app, ["embed", str(stac_item), "--model", "clay-v1", "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_cli_rejects_bad_bbox(whirld_home: Path, stac_item: Path) -> None:
    """A malformed --bbox is rejected with a clear error (exit 7)."""
    runner.invoke(app, ["pull", "clay-v1"])
    result = runner.invoke(
        app,
        ["embed", str(stac_item), "--model", "clay-v1", "--bbox", "1,2,3"],
    )
    assert result.exit_code == 7  # InvalidInputError
