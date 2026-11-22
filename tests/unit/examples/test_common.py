"""Display and output-safety checks for shared example helpers."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from examples.common import OverlayPolygon, Preview, RasterImage
from whirld.errors import InvalidInputError


def test_display_stretch_handles_nodata_without_changing_input() -> None:
    """Preview transforms must leave model input untouched and produce finite RGB."""
    pixels = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
    pixels[0, 0, 0] = np.nan
    before = pixels.copy()
    rgb = Preview.stretch(pixels)
    assert rgb.dtype == np.uint8
    assert rgb.shape == pixels.shape
    assert rgb[0, 0, 0] == 0
    np.testing.assert_array_equal(pixels, before)


def test_preview_projects_overlay_and_escapes_text(tmp_path: Path) -> None:
    """A standalone preview retains pixels, projected locations and safe labels."""
    raster = RasterImage(
        rgb=np.full((3, 10, 10), 128, dtype=np.uint8),
        transform=from_origin(-1, 1, 0.2, 0.2),
        crs="EPSG:4326",
    )
    ring = OverlayPolygon(
        coordinates=[[-1, 1], [1, 1], [1, -1], [-1, -1], [-1, 1]],
        label="<script>alert(1)</script>",
    )
    path = tmp_path / "preview.html"
    Preview.write_raster(path, "A & B", raster, [ring], "Test <caption>")
    html = path.read_text()
    assert 'points="0.00,0.00 10.00,0.00 10.00,10.00 0.00,10.00 0.00,0.00"' in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "A &amp; B" in html
    assert "data:image/png;base64," in html
    with rasterio.open(path.with_suffix(".png")) as source:
        np.testing.assert_array_equal(source.read(), raster.rgb)


def test_preview_rejects_wrong_rgb_contract(tmp_path: Path) -> None:
    """Reject float or wrongly ordered display arrays instead of corrupting colors."""
    image = RasterImage(
        rgb=np.ones((3, 10, 10)), transform=from_origin(0, 10, 1, 1), crs="EPSG:32610"
    )
    with pytest.raises(InvalidInputError, match="uint8"):
        Preview.write_raster(tmp_path / "bad.html", "Bad", image, [], "Bad")


def test_runtime_verifies_weights_and_preserves_previous_outputs(
    whirld_home: Path, tmp_path: Path
) -> None:
    """Reject altered checkpoints before creating outputs and never reuse a run dir."""
    import whirld
    from examples.common import ExampleRuntime
    from whirld.core.fetch import ModelStore

    manifest = whirld.pull("clay-v1")
    output = tmp_path / "run"
    ExampleRuntime(output).prepare("clay-v1")
    assert (output / "model.json").exists()
    with pytest.raises(FileExistsError):
        ExampleRuntime(output).prepare("clay-v1")
    weights = ModelStore().paths.model_file("clay-v1", manifest.weights_file)
    original = weights.read_bytes()
    weights.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    failed_output = tmp_path / "invalid-run"
    with pytest.raises(InvalidInputError, match="checksum"):
        ExampleRuntime(failed_output).prepare("clay-v1")
    assert not failed_output.exists()


def test_edge_chip_label_stays_in_visible_footprint(tmp_path: Path) -> None:
    """A partially padded chip keeps its label inside the cropped display image."""
    raster = RasterImage(
        rgb=np.full((3, 100, 100), 128, dtype=np.uint8),
        transform=from_origin(0, 1, 0.01, 0.01),
        crs="EPSG:4326",
    )
    ring = OverlayPolygon(
        coordinates=[[0.8, 1], [1.2, 1], [1.2, 0], [0.8, 0], [0.8, 1]],
        label="chip 3: urban area 0.90",
    )
    path = tmp_path / "edge.html"
    Preview.write_raster(path, "Edge", raster, [ring], "Partial chip")
    text = path.read_text()
    assert '<tspan x="90.00"' in text
    assert "urban area 0.90</tspan>" in text
    assert '<tspan x="100.00"' not in text
