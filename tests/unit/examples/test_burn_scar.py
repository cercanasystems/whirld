"""Hermetic source-unit, geometry and mask checks for the HLS example."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from examples.burn_scar import BurnScarExample, HlsSample
from whirld.core.contract import translate
from whirld.core.registry import Registry
from whirld.errors import InvalidInputError
from whirld.io.stac import read_stac_item


def _source(path: Path) -> None:
    """Write a 512-square six-band reflectance raster with known physical values.

    Args:
        path: Temporary GeoTIFF destination for an independent 0.2-reflectance input.
    """
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=512,
        height=512,
        count=6,
        dtype="float32",
        crs="EPSG:32610",
        transform=from_origin(564870, 4287150, 30, 30),
        nodata=0,
    ) as dataset:
        dataset.write(np.full((6, 512, 512), 0.2, dtype=np.float32))


def _mask(path: Path, *, shifted: bool = False, value: int = 1) -> None:
    """Write a binary-or-invalid mask for spatial and class contract tests.

    Args:
        path: Temporary mask output.
        shifted: Move the mask 30 m east to simulate a georeferencing regression.
        value: Value assigned to all pixels; 0 and 1 are valid, 2 is invalid.
    """
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=512,
        height=512,
        count=1,
        dtype="uint8",
        crs="EPSG:32610",
        transform=from_origin(564900 if shifted else 564870, 4287150, 30, 30),
        nodata=255,
    ) as dataset:
        dataset.write(np.full((512, 512), value, dtype=np.uint8), 1)


def test_reflectance_is_normalized_exactly_once(
    tmp_path: Path, whirld_home: Path
) -> None:
    """Explicit scale 1 preserves physical reflectance before Prithvi normalization."""
    image = tmp_path / "sample.tif"
    item = tmp_path / "item.json"
    _source(image)
    HlsSample.write_item(image, item)
    entry = Registry().get("prithvi-burn-scar")
    source = read_stac_item(str(item), entry=entry, sensor="hls")
    prepared = translate(source, entry, "hls")
    normalization = entry.band_contract.normalization
    for index in range(6):
        expected = (0.2 - normalization.mean[index]) / normalization.std[index]
        np.testing.assert_allclose(prepared.data[index], expected, rtol=1e-5)


@pytest.mark.parametrize("value", [0, 1])
def test_valid_masks_keep_background_and_burn_classes(
    tmp_path: Path, value: int
) -> None:
    """Class zero is valid background; class one is counted as a prediction."""
    image, mask = tmp_path / "image.tif", tmp_path / "mask.tif"
    _source(image)
    _mask(mask, value=value)
    burned, valid = BurnScarExample.validate_mask(image, mask)
    assert valid == 512 * 512
    assert burned == valid * value


@pytest.mark.parametrize("failure", ["grid", "class", "empty"])
def test_invalid_prediction_is_rejected(tmp_path: Path, failure: str) -> None:
    """Never present shifted, invalid-class, or wholly missing masks as success."""
    image, mask = tmp_path / "image.tif", tmp_path / "mask.tif"
    _source(image)
    _mask(
        mask,
        shifted=failure == "grid",
        value=2 if failure == "class" else 255 if failure == "empty" else 1,
    )
    with pytest.raises(InvalidInputError):
        BurnScarExample.validate_mask(image, mask)


def test_overlay_only_changes_predicted_pixels(tmp_path: Path) -> None:
    """Original and overlay use one stretch and preserve the source coordinate grid."""
    image, mask = tmp_path / "image.tif", tmp_path / "mask.tif"
    _source(image)
    _mask(mask, value=0)
    original, unburned = BurnScarExample.preview_images(image, mask)
    np.testing.assert_array_equal(original.rgb, unburned.rgb)
    _mask(mask, value=1)
    original, burned = BurnScarExample.preview_images(image, mask)
    assert np.any(original.rgb != burned.rgb)
    assert original.transform == burned.transform
    assert original.crs == burned.crs


def test_source_missing_pixels_must_remain_missing(tmp_path: Path) -> None:
    """A missing input band invalidates its output pixel instead of becoming class 0."""
    image, mask = tmp_path / "image.tif", tmp_path / "mask.tif"
    _source(image)
    _mask(mask, value=0)
    with rasterio.open(image, "r+") as source:
        band = source.read(1)
        band[0, 0] = 0
        source.write(band, 1)
    with pytest.raises(InvalidInputError, match="validity"):
        BurnScarExample.validate_mask(image, mask)
    with rasterio.open(mask, "r+") as target:
        values = target.read(1)
        values[0, 0] = 255
        target.write(values, 1)
    burned, valid = BurnScarExample.validate_mask(image, mask)
    assert burned == 0
    assert valid == 512 * 512 - 1
