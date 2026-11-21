"""Radiometric correctness tests using small, real GeoTIFF/STAC fixtures."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from whirld.core.contract import TranslatedRaster, translate
from whirld.core.registry import ModelEntry, Registry
from whirld.errors import InvalidInputError, RegistryError
from whirld.io.stac import read_stac_item


@pytest.fixture
def reflectance_entry(whirld_home: Path) -> ModelEntry:
    """Return a two-band contract whose normalized outputs equal reflectance."""
    raw = Registry().get("clay-v1").model_dump()
    sensor = raw["band_contract"]["sensors"]["sentinel-2-l2a"]
    sensor["bands"] = ["B02", "B03"]
    sensor["aliases"] = ["blue", "green"]
    raw["band_contract"]["sensors"] = {"sentinel-2-l2a": sensor}
    raw["band_contract"]["normalization"] = {
        "type": "per_band_zscore",
        "scale": 0.0001,
        "mean": [0.0, 0.0],
        "std": [1.0, 1.0],
    }
    return ModelEntry.model_validate(raw)


def _asset(
    path: Path,
    data: np.ndarray,
    *,
    metadata: list[dict[str, Any]] | None = None,
    nodata: float | None = None,
    mask: np.ndarray | None = None,
    resolution: float = 10.0,
) -> dict[str, Any]:
    """Write pixels and optional TIFF validity metadata; return a STAC asset."""
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=data.shape[2],
            height=data.shape[1],
            count=data.shape[0],
            dtype=data.dtype,
            crs="EPSG:32630",
            transform=from_origin(320000.0, 5822560.0, resolution, resolution),
            nodata=nodata,
        ) as dst:
            dst.write(data)
            if mask is not None:
                dst.write_mask(mask.astype(np.uint8) * 255)
    result: dict[str, Any] = {"href": path.as_uri()}
    if metadata is not None:
        result["raster:bands"] = metadata
    return result


def _item(path: Path, assets: dict[str, Any]) -> Path:
    """Write a minimal item with fixed sensor and acquisition metadata."""
    path.write_text(
        json.dumps(
            {
                "type": "Feature",
                "id": path.stem,
                "properties": {
                    "platform": "sentinel-2a",
                    "datetime": "2024-06-01T10:00:00Z",
                },
                "assets": assets,
            }
        ),
        encoding="utf-8",
    )
    return path


def _translate(
    item: Path,
    entry: ModelEntry,
    *,
    bbox: tuple[float, float, float, float] | None = None,
) -> TranslatedRaster:
    """Read an item through the public reader and apply its model contract."""
    sensor = next(iter(entry.band_contract.sensors))
    raster = read_stac_item(str(item), entry=entry, sensor=sensor, bbox=bbox)
    return translate(raster, entry, sensor)


def _constant_item(
    directory: Path,
    entry: ModelEntry,
    value: float,
    metadata: dict[str, Any] | None,
) -> Path:
    """Write all required bands with the same stored value and radiometry."""
    directory.mkdir()
    sensor = next(iter(entry.band_contract.sensors.values()))
    assets = {
        band: _asset(
            directory / f"{band}.tif",
            np.full((4, 4), value, dtype=np.float32),
            metadata=None if metadata is None else [metadata],
        )
        for band in sensor.bands
    }
    return _item(directory / "item.json", assets)


def test_different_band_scale_offsets(
    tmp_path: Path, reflectance_entry: ModelEntry
) -> None:
    """Each band uses its own physical transform, with no second scaling."""
    assets = {
        "B02": _asset(
            tmp_path / "blue.tif",
            np.full((4, 4), 3000, dtype=np.uint16),
            metadata=[{"scale": 0.0001, "offset": -0.1}],
        ),
        "B03": _asset(
            tmp_path / "green.tif",
            np.full((4, 4), 1000, dtype=np.uint16),
            metadata=[{"scale": 0.0002, "offset": 0.1}],
        ),
    }
    out = _translate(_item(tmp_path / "item.json", assets), reflectance_entry)
    np.testing.assert_allclose(out.data[0], 0.2, atol=1e-7)
    np.testing.assert_allclose(out.data[1], 0.3, atol=1e-7)


def test_radiometry_follows_shuffled_raster_band_descriptions(
    tmp_path: Path, reflectance_entry: ModelEntry
) -> None:
    """Translation selects each band's radiometry along with its shuffled pixels."""
    assets = {
        "B02": _asset(
            tmp_path / "blue.tif",
            np.full((4, 4), 3000, dtype=np.uint16),
            metadata=[{"scale": 0.0001, "offset": -0.1}],
        ),
        "B03": _asset(
            tmp_path / "green.tif",
            np.full((4, 4), 1000, dtype=np.uint16),
            metadata=[{"scale": 0.0002, "offset": 0.1}],
        ),
    }
    item = _item(tmp_path / "item.json", assets)
    raster = read_stac_item(str(item), entry=reflectance_entry)
    assert raster.band_radiometry is not None
    shuffled = replace(
        raster,
        data=raster.data[::-1],
        band_descriptions=raster.band_descriptions[::-1],
        band_radiometry=raster.band_radiometry[::-1],
    )
    out = translate(shuffled, reflectance_entry, "sentinel-2-l2a")
    np.testing.assert_allclose(out.data[0], 0.2, atol=1e-7)
    np.testing.assert_allclose(out.data[1], 0.3, atol=1e-7)


@pytest.mark.parametrize(
    ("stored", "metadata"),
    [
        (3000.0, {"scale": 0.0001, "offset": -0.1}),
        (2000.0, {"scale": 0.0001, "offset": 0.0}),
        (1000.0, {"scale": 0.0002}),
        (0.2, {"scale": 1.0}),
        (0.1, {"offset": 0.1}),
        (999.0, {"scale": 0.0, "offset": 0.2}),
    ],
)
def test_equivalent_physical_encodings(
    tmp_path: Path,
    reflectance_entry: ModelEntry,
    stored: float,
    metadata: dict[str, Any],
) -> None:
    """Different DN encodings of 20% reflectance produce identical inputs."""
    item = _constant_item(tmp_path / "encoded", reflectance_entry, stored, metadata)
    expected = _constant_item(tmp_path / "legacy", reflectance_entry, 2000.0, None)
    out = _translate(item, reflectance_entry)
    reference = _translate(expected, reflectance_entry)
    np.testing.assert_allclose(out.data, reference.data, atol=1e-7)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("clay-v1.5", (2000.0 - 1105.0) / 1809.0),
        ("clay-v1", (0.2 - 0.0379) / 0.0233),
        ("remoteclip", (0.2 - 0.48145466) / 0.26862954),
        ("prithvi-burn-scar", (0.2 - 0.03335) / 0.02269),
    ],
)
def test_bundled_model_input_units(
    whirld_home: Path, tmp_path: Path, model: str, expected: float
) -> None:
    """Real Clay receives canonical DN; other bundled models receive reflectance."""
    entry = Registry().get(model)
    item = _constant_item(
        tmp_path / model, entry, 3000.0, {"scale": 0.0001, "offset": -0.1}
    )
    out = _translate(item, entry)
    np.testing.assert_allclose(out.data[0], expected, rtol=1e-6, atol=1e-6)


def test_cached_clay_entry_without_new_convention(
    whirld_home: Path, tmp_path: Path
) -> None:
    """A pre-upgrade cached Clay entry still converts reflectance to Clay DN."""
    raw = Registry().get("clay-v1.5").model_dump()
    for sensor in raw["band_contract"]["sensors"].values():
        sensor.pop("canonical_reflectance_scale", None)
    entry = ModelEntry.model_validate(raw)
    item = _constant_item(
        tmp_path / "cached", entry, 3000.0, {"scale": 0.0001, "offset": -0.1}
    )
    out = _translate(item, entry)
    np.testing.assert_allclose(out.data[0], (2000.0 - 1105.0) / 1809.0, rtol=1e-6)


@pytest.mark.parametrize("metadata", [None, {}, {"nodata": -9999.0}])
def test_absent_transform_preserves_legacy_dn(
    tmp_path: Path, reflectance_entry: ModelEntry, metadata: dict[str, Any] | None
) -> None:
    """Absent scale/offset keeps raw DN semantics, including nodata-only metadata."""
    item = _constant_item(tmp_path / "legacy", reflectance_entry, 2000.0, metadata)
    np.testing.assert_allclose(_translate(item, reflectance_entry).data, 0.2, atol=1e-7)


def test_explicit_sensor_convention_controls_conversion(
    tmp_path: Path, reflectance_entry: ModelEntry
) -> None:
    """New sensors can declare canonical units without inheriting Sentinel DN."""
    entry = reflectance_entry.model_copy(deep=True)
    contract = entry.band_contract.sensors.pop("sentinel-2-l2a")
    contract.canonical_reflectance_scale = 1.0
    entry.band_contract.sensors["custom-reflectance"] = contract
    entry.band_contract.normalization.scale = 1.0
    item = _constant_item(
        tmp_path / "custom", entry, 3000.0, {"scale": 0.0001, "offset": -0.1}
    )
    np.testing.assert_allclose(_translate(item, entry).data, 0.2, atol=1e-7)


def test_unknown_sensor_requires_explicit_canonical_units(
    tmp_path: Path, reflectance_entry: ModelEntry
) -> None:
    """An unfamiliar sensor's model input units cannot be guessed from scale."""
    entry = reflectance_entry.model_copy(deep=True)
    contract = entry.band_contract.sensors.pop("sentinel-2-l2a")
    contract.canonical_reflectance_scale = None
    entry.band_contract.sensors["unknown-sensor"] = contract
    item = _constant_item(
        tmp_path / "unknown", entry, 3000.0, {"scale": 0.0001, "offset": -0.1}
    )
    with pytest.raises(RegistryError, match="canonical_reflectance_scale"):
        _translate(item, entry)


@pytest.mark.parametrize("field", ["scale", "offset"])
@pytest.mark.parametrize("value", [None, True, "invalid", float("nan"), float("inf")])
def test_invalid_radiometry_is_domain_error(
    tmp_path: Path, reflectance_entry: ModelEntry, field: str, value: Any
) -> None:
    """Invalid declared numeric transforms fail clearly before inference."""
    item = _constant_item(tmp_path / "bad", reflectance_entry, 2000.0, {field: value})
    with pytest.raises(InvalidInputError):
        _translate(item, reflectance_entry)


def test_finite_transform_overflow_is_not_silent_nodata(
    tmp_path: Path, reflectance_entry: ModelEntry
) -> None:
    """Unrepresentable valid observations must fail rather than become padding."""
    item = _constant_item(
        tmp_path / "overflow", reflectance_entry, 2000.0, {"scale": 1e300}
    )
    with pytest.raises(InvalidInputError, match="overflows"):
        _translate(item, reflectance_entry)


def test_normalization_overflow_is_not_silent_nodata(
    tmp_path: Path, reflectance_entry: ModelEntry
) -> None:
    """Model normalization cannot hide invalid arithmetic as missing imagery."""
    reflectance_entry.band_contract.normalization.std = [0.0, 0.0]
    item = _constant_item(
        tmp_path / "bad-norm", reflectance_entry, 2000.0, {"scale": 0.0001}
    )
    with pytest.raises(InvalidInputError, match="normalization"):
        _translate(item, reflectance_entry)


def test_multiband_asset_uses_selected_band_metadata(
    tmp_path: Path, reflectance_entry: ModelEntry
) -> None:
    """Band names select both the correct pixel plane and its radiometry."""
    asset = _asset(
        tmp_path / "stack.tif",
        np.stack(
            [
                np.full((4, 4), 1000, dtype=np.uint16),
                np.full((4, 4), 3000, dtype=np.uint16),
            ]
        ),
        metadata=[{"scale": 0.0002, "offset": 0.1}, {"scale": 0.0001, "offset": -0.1}],
    )
    asset["eo:bands"] = [{"name": "B03"}, {"name": "B02"}]
    item = _item(tmp_path / "item.json", {"multispectral": asset})
    out = _translate(item, reflectance_entry)
    np.testing.assert_allclose(out.data[0], 0.2, atol=1e-7)
    np.testing.assert_allclose(out.data[1], 0.3, atol=1e-7)


def test_stac_11_band_overrides_asset_radiometry_defaults(
    tmp_path: Path, reflectance_entry: ModelEntry
) -> None:
    """STAC 1.1 named bands retain their index and override asset-level defaults."""
    asset = _asset(
        tmp_path / "stack.tif",
        np.stack(
            [
                np.full((4, 4), 1000, dtype=np.uint16),
                np.full((4, 4), 3000, dtype=np.uint16),
            ]
        ),
    )
    asset["raster:scale"] = 0.0001
    asset["raster:offset"] = -0.1
    asset["bands"] = [
        {"name": "B03", "raster:scale": 0.0002, "raster:offset": 0.1},
        {"name": "B02"},
    ]
    item = _item(tmp_path / "item.json", {"multispectral": asset})
    out = _translate(item, reflectance_entry)
    np.testing.assert_allclose(out.data[0], 0.2, atol=1e-7)
    np.testing.assert_allclose(out.data[1], 0.3, atol=1e-7)


@pytest.mark.parametrize("value", [None, True, "unknown", float("inf")])
def test_invalid_nodata_is_rejected(
    tmp_path: Path, reflectance_entry: ModelEntry, value: Any
) -> None:
    """Invalid nodata declarations cannot silently change valid observations."""
    item = _constant_item(
        tmp_path / "bad-nodata", reflectance_entry, 2000.0, {"nodata": value}
    )
    with pytest.raises(InvalidInputError):
        _translate(item, reflectance_entry)


@pytest.mark.parametrize("sentinel", ["nan", "inf", "-inf"])
def test_special_stac_nodata_values(
    tmp_path: Path, reflectance_entry: ModelEntry, sentinel: str
) -> None:
    """STAC string forms of nonfinite nodata remain missing after conversion."""
    data = np.full((4, 4), 3000, dtype=np.float32)
    data[1, 1] = float(sentinel)
    asset = _asset(
        tmp_path / "band.tif",
        data,
        metadata=[{"scale": 0.0001, "offset": -0.1, "nodata": sentinel}],
    )
    item = _item(tmp_path / "item.json", {"B02": asset, "B03": asset})
    out = _translate(item, reflectance_entry)
    np.testing.assert_allclose(out.data[:, 1, 1], 0.0)
    np.testing.assert_allclose(out.data[:, 0, 0], 0.2, atol=1e-7)


@pytest.mark.parametrize("nodata_source", ["stac", "tiff", "mask", "stac-and-mask"])
def test_missing_pixels_receive_normalized_fill(
    tmp_path: Path, reflectance_entry: ModelEntry, nodata_source: str
) -> None:
    """Missing samples never become offset observations or normalized DN zeros."""
    entry = reflectance_entry.model_copy(deep=True)
    entry.band_contract.normalization.mean = [0.1, 0.1]
    entry.band_contract.normalization.std = [0.2, 0.2]
    entry.band_contract.nodata_fill = -7.0
    data = np.full((4, 4), 3000, dtype=np.float32)
    data[1, 1] = 0.0
    mask = np.ones((4, 4), dtype=bool)
    mask[1, 1] = False
    metadata: dict[str, Any] = {"scale": 0.0001, "offset": -0.1}
    if nodata_source in {"stac", "stac-and-mask"}:
        metadata["nodata"] = 0.0
    if nodata_source == "stac-and-mask":
        mask[2, 2] = False
    asset = _asset(
        tmp_path / "band.tif",
        data,
        metadata=[metadata],
        nodata=0.0 if nodata_source == "tiff" else None,
        mask=mask if nodata_source in {"mask", "stac-and-mask"} else None,
    )
    item = _item(tmp_path / "item.json", {"B02": asset, "B03": asset})
    out = _translate(item, entry)
    np.testing.assert_allclose(out.data[:, 1, 1], -7.0)
    np.testing.assert_allclose(out.data[:, 0, 0], 0.5, atol=1e-6)
    if nodata_source == "stac-and-mask":
        np.testing.assert_allclose(out.data[:, 2, 2], -7.0)
    assert np.isfinite(out.data).all()


def test_stac_nodata_overrides_tiff_nodata(
    tmp_path: Path, reflectance_entry: ModelEntry
) -> None:
    """Declared asset nodata takes precedence over the TIFF's different sentinel."""
    data = np.full((4, 4), 3000, dtype=np.float32)
    data[1, 1] = -9999.0
    data[2, 2] = 0.0
    asset = _asset(
        tmp_path / "band.tif",
        data,
        metadata=[{"scale": 0.0001, "offset": -0.1, "nodata": -9999.0}],
        nodata=0.0,
    )
    item = _item(tmp_path / "item.json", {"B02": asset, "B03": asset})
    out = _translate(item, reflectance_entry)
    np.testing.assert_allclose(out.data[:, 1, 1], 0.0)
    np.testing.assert_allclose(out.data[:, 0, 0], 0.2, atol=1e-7)
    # The superseded TIFF sentinel is a valid observation under the STAC contract.
    np.testing.assert_allclose(out.data[:, 2, 2], -0.1, atol=1e-7)


def test_nodata_excluded_from_bilinear_resampling(
    tmp_path: Path, reflectance_entry: ModelEntry
) -> None:
    """Warping a coarse band must not mix missing sentinels into valid pixels."""
    data = np.full((4, 4), 3000, dtype=np.float32)
    data[:, :2] = -9999.0
    asset = _asset(
        tmp_path / "coarse.tif",
        data,
        metadata=[{"scale": 0.0001, "offset": -0.1, "nodata": -9999.0}],
        resolution=20.0,
    )
    item = _item(tmp_path / "item.json", {"B02": asset, "B03": asset})
    out = _translate(item, reflectance_entry)
    assert out.data.shape == (2, 8, 8)
    assert np.any(out.data == 0.0)
    assert np.any(np.isclose(out.data, 0.2, atol=1e-7))
    assert np.all((out.data == 0.0) | np.isclose(out.data, 0.2, atol=1e-7))


def test_explicit_mask_and_nodata_excluded_before_resampling(
    tmp_path: Path, reflectance_entry: ModelEntry
) -> None:
    """Extreme masked values and STAC nodata cannot bleed into neighboring pixels."""
    data = np.full((4, 4), 3000, dtype=np.float32)
    data[:, 0] = -9999.0
    data[:, 1] = 1_000_000.0
    mask = np.ones((4, 4), dtype=bool)
    mask[:, 1] = False
    asset = _asset(
        tmp_path / "coarse-masked.tif",
        data,
        metadata=[{"scale": 0.0001, "offset": -0.1, "nodata": -9999.0}],
        mask=mask,
        resolution=20.0,
    )
    item = _item(tmp_path / "item.json", {"B02": asset, "B03": asset})
    out = _translate(item, reflectance_entry)
    assert out.data.shape == (2, 8, 8)
    assert np.any(out.data == 0.0)
    assert np.any(np.isclose(out.data, 0.2, atol=1e-7))
    assert np.all((out.data == 0.0) | np.isclose(out.data, 0.2, atol=1e-7))


def test_multiband_nodata_is_excluded_independently_before_resampling(
    tmp_path: Path, reflectance_entry: ModelEntry
) -> None:
    """A valid neighboring band cannot make another band's nodata interpolate.

    Each band has its own sentinel and missing column. Warping the entire source
    with one selected band's sentinel can invoke GDAL unified validity and mix
    those missing values into otherwise constant reflectance observations.
    """
    data = np.full((2, 4, 4), 3000, dtype=np.float32)
    data[0, :, 1] = -9999.0
    data[1, :, 2] = -8888.0
    asset = _asset(
        tmp_path / "coarse-multiband.tif",
        data,
        metadata=[
            {"scale": 0.0001, "offset": -0.1, "nodata": -9999.0},
            {"scale": 0.0001, "offset": -0.1, "nodata": -8888.0},
        ],
        resolution=20.0,
    )
    asset["eo:bands"] = [{"name": "B02"}, {"name": "B03"}]
    item = _item(tmp_path / "item.json", {"multispectral": asset})
    out = _translate(item, reflectance_entry)
    assert out.data.shape == (2, 8, 8)
    for band in out.data:
        assert np.any(band == 0.0)
        assert np.any(np.isclose(band, 0.2, atol=1e-7))
        assert np.all((band == 0.0) | np.isclose(band, 0.2, atol=1e-7))
    assert not np.array_equal(out.data[0] == 0.0, out.data[1] == 0.0)


def test_bbox_outside_asset_is_missing_not_offset_value(
    tmp_path: Path, reflectance_entry: ModelEntry
) -> None:
    """An expanded bbox fills outside coverage even when the TIFF has no nodata."""
    item = _constant_item(
        tmp_path / "bounded",
        reflectance_entry,
        3000.0,
        {"scale": 0.0001, "offset": -0.1},
    )
    bbox = transform_bounds(
        "EPSG:32630", "EPSG:4326", 319980.0, 5822500.0, 320060.0, 5822580.0
    )
    out = _translate(item, reflectance_entry, bbox=bbox)
    assert out.data.shape[1] > 4
    np.testing.assert_allclose(out.data[:, 0, 0], 0.0)
    assert np.any(np.isclose(out.data, 0.2, atol=1e-7))
    assert np.isfinite(out.data).all()
