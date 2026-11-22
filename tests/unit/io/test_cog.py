"""Direct COG metadata selection and real GDAL reads obey the model contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.env import get_gdal_config
from rasterio.transform import from_origin

from whirld.core.chips import chip_raster
from whirld.core.contract import translate
from whirld.core.options import CogOptions
from whirld.core.registry import ModelEntry, Registry
from whirld.errors import InvalidInputError, UnsupportedSensorError
from whirld.io.cog import CogSceneSource, CogWindowReader, read_remote_cog
from whirld.io.scene import RasterScene
from whirld.io.stac_auth import StacAccessPolicy

_SENSOR = "sentinel-2-l2a"
_URL = "https://example.invalid/scene.tif?signature=private-value"


@pytest.fixture
def cog_entry(whirld_home: Path) -> ModelEntry:
    """Create a two-band contract with independently observable normalized bands."""
    raw = Registry().get("clay-v1").model_dump()
    sensor = raw["band_contract"]["sensors"][_SENSOR]
    sensor["bands"] = ["B02", "B03"]
    sensor["aliases"] = ["blue", "green"]
    raw["band_contract"]["sensors"] = {_SENSOR: sensor}
    raw["band_contract"]["normalization"] = {
        "type": "per_band_zscore",
        "scale": 0.0001,
        "mean": [0.1, 0.2],
        "std": [0.2, 0.4],
    }
    raw["band_contract"]["nodata_fill"] = -7.0
    return ModelEntry.model_validate(raw)


class _LocalCog:
    """Map one remote URL to a real local TIFF while keeping GDAL reads intact.

    HTTP and credential behavior is covered by the real range-server integration
    suite; this fixture isolates deterministic metadata/pixel behavior.
    """

    def __init__(self, path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Install URL translation and observe dataset/cache resource lifetimes."""
        self.path = path
        self.opened: list[Any] = []
        self.cache_values: list[int] = []
        original_open = rasterio.open

        def open_local(location: Any, *args: Any, **kwargs: Any) -> Any:
            """Translate only the declared remote URL, retaining real VRT opens."""
            if location == "/vsicurl/" + _URL:
                self.cache_values.append(get_gdal_config("GDAL_CACHEMAX"))
                location = self.path
            dataset = original_open(location, *args, **kwargs)
            self.opened.append(dataset)
            return dataset

        monkeypatch.setattr(rasterio, "open", open_local)

    def write(
        self,
        *,
        names: tuple[str | None, ...] = ("GREEN", "B02", "unused"),
        scales: tuple[float, ...] = (1.0, 1.0, 1.0),
        offsets: tuple[float, ...] = (0.0, 0.0, 0.0),
        masks: bool = False,
        crs: str | None = "EPSG:32630",
    ) -> np.ndarray:
        """Write reordered source bands, TIFF metadata, optional nodata and masks."""
        pixels = np.stack(
            [
                2000 + np.arange(7 * 9).reshape(7, 9) * 10,
                1000 + np.arange(7 * 9).reshape(7, 9) * 20,
                np.full((7, 9), 3000),
            ]
        ).astype(np.float32)
        if masks:
            pixels[:, 1, 2] = 0
            pixels[:, 3, 4] = 1_000_000
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(
                self.path,
                "w",
                driver="GTiff",
                width=9,
                height=7,
                count=3,
                dtype="float32",
                crs=crs,
                transform=from_origin(320000, 5822560, 10, 10),
                nodata=0,
            ) as dataset:
                dataset.write(pixels)
                dataset.descriptions = names
                dataset.scales = scales
                dataset.offsets = offsets
                dataset.update_tags(TIFFTAG_DATETIME="2024:06:01 12:00:00")
                if masks:
                    valid = np.ones((7, 9), dtype=np.uint8) * 255
                    valid[3, 4] = 0
                    dataset.write_mask(valid)
        return pixels

    def resolve(self, entry: ModelEntry, cog: CogOptions | None = None) -> RasterScene:
        """Resolve the real file header without requesting any pixels."""
        return CogSceneSource(_URL, cog).resolve(entry, _SENSOR, StacAccessPolicy(_URL))


@pytest.fixture
def local_cog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _LocalCog:
    """Provide isolated real TIFF datasets behind the validated remote source URL."""
    return _LocalCog(tmp_path / "source.tif", monkeypatch)


def test_description_mapping_identity_dn_and_metadata(
    local_cog: _LocalCog, cog_entry: ModelEntry
) -> None:
    """Unique case-insensitive names reorder pixels; identity TIFF values stay DN."""
    pixels = local_cog.write()
    scene = local_cog.resolve(cog_entry)
    assert [band.index for band in scene.bands] == [2, 1]
    assert all(not band.radiometry.has_transform for band in scene.bands)
    raster = read_remote_cog(_URL, entry=cog_entry, sensor=_SENSOR)
    translated = translate(raster, cog_entry, _SENSOR)
    expected = (pixels[[1, 0]] * 0.0001 - np.array([0.1, 0.2])[:, None, None]) / (
        np.array([0.2, 0.4])[:, None, None]
    )
    np.testing.assert_allclose(translated.data, expected, rtol=1e-6, atol=1e-6)
    assert raster.tags["TIFFTAG_DATETIME"] == "2024:06:01 12:00:00"
    assert raster.band_descriptions == ["B02", "B03"]


@pytest.mark.parametrize(
    ("options", "expected_scales", "expected_offsets"),
    [
        (None, [0.0001, 0.0002], [-0.1, 0.1]),
        (CogOptions(scale=0.0003), [0.0003, 0.0003], [-0.1, 0.1]),
        (CogOptions(offset=0.2), [0.0001, 0.0002], [0.2, 0.2]),
        (
            CogOptions(scale=(0.0004, 0.0005, 0.0006), offset=(0.3, 0.4, 0.5)),
            [0.0005, 0.0004],
            [0.4, 0.3],
        ),
    ],
)
def test_tiff_radiometry_and_per_field_source_position_overrides(
    local_cog: _LocalCog,
    cog_entry: ModelEntry,
    options: CogOptions | None,
    expected_scales: list[float],
    expected_offsets: list[float],
) -> None:
    """TIFF transforms and independent overrides apply before contract normalization."""
    pixels = local_cog.write(scales=(0.0002, 0.0001, 1.0), offsets=(0.1, -0.1, 0.0))
    raster = read_remote_cog(_URL, entry=cog_entry, sensor=_SENSOR, cog=options)
    transformed = translate(raster, cog_entry, _SENSOR)
    expected_reflectance = (
        pixels[[1, 0]].astype(np.float64) * np.array(expected_scales)[:, None, None]
        + np.array(expected_offsets)[:, None, None]
    )
    expected = (expected_reflectance - np.array([0.1, 0.2])[:, None, None]) / (
        np.array([0.2, 0.4])[:, None, None]
    )
    np.testing.assert_allclose(transformed.data, expected, rtol=1e-6, atol=1e-6)


def test_explicit_identity_declares_reflectance(
    local_cog: _LocalCog, cog_entry: ModelEntry
) -> None:
    """Explicit scale one is a physical transform, distinct from TIFF defaults."""
    pixels = local_cog.write()
    scene = local_cog.resolve(cog_entry, CogOptions(scale=1.0))
    assert all(band.radiometry.has_transform for band in scene.bands)
    raster = read_remote_cog(
        _URL, entry=cog_entry, sensor=_SENSOR, cog=CogOptions(scale=1.0)
    )
    expected = (pixels[[1, 0]] - np.array([0.1, 0.2])[:, None, None]) / (
        np.array([0.2, 0.4])[:, None, None]
    )
    np.testing.assert_allclose(
        translate(raster, cog_entry, _SENSOR).data, expected, rtol=1e-6
    )


@pytest.mark.parametrize("overlap", [0, 2])
def test_masks_nodata_and_edge_chips_match_eager(
    local_cog: _LocalCog, cog_entry: ModelEntry, overlap: int
) -> None:
    """TIFF masks and nodata survive shared interpolation and chip padding."""
    local_cog.write(masks=True)
    raster = read_remote_cog(_URL, entry=cog_entry, sensor=_SENSOR)
    transformed = translate(raster, cog_entry, _SENSOR)
    assert np.all(transformed.data[:, 1, 2] == -7)
    assert np.all(transformed.data[:, 3, 4] == -7)
    expected, geometry = chip_raster(transformed, 4, overlap=overlap, nodata_fill=-7)
    with CogWindowReader(
        _URL,
        entry=cog_entry,
        sensor=_SENSOR,
        chip_size=4,
        overlap=overlap,
        batch_size=2,
    ) as reader:
        batches = list(reader.batches())
        assert reader.metadata.timestamp == "2024:06:01 12:00:00"
    np.testing.assert_allclose(
        np.concatenate([batch.pixels for batch in batches]), expected, atol=1e-6
    )
    assert [chip for batch in batches for chip in batch.chips] == geometry.chips
    assert all(len(batch.chips) <= 2 for batch in batches)


@pytest.mark.parametrize(
    "names",
    [
        (None, None, None),
        ("green", None, "unused"),
        ("green", "blue", "BLUE"),
        ("green", "blue", "B02"),
        ("green", "red", "unused"),
    ],
)
def test_ambiguous_partial_or_missing_names_require_override(
    local_cog: _LocalCog, cog_entry: ModelEntry, names: tuple[str | None, ...]
) -> None:
    """A source never guesses spectral meaning from stored band positions."""
    local_cog.write(names=names)
    with pytest.raises(InvalidInputError):
        local_cog.resolve(cog_entry)
    scene = local_cog.resolve(cog_entry, CogOptions(bands=("green", "blue", "unused")))
    assert [band.index for band in scene.bands] == [2, 1]


@pytest.mark.parametrize(
    "options",
    [
        CogOptions(bands=("green", "blue")),
        CogOptions(scale=(0.0001, 0.0002)),
        CogOptions(offset=(0.1, 0.2)),
    ],
)
def test_source_position_overrides_require_full_count(
    local_cog: _LocalCog, cog_entry: ModelEntry, options: CogOptions
) -> None:
    """Overrides describe all source positions, including unused stored bands."""
    local_cog.write()
    with pytest.raises(InvalidInputError, match="one value per source band"):
        local_cog.resolve(cog_entry, options)


@pytest.mark.parametrize("sensor", [None, "not-a-sensor"])
def test_explicit_supported_sensor_required_before_open(
    local_cog: _LocalCog, cog_entry: ModelEntry, sensor: str | None
) -> None:
    """Missing/unsupported sensors fail without initiating a TIFF header read."""
    with pytest.raises(UnsupportedSensorError):
        with CogWindowReader(_URL, entry=cog_entry, sensor=sensor, chip_size=4):
            pytest.fail("Sensor validation must prevent entry")
    assert not local_cog.opened


@pytest.mark.parametrize(
    "url",
    [
        "file:///scene.tif",
        "ftp://example.invalid/scene.tif",
        "https://user:secret@example.invalid/scene.tif",
        "https://example.invalid:invalid/scene.tif",
        "https://example.invalid/scene.tif#secret",
        "https:///scene.tif",
    ],
)
def test_remote_urls_reject_unsafe_or_invalid_locations(url: str) -> None:
    """Invalid URLs produce generic diagnostics without echoing secrets."""
    with pytest.raises(InvalidInputError) as caught:
        CogSceneSource(url)
    assert url not in str(caught.value)
    assert "secret" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_header_cache_limits_and_failure_cleanup(
    local_cog: _LocalCog, cog_entry: ModelEntry
) -> None:
    """Caches are bounded before metadata discovery and restored after failure."""
    local_cog.write()
    local_cog.cache_values.clear()
    with rasterio.Env(GDAL_CACHEMAX=9 * 1024 * 1024):
        with pytest.raises(RuntimeError, match="consumer"):
            with CogWindowReader(
                _URL, entry=cog_entry, sensor=_SENSOR, chip_size=4, cache_mb=2
            ) as reader:
                next(reader.batches())
                raise RuntimeError("consumer")
        assert get_gdal_config("GDAL_CACHEMAX") == 9 * 1024 * 1024
    assert len(local_cog.cache_values) >= 4
    assert all(value == 2 * 1024 * 1024 for value in local_cog.cache_values)
    assert all(dataset.closed for dataset in local_cog.opened)
    with pytest.raises(InvalidInputError, match="Open"):
        reader.read_raster()


@pytest.mark.parametrize("geometry", ["missing-crs", "geographic", "rotated"])
def test_unsupported_geometry_is_explicit(
    local_cog: _LocalCog, cog_entry: ModelEntry, geometry: str
) -> None:
    """Direct COGs enforce the existing metric, affine streaming grid contract."""
    local_cog.write(crs=None if geometry == "missing-crs" else "EPSG:32630")
    with rasterio.open(local_cog.path, "r+") as dataset:
        if geometry == "geographic":
            dataset.crs = "EPSG:4326"
        elif geometry == "rotated":
            dataset.transform = rasterio.Affine(10, 1, 320000, 1, -10, 5822560)
    with pytest.raises(InvalidInputError):
        read_remote_cog(_URL, entry=cog_entry, sensor=_SENSOR)
    if geometry == "missing-crs":
        source = read_remote_cog(
            _URL, entry=cog_entry, sensor=_SENSOR, crs="EPSG:32630"
        )
        assert source.crs == "EPSG:32630"


@pytest.mark.parametrize("field", ["scales", "offsets"])
def test_nonfinite_tiff_transforms_fail_without_echoing_metadata(
    local_cog: _LocalCog, cog_entry: ModelEntry, field: str
) -> None:
    """Automatic TIFF radiometry must be finite before pixel normalization."""
    local_cog.write()
    with rasterio.open(local_cog.path, "r+") as dataset:
        setattr(dataset, field, (float("nan"), 1.0, 1.0))
    with pytest.raises(InvalidInputError, match="finite"):
        local_cog.resolve(cog_entry)


def test_one_source_band_cannot_satisfy_two_contract_bands(
    local_cog: _LocalCog, cog_entry: ModelEntry
) -> None:
    """An overlapping native/alias vocabulary cannot silently reuse observations."""
    local_cog.write(names=("green", "B02", "unused"))
    raw = cog_entry.model_dump()
    raw["band_contract"]["sensors"][_SENSOR]["aliases"] = ["green", "green"]
    ambiguous = ModelEntry.model_validate(raw)
    with pytest.raises(InvalidInputError, match="exactly one|ambiguous"):
        local_cog.resolve(ambiguous)
