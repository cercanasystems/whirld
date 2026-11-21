"""Windowed STAC reads preserve eager raster, mask, and chip semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.env import get_gdal_config
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from whirld.core.chips import chip_raster
from whirld.core.contract import translate
from whirld.core.registry import ModelEntry, Registry
from whirld.errors import InvalidInputError
from whirld.io.stac import read_stac_item
from whirld.io.stac_stream import StacWindowReader


@pytest.fixture
def stream_entry(whirld_home: Path) -> ModelEntry:
    """Two-band normalized contract with distinguishable radiometry and padding."""
    raw = Registry().get("clay-v1").model_dump()
    sensor = raw["band_contract"]["sensors"]["sentinel-2-l2a"]
    sensor["bands"] = ["B02", "B03"]
    sensor["aliases"] = ["blue", "green"]
    raw["band_contract"]["sensors"] = {"sentinel-2-l2a": sensor}
    raw["band_contract"]["normalization"] = {
        "type": "per_band_zscore",
        "scale": 0.0001,
        "mean": [0.1, 0.2],
        "std": [0.2, 0.4],
    }
    raw["band_contract"]["nodata_fill"] = -7.0
    return ModelEntry.model_validate(raw)


def _write_asset(
    path: Path,
    pixels: np.ndarray,
    resolution: float,
    *,
    mask: np.ndarray | None = None,
    crs: str | None = "EPSG:32630",
) -> dict[str, Any]:
    """Write real gradients, differing TIFF nodata, and optional dataset masks."""
    if pixels.ndim == 2:
        pixels = pixels[np.newaxis]
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=pixels.shape[2],
            height=pixels.shape[1],
            count=pixels.shape[0],
            dtype="float32",
            crs=crs,
            transform=from_origin(320000, 5822560, resolution, resolution),
            nodata=0,
        ) as dataset:
            dataset.write(pixels.astype(np.float32))
            if mask is not None:
                dataset.write_mask(mask.astype(np.uint8) * 255)
    return {"href": path.as_uri()}


def _write_item(path: Path, assets: dict[str, Any]) -> Path:
    """Write item metadata whose time must survive every streaming batch."""
    path.write_text(
        json.dumps(
            {
                "type": "Feature",
                "id": "sentinel-2-stream",
                "properties": {"datetime": "2024-06-01T12:00:00+02:00"},
                "assets": assets,
            }
        )
    )
    return path


def _mixed_item(directory: Path) -> Path:
    """Different resolutions and valid gradients expose interpolation seams."""
    fine = 1000 + np.arange(13 * 17).reshape(13, 17) * 19
    fine[2, 3] = -9999
    fine[4, 5] = 0  # valid because STAC supersedes the TIFF nodata sentinel
    coarse = 2000 + np.arange(7 * 9).reshape(7, 9) * 31
    coarse[2, 3] = -8888
    mask = np.ones_like(coarse, dtype=bool)
    mask[4, 5] = False
    coarse[4, 5] = 1_000_000
    blue = _write_asset(directory / "blue.tif", fine, 10)
    blue["raster:bands"] = [{"scale": 0.0001, "offset": -0.1, "nodata": -9999}]
    green = _write_asset(directory / "green.tif", coarse, 20, mask=mask)
    green["raster:bands"] = [{"scale": 0.0002, "offset": 0.1, "nodata": -8888}]
    # Reverse asset order: the contract, not item ordering, determines pixels.
    return _write_item(directory / "item.json", {"B03": green, "B02": blue})


@pytest.mark.parametrize("overlap", [0, 2])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("bounded", [False, True])
def test_window_batches_match_eager_mixed_resolution(
    tmp_path: Path,
    stream_entry: ModelEntry,
    overlap: int,
    batch_size: int,
    bounded: bool,
) -> None:
    """Every normalized sample and global padded chip bound matches eager reads."""
    item = _mixed_item(tmp_path)
    bbox = (
        transform_bounds("EPSG:32630", "EPSG:4326", 319980, 5822400, 320190, 5822580)
        if bounded
        else None
    )
    raster = read_stac_item(str(item), entry=stream_entry, bbox=bbox)
    translated = translate(raster, stream_entry, "sentinel-2-l2a")
    expected, expected_layout = chip_raster(
        translated, 6, overlap=overlap, nodata_fill=-7
    )
    with StacWindowReader(
        str(item),
        entry=stream_entry,
        chip_size=6,
        overlap=overlap,
        batch_size=batch_size,
        bbox=bbox,
        cache_mb=2,
    ) as reader:
        assert reader.sensor == "sentinel-2-l2a"
        assert reader.crs == raster.crs
        assert reader.metadata.timestamp == raster.tags.get("TIFFTAG_DATETIME")
        assert reader.chip_count == len(expected_layout.chips)
        assert reader.n_rows == expected_layout.n_rows
        assert reader.n_cols == expected_layout.n_cols
        assert reader.chip_size == 6
        assert reader.resolution_m == 10
        batches = list(reader.batches())
    assert all(0 < len(batch.chips) <= batch_size for batch in batches)
    actual = np.concatenate([batch.pixels for batch in batches])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert [chip for batch in batches for chip in batch.chips] == expected_layout.chips
    assert np.any(actual == -7)
    assert np.isfinite(actual).all()


def test_multiband_masks_and_per_band_nodata_match_eager(
    tmp_path: Path, stream_entry: ModelEntry
) -> None:
    """Selected-band VRT adapters preserve distinct nodata and a dataset mask."""
    pixels = 3000 + np.arange(2 * 7 * 9).reshape(2, 7, 9) * 23
    pixels[0, :, 2] = -8888
    pixels[1, :, 4] = -9999
    mask = np.ones((7, 9), dtype=bool)
    mask[5, 5] = False
    pixels[:, 5, 5] = 1_000_000
    asset = _write_asset(tmp_path / "stack.tif", pixels, 20, mask=mask)
    asset["eo:bands"] = [{"name": "B03"}, {"name": "B02"}]
    asset["raster:bands"] = [
        {"scale": 0.0002, "offset": 0.1, "nodata": -8888},
        {"scale": 0.0001, "offset": -0.1, "nodata": -9999},
    ]
    item = _write_item(tmp_path / "item.json", {"stack": asset})
    expected, geometry = chip_raster(
        translate(
            read_stac_item(str(item), entry=stream_entry),
            stream_entry,
            "sentinel-2-l2a",
        ),
        5,
        overlap=1,
        nodata_fill=-7,
    )
    with StacWindowReader(
        str(item), entry=stream_entry, chip_size=5, overlap=1, batch_size=3
    ) as reader:
        batches = list(reader.batches())
    np.testing.assert_allclose(
        np.concatenate([batch.pixels for batch in batches]),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )
    assert [chip for batch in batches for chip in batch.chips] == geometry.chips


def test_missing_crs_override_reaches_warped_source(
    tmp_path: Path, stream_entry: ModelEntry
) -> None:
    """An explicit CRS is applied to a source lacking CRS, not only its output."""
    pixels = np.arange(7 * 9).reshape(7, 9) + 1000
    asset = _write_asset(tmp_path / "unlocated.tif", pixels, 20, crs=None)
    item = _write_item(tmp_path / "item.json", {"B02": asset, "B03": asset})
    with pytest.raises(InvalidInputError, match="no CRS"):
        with StacWindowReader(str(item), entry=stream_entry, chip_size=6):
            pass
    with StacWindowReader(
        str(item), entry=stream_entry, chip_size=6, crs="EPSG:32630"
    ) as reader:
        batch = next(reader.batches())
        assert reader.crs == "EPSG:32630"
        assert np.isfinite(batch.pixels).all()
        assert np.any(batch.pixels != -7)


def test_reader_restores_resources_after_consumer_failure(
    tmp_path: Path, stream_entry: ModelEntry
) -> None:
    """Failure restores cache settings and closes the reader's source datasets."""
    item = _mixed_item(tmp_path)
    reader = StacWindowReader(str(item), entry=stream_entry, chip_size=6, cache_mb=2)
    with pytest.raises(InvalidInputError, match="Enter"):
        _ = reader.chip_count
    with pytest.raises(InvalidInputError, match="must be open"):
        next(reader.batches())
    with rasterio.Env(GDAL_CACHEMAX=9 * 1024 * 1024, VSI_CACHE_SIZE="123456"):
        with pytest.raises(RuntimeError, match="consumer"):
            with reader:
                assert get_gdal_config("GDAL_CACHEMAX") == 2 * 1024 * 1024
                assert int(get_gdal_config("VSI_CACHE_SIZE")) == 1024 * 1024
                next(reader.batches())
                raise RuntimeError("consumer failed")
        assert get_gdal_config("GDAL_CACHEMAX") == 9 * 1024 * 1024
        assert int(get_gdal_config("VSI_CACHE_SIZE")) == 123456
    with pytest.raises(InvalidInputError, match="must be open"):
        next(reader.batches())


@pytest.mark.parametrize(
    "options",
    [
        {"chip_size": 0},
        {"chip_size": 4, "overlap": 4},
        {"chip_size": 4, "batch_size": 0},
        {"chip_size": 4, "cache_mb": 0},
        {"chip_size": 4, "cache_mb": True},
    ],
)
def test_reader_rejects_invalid_limits_before_io(
    stream_entry: ModelEntry, options: dict[str, Any]
) -> None:
    """Configuration errors cannot initiate item or raster network access."""
    with pytest.raises(InvalidInputError):
        StacWindowReader("https://unused.invalid/item", entry=stream_entry, **options)


def test_failed_open_closes_prior_sources_and_restores_cache(
    tmp_path: Path, stream_entry: ModelEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later band's open failure releases earlier opened assets and GDAL state."""
    item = _mixed_item(tmp_path)
    document = json.loads(item.read_text())
    document["assets"]["B03"]["href"] = (tmp_path / "missing.tif").as_uri()
    item.write_text(json.dumps(document))
    original_open = rasterio.open
    opened: list[Any] = []

    def track_open(*args: Any, **kwargs: Any) -> Any:
        """Observe real dataset lifetimes while leaving raster operations intact."""
        dataset = original_open(*args, **kwargs)
        opened.append(dataset)
        return dataset

    monkeypatch.setattr(rasterio, "open", track_open)
    with rasterio.Env(GDAL_CACHEMAX=9 * 1024 * 1024):
        with pytest.raises(InvalidInputError, match="open STAC"):
            with StacWindowReader(
                str(item), entry=stream_entry, chip_size=6, cache_mb=2
            ):
                pytest.fail("missing second asset must fail during entry")
        assert get_gdal_config("GDAL_CACHEMAX") == 9 * 1024 * 1024
    assert len(opened) >= 2
    assert all(dataset.closed for dataset in opened)


@pytest.mark.parametrize("authorize_foreign", [False, True])
def test_each_asset_read_uses_its_own_origin_policy(
    tmp_path: Path, stream_entry: ModelEntry, authorize_foreign: bool
) -> None:
    """Interleaved retained VRTs cannot inherit another asset origin's token."""
    from tests.unit.test_stac_auth import _server

    pixels = np.arange(256 * 256, dtype=np.float32).reshape(256, 256) + 1000
    _write_asset(tmp_path / "remote.tif", pixels, 10)
    content = (tmp_path / "remote.tif").read_bytes()
    primary_requests: list[tuple[str, str | None]] = []
    foreign_requests: list[tuple[str, str | None]] = []
    primary_routes: dict[str, tuple[int, dict[str, str], bytes]] = {}
    with _server({"/green.tif": (200, {}, content)}, foreign_requests) as foreign:
        with _server(primary_routes, primary_requests) as primary:
            document = {
                "type": "Feature",
                "id": "sentinel-2-origin-test",
                "properties": {},
                "assets": {
                    "B02": {"href": primary + "/blue.tif"},
                    "B03": {"href": foreign + "/green.tif"},
                },
            }
            primary_routes["/item"] = (200, {}, json.dumps(document).encode())
            primary_routes["/blue.tif"] = (200, {}, content)
            with StacWindowReader(
                primary + "/item",
                entry=stream_entry,
                chip_size=32,
                batch_size=3,
                token="test-token",
                token_origins=(foreign,) if authorize_foreign else (),
                cache_mb=2,
            ) as reader:
                assert sum(len(batch.chips) for batch in reader.batches()) == 64
    assert len([path for path, _ in primary_requests if path == "/blue.tif"]) > 2
    assert len(foreign_requests) > 2  # includes lazy pixel ranges, not only headers
    assert all(token == "Bearer test-token" for _, token in primary_requests)
    assert all(
        token == ("Bearer test-token" if authorize_foreign else None)
        for _, token in foreign_requests
    )


@pytest.mark.parametrize("resolution", [10.0, 15.0, 20.0])
def test_fractional_interior_windows_match_independent_bilinear_oracle(
    tmp_path: Path, stream_entry: ModelEntry, resolution: float
) -> None:
    """Warp kernels cannot change with chunk size on fractional interior windows.

    A tiny bbox far inside a larger nonlinear source exposes GDAL's automatic
    per-chunk scale estimates. The oracle computes four-neighbor bilinear weights
    directly from pixel-center coordinates, without GDAL or Whirld warp helpers.
    """
    rng = np.random.default_rng(42)
    pixels = rng.integers(1, 10000, (512, 512), dtype=np.uint16)
    asset = _write_asset(tmp_path / "nonlinear.tif", pixels, resolution)
    item = _write_item(tmp_path / "item.json", {"B02": asset, "B03": asset})
    bbox = transform_bounds(
        "EPSG:32630",
        "EPSG:4326",
        321234.32,
        5819942.52,
        321264.32,
        5819992.52,
    )
    raster = read_stac_item(str(item), entry=stream_entry, bbox=bbox)
    native_transform = from_origin(320000, 5822560, resolution, resolution)
    expected_raw = np.empty((raster.height, raster.width), dtype=np.float32)
    for row in range(raster.height):
        for col in range(raster.width):
            x, y = raster.transform * (col + 0.5, row + 0.5)
            source_col, source_row = ~native_transform * (x, y)
            source_col -= 0.5
            source_row -= 0.5
            left, top = int(np.floor(source_col)), int(np.floor(source_row))
            dx, dy = source_col - left, source_row - top
            neighbors = pixels[top : top + 2, left : left + 2].astype(np.float64)
            expected_raw[row, col] = (
                neighbors[0, 0] * (1 - dx) * (1 - dy)
                + neighbors[0, 1] * dx * (1 - dy)
                + neighbors[1, 0] * (1 - dx) * dy
                + neighbors[1, 1] * dx * dy
            )
    for band in raster.data:
        np.testing.assert_allclose(band, expected_raw, rtol=1e-6, atol=1e-4)
    eager, geometry = chip_raster(
        translate(raster, stream_entry, "sentinel-2-l2a"),
        2,
        overlap=1,
        nodata_fill=-7,
    )
    with StacWindowReader(
        str(item),
        entry=stream_entry,
        chip_size=2,
        overlap=1,
        batch_size=2,
        bbox=bbox,
        cache_mb=2,
    ) as reader:
        streamed = list(reader.batches())
    np.testing.assert_allclose(
        np.concatenate([batch.pixels for batch in streamed]),
        eager,
        rtol=1e-6,
        atol=1e-6,
    )
    assert [chip for batch in streamed for chip in batch.chips] == geometry.chips


@pytest.mark.parametrize("geometry", ["different-crs", "rotated"])
def test_streaming_rejects_assets_requiring_general_reprojection(
    tmp_path: Path, stream_entry: ModelEntry, geometry: str
) -> None:
    """Unsupported secondary grids fail explicitly; eager general warps still work."""
    item = _mixed_item(tmp_path)
    with rasterio.open(tmp_path / "green.tif", "r+") as source:
        if geometry == "different-crs":
            source.crs = "EPSG:32631"
        else:
            source.transform = rasterio.Affine(20, 1, 320000, 1, -20, 5822560)
    eager = read_stac_item(str(item), entry=stream_entry)
    assert eager.data.shape == (2, 13, 17)
    with pytest.raises(InvalidInputError, match="same CRS.*north-up"):
        with StacWindowReader(str(item), entry=stream_entry, chip_size=4):
            pytest.fail("General reprojection must be rejected before streaming")


def test_band_handle_requires_open_lifetime(tmp_path: Path) -> None:
    """The typed handle rejects reads both before acquisition and after release."""
    from rasterio.windows import Window

    from whirld.io.radiometry import StacBand
    from whirld.io.stac import _Grid
    from whirld.io.stac_auth import StacAccessPolicy
    from whirld.io.stac_stream import _BandWindowReader

    asset = _write_asset(tmp_path / "handle.tif", np.ones((4, 4)), 10)
    reader = _BandWindowReader(
        StacBand(href=asset["href"], index=1),
        _Grid("EPSG:32630", from_origin(320000, 5822560, 10, 10), 4, 4),
        StacAccessPolicy(asset["href"]),
        warp_mb=1,
        crs_override=None,
    )
    window = Window(0, 0, 2, 2)
    with pytest.raises(InvalidInputError, match="Open the STAC band reader"):
        reader.read(window)
    with reader:
        np.testing.assert_array_equal(reader.read(window), np.ones((2, 2)))
    with pytest.raises(InvalidInputError, match="Open the STAC band reader"):
        reader.read(window)
