"""Local bounded windows retain independently computed eager raster semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from whirld.core.chips import chip_raster
from whirld.core.contract import translate
from whirld.core.registry import ModelEntry, Registry
from whirld.errors import InvalidInputError, UnsupportedSensorError
from whirld.io.local import LocalRasterAccessPolicy, LocalSceneSource, LocalWindowReader
from whirld.io.raster import read_raster


@pytest.fixture
def local_entry(whirld_home: Path) -> ModelEntry:
    """Make a two-band contract with nontrivial normalization and padding."""
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


def _write(
    path: Path,
    resolution: tuple[float, float] = (10, 10),
    *,
    mask: str | None = None,
    nonfinite: bool = False,
    names: tuple[str | None, ...] = ("B03", "B02"),
    crs: str | None = "EPSG:32630",
    shape: tuple[int, int] = (7, 9),
) -> Path:
    """Write distinguishable pixels, named reverse order, and optional sidecars."""
    pixels = (
        1000
        + np.arange(len(names) * shape[0] * shape[1]).reshape(len(names), *shape) * 19
    ).astype(np.float32)
    pixels[0, 2, 3] = -9999
    if nonfinite:
        pixels[0, 3, 5] = np.inf
        pixels[1, 4, 2] = -np.inf
        pixels[1, 2, 7] = np.nan
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=mask != "external"):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=shape[1],
            height=shape[0],
            count=len(names),
            dtype="float32",
            crs=crs,
            transform=from_origin(320000, 5822560, *resolution),
            nodata=-9999,
        ) as dataset:
            dataset.write(pixels)
            dataset.descriptions = names
            dataset.scales = tuple(0.005 for _ in names)
            dataset.offsets = tuple(-100 for _ in names)
            dataset.update_tags(
                TIFFTAG_IMAGEDESCRIPTION="sentinel-2-l2a",
                TIFFTAG_DATETIME="2024:06:01 12:00:00",
            )
            if mask:
                valid = np.ones(shape, dtype=np.uint8) * 255
                valid[5, 5] = 0
                dataset.write_mask(valid)
    return path


@pytest.mark.parametrize(
    "resolution",
    [
        (10, 10),
        (10.04, 9.98),
        (20, 30),
        (13, 17),
        (37, 23),
        (3, 5),
        (1, 2),
        (10.05, 10.05),
        (10.051, 9.99),
    ],
)
@pytest.mark.parametrize("mask", [None, "internal", "external"])
@pytest.mark.parametrize("nonfinite", [False, True])
def test_local_batches_equal_independent_eager(
    tmp_path: Path,
    local_entry: ModelEntry,
    resolution: tuple[float, float],
    mask: str | None,
    nonfinite: bool,
) -> None:
    """Affine, masks, radiometry defaults, interpolation, and chip padding agree."""
    path = _write(tmp_path / "pixels.tif", resolution, mask=mask, nonfinite=nonfinite)
    translated = translate(read_raster(path), local_entry, "sentinel-2-l2a")
    expected, layout = chip_raster(translated, 5, overlap=1, nodata_fill=-7)
    with LocalWindowReader(
        path, entry=local_entry, chip_size=5, overlap=1, batch_size=3, cache_mb=2
    ) as reader:
        assert reader.transform == translated.transform
        assert (reader.width, reader.height) == (
            translated.data.shape[2],
            translated.data.shape[1],
        )
        assert reader.metadata.timestamp == "2024:06:01 12:00:00"
        batches = list(reader.batches(include_validity=True))
    assert [chip for batch in batches for chip in batch.chips] == layout.chips
    np.testing.assert_allclose(
        np.concatenate([batch.pixels for batch in batches]),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("names", [(None, None), (" B03 ", "b02"), (None, None, None)])
def test_local_selection_retains_eager_fallback(
    tmp_path: Path, local_entry: ModelEntry, names: tuple[str | None, ...]
) -> None:
    """Unlabeled extra bands and case-insensitive native names match eager rules."""
    path = _write(tmp_path / "selection.tif", names=names)
    source = LocalSceneSource(path)
    scene = source.resolve(local_entry, None, LocalRasterAccessPolicy(str(path)))
    expected = [2, 1] if names[0] else [1, 2]
    assert [band.index for band in scene.bands] == expected
    assert all(not band.radiometry.has_transform for band in scene.bands)
    assert scene.kind == "geotiff"


@pytest.mark.parametrize("names", [("B02", None), ("B02", "b02"), ("blue", "green")])
def test_local_partial_duplicate_or_alias_descriptions_rejected(
    tmp_path: Path, local_entry: ModelEntry, names: tuple[str | None, ...]
) -> None:
    """Native local metadata never borrows COG alias or partial-name fallback."""
    path = _write(tmp_path / "selection.tif", names=names)
    with pytest.raises(UnsupportedSensorError):
        with LocalWindowReader(path, entry=local_entry, chip_size=5):
            pass


def test_missing_crs_override_and_bbox(tmp_path: Path, local_entry: ModelEntry) -> None:
    """Missing CRS can be assigned while local streaming crop requests fail early."""
    path = _write(tmp_path / "unreferenced.tif", crs=None)
    with pytest.raises(InvalidInputError, match="no CRS"):
        with LocalWindowReader(path, entry=local_entry, chip_size=5):
            pass
    with LocalWindowReader(
        path, entry=local_entry, chip_size=5, crs="EPSG:32630"
    ) as reader:
        assert reader.crs == "EPSG:32630"
        assert next(reader.batches()).pixels.shape[1:] == (2, 5, 5)
    with pytest.raises(InvalidInputError, match="bbox"):
        LocalWindowReader(path, entry=local_entry, chip_size=5, bbox=(0, 0, 1, 1))


@pytest.mark.parametrize(
    "location",
    [
        "missing.tif",
        "https://example.test/a.tif",
        "/vsicurl/https://example.test/a.tif",
        "NETCDF:thing:var",
    ],
)
def test_nonfilesystem_locations_rejected_before_gdal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, location: str
) -> None:
    """GDAL receives no missing, URL, VSI or subdataset input strings."""
    monkeypatch.chdir(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("GDAL must not open unbound locations")

    monkeypatch.setattr(rasterio, "open", forbidden)
    with pytest.raises(InvalidInputError):
        LocalSceneSource(location)
    with pytest.raises(InvalidInputError):
        LocalSceneSource(tmp_path)


def test_local_vrt_driver_guard_and_path_binding(
    tmp_path: Path, local_entry: ModelEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative identities stay bound and VRT content cannot invoke source drivers."""
    _write(tmp_path / "local#?%.tif")
    monkeypatch.chdir(tmp_path)
    reader = LocalWindowReader("local#?%.tif", entry=local_entry, chip_size=5)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    with reader:
        assert next(reader.batches()).pixels.shape[1:] == (2, 5, 5)
    vrt = tmp_path / "pretend.tif"
    vrt.write_text(
        '<VRTDataset rasterXSize="1" rasterYSize="1"><VRTRasterBand dataType="Byte" band="1"><SimpleSource><SourceFilename>/vsicurl/http://127.0.0.1:9/forbidden</SourceFilename><SourceBand>1</SourceBand></SimpleSource></VRTRasterBand></VRTDataset>'
    )
    with pytest.raises(InvalidInputError, match="GeoTIFF"):
        with LocalWindowReader(vrt, entry=local_entry, chip_size=5):
            pass


def test_pam_metadata_discovered_and_header_read_is_pixel_free(
    tmp_path: Path, local_entry: ModelEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metadata resolution sees sidecars without calling any pixel-read method."""
    path = _write(tmp_path / "pam.tif", names=(None, None))
    with rasterio.open(path, "r+") as dataset:
        dataset.update_tags(TIFFTAG_IMAGEDESCRIPTION="")
    Path(str(path) + ".aux.xml").write_text(
        '<PAMDataset><Metadata><MDI key="TIFFTAG_IMAGEDESCRIPTION">'
        'sentinel-2-l2a</MDI><MDI key="TIFFTAG_DATETIME">'
        "2025:02:03 04:05:06</MDI></Metadata></PAMDataset>"
    )
    opened = rasterio.open

    class HeaderOnly:
        """Delegate header access but fail any attempt to load pixels."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.dataset = opened(*args, **kwargs)

        def __enter__(self) -> HeaderOnly:
            return self

        def __exit__(self, *args: object) -> None:
            self.dataset.close()

        def __getattr__(self, name: str) -> object:
            if name in {"read", "read_masks", "dataset_mask"}:
                raise AssertionError("Header resolution must not read pixels")
            return getattr(self.dataset, name)

    monkeypatch.setattr(rasterio, "open", HeaderOnly)
    scene = LocalSceneSource(path).resolve(
        local_entry, None, LocalRasterAccessPolicy(str(path))
    )
    assert scene.sensor == "sentinel-2-l2a"
    assert scene.metadata.timestamp == "2025:02:03 04:05:06"


def test_owned_local_environments_and_handles_restore(
    tmp_path: Path, local_entry: ModelEntry
) -> None:
    """Exceptions release all source/VRT handles and scoped local GDAL options."""
    from rasterio.env import get_gdal_config

    path = _write(tmp_path / "mask.tif", mask="external")
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_HEADERS="Authorization: Bearer outer",
        GDAL_CACHEMAX=123456,
    ):
        reader = LocalWindowReader(path, entry=local_entry, chip_size=5, cache_mb=2)
        with pytest.raises(RuntimeError, match="consumer"):
            with reader:
                assert get_gdal_config("GDAL_CACHEMAX") == 2 * 1024 * 1024
                assert next(reader.batches()).pixels.shape[1:] == (2, 5, 5)
                raise RuntimeError("consumer")
        assert get_gdal_config("GDAL_DISABLE_READDIR_ON_OPEN") == "EMPTY_DIR"
        assert get_gdal_config("GDAL_HTTP_HEADERS") == "Authorization: Bearer outer"
        assert get_gdal_config("GDAL_CACHEMAX") == 123456
        assert reader._readers == []
        assert reader._resources is None
    path.unlink()


@pytest.mark.parametrize("resolution", [(3.3, 5.7), (4, 4), (13.3, 18.7)])
@pytest.mark.parametrize("chip_size", [5, 17, 64])
def test_downsampling_kernel_stays_equal_across_windows(
    tmp_path: Path,
    local_entry: ModelEntry,
    resolution: tuple[float, float],
    chip_size: int,
) -> None:
    """Fixed local scales reproduce eager edge clipping without chip-boundary seams."""
    path = _write(
        tmp_path / "large.tif",
        resolution,
        shape=(101, 117),
        nonfinite=True,
        mask="external",
    )
    translated = translate(read_raster(path), local_entry, "sentinel-2-l2a")
    expected, _ = chip_raster(translated, chip_size, overlap=1, nodata_fill=-7)
    with LocalWindowReader(
        path, entry=local_entry, chip_size=chip_size, overlap=1, batch_size=3
    ) as reader:
        batches = list(reader.batches())
    np.testing.assert_allclose(
        np.concatenate([batch.pixels for batch in batches]),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_all_source_opens_are_gtiff_and_chip_reads_are_bounded(
    tmp_path: Path, local_entry: ModelEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Driver selection precedes external access and pixel reads carry chip windows."""
    from typing import Any

    import whirld.io.stac_stream as module

    path = _write(tmp_path / "bounded.tif", (20, 30), shape=(101, 117))
    opened = rasterio.open
    warped = module.WarpedVRT
    sources: list[Any] = []
    warps: list[Any] = []
    windows: list[Any] = []

    def tracked_open(*args: Any, **kwargs: Any) -> Any:
        """Record GTiff source lifetimes while retaining real Rasterio objects."""
        assert kwargs.get("driver") == "GTiff"
        dataset = opened(*args, **kwargs)
        sources.append(dataset)
        return dataset

    class TrackedWarp:
        """Delegate warp reads while recording bounded requests and cleanup."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Acquire an ordinary warp whose lazy reads remain observable."""
            self.dataset = warped(*args, **kwargs)
            warps.append(self.dataset)

        def __enter__(self) -> TrackedWarp:
            """Return this tracked reader without loading pixels."""
            return self

        def __exit__(self, *args: Any) -> None:
            """Release the underlying real warp on every exit path."""
            self.dataset.close()

        def read(self, *args: Any, **kwargs: Any) -> Any:
            """Record each window and delegate an actual masked pixel read."""
            windows.append(kwargs["window"])
            return self.dataset.read(*args, **kwargs)

    monkeypatch.setattr(rasterio, "open", tracked_open)
    monkeypatch.setattr(module, "WarpedVRT", TrackedWarp)
    reader = LocalWindowReader(
        path, entry=local_entry, chip_size=17, batch_size=2, cache_mb=2
    )
    with reader:
        for batch in reader.batches():
            assert len(batch.chips) <= 2
    assert windows
    assert all(0 < window.width <= 17 and 0 < window.height <= 17 for window in windows)
    assert sources and all(dataset.closed for dataset in sources)
    assert warps and all(dataset.closed for dataset in warps)
