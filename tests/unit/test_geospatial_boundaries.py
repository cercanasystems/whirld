"""Regression coverage for reviewed input, spatial, and validity boundaries."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from whirld.core.chips import ChipMeta, ChipSet
from whirld.core.contract import translate
from whirld.core.registry import Registry
from whirld.errors import InvalidInputError, UnsupportedSensorError
from whirld.io.output import build_feature_collection, mask_to_geotiff_bytes
from whirld.io.raster import RasterSource, read_raster, read_raster_from_bytes


def _source() -> RasterSource:
    """Return six constant labeled bands on a small north-up metric grid."""
    return RasterSource(
        data=np.full((6, 4, 4), 1000, dtype=np.float32),
        crs="EPSG:32630",
        transform=from_origin(500000, 1000, 10, 10),
        band_descriptions=["B02", "B03", "B04", "B08", "B11", "B12"],
    )


@pytest.mark.parametrize(
    "descriptions",
    [
        ["B03", "B04", "B08", "B11", "B12", "B01"],
        [None, "B03", "B04", "B08", "B11", "B12"],
        ["B02", "b02", "B04", "B08", "B11", "B12"],
    ],
)
def test_missing_and_ambiguous_named_bands_fail(
    whirld_home: Path, descriptions: list[str | None]
) -> None:
    """Labels cannot silently fall back to duplicate or incorrect band indices."""
    raster = _source()
    raster.band_descriptions = descriptions
    with pytest.raises(UnsupportedSensorError, match="Required band|Ambiguous"):
        translate(raster, Registry().get("clay-v1"), "sentinel-2-l2a")


def test_wholly_unlabeled_contract_order_remains_supported(whirld_home: Path) -> None:
    """The documented positional convention applies only to unlabeled rasters."""
    raster = _source()
    entry = Registry().get("clay-v1")
    labeled = translate(raster, entry, "sentinel-2-l2a")
    raster.band_descriptions = [None] * 6
    unlabeled = translate(raster, entry, "sentinel-2-l2a")
    np.testing.assert_array_equal(labeled.data, unlabeled.data)


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:2263"])
def test_nonmetric_coordinates_rejected(whirld_home: Path, crs: str) -> None:
    """Degrees and US survey feet must not become meter-sized pixels."""
    raster = _source()
    raster.crs = crs
    with pytest.raises(InvalidInputError, match="meter units"):
        translate(raster, Registry().get("clay-v1"), "sentinel-2-l2a")


@pytest.mark.parametrize(
    "transform",
    [rasterio.Affine(10, 1, 500000, 0, -10, 1000), from_origin(0, 0, 10, -10)],
)
def test_unsupported_affine_rejected(
    whirld_home: Path, transform: rasterio.Affine
) -> None:
    """Rotation and south-up grids fail explicitly rather than corrupting extent."""
    raster = _source()
    raster.transform = transform
    with pytest.raises(InvalidInputError, match="north-up"):
        translate(raster, Registry().get("clay-v1"), "sentinel-2-l2a")


def test_anisotropic_pixels_use_both_dimensions(whirld_home: Path) -> None:
    """A 40-by-80-meter footprint becomes 4-by-8 ten-meter cells."""
    raster = _source()
    raster.transform = from_origin(500000, 1000, 10, 20)
    output = translate(raster, Registry().get("clay-v1"), "sentinel-2-l2a")
    assert output.data.shape == (6, 8, 4)
    assert output.transform == from_origin(500000, 1000, 10, 10)
    assert rasterio.transform.array_bounds(8, 4, output.transform) == (
        500000,
        920,
        500040,
        1000,
    )


@pytest.mark.parametrize("resample", [False, True])
def test_local_upload_and_stac_validity_parity(
    whirld_home: Path, tmp_path: Path, resample: bool
) -> None:
    """Internal masks and nodata survive reading and cannot pollute interpolation."""
    from whirld.io.radiometry import BandRadiometry

    raster = _source()
    raster.data[0, 0, 0] = -9999
    raster.data[:, 3, 3] = 60000
    raster.transform = from_origin(
        500000, 1000, 20 if resample else 10, 20 if resample else 10
    )
    path = tmp_path / "masked.tif"
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=4,
            height=4,
            count=6,
            dtype="float32",
            crs=raster.crs,
            transform=raster.transform,
            nodata=-9999,
        ) as dst:
            dst.write(raster.data)
            dst.descriptions = tuple(raster.band_descriptions)
            mask = np.full((4, 4), 255, dtype=np.uint8)
            mask[3, 3] = 0
            dst.write_mask(mask)
    local = read_raster(path)
    upload = read_raster_from_bytes(path.read_bytes())
    assert np.isnan(local.data[0, 0, 0])
    assert np.isnan(local.data[:, 3, 3]).all()
    np.testing.assert_array_equal(local.data, upload.data)
    stac = _source()
    stac.data = local.data.copy()
    stac.transform = local.transform
    stac.band_radiometry = [BandRadiometry() for _ in range(6)]
    entry = Registry().get("clay-v1")
    outputs = [
        translate(source, entry, "sentinel-2-l2a") for source in (local, upload, stac)
    ]
    expected = (
        1000 * entry.band_contract.normalization.scale
        - entry.band_contract.normalization.mean[0]
    ) / entry.band_contract.normalization.std[0]
    first = outputs[0]
    assert first.valid_mask is not None and not first.valid_mask[0, 0]
    assert first.data[0, 0, 0] == entry.band_contract.nodata_fill
    valid = first.data[0] != entry.band_contract.nodata_fill
    np.testing.assert_allclose(first.data[0][valid], expected, atol=1e-5)
    for output in outputs[1:]:
        np.testing.assert_array_equal(first.data, output.data)
        np.testing.assert_array_equal(first.valid_mask, output.valid_mask)


@pytest.mark.parametrize(
    "source", ["/private/server-data.tif", "http://127.0.0.1:9/private.tif"]
)
def test_uploaded_vrt_rejected_before_external_read(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    """A .tif label cannot authorize a VRT containing server-side references."""
    import whirld.io.raster as reader

    def unexpected_extract(*args: object, **kwargs: object) -> None:
        pytest.fail("Untrusted VRT was opened and reached pixel extraction")

    monkeypatch.setattr(reader, "_extract", unexpected_extract)
    vrt = f"""<VRTDataset rasterXSize="1" rasterYSize="1">
      <SRS>EPSG:32630</SRS><GeoTransform>0,10,0,10,0,-10</GeoTransform>
      <VRTRasterBand dataType="Byte" band="1"><SimpleSource>
      <SourceFilename>{source}</SourceFilename><SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="1" ySize="1"/>
      <DstRect xOff="0" yOff="0" xSize="1" ySize="1"/>
      </SimpleSource></VRTRasterBand></VRTDataset>""".encode()
    with pytest.raises(InvalidInputError, match="valid GeoTIFF"):
        read_raster_from_bytes(vrt, label="imagery.tif")


def test_geojson_uses_known_utm_central_meridian() -> None:
    """UTM zone 30's equatorial false easting represents longitude -3, latitude 0."""
    chips = ChipSet(
        chips=[ChipMeta(index=0, row=0, col=0, bbox=[500000, 0, 500010, 10])],
        chip_size_px=1,
        n_rows=1,
        n_cols=1,
    )
    result = build_feature_collection(
        np.array([[1.0]]),
        chips,
        queries=["forest"],
        model="remoteclip",
        model_version="1",
        crs="EPSG:32630",
    )
    ring = result["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == pytest.approx((-3.0, 0.0), abs=1e-8)
    assert ring[1][0] > -3.0 and ring[2][1] > 0
    assert "crs" not in result


def test_mask_zero_is_valid_and_255_is_missing() -> None:
    """Ordinary masked raster reads distinguish valid background from missing data."""
    mask = np.array([[0, 1], [255, 0]], dtype=np.uint8)
    payload = mask_to_geotiff_bytes(
        mask, from_origin(500000, 1000, 10, 10), "EPSG:32630"
    )
    with rasterio.MemoryFile(payload) as memfile, memfile.open() as dataset:
        result = dataset.read(1, masked=True)
        assert result.count() == 3
        np.testing.assert_array_equal(result.mask, [[False, False], [True, False]])
