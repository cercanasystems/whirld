"""Unit tests for the STAC item reader (``whirld.io.stac``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whirld.core.registry import Registry
from whirld.errors import InvalidInputError, UnsupportedSensorError
from whirld.io import stac


def _entry():
    """The clay-v1 reference entry (S2/Landsat band contract)."""
    return Registry().get("clay-v1")


def test_reads_item_keyed_by_native_band(whirld_home: Path, stac_item: Path) -> None:
    """A STAC item with assets keyed by native band id assembles a 6-band raster."""
    raster = stac.read_stac_item(str(stac_item), entry=_entry())
    assert raster.band_count == 6
    assert raster.band_descriptions == ["B02", "B03", "B04", "B08", "B11", "B12"]
    assert raster.crs == "EPSG:32630"
    # The item's datetime is carried into tags for metadata-aware models.
    assert raster.tags["TIFFTAG_DATETIME"] == "2024-06-01T10:00:00Z"


def test_reads_item_keyed_by_common_name(
    whirld_home: Path, stac_item_common: Path
) -> None:
    """Assets keyed by spectral common name + eo:bands resolve via the alias tier."""
    raster = stac.read_stac_item(str(stac_item_common), entry=_entry())
    assert raster.band_count == 6
    assert raster.band_descriptions == ["B02", "B03", "B04", "B08", "B11", "B12"]


def test_sensor_inferred_from_item_properties(
    whirld_home: Path, stac_item: Path
) -> None:
    """The sensor is inferred from the item's platform/collection when not given."""
    item = json.loads(Path(stac_item).read_text())
    sensor = stac._resolve_sensor(item, _entry(), None)
    assert sensor == "sentinel-2-l2a"


def test_sensor_override_validated(whirld_home: Path, stac_item: Path) -> None:
    """An unsupported --sensor override is rejected with the supported list."""
    item = json.loads(Path(stac_item).read_text())
    with pytest.raises(UnsupportedSensorError) as exc:
        stac._resolve_sensor(item, _entry(), "naip")
    assert "supports" in exc.value.message


def test_sensor_uninferable_asks_for_override(whirld_home: Path) -> None:
    """An item with no platform hints raises asking the user to pass --sensor."""
    item = {"type": "Feature", "id": "x", "properties": {}, "assets": {}}
    with pytest.raises(UnsupportedSensorError) as exc:
        stac._resolve_sensor(item, _entry(), None)
    assert "--sensor" in exc.value.message


def test_missing_asset_lists_available(whirld_home: Path, tmp_path: Path) -> None:
    """A required band with no matching asset errors and lists what is available."""
    assets = {"B02": {"href": "file:///x/B02.tif"}}  # missing the other 5 bands
    item = {"type": "Feature", "id": "partial", "assets": assets}
    with pytest.raises(UnsupportedSensorError) as exc:
        stac._match_asset(assets, "B03", "green", item)
    assert "B02" in exc.value.message  # available assets are listed


def test_bearer_token_set_on_fetch(
    whirld_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token produces an Authorization: Bearer header on the item fetch."""
    captured: dict[str, str] = {}

    class _Resp:
        def read(self):  # type: ignore[no-untyped-def]
            return json.dumps({"type": "Feature", "id": "x", "assets": {}}).encode(
                "utf-8"
            )

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *a):  # type: ignore[no-untyped-def]
            return False

    def _fake_urlopen(request, timeout=None):  # type: ignore[no-untyped-def]
        captured.update(request.headers)
        return _Resp()

    class _Opener:
        open = staticmethod(_fake_urlopen)

    monkeypatch.setattr(stac, "build_opener", lambda *_: _Opener())
    stac._fetch_item("https://example.com/item.json", "secret-token")
    # urllib capitalizes header keys.
    assert captured.get("Authorization") == "Bearer secret-token"


def test_non_feature_json_rejected(whirld_home: Path, tmp_path: Path) -> None:
    """A JSON document that is not a STAC Feature is rejected."""
    bad = tmp_path / "not_item.json"
    bad.write_text(json.dumps({"type": "Collection"}))
    with pytest.raises(InvalidInputError):
        stac._fetch_item(str(bad), None)


def test_gdal_path_translations() -> None:
    """Asset hrefs map to the right GDAL virtual-filesystem paths."""
    assert stac._gdal_path("https://h/x.tif") == "/vsicurl/https://h/x.tif"
    assert stac._gdal_path("s3://bucket/x.tif") == "/vsis3/bucket/x.tif"
    assert stac._gdal_path("file:///tmp/x.tif") == "/tmp/x.tif"
    assert stac._gdal_path("/tmp/x.tif") == "/tmp/x.tif"
    assert stac._gdal_path("file:///tmp/space%20name.tif") == "/tmp/space name.tif"
    assert stac._gdal_path("/tmp/literal%20name.tif") == "/tmp/literal%20name.tif"


def test_file_item_url_decodes_spaces(stac_item: Path, tmp_path: Path) -> None:
    """Percent-encoded file URLs identify the same local JSON as a normal path."""
    path = tmp_path / "item with spaces.json"
    path.write_bytes(stac_item.read_bytes())
    assert stac._fetch_item(path.as_uri(), None) == stac._fetch_item(str(path), None)


def test_stac_grid_preserves_partial_edge_extent() -> None:
    """A partial final cell is retained, rather than truncating the source edge."""
    from types import SimpleNamespace

    from rasterio.transform import from_origin

    ref = SimpleNamespace(bounds=(0, 0, 101, 101), transform=from_origin(0, 101, 1, 1))
    grid = stac._output_grid(ref, "EPSG:32630", 10, None)
    assert (grid.width, grid.height) == (11, 11)


def test_stac_geographic_grid_rejected_before_resampling() -> None:
    """An angular reference grid cannot be interpreted as a meter-resolution grid."""
    from types import SimpleNamespace

    from rasterio.transform import from_origin

    ref = SimpleNamespace(
        bounds=(-1, 50, -0.99, 50.01), transform=from_origin(-1, 50.01, 0.0001, 0.0001)
    )
    with pytest.raises(InvalidInputError):
        stac._output_grid(ref, "EPSG:4326", 10, None)
