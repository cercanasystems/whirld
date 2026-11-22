"""Scoped raster credentials clear file-based headers without GDAL read warnings."""

import logging
from pathlib import Path

import pytest
import rasterio
from rasterio.env import get_gdal_config
from rasterio.windows import Window

from tests.integration.io.test_cog_input import CogScene, CogServer
from whirld.core.registry import Registry
from whirld.io.cog import CogWindowReader


def test_header_file_is_cleared_without_empty_filename_warning(
    tmp_path: Path,
    whirld_home: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Actual HTTP requests prove inherited headers are usable, then safely blocked."""
    scene = CogScene(tmp_path / "scene", size=128)
    scene.create()
    headers = tmp_path / "inherited.headers"
    headers.write_text("Authorization: Bearer synthetic-inherited-secret\n")
    with (
        CogServer(scene) as server,
        rasterio.Env(
            GDAL_HTTP_HEADERS="",
            GDAL_HTTP_HEADER_FILE=str(headers),
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        ),
    ):
        # Prove that this GDAL build would actually forward the inherited file.
        with rasterio.open("/vsicurl/" + server.url + "?control=header-file") as source:
            assert source.read(1, window=Window(0, 0, 1, 1)).shape == (1, 1)
        assert any(
            record.authorization == "Bearer synthetic-inherited-secret"
            for record in server.records
        )
        server.records.clear()
        with caplog.at_level(logging.DEBUG):
            with CogWindowReader(
                server.url + "?policy=empty-headers",
                entry=Registry().get("clay-v1"),
                sensor="sentinel-2-l2a",
                chip_size=32,
                batch_size=2,
            ) as reader:
                assert get_gdal_config("GDAL_HTTP_HEADER_FILE") == str(headers)
                assert sum(len(batch.chips) for batch in reader.batches()) == 16
        assert get_gdal_config("GDAL_HTTP_HEADER_FILE") == str(headers)
        assert get_gdal_config("GDAL_HTTP_HEADERS") == ""
        assert server.records
        assert all(record.authorization is None for record in server.records)
    assert not any("Cannot read" in record.getMessage() for record in caplog.records)
