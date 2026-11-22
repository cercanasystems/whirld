"""Signed URLs remain private in actual GDAL header and lazy-read diagnostics."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

import pytest

from tests.integration.io.test_cog_input import CogScene, CogServer
from whirld.core.registry import Registry
from whirld.errors import InvalidInputError
from whirld.io.cog import CogWindowReader, read_remote_cog


def test_malformed_remote_tiff_redacts_debug_error(
    tmp_path: Path, whirld_home: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """GDAL's unsupported-format message contains a URL before scoped redaction."""
    scene = CogScene(tmp_path)
    scene.path.write_bytes(b"not a TIFF image" * 1024)
    with CogServer(scene) as server, caplog.at_level(logging.DEBUG):
        url = server.url + "?token=synthetic-malformed-query-secret"
        with pytest.raises(InvalidInputError) as caught:
            read_remote_cog(
                url, entry=Registry().get("clay-v1"), sensor="sentinel-2-l2a"
            )
    diagnostic = "".join(traceback.format_exception(caught.value)) + caplog.text
    assert "synthetic-malformed-query-secret" not in diagnostic
    assert "[redacted]" in caplog.text
    assert "GDAL signalled an error" in caplog.text


def test_lazy_read_lifetime_redacts_urls_and_bearer(
    tmp_path: Path, whirld_home: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The diagnostic guard remains active after header discovery through batches."""
    scene = CogScene(tmp_path, size=128)
    scene.create()
    token = "synthetic-bearer-credential"
    with CogServer(scene, token=token) as server, caplog.at_level(logging.DEBUG):
        url = server.url + "?sig=synthetic-lazy-query-secret"
        with CogWindowReader(
            url,
            entry=Registry().get("clay-v1"),
            sensor="sentinel-2-l2a",
            chip_size=32,
            batch_size=2,
            token=token,
        ) as reader:
            logger = logging.getLogger("rasterio._env")
            logger.debug("Retained dataset %s authorization Bearer %s", url, token)
            assert sum(len(batch.chips) for batch in reader.batches()) == 16
    assert "synthetic-lazy-query-secret" not in caplog.text
    assert token not in caplog.text
    assert "[redacted]" in caplog.text
