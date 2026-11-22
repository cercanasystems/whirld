"""Raster diagnostic redaction preserves logging ownership and thread isolation."""

from __future__ import annotations

import logging
from threading import Thread

import pytest
import rasterio
from rasterio.env import get_gdal_config

from whirld.io.diagnostics import RasterDiagnostics
from whirld.io.radiometry import RasterBand


def test_nested_failure_restores_only_owned_filters(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nested readers restore their exact logger/GDAL state even after failure."""
    logger = logging.getLogger("rasterio._env")
    with caplog.at_level(logging.DEBUG):
        filters = list(logger.filters)
        handlers = list(logger.handlers)
        level = logger.level
        with rasterio.Env(CPL_DEBUG=True, CPL_CURL_VERBOSE=True):
            with RasterDiagnostics("outer-credential"):
                assert get_gdal_config("CPL_DEBUG") is False
                assert get_gdal_config("CPL_CURL_VERBOSE") is False
                with pytest.raises(RuntimeError):
                    with RasterDiagnostics("inner-credential"):
                        logger.debug(
                            "Bearer %s %s", "inner-credential", "outer-credential"
                        )
                        raise RuntimeError("consumer")
                logger.debug("Bearer %s", "outer-credential")
            assert get_gdal_config("CPL_DEBUG") is True
            assert get_gdal_config("CPL_CURL_VERBOSE") is True
        assert logger.filters == filters
        assert logger.handlers == handlers
        assert logger.level == level
    assert "inner-credential" not in caplog.text
    assert "outer-credential" not in caplog.text
    assert "[redacted]" in caplog.text


def test_redaction_does_not_modify_other_threads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A reader's diagnostics policy cannot change concurrent unrelated records."""
    logger = logging.getLogger("rasterio._env")

    def emit_foreign() -> None:
        """Emit a record from a thread that does not own the active reader."""
        logger.debug("https://example.invalid/other?value=foreign-visible")

    with caplog.at_level(logging.DEBUG):
        with RasterDiagnostics():
            logger.debug("https://example.invalid/scene?value=owner-secret")
            worker = Thread(target=emit_foreign)
            worker.start()
            worker.join(timeout=5)
            assert not worker.is_alive()
    assert "owner-secret" not in caplog.text
    assert "foreign-visible" in caplog.text


def test_redaction_removes_signed_urls_from_exception_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Third-party exception tracebacks cannot reintroduce a redacted query."""
    logger = logging.getLogger("rasterio._env")
    with caplog.at_level(logging.DEBUG), RasterDiagnostics():
        try:
            raise RuntimeError("https://example.invalid/scene?sig=exception-secret")
        except RuntimeError:
            logger.debug("Driver failure", exc_info=True)
    assert "exception-secret" not in caplog.text
    assert "Driver failure" in caplog.text
    assert "[redacted]" in caplog.text


def test_band_representation_hides_signed_url() -> None:
    """Typed scene reprs cannot accidentally expose nested source credentials."""
    band = RasterBand(href="https://example.invalid/scene?sig=repr-secret")
    assert "repr-secret" not in repr(band)
