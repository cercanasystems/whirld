"""Resource-owned redaction for third-party raster diagnostics."""

from __future__ import annotations

import logging
import re
import threading
import traceback
from contextlib import ExitStack
from types import TracebackType

import rasterio


class RasterDiagnostics(logging.Filter):
    """Redact remote credentials in Rasterio messages from the owning thread.

    Rasterio logs GDAL errors before raising them, so sanitizing exception text
    alone cannot protect signed URLs. Each context owns its filters and restores
    only those filters on exit; logger levels, handlers and unrelated threads are
    untouched. Contexts can nest or overlap on different threads.

    Args:
        token: Optional bearer credential to remove from diagnostic text.
    """

    def __init__(self, token: str | None = None) -> None:
        """Bind optional credentials without modifying process logging state."""
        super().__init__()
        self._token = token
        self._thread: int | None = None
        self._loggers: list[logging.Logger] = []
        self._resources = ExitStack()

    def __enter__(self) -> RasterDiagnostics:
        """Attach owned filters and disable raw GDAL/curl debug output.

        Returns:
            This redactor, active until context exit. Rasterio's imported logger
            namespace is fixed before opening datasets; filter each logger because
            Python does not apply parent logger filters to propagated records.
        """
        self._thread = threading.get_ident()
        names = {
            "rasterio",
            "rasterio._env",
            "rasterio._io",
            "rasterio._base",
            "rasterio._warp",
            "rasterio._err",
        }
        names.update(
            name
            for name in logging.Logger.manager.loggerDict.copy()
            if name.startswith("rasterio.")
        )
        try:
            for name in names:
                logger = logging.getLogger(name)
                logger.addFilter(self)
                self._loggers.append(logger)
            self._resources.enter_context(
                rasterio.Env(CPL_DEBUG=False, CPL_CURL_VERBOSE=False)
            )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _redact(self, message: str) -> str:
        """Remove URL queries and a configured bearer token from one message."""
        message = re.sub(
            r"(https?://[^\s\"'<>?]+)\?[^\s\"'<>]*",
            r"\1?[redacted]",
            message,
        )
        if self._token:
            message = message.replace(self._token, "[redacted]")
        return message

    def filter(self, record: logging.LogRecord) -> bool:
        """Sanitize this thread's messages while retaining level and attribution.

        Args:
            record: Rasterio diagnostic about to reach configured handlers.

        Returns:
            True for every record; redaction changes message content only and
            does not suppress diagnostics or records from other threads.
        """
        if record.thread != self._thread:
            return True
        record.msg = self._redact(record.getMessage())
        record.args = ()
        if record.exc_info:
            record.exc_text = self._redact(
                "".join(traceback.format_exception(*record.exc_info))
            )
        if record.stack_info:
            record.stack_info = self._redact(record.stack_info)
        return True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore GDAL options and remove only this context's own filters."""
        try:
            self._resources.close()
        finally:
            for logger in self._loggers:
                logger.removeFilter(self)
            self._loggers.clear()
            self._thread = None
