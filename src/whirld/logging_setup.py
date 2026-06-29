"""Logging configuration and the structured local usage record.

Two distinct sinks (PRD section 16):

1. **Application log** — human-readable diagnostics. DEBUG to a rotating file
   (``~/.whirld/logs/whirld.log``, 10 MB x 3) and INFO to stderr by default.
   Verbosity overridable by flags or ``WHIRLD_LOG_LEVEL``.
2. **Usage record** — one JSON object per inference run appended to
   ``~/.whirld/logs/usage.jsonl`` (1 MB x 2). Local-only, never transmitted, and
   deliberately free of file paths, CRS, geographic data, or any PII. On error it
   records only the exception *class name*, never the message or traceback.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import platform
import sys
from typing import Any

from . import config
from ._version import __version__

_APP_LOGGER_NAME = "whirld"
_USAGE_LOGGER_NAME = "whirld.usage"

_APP_LOG_MAX_BYTES = 10 * 1024 * 1024
_APP_LOG_BACKUPS = 3
_USAGE_LOG_MAX_BYTES = 1 * 1024 * 1024
_USAGE_LOG_BACKUPS = 2

_configured = False
_configured_home: str | None = None


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the root ``whirld`` logger.

    Args:
        name: Optional dotted suffix (e.g. ``"core.fetch"``). When ``None``,
            the root application logger is returned.

    Returns:
        A configured :class:`logging.Logger`.
    """
    if name is None:
        return logging.getLogger(_APP_LOGGER_NAME)
    return logging.getLogger(f"{_APP_LOGGER_NAME}.{name}")


def configure_logging(
    *,
    verbose: bool = False,
    quiet: bool = False,
    force: bool = False,
) -> None:
    """Configure application logging sinks.

    Idempotent unless ``force`` is set. Resolves the stderr level from (in
    precedence order): ``quiet`` flag, ``verbose`` flag, ``WHIRLD_LOG_LEVEL``
    environment variable, then the INFO default. The file handler always logs at
    DEBUG.

    Args:
        verbose: If true, emit DEBUG and above to stderr.
        quiet: If true, emit WARNING and above to stderr (wins over ``verbose``).
        force: Re-run configuration even if already configured.
    """
    global _configured, _configured_home
    paths = config.get_paths()
    # Rebind if the home changed (e.g. WHIRLD_HOME override) so logs never go to
    # a stale path; otherwise honor the idempotency guard.
    if _configured and not force and _configured_home == str(paths.home):
        return

    paths.ensure_dirs()

    logger = logging.getLogger(_APP_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        paths.app_log,
        maxBytes=_APP_LOG_MAX_BYTES,
        backupCount=_APP_LOG_BACKUPS,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(_resolve_stderr_level(verbose=verbose, quiet=quiet))
    stderr_handler.setFormatter(fmt)
    logger.addHandler(stderr_handler)

    _configured = True
    _configured_home = str(paths.home)


def _resolve_stderr_level(*, verbose: bool, quiet: bool) -> int:
    """Resolve the stderr handler level from flags and environment.

    Args:
        verbose: DEBUG when true.
        quiet: WARNING when true (takes precedence over ``verbose``).

    Returns:
        A :mod:`logging` level integer.
    """
    if quiet:
        return logging.WARNING
    if verbose:
        return logging.DEBUG
    env_level = config.get_log_level()
    if env_level:
        return logging.getLevelNamesMapping().get(env_level.upper(), logging.INFO)
    return logging.INFO


def _usage_logger() -> logging.Logger:
    """Return the dedicated rotating usage-record logger.

    The handler is (re)bound whenever the target ``usage.jsonl`` path changes, so
    records always land under the active ``WHIRLD_HOME`` rather than the path that
    happened to be active when the logger was first created.
    """
    paths = config.get_paths()
    paths.ensure_dirs()
    target = str(paths.usage_log.resolve())

    logger = logging.getLogger(_USAGE_LOGGER_NAME)
    if logger.handlers:
        existing = getattr(logger.handlers[0], "baseFilename", None)
        if existing == target:
            return logger
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.handlers.RotatingFileHandler(
        paths.usage_log,
        maxBytes=_USAGE_LOG_MAX_BYTES,
        backupCount=_USAGE_LOG_BACKUPS,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def record_usage(
    *,
    command: str,
    model: str,
    model_version: str,
    input_type: str,
    sensor_detected: str | None,
    chip_count: int | None,
    device: str,
    quantized: bool,
    duration_ms: int,
    error: str | None = None,
) -> dict[str, Any]:
    """Append a structured usage record to ``usage.jsonl``.

    Only non-sensitive fields are written: no file paths, no CRS, no geographic
    data, no user-identifiable information. On error, ``error`` must be the
    exception *class name* only.

    Args:
        command: Top-level command (``embed``, ``segment``, ``classify``).
        model: Model identifier.
        model_version: Resolved model version.
        input_type: ``geotiff`` or ``stac``.
        sensor_detected: Detected sensor key, or ``None`` if detection failed.
        chip_count: Number of chips processed, or ``None`` on early failure.
        device: Inference device (``cuda``, ``mps``, ``cpu``).
        quantized: Whether a quantized variant was used.
        duration_ms: Wall-clock duration in milliseconds.
        error: Exception class name on failure, else ``None``.

    Returns:
        The record dictionary that was written (useful for tests).
    """
    record: dict[str, Any] = {
        "timestamp": _utc_now_iso(),
        "command": command,
        "model": model,
        "model_version": model_version,
        "input_type": input_type,
        "sensor_detected": sensor_detected,
        "chip_count": chip_count,
        "device": device,
        "quantized": quantized,
        "duration_ms": duration_ms,
        "whirld_version": __version__,
        "python_version": platform.python_version(),
        "os": sys.platform,
        "error": error,
    }
    _usage_logger().info(json.dumps(record))
    return record


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with ``Z`` suffix."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
