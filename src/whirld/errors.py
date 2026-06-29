"""Domain-specific exception hierarchy and process exit-code mapping.

Whirld raises typed exceptions throughout the library layer; the CLI layer
catches :class:`WhirldError` and converts it to the documented process exit
code (PRD section 13). Every exception carries an actionable, human-readable
message so the CLI can print it verbatim.

Exit codes (PRD section 13):

======  ==========================================
 Code    Meaning
======  ==========================================
 0       Success
 1       General error
 2       Model not found in registry
 3       Model not installed (needs ``whirld pull``)
 4       Unsupported sensor for model
 5       Network error (download failed)
 6       Checksum mismatch
 7       Invalid input file
 8       Insufficient memory
 9       Security policy violation (Whirld extension)
======  ==========================================
"""

from __future__ import annotations


class WhirldError(Exception):
    """Base class for all Whirld errors.

    Attributes:
        exit_code: Process exit code the CLI should return for this error.

    Args:
        message: Actionable, user-facing description of what went wrong, why,
            and what to do about it.
    """

    exit_code: int = 1

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ModelNotFoundError(WhirldError):
    """Requested model has no entry in the registry."""

    exit_code = 2


class ModelNotInstalledError(WhirldError):
    """Requested model is in the registry but not pulled to the local cache."""

    exit_code = 3


class UnsupportedSensorError(WhirldError):
    """Detected (or supplied) sensor is not declared in the model's band contract."""

    exit_code = 4


class NetworkError(WhirldError):
    """A network operation (registry fetch or weight download) failed."""

    exit_code = 5


class ChecksumMismatchError(WhirldError):
    """A downloaded artifact's sha256 did not match the registry-declared value."""

    exit_code = 6


class InvalidInputError(WhirldError):
    """Input raster is missing, malformed, or lacks a required CRS."""

    exit_code = 7


class InsufficientMemoryError(WhirldError):
    """Inference could not proceed because of insufficient memory."""

    exit_code = 8


class SecurityError(WhirldError):
    """A model violates Whirld's security policy (e.g. untrusted pickle weights).

    Exit code 9 — a Whirld extension beyond the PRD's documented 1–8.
    """

    exit_code = 9


class RegistryError(WhirldError):
    """Registry data is missing or fails schema validation (general error)."""

    exit_code = 1
