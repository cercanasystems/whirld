"""Verified public evaluation artifacts with bounded downloads and atomic caching."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from ..core.fetch import sha256_file
from .errors import EvaluationError
from .schemas import AssetPin


class EvaluationAssetCache:
    """Own content-addressed evaluation files; never trust a cache hit by name alone."""

    def __init__(self, directory: Path, offline: bool = False) -> None:
        """Bind a cache location and network policy without touching the filesystem."""
        self.directory = directory.resolve()
        self.offline = offline

    def path(self, pin: AssetPin) -> Path:
        """Return a filename derived solely from the validated expected SHA-256."""
        return self.directory / pin.sha256

    def fetch(self, pin: AssetPin) -> Path:
        """Verify an existing file or atomically cache an exact pinned download.

        Args:
            pin: Public URL, SHA-256 and exact positive byte length.

        Returns:
            An absolute local path whose bytes match both expected checks.

        Raises:
            EvaluationError: Cached/downloaded data is corrupt, unavailable or
                missing in offline mode. A corrupt hit is never silently reused.
            OSError: Local cache storage cannot be created or written.
        """
        target = self.path(pin)
        if any(path != target for path in self.directory.glob(pin.sha256 + "*")):
            raise EvaluationError(
                "Pinned evaluation files must be standalone; unpinned sidecar found."
            )
        if target.exists():
            if (
                target.is_file()
                and target.stat().st_size == pin.size_bytes
                and sha256_file(target) == pin.sha256
            ):
                return target
            raise EvaluationError("Cached evaluation artifact differs from its pin.")
        if self.offline:
            raise EvaluationError("Evaluation artifact missing from offline cache.")
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with (
                urlopen(pin.url, timeout=60) as response,
                tempfile.NamedTemporaryFile(
                    dir=self.directory, prefix=".download-", delete=False
                ) as output,
            ):
                temporary = Path(output.name)
                received = 0
                digest = hashlib.sha256()
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > pin.size_bytes:
                        raise EvaluationError(
                            "Evaluation download exceeds pinned size."
                        )
                    digest.update(block)
                    output.write(block)
            if received != pin.size_bytes or digest.hexdigest() != pin.sha256:
                raise EvaluationError("Evaluation download size or SHA-256 differs.")
            os.replace(temporary, target)
            return target
        except (URLError, TimeoutError) as exc:
            raise EvaluationError("Evaluation artifact download failed.") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
