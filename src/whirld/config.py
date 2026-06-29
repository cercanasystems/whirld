"""Cache/storage layout and path resolution for Whirld.

All on-disk state lives under a single home directory (default ``~/.whirld``,
overridable via the ``WHIRLD_HOME`` environment variable), laid out per PRD
section 11::

    ~/.whirld/
      registry/
        models/<name>.yaml
        schema/model.schema.json
        last_updated
      models/
        <name>/<weights>
        <name>/manifest.json
      logs/
        whirld.log
        usage.jsonl

The :class:`WhirldPaths` object is the single source of truth for these
locations; nothing else in the codebase should hard-code ``~/.whirld``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

_ENV_HOME = "WHIRLD_HOME"
_ENV_LOG_LEVEL = "WHIRLD_LOG_LEVEL"
_ENV_STAC_TOKEN = "WHIRLD_STAC_TOKEN"


@dataclass(frozen=True)
class WhirldPaths:
    """Resolved filesystem locations for a Whirld home directory.

    Args:
        home: Root directory containing all Whirld state.
    """

    home: Path

    @property
    def registry_dir(self) -> Path:
        """Directory holding the synced registry (models + schema)."""
        return self.home / "registry"

    @property
    def registry_models_dir(self) -> Path:
        """Directory holding per-model registry YAML files."""
        return self.registry_dir / "models"

    @property
    def registry_schema_dir(self) -> Path:
        """Directory holding the registry JSON Schema."""
        return self.registry_dir / "schema"

    @property
    def registry_last_updated(self) -> Path:
        """Timestamp file recording the last registry refresh."""
        return self.registry_dir / "last_updated"

    @property
    def models_dir(self) -> Path:
        """Directory holding downloaded/materialized model weights."""
        return self.home / "models"

    @property
    def logs_dir(self) -> Path:
        """Directory holding application and usage logs."""
        return self.home / "logs"

    @property
    def app_log(self) -> Path:
        """Path to the rotating application log file."""
        return self.logs_dir / "whirld.log"

    @property
    def usage_log(self) -> Path:
        """Path to the structured local usage record (JSONL)."""
        return self.logs_dir / "usage.jsonl"

    def model_dir(self, name: str) -> Path:
        """Return the cache directory for a single model.

        Args:
            name: Machine-readable model identifier (e.g. ``clay-v1``).

        Returns:
            Path to ``<home>/models/<name>`` (not guaranteed to exist).
        """
        return self.models_dir / name

    def model_manifest(self, name: str) -> Path:
        """Return the ``manifest.json`` path for a cached model."""
        return self.model_dir(name) / "manifest.json"

    def ensure_dirs(self) -> None:
        """Create the home, models, and logs directories if absent.

        The registry directory is created on demand by the registry seeding
        logic, so it is intentionally not created here.
        """
        self.home.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def get_home() -> Path:
    """Resolve the Whirld home directory.

    Honors the ``WHIRLD_HOME`` environment variable; otherwise defaults to
    ``~/.whirld``. The returned path is expanded and absolute but not created.

    Returns:
        The resolved home directory path.
    """
    override = os.environ.get(_ENV_HOME)
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".whirld").resolve()


def get_paths() -> WhirldPaths:
    """Return a :class:`WhirldPaths` for the current environment.

    Returns:
        Path bundle rooted at :func:`get_home`.
    """
    return WhirldPaths(home=get_home())


def get_log_level() -> str | None:
    """Return the ``WHIRLD_LOG_LEVEL`` override, if set."""
    return os.environ.get(_ENV_LOG_LEVEL)


def get_stac_token() -> str | None:
    """Return the ``WHIRLD_STAC_TOKEN`` bearer token, if set.

    Used by the deferred STAC input path. Never logged or persisted.
    """
    return os.environ.get(_ENV_STAC_TOKEN)


def bundled_registry_dir() -> Path:
    """Return the path to the registry bundled inside the installed package.

    This is the seed copied into ``~/.whirld/registry`` on first use so that
    Whirld works fully offline out of the box.

    Returns:
        Path to the packaged ``registry_data`` directory.
    """
    return Path(str(resources.files("whirld") / "registry_data"))
