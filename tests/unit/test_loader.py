"""Unit tests for registry-driven backend selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from whirld.core.fetch import Manifest
from whirld.core.registry import Registry
from whirld.errors import WhirldError
from whirld.models.loader import load_backend


def test_every_bundled_model_declares_a_backend(whirld_home: Path) -> None:
    """Adding a model is data: every registry entry names its backend."""
    registry = Registry()
    for name in registry.available():
        entry = registry.get(name)
        assert entry.backend, f"{name} declares no backend"


def test_reference_backend_resolves_from_entry(whirld_home: Path) -> None:
    """clay-v1 (backend: clay-reference) loads its numpy reference backend."""
    from whirld.core.fetch import pull
    from whirld.models.clay import ClayBackend

    manifest = pull("clay-v1")
    entry = Registry().get("clay-v1")
    assert entry.backend == "clay-reference"
    backend = load_backend(entry, manifest, device="cpu")
    assert isinstance(backend, ClayBackend)


def test_unknown_backend_errors(whirld_home: Path) -> None:
    """An entry naming an unknown backend raises a clear error."""
    entry = Registry().get("clay-v1").model_copy(update={"backend": "no-such"})
    dummy = Manifest(
        name="clay-v1",
        version="1.0.0",
        source_type="reference",
        sha256="0" * 64,
        weights_file="w",
    )
    with pytest.raises(WhirldError) as excinfo:
        load_backend(entry, dummy)
    assert "no-such" in excinfo.value.message


def test_missing_backend_errors(whirld_home: Path) -> None:
    """An entry with an empty backend raises a clear error."""
    entry = Registry().get("clay-v1").model_copy(update={"backend": ""})
    dummy = Manifest(
        name="clay-v1",
        version="1.0.0",
        source_type="reference",
        sha256="0" * 64,
        weights_file="w",
    )
    with pytest.raises(WhirldError):
        load_backend(entry, dummy)


def test_backend_dispatch_is_a_closed_allowlist(
    whirld_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dynamic/module-path backend id is rejected — YAML never names code to import.

    Security invariant: ``load_backend`` dispatches by an allowlisted id to a hardcoded
    branch; it must not import a module path supplied in the registry. We assert both
    rejection and that no import was attempted for the malicious id.
    """
    import importlib

    dummy = Manifest(
        name="clay-v1",
        version="1.0.0",
        source_type="reference",
        sha256="0" * 64,
        weights_file="w",
    )
    imported: list[str] = []
    real_import = importlib.import_module

    def _spy(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _spy)

    for evil in ("whirld.models.clay", "os.system", "subprocess", "../etc/passwd"):
        entry = Registry().get("clay-v1").model_copy(update={"backend": evil})
        with pytest.raises(WhirldError):
            load_backend(entry, dummy)
    # No attacker-controlled module path was imported.
    assert not any(m in ("os.system", "subprocess", "../etc/passwd") for m in imported)
