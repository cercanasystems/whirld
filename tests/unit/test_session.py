"""Unit tests for the warm ModelSession cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from whirld.core.fetch import pull
from whirld.core.session import LoadedModel, ModelSession
from whirld.errors import ModelNotFoundError, ModelNotInstalledError
from whirld.models.clay import ClayBackend


def test_get_loads_and_caches(whirld_home: Path) -> None:
    """get() returns a LoadedModel and caches the same backend instance."""
    pull("clay-v1")
    session = ModelSession(device="cpu")
    first = session.get("clay-v1")
    assert isinstance(first, LoadedModel)
    assert isinstance(first.backend, ClayBackend)
    second = session.get("clay-v1")
    assert first.backend is second.backend  # warm: same instance


def test_loaded_reflects_cache(whirld_home: Path) -> None:
    """`loaded` lists resident models; empty before first get()."""
    pull("clay-v1")
    session = ModelSession(device="cpu")
    assert session.loaded == []
    session.get("clay-v1")
    assert session.loaded == ["clay-v1"]


def test_preload(whirld_home: Path) -> None:
    """preload() eagerly loads the requested models."""
    pull("clay-v1")
    session = ModelSession(device="cpu")
    session.preload(["clay-v1"])
    assert "clay-v1" in session.loaded


def test_clear(whirld_home: Path) -> None:
    """clear() drops all resident models."""
    pull("clay-v1")
    session = ModelSession(device="cpu")
    session.get("clay-v1")
    session.clear()
    assert session.loaded == []


def test_get_unknown_model_raises(whirld_home: Path) -> None:
    """An unknown model surfaces ModelNotFoundError."""
    session = ModelSession(device="cpu")
    with pytest.raises(ModelNotFoundError):
        session.get("no-such-model")


def test_get_not_installed_raises(whirld_home: Path) -> None:
    """A known but un-pulled model surfaces ModelNotInstalledError."""
    session = ModelSession(device="cpu")
    with pytest.raises(ModelNotInstalledError):
        session.get("clay-v1")
