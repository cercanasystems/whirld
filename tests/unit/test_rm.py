"""Unit tests for model removal (core helpers)."""

from __future__ import annotations

from pathlib import Path

import pytest

from whirld.core.fetch import is_installed, pull, remove_all, remove_model
from whirld.errors import ModelNotInstalledError


def test_remove_model(whirld_home: Path) -> None:
    """remove_model deletes a pulled model's directory."""
    pull("clay-v1")
    assert is_installed("clay-v1")
    remove_model("clay-v1")
    assert not is_installed("clay-v1")


def test_remove_model_not_installed(whirld_home: Path) -> None:
    """Removing an un-installed model raises ModelNotInstalledError."""
    with pytest.raises(ModelNotInstalledError):
        remove_model("clay-v1")


def test_remove_all(whirld_home: Path) -> None:
    """remove_all clears every cached model and reports their names."""
    pull("clay-v1")
    removed = remove_all()
    assert removed == ["clay-v1"]
    assert not is_installed("clay-v1")


def test_remove_all_empty(whirld_home: Path) -> None:
    """remove_all on an empty cache returns an empty list."""
    assert remove_all() == []
