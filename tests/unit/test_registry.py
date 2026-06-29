"""Unit tests for registry loading, validation, and seeding."""

from __future__ import annotations

from pathlib import Path

import pytest

from whirld.core.registry import Registry
from whirld.errors import ModelNotFoundError, RegistryError


def test_seed_and_available(whirld_home: Path) -> None:
    """A fresh registry seeds the bundled models and lists them."""
    registry = Registry()
    assert "clay-v1" in registry.available()
    assert (whirld_home / "registry" / "models" / "clay-v1.yaml").exists()


def test_get_clay_entry(whirld_home: Path) -> None:
    """The clay-v1 entry parses into a typed model with expected fields."""
    entry = Registry().get("clay-v1")
    assert entry.name == "clay-v1"
    assert entry.category == "embedding"
    assert entry.output.embed_dim == 512
    assert "sentinel-2-l2a" in entry.supported_sensors()
    # Normalization arrays align with the six-band sensors.
    assert len(entry.band_contract.normalization.mean) == 6
    assert len(entry.band_contract.normalization.std) == 6


def test_get_unknown_model_raises(whirld_home: Path) -> None:
    """An unknown model raises ModelNotFoundError (exit code 2)."""
    with pytest.raises(ModelNotFoundError) as excinfo:
        Registry().get("does-not-exist")
    assert excinfo.value.exit_code == 2
    assert "Available models" in excinfo.value.message


def test_malformed_yaml_raises(whirld_home: Path) -> None:
    """A malformed registry YAML raises RegistryError."""
    registry = Registry()
    bad = registry.models_dir / "broken.yaml"
    bad.write_text("name: broken\nsource: [unterminated", encoding="utf-8")
    with pytest.raises(RegistryError):
        registry.get("broken")


def test_invalid_schema_raises(whirld_home: Path) -> None:
    """A schema-invalid entry (missing required fields) raises RegistryError."""
    registry = Registry()
    bad = registry.models_dir / "invalid.yaml"
    bad.write_text("name: invalid\ndisplay_name: x\n", encoding="utf-8")
    with pytest.raises(RegistryError):
        registry.get("invalid")
