"""Registry document construction owns parsing, schema, and model invariants."""

from pathlib import Path

import pytest
import yaml

from scripts.check_registry import check_entry
from whirld import config
from whirld.core.registry import ModelEntry, Registry
from whirld.errors import RegistryError


@pytest.mark.parametrize(
    "contents", [b"[", b"\xff", b"null", b"[one, two]", b"name: incomplete"]
)
def test_invalid_document_is_a_domain_error(tmp_path: Path, contents: bytes) -> None:
    """Malformed or schema-invalid documents share one caller-facing error type."""
    path = tmp_path / "entry.yaml"
    path.write_bytes(contents)
    with pytest.raises(RegistryError):
        ModelEntry.from_registry_file(path)


def test_missing_document_is_a_domain_error(tmp_path: Path) -> None:
    """File construction does not leak an OSError through the registry boundary."""
    with pytest.raises(RegistryError):
        ModelEntry.from_registry_file(tmp_path / "missing.yaml")


def test_document_constructor_matches_runtime_registry(whirld_home: Path) -> None:
    """The checker and runtime share the same typed, fully validated entry."""
    registry = Registry()
    path = registry.models_dir / "clay-v1.yaml"
    assert registry.get("clay-v1") == ModelEntry.from_registry_file(path)
    assert registry.paths == config.get_paths()


def test_checker_preserves_review_and_validate_only_verdicts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Schema-valid curated checkpoints still need review in governance mode."""
    path = config.bundled_registry_dir() / "models/clay-v1.5.yaml"
    assert check_entry(path) == 1
    assert "REVIEW" in capsys.readouterr().out
    assert check_entry(path, validate_only=True) == 0
    assert "VALID" in capsys.readouterr().out


def test_checker_preserves_safe_community_verdict(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Safe community entries remain eligible without relaxing model constraints."""
    path = config.bundled_registry_dir() / "models/clay-v1.5.yaml"
    document = yaml.safe_load(path.read_text())
    document["trust"] = "community"
    document["source"]["files"] = ["weights.safetensors"]
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(yaml.safe_dump(document))
    assert check_entry(candidate) == 0
    assert "AUTO-MERGE" in capsys.readouterr().out
