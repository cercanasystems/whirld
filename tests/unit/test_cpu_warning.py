"""Unit tests for the CPU full-precision runtime warning (PRD section 12.2)."""

from __future__ import annotations

import logging

import pytest

from whirld.api import _maybe_warn_cpu
from whirld.core.fetch import Manifest


def _manifest(source_type: str = "huggingface", quantized: bool = False) -> Manifest:
    return Manifest(
        name="clay-v1.5",
        version="1.5.0",
        source_type=source_type,
        sha256="0" * 64,
        weights_file="w.ckpt",
        quantized=quantized,
    )


def test_warns_for_real_model_on_cpu(caplog: pytest.LogCaptureFixture) -> None:
    """A real full-precision model on CPU triggers the warning with an estimate."""
    with caplog.at_level(logging.WARNING, logger="whirld.api"):
        _maybe_warn_cpu("clay-v1.5", _manifest(), "cpu", 24, no_warnings=False)
    assert any("full-precision on CPU" in r.message for r in caplog.records)
    assert any("--device" in r.message for r in caplog.records)


def test_silent_with_no_warnings(caplog: pytest.LogCaptureFixture) -> None:
    """--no-warnings suppresses it."""
    with caplog.at_level(logging.WARNING, logger="whirld.api"):
        _maybe_warn_cpu("clay-v1.5", _manifest(), "cpu", 24, no_warnings=True)
    assert not caplog.records


def test_silent_for_reference_backend(caplog: pytest.LogCaptureFixture) -> None:
    """The numpy reference backend (source=reference) never warns."""
    with caplog.at_level(logging.WARNING, logger="whirld.api"):
        _maybe_warn_cpu("clay-v1", _manifest("reference"), "cpu", 24, no_warnings=False)
    assert not caplog.records


def test_silent_on_non_cpu(caplog: pytest.LogCaptureFixture) -> None:
    """No warning when running on a GPU/MPS device."""
    with caplog.at_level(logging.WARNING, logger="whirld.api"):
        _maybe_warn_cpu("clay-v1.5", _manifest(), "mps", 24, no_warnings=False)
    assert not caplog.records


def test_silent_for_quantized(caplog: pytest.LogCaptureFixture) -> None:
    """Quantized variants are exempt from the full-precision warning."""
    with caplog.at_level(logging.WARNING, logger="whirld.api"):
        _maybe_warn_cpu(
            "clay-v1.5", _manifest(quantized=True), "cpu", 24, no_warnings=False
        )
    assert not caplog.records
