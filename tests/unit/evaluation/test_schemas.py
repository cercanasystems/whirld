"""Strict frozen protocols and spatial partition constraints for evaluation."""

from __future__ import annotations

import copy
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from whirld.evaluation.schemas import (
    AssetPin,
    ConfusionCounts,
    EvaluationCase,
    EvaluationOptions,
    EvaluationSuite,
    MetricSummary,
    SpatialBounds,
)


@pytest.fixture
def suite_payload() -> dict[str, Any]:
    """Return independent JSON fields from the packaged source-pinned suite."""
    return json.loads(
        files("whirld.evaluation").joinpath("data/hls-burn-scars-mini.json").read_text()
    )


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/image.tif",
        "ftp://example.invalid/image.tif",
        "/tmp/image.tif",
        "https:///missing-host.tif",
        "https://user:secret@example.invalid/image.tif",
        "https://@example.invalid/image.tif",
        "https://:@example.invalid/image.tif",
        "https://example.invalid:broken/image.tif",
        "https://[broken/image.tif",
        "https://example.invalid/image.tif#fragment",
    ],
)
def test_asset_urls_reject_ambiguous_or_private_identity(url: str) -> None:
    """Pins accept public HTTP locations without credentials or fragments."""
    with pytest.raises(ValidationError):
        AssetPin(url=url, sha256="0" * 64, size_bytes=1)


def test_asset_url_preserves_signed_query_without_mutation() -> None:
    """Legitimate query-bearing public locations keep their exact serialized URL."""
    url = "https://example.invalid/pinned/image.tif?download=true&signature=abc"
    pin = AssetPin(url=url, sha256="0" * 64, size_bytes=1)
    assert pin.url == url
    with pytest.raises(ValidationError):
        pin.size_bytes = 2


@pytest.mark.parametrize("digest", ["0" * 63, "A" * 64, "g" * 64, "sha256:abc"])
def test_digest_requires_exact_lowercase_sha256(digest: str) -> None:
    """Human-readable aliases or malformed digests cannot identify cached bytes."""
    with pytest.raises(ValidationError):
        AssetPin(url="https://example.invalid/file", sha256=digest, size_bytes=1)


@pytest.mark.parametrize("size", [True, 0, -1, 1.5, "12"])
def test_asset_sizes_require_strict_positive_integers(size: Any) -> None:
    """Download limits cannot accept coerced booleans, strings or fractions."""
    with pytest.raises(ValidationError):
        AssetPin(url="https://example.invalid/file", sha256="0" * 64, size_bytes=size)


@pytest.mark.parametrize("field", list(ConfusionCounts.model_fields))
@pytest.mark.parametrize("value", [-1, True, 0.5, "2"])
def test_confusion_counts_reject_lossy_or_negative_values(
    field: str, value: Any
) -> None:
    """Every denominator contribution retains exact nonnegative integer semantics."""
    with pytest.raises(ValidationError):
        ConfusionCounts.model_validate({field: value})


def test_extra_fields_and_nonfinite_ratios_fail() -> None:
    """Reports cannot silently discard unknown fields or serialize NaN/Infinity."""
    with pytest.raises(ValidationError):
        ConfusionCounts.model_validate({"unexpected": 1})
    for invalid in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValidationError):
            MetricSummary.model_validate(
                MetricSummary.from_counts(ConfusionCounts()).model_dump()
                | {"precision": invalid}
            )


@pytest.mark.parametrize(
    "options",
    [
        {"threshold": 0.49},
        {"threshold": 0.51},
        {"threshold": float("nan")},
        {"threshold": True},
        {"threshold": "0.5"},
        {"batch_size": 0},
        {"batch_size": True},
        {"cache_mb": -1},
        {"threads": 1.5},
        {"split": "train"},
        {"unknown": True},
    ],
)
def test_fixed_evaluation_options_reject_tuning_or_ambiguous_settings(
    tmp_path: Path, options: dict[str, Any]
) -> None:
    """The test protocol cannot tune threshold or coerce its bounded CPU settings."""
    with pytest.raises(ValidationError):
        EvaluationOptions.model_validate(
            {"output_dir": tmp_path / "output", "cache_dir": tmp_path / "cache"}
            | options
        )


def test_evaluation_defaults_use_validation_and_fixed_threshold(tmp_path: Path) -> None:
    """Held-out test cases require an explicit split selection and never tune."""
    options = EvaluationOptions(
        output_dir=tmp_path / "out", cache_dir=tmp_path / "cache"
    )
    assert options.split == "validation" and options.threshold == 0.5
    assert options.batch_size == 1 and options.cache_mb == 8 and options.threads == 2
    assert options.offline is False
    with pytest.raises(ValidationError):
        options.threshold = 0.6


@pytest.mark.parametrize(
    "change",
    [
        "duplicate-id",
        "duplicate-source",
        "duplicate-image",
        "duplicate-label",
        "same-group",
        "footprint-overlap",
        "one-split",
        "duplicate-split-pin",
        "bands",
    ],
)
def test_suite_rejects_identity_or_spatial_partition_conflicts(
    suite_payload: dict[str, Any], change: str
) -> None:
    """Validation/test separation operates on groups and footprints, not tile IDs."""
    payload = copy.deepcopy(suite_payload)
    first, second, third = payload["cases"][:3]
    if change == "duplicate-id":
        second["id"] = first["id"]
    elif change == "duplicate-source":
        second["source_name"] = first["source_name"]
    elif change == "duplicate-image":
        second["image"]["sha256"] = first["image"]["sha256"]
    elif change == "duplicate-label":
        second["label"]["sha256"] = first["label"]["sha256"]
    elif change == "same-group":
        third["spatial_group"] = first["spatial_group"]
    elif change == "footprint-overlap":
        third["bounds"] = first["bounds"]
    elif change == "one-split":
        payload["cases"] = payload["cases"][:2]
    elif change == "duplicate-split-pin":
        payload["split_pins"][0]["split"] = "validation"
    else:
        payload["band_names"] = list(reversed(payload["band_names"]))
    with pytest.raises(ValidationError):
        EvaluationSuite.model_validate(payload)


@pytest.mark.parametrize("identity", ["../escape", "/absolute", "contains space", ""])
def test_case_identifiers_cannot_escape_owned_outputs(
    suite_payload: dict[str, Any], identity: str
) -> None:
    """Only bounded plain identifiers become output artifact filenames."""
    case = suite_payload["cases"][0] | {"id": identity}
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(case)


@pytest.mark.parametrize(
    "bounds",
    [
        {"west": 2, "east": 1, "south": 0, "north": 1},
        {"west": 0, "east": 1, "south": 1, "north": 1},
        {"west": -181, "east": 1, "south": 0, "north": 1},
        {"west": 0, "east": 1, "south": 0, "north": float("nan")},
    ],
)
def test_bounds_require_finite_positive_geographic_extents(
    bounds: dict[str, Any],
) -> None:
    """Reversed, empty, out-of-range or undefined footprint records fail."""
    with pytest.raises(ValidationError):
        SpatialBounds.model_validate(bounds)


def test_touching_footprint_edges_do_not_overlap() -> None:
    """Partition checks reject shared area while allowing adjacent extents."""
    first = SpatialBounds(west=0, east=1, south=0, north=1)
    neighbor = SpatialBounds(west=1, east=2, south=0, north=1)
    intersection = SpatialBounds(west=0.5, east=1.5, south=0, north=1)
    assert not first.overlaps(neighbor)
    assert first.overlaps(intersection) and intersection.overlaps(first)
