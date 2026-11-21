"""Typed acquisition metadata preserves eager parsing and override semantics."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from whirld.core.inference_metadata import AcquisitionMetadata, resolve_datetime


@pytest.mark.parametrize(
    "timestamp,override,hour",
    [
        ("2024-06-01T12:00:00+02:00", None, 10),
        ("2024:06:01 10:00:00", None, 10),
        ("2024-06-01T10:00:00", "invalid", 10),
        ("invalid", "2024-06-01T11:00:00Z", 11),
        ("2024-06-01T10:00:00Z", "2024-06-01T11:00:00Z", 11),
        (None, None, None),
        ("invalid", "also invalid", None),
    ],
)
def test_typed_metadata_preserves_datetime_policy(
    timestamp: str | None, override: str | None, hour: int | None
) -> None:
    """Offsets, TIFF dates, malformed fallback and overrides retain UTC semantics."""
    metadata = AcquisitionMetadata(timestamp=timestamp)
    expected = (
        datetime(2024, 6, 1, hour, tzinfo=timezone.utc) if hour is not None else None
    )
    assert metadata.resolve(override) == expected
    tags = {"TIFFTAG_DATETIME": timestamp} if timestamp is not None else {}
    assert resolve_datetime(override, tags) == expected
    assert metadata.timestamp == timestamp


def test_acquisition_metadata_cannot_be_mutated() -> None:
    """A consumer cannot change the source timestamp shared by reader batches."""
    metadata = AcquisitionMetadata(timestamp="2024-06-01T10:00:00Z")
    with pytest.raises(ValidationError, match="frozen"):
        metadata.timestamp = "2025-01-01T00:00:00Z"
