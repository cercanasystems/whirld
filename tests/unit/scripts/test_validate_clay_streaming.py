"""Hermetic acceptance failures for the real-model validation harness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from scripts.validate_clay_streaming import (
    BASE_BBOX,
    EmbeddingComparison,
    RunMeasurement,
    ValidationFailure,
    ValidationOptions,
)


def _artifact(directory: Path, name: str, *, batch: int = 1) -> RunMeasurement:
    """Write independent known outputs and return their resource record.

    Args:
        directory: Temporary directory for one tiny NPY and JSON artifact pair.
        name: Filename stem and mode selector (eager or stream).
        batch: Recorded inference batch size.

    Returns:
        A four-chip float32 output record, without model or network dependencies.
    """
    output = directory / f"{name}.npy"
    meta = directory / f"{name}_meta.json"
    np.save(output, np.ones((4, 1024), dtype=np.float32))
    meta.write_text(
        json.dumps(
            {
                "model": "clay-v1.5",
                "model_version": "1.5.0",
                "whirld_version": "0.1.0",
                "timestamp": name,
                "crs": "EPSG:32630",
                "embed_dim": 1024,
                "chip_size_px": 256,
                "resolution_m": 10.0,
                "chips": [
                    {"index": i, "row": 0, "col": i, "bbox": [i, 0, i + 1, 1]}
                    for i in range(4)
                ],
            }
        ),
        encoding="utf-8",
    )
    return RunMeasurement(
        mode="eager" if name == "eager" else "stream",
        region="base",
        bbox=BASE_BBOX,
        batch_size=batch,
        chip_size=256,
        chip_count=4,
        embed_dim=1024,
        dtype="float32",
        finite=True,
        output_path=output,
        meta_path=meta,
        peak_rss_bytes=1024,
        elapsed_seconds=2.0,
        chips_per_second=2.0,
    )


def test_comparison_allows_timestamp_and_small_numeric_difference(
    tmp_path: Path,
) -> None:
    """Report measured error while ignoring only per-run timestamps."""
    eager = _artifact(tmp_path, "eager")
    streamed = _artifact(tmp_path, "stream", batch=3)
    array = np.load(streamed.output_path)
    array[0, 0] += 1e-5
    np.save(streamed.output_path, array)
    report = EmbeddingComparison.compare(eager, streamed)
    assert report.batch_size == 3
    assert report.chip_count == 4
    assert 0 < report.rmse < report.max_abs_error < 2e-5


@pytest.mark.parametrize("failure", ["values", "geometry", "order", "count", "model"])
def test_comparison_rejects_incorrect_outputs(tmp_path: Path, failure: str) -> None:
    """Reject numerical, geometric, ordering, identity and count regressions."""
    eager = _artifact(tmp_path, "eager")
    streamed = _artifact(tmp_path, "stream")
    metadata = json.loads(streamed.meta_path.read_text())
    if failure == "values":
        np.save(streamed.output_path, np.zeros((4, 1024), dtype=np.float32))
    elif failure == "geometry":
        metadata["chips"][0]["bbox"][0] += 1
    elif failure == "order":
        metadata["chips"].reverse()
    elif failure == "model":
        metadata["model"] = "clay-v1"
    else:
        streamed = streamed.model_copy(update={"chip_count": 5})
    streamed.meta_path.write_text(json.dumps(metadata))
    with pytest.raises(ValidationFailure):
        EmbeddingComparison.compare(eager, streamed)


@pytest.mark.parametrize("failure", ["dtype", "dimension", "empty", "nan", "infinity"])
def test_array_validation_rejects_invalid_embedding(
    tmp_path: Path, failure: str
) -> None:
    """Reject invalid model output before numerical comparison can hide it."""
    array = np.ones((4, 1024), dtype=np.float32)
    if failure == "dtype":
        array = array.astype(np.float64)
    elif failure == "dimension":
        array = array[:, :512]
    elif failure == "empty":
        array = array[:0]
    else:
        array[0, 0] = np.nan if failure == "nan" else np.inf
    path = tmp_path / "invalid.npy"
    np.save(path, array)
    with pytest.raises(ValidationFailure):
        EmbeddingComparison.validate_array(path)


def test_unknown_chip_metadata_is_not_silently_dropped(tmp_path: Path) -> None:
    """Exact comparison rejects schema drift even in nested chip records."""
    eager = _artifact(tmp_path, "eager")
    streamed = _artifact(tmp_path, "stream")
    metadata = json.loads(streamed.meta_path.read_text())
    metadata["chips"][0]["unexpected"] = "changed"
    streamed.meta_path.write_text(json.dumps(metadata))
    with pytest.raises(ValidationError):
        EmbeddingComparison.compare(eager, streamed)


@pytest.mark.parametrize(
    "options",
    [
        {"chip_size": 7},
        {"batch_sizes": ()},
        {"batch_sizes": (1, 1)},
        {"batch_sizes": (2, 3)},
        {"threads": 0},
        {"timeout": -1},
    ],
)
def test_invalid_experiment_options_fail(
    tmp_path: Path, options: dict[str, object]
) -> None:
    """Reject meaningless experiment settings before starting costly workers."""
    with pytest.raises(ValidationError):
        ValidationOptions(output_dir=tmp_path, **options)
