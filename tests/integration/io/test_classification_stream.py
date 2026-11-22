"""Process-memory evidence for classification result staging and external sorting."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class WriterMemoryResult(BaseModel):
    """A fresh writer worker's published count, output bytes and process peak RSS."""

    model_config = ConfigDict(frozen=True)
    features: int
    output_bytes: int
    peak_rss_bytes: int


_WRITER_WORKER = """
import json
import resource
import sys
from pathlib import Path

import numpy as np

from whirld.core.chips import ChipMeta
from whirld.io.classification_stream import (
    ClassificationMetadata,
    ClassificationStreamWriter,
)

count = int(sys.argv[1])
output = Path(sys.argv[2])
metadata = ClassificationMetadata(
    model="memory-fixture", model_version="1", crs="EPSG:4326",
    queries=("farmland", "forest"),
)
with ClassificationStreamWriter(
    output, chip_count=count, metadata=metadata, top_k=None,
) as writer:
    for start in range(0, count, 64):
        indices = range(start, min(start + 64, count))
        chips = []
        probabilities = []
        for index in indices:
            longitude = -120 + (index % 100) * 0.01
            latitude = 30 + (index // 100 % 100) * 0.01
            chips.append(ChipMeta(
                index=index, row=index // 100, col=index % 100,
                bbox=[longitude, latitude, longitude + 0.01, latitude + 0.01],
            ))
            # Pseudorandom primary scores require global external sorting, while
            # repeated ties also exercise the secondary global chip-index key.
            score = 0.5 if index % 17 == 0 else (index * 2053 % 65521) / 65521
            probabilities.append((score, 1 - score))
        writer.write(np.asarray(probabilities, dtype=np.float32), chips)
    writer.finish()
    features = writer.feature_count
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
sys.stdout.write(json.dumps({
    "features": features,
    "output_bytes": output.stat().st_size,
    "peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
}))
"""


def test_all_result_external_sort_memory_plateaus(tmp_path: Path) -> None:
    """Large globally ranked outputs grow on disk while fresh-worker RSS stays flat.

    Each worker generates only 64-chip/two-query batches. The largest output has
    64000 features, far exceeding the writer's 2 MiB SQLite page cache, and every
    feature qualifies. No worker or parent parses/loads the completed GeoJSON.
    An implementation retaining completed feature dictionaries would grow by
    well over the allowed 32 MiB as counts increase fourfold between workers.
    """
    results: list[WriterMemoryResult] = []
    for count in (4000, 16000, 64000):
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                _WRITER_WORKER,
                str(count),
                str(tmp_path / f"classified-{count}.geojson"),
            ],
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        measurement = WriterMemoryResult.model_validate_json(process.stdout)
        assert measurement.features == count
        results.append(measurement)
    assert results[1].features == 4 * results[0].features
    assert results[2].features == 4 * results[1].features
    assert results[1].output_bytes > 3.5 * results[0].output_bytes
    assert results[2].output_bytes > 3.5 * results[1].output_bytes
    assert results[2].output_bytes > 16 * 1024 * 1024
    assert (
        results[1].peak_rss_bytes - results[0].peak_rss_bytes < 32 * 1024 * 1024
    ), results
    assert (
        results[2].peak_rss_bytes - results[1].peak_rss_bytes < 32 * 1024 * 1024
    ), results
    assert not list(tmp_path.glob(".whirld-classify-*"))
