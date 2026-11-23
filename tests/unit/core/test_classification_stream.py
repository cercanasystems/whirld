"""Classification coordinator resource, context and bounded-call contracts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import cast

import numpy as np
import pytest

from whirld.core import classification_stream
from whirld.core.chips import ChipMeta
from whirld.core.classification_stream import ClassificationStream
from whirld.core.fetch import Manifest, ModelStore
from whirld.core.inference_metadata import AcquisitionMetadata
from whirld.core.options import (
    ClassificationQueries,
    StreamingClassificationOptions,
)
from whirld.errors import InvalidInputError, WhirldError
from whirld.io.stac_stream import ChipBatch, RasterWindowReader
from whirld.models.base import InferenceContext, ModelBackend


class TestStore(ModelStore):
    """Use actual bundled registry with a synthetic manifest and no downloads."""

    __test__ = False

    def load_manifest(self, name: str) -> Manifest:
        """Return typed installation provenance for the requested fixture model."""
        return Manifest(
            name=name,
            version="1.0.0",
            source_type="reference",
            sha256="0" * 64,
            weights_file="weights.bin",
        )


class TestReader:
    """Generate a five-chip scene lazily while observing explicit lifetime."""

    __test__ = False
    sensor = "sentinel-2-l2a"
    chip_count = 5
    crs = "EPSG:32630"
    resolution_m = 10.0
    n_rows = 1
    n_cols = 5
    metadata = AcquisitionMetadata(timestamp="2024-06-01T10:00:00Z")

    def __init__(self, batch_size: int) -> None:
        """Bind batch size and initialize observed resource state."""
        self.batch_size = batch_size
        self.opened = False
        self.closed = False

    def __enter__(self) -> TestReader:
        """Mark source resources active before yielding pixel windows."""
        self.opened = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Mark source resources closed for successful and failed consumers."""
        self.closed = True

    def batches(self) -> Iterator[ChipBatch]:
        """Yield independent bounded arrays and global chip coordinates."""
        assert self.opened and not self.closed
        for start in range(0, self.chip_count, self.batch_size):
            indices = range(start, min(self.chip_count, start + self.batch_size))
            chips = [
                ChipMeta(
                    index=i,
                    row=0,
                    col=i,
                    bbox=[500000 + i * 2240, 4497760, 502240 + i * 2240, 4500000],
                )
                for i in indices
            ]
            pixels = np.stack(
                [np.full((3, 224, 224), chip.index / 10, np.float32) for chip in chips]
            )
            yield ChipBatch(pixels=pixels, chips=chips)


class TestBackend(ModelBackend):
    """Record contexts and return deterministic scores independent of batch size."""

    __test__ = False

    def __init__(self, failure: type[BaseException] | None = None) -> None:
        """Bind optional later-call failure and initialize bounded-call records."""
        super().__init__("remoteclip", "1.0.0", "cpu")
        self.failure = failure
        self.contexts: list[InferenceContext] = []
        self.counts: list[int] = []

    def embed(
        self, chips: np.ndarray, context: InferenceContext | None = None
    ) -> np.ndarray:
        """Reject accidental embedding dispatch in a classification-only test."""
        raise AssertionError("classification must not call embed")

    def classify(
        self,
        chips: np.ndarray,
        queries: list[str],
        context: InferenceContext | None = None,
    ) -> np.ndarray:
        """Observe each inference call and fail only after staging has started."""
        assert context is not None
        self.contexts.append(context)
        self.counts.append(len(chips))
        if self.failure is not None and len(self.counts) == 2:
            raise self.failure("inference stopped")
        primary = chips[:, 0, 0, 0]
        columns = [primary]
        if len(queries) > 1:
            columns += [(1 - primary) / (len(queries) - 1)] * (len(queries) - 1)
        return np.stack(columns, axis=1)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_one_backend_receives_bounded_contexts_and_global_order(
    whirld_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
) -> None:
    """Partial final batches retain per-chip context and one backend lifetime."""
    reader = TestReader(batch_size)
    backend = TestBackend()
    loads: list[ModelBackend] = []
    stream = ClassificationStream(TestStore(), "cpu")

    def load(*args: object, **kwargs: object) -> ModelBackend:
        """Assert source lifetime spans model load and retain one load record."""
        assert reader.opened and not reader.closed
        loads.append(backend)
        return backend

    monkeypatch.setattr(classification_stream, "load_backend", load)
    monkeypatch.setattr(
        stream, "_reader", lambda *args: cast(RasterWindowReader, reader)
    )
    result = stream.run(
        "item.json",
        model="remoteclip",
        queries=["forest", "water"],
        output=tmp_path / "ranked.geojson",
        options=StreamingClassificationOptions(batch_size=batch_size, top_k=3),
    )
    assert loads == [backend]
    assert backend.counts == [batch_size] * (5 // batch_size) + (
        [5 % batch_size] if 5 % batch_size else []
    )
    assert all(c.options.batch_size == batch_size for c in backend.contexts)
    assert all(c.gsd_m == 10 for c in backend.contexts)
    assert all(c.acquisition_datetime is not None for c in backend.contexts)
    assert sum(len(c.latlons or []) for c in backend.contexts) == 5
    assert reader.closed
    assert (result.chip_count, result.feature_count) == (5, 3)
    assert result.queries == ("forest", "water")
    payload = json.loads(result.output_path.read_text())
    assert [f["properties"]["chip_index"] for f in payload["features"]] == [4, 3, 2]
    usage = json.loads((whirld_home / "logs/usage.jsonl").read_text().splitlines()[-1])
    assert usage["command"] == "classify"
    assert usage["input_type"] == "stac"
    assert usage["chip_count"] == 5
    assert usage["error"] is None


@pytest.mark.parametrize("failure", [WhirldError, KeyboardInterrupt])
def test_inference_failure_closes_reader_and_preserves_output(
    whirld_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
) -> None:
    """A failed later batch cannot publish staged features or leak source handles."""
    stream = ClassificationStream(TestStore(), "cpu")
    reader = TestReader(2)
    backend = TestBackend(failure)
    monkeypatch.setattr(classification_stream, "load_backend", lambda *a, **kw: backend)
    monkeypatch.setattr(stream, "_reader", lambda *a: cast(RasterWindowReader, reader))
    output = tmp_path / "ranked.geojson"
    output.write_bytes(b"prior output")
    with pytest.raises(failure, match="inference stopped"):
        stream.run(
            "item.json",
            model="remoteclip",
            queries=["forest"],
            output=output,
            options=StreamingClassificationOptions(batch_size=2),
        )
    assert reader.closed
    assert backend.counts == [2, 2]
    assert output.read_bytes() == b"prior output"
    assert not list(tmp_path.glob(".whirld-classify-*"))
    usage = json.loads((whirld_home / "logs/usage.jsonl").read_text().splitlines()[-1])
    assert usage["error"] == failure.__name__


@pytest.mark.parametrize("model", ["clay-v1", "prithvi-burn-scar"])
def test_unsupported_model_rejected_before_source_or_backend(
    whirld_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    """Embedding/segmentation entries cannot open classification sources."""
    stream = ClassificationStream(TestStore(), "cpu")

    def unexpected(*args: object, **kwargs: object) -> None:
        """Fail the test if invalid model dispatch performs source or backend I/O."""
        raise AssertionError("unexpected source/backend call")

    monkeypatch.setattr(stream, "_reader", unexpected)
    monkeypatch.setattr(classification_stream, "load_backend", unexpected)
    with pytest.raises(InvalidInputError, match="classification model"):
        stream.run(
            "item.json",
            model=model,
            queries=["forest"],
            output=tmp_path / "unused.geojson",
            options=StreamingClassificationOptions(),
        )


@pytest.mark.parametrize("location", ["scene.tif", "/tmp/local.tiff"])
def test_local_bbox_rejected_before_store_io(location: str, tmp_path: Path) -> None:
    """Unsupported local crops fail before model installation or raster reads."""
    with pytest.raises(InvalidInputError, match="bbox"):
        ClassificationStream(cast(ModelStore, None), "cpu").run(
            location,
            model="remoteclip",
            queries=["forest"],
            output=tmp_path / "unused.geojson",
            options=StreamingClassificationOptions(bbox=(-1.0, 50.0, 0.0, 51.0)),
        )


@pytest.mark.parametrize(
    "values",
    [
        {"batch_size": 0},
        {"batch_size": True},
        {"cache_mb": 0},
        {"top_k": 0},
        {"top_k": True},
        {"threshold": float("nan")},
        {"threshold": 1.1},
        {"bbox": (2.0, 1.0, 1.0, 2.0)},
    ],
)
def test_invalid_options_reject_ambiguous_or_unbounded_calls(
    values: dict[str, object],
) -> None:
    """Schema validation rejects invalid scalar domains before any source I/O."""
    with pytest.raises(InvalidInputError):
        StreamingClassificationOptions.from_arguments(**values)


def test_prompt_validation_preserves_eager_text_order_and_duplicates() -> None:
    """Blank filtering cannot strip real prompt text or collapse duplicate labels."""
    result = ClassificationQueries.from_arguments(
        queries=["", " ", " forest ", "forest"]
    )
    assert result.queries == [" forest ", "forest"]
    with pytest.raises(InvalidInputError):
        ClassificationQueries.from_arguments(queries=[" "])
    with pytest.raises(InvalidInputError):
        ClassificationQueries.from_arguments(queries=[True])
