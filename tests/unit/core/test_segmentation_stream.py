"""Segmentation coordinator owns bounded inference, validity and resource lifetimes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import cast

import numpy as np
import pytest
from rasterio.transform import from_origin

from whirld.core import segmentation_stream
from whirld.core.chips import ChipMeta
from whirld.core.fetch import Manifest, ModelStore
from whirld.core.inference_metadata import AcquisitionMetadata
from whirld.core.options import CogOptions, StreamingSegmentationOptions
from whirld.core.segmentation_stream import SegmentationStream
from whirld.errors import InvalidInputError, UnsupportedOperationError, WhirldError
from whirld.io.segmentation_stream import SegmentationMetadata
from whirld.io.stac_stream import ChipBatch, RasterWindowReader
from whirld.models.base import InferenceContext, ModelBackend


class FixtureStore(ModelStore):
    """Resolve real registry contracts with typed synthetic installation provenance."""

    def __init__(self) -> None:
        """Retain manifest call observations without downloading checkpoints."""
        super().__init__()
        self.manifest_calls: list[str] = []

    def load_manifest(self, name: str) -> Manifest:
        """Return a tiny fixture manifest owned by this store."""
        self.manifest_calls.append(name)
        return Manifest(
            name=name,
            version="2.0.0",
            source_type="reference",
            sha256="0" * 64,
            weights_file="weights.bin",
        )


class FixtureReader:
    """Yield five native chips while observing validity requests and cleanup."""

    sensor = "hls"
    chip_count = 5
    crs = "EPSG:32630"
    resolution_m = 30.0
    chip_size = 512
    width = 2560
    height = 500
    n_rows = 1
    n_cols = 5
    transform = from_origin(500000, 4500000, 30, 30)
    metadata = AcquisitionMetadata(timestamp="2024-06-01T10:00:00Z")

    def __init__(self, batch_size: int, *, missing_validity: bool = False) -> None:
        """Bind bounded iteration and optional broken source contract."""
        self.batch_size = batch_size
        self.missing_validity = missing_validity
        self.opened = False
        self.closed = False
        self.requested_validity: bool | None = None

    def __enter__(self) -> FixtureReader:
        """Activate owned source handles before backend loading."""
        self.opened = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Observe closure regardless of successful or failed consumption."""
        self.closed = True

    def batches(self, *, include_validity: bool = False) -> Iterator[ChipBatch]:
        """Generate bounded pixels and padded-false source validity."""
        assert self.opened and not self.closed
        self.requested_validity = include_validity
        for start in range(0, 5, self.batch_size):
            chips = [
                ChipMeta(
                    index=i,
                    row=0,
                    col=i,
                    bbox=[500000 + i * 15360, 4484640, 515360 + i * 15360, 4500000],
                )
                for i in range(start, min(start + self.batch_size, 5))
            ]
            pixels = np.full((len(chips), 6, 512, 512), 0.25, np.float32)
            valid = np.ones((len(chips), 512, 512), dtype=bool)
            valid[:, 500:, :] = False
            valid[:, 0, 0] = False
            yield ChipBatch(
                pixels=pixels,
                chips=chips,
                valid_masks=None if self.missing_validity else valid,
            )


class FixtureBackend(ModelBackend):
    """Observe native bounded calls and optionally fail after one staged batch."""

    classes = 2

    def __init__(self, failure: type[BaseException] | None = None) -> None:
        """Bind failure mode and small invocation metadata observations."""
        super().__init__("prithvi-burn-scar", "2.0.0", "cpu")
        self.failure = failure
        self.counts: list[int] = []
        self.contexts: list[InferenceContext] = []
        self.heads: list[str | None] = []
        self.thresholds: list[float] = []

    def embed(
        self, chips: np.ndarray, context: InferenceContext | None = None
    ) -> np.ndarray:
        """Reject paradigm dispatch errors immediately."""
        raise AssertionError("segmentation must not call embed")

    def segment(
        self,
        chips: np.ndarray,
        head: str | None = None,
        threshold: float = 0.5,
        context: InferenceContext | None = None,
    ) -> np.ndarray:
        """Retain contexts and produce positive masks from native-sized pixels."""
        assert chips.shape[1:] == (6, 512, 512)
        assert context is not None
        self.counts.append(len(chips))
        self.contexts.append(context)
        self.heads.append(head)
        self.thresholds.append(threshold)
        if self.failure is not None and len(self.counts) == 2:
            raise self.failure("inference stopped")
        return np.ones((len(chips), 512, 512), dtype=np.uint8)


class FixtureWriter:
    """Observe exact staged-writer contract while keeping previous output intact."""

    def __init__(
        self, output: Path, *, chip_count: int, metadata: SegmentationMetadata
    ) -> None:
        """Bind typed output metadata without opening its final destination."""
        self.output = output
        self.chip_count = chip_count
        self.metadata = metadata
        self.opened = False
        self.closed = False
        self.finished = False
        self.valid_pixel_count = 0
        self.positive_pixel_count = 0
        self.indices: list[int] = []

    def __enter__(self) -> FixtureWriter:
        """Activate the staged writer resource."""
        self.opened = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Observe resource closure, including failed/incomplete inference."""
        self.closed = True

    def write(
        self, masks: np.ndarray, chips: list[ChipMeta], valid: np.ndarray
    ) -> None:
        """Assert explicit padded validity reaches output unchanged."""
        assert self.opened and not self.closed and not self.finished
        assert valid.dtype == bool
        assert not valid[:, 500:, :].any()
        assert not valid[:, 0, 0].any()
        assert masks.shape == valid.shape
        self.indices.extend(chip.index for chip in chips)
        self.valid_pixel_count += int(valid.sum())
        self.positive_pixel_count += int(((masks > 0) & valid).sum())

    def finish(self) -> Path:
        """Publish only after every chip has traversed the writer boundary."""
        assert self.indices == list(range(self.chip_count))
        self.output.write_bytes(b"published mask")
        self.finished = True
        return self.output


class Harness:
    """Own fake dependency seams and expose small coordinator observations."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        batch_size: int,
        *,
        failure: type[BaseException] | None = None,
        missing_validity: bool = False,
    ) -> None:
        """Connect reader/backend/writer seams while retaining real option schemas."""
        self.store = FixtureStore()
        self.stream = SegmentationStream(self.store, "cpu")
        self.reader = FixtureReader(batch_size, missing_validity=missing_validity)
        self.backend = FixtureBackend(failure)
        self.loads = 0
        self.writer: FixtureWriter | None = None
        monkeypatch.setattr(
            self.stream, "_reader", lambda *args: cast(RasterWindowReader, self.reader)
        )
        monkeypatch.setattr(segmentation_stream, "load_backend", self.load)
        monkeypatch.setattr(
            segmentation_stream, "SegmentationStreamWriter", self.make_writer
        )

    def load(self, *args: object, **kwargs: object) -> ModelBackend:
        """Ensure reader resources precede backend loading and one store is shared."""
        assert self.reader.opened and not self.reader.closed
        assert kwargs["store"] is self.store
        self.loads += 1
        return self.backend

    def make_writer(
        self, output: Path, *, chip_count: int, metadata: SegmentationMetadata
    ) -> FixtureWriter:
        """Bind the typed staged writer to the actual resolved source grid."""
        self.writer = FixtureWriter(output, chip_count=chip_count, metadata=metadata)
        return self.writer


def usage(home: Path) -> dict[str, object]:
    """Read the last small structured usage record for branch outcome assertions."""
    return cast(
        dict[str, object],
        json.loads((home / "logs/usage.jsonl").read_text().splitlines()[-1]),
    )


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_owned_run_passes_native_context_validity_and_usage(
    whirld_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
) -> None:
    """One backend receives bounded native chips and context through publication."""
    h = Harness(monkeypatch, batch_size)
    result = h.stream.run(
        "item.json",
        model="prithvi-burn-scar",
        head="burn-scar",
        output=tmp_path / "mask.tif",
        options=StreamingSegmentationOptions(batch_size=batch_size, threshold=0.7),
    )
    assert h.loads == 1
    assert sum(h.backend.counts) == 5
    assert max(h.backend.counts) <= batch_size
    assert h.backend.counts[-1] == (5 % batch_size or batch_size)
    assert h.reader.requested_validity is True and h.reader.closed
    assert h.writer is not None and h.writer.finished and h.writer.closed
    assert h.writer.metadata.width == 2560 and h.writer.metadata.height == 500
    assert h.writer.metadata.chip_size_px == 512
    assert result.chip_count == 5 and result.classes == 2
    assert (
        result.valid_pixel_count == result.positive_pixel_count == 5 * (500 * 512 - 1)
    )
    assert result.transform == (30, 0, 500000, 0, -30, 4500000)
    assert h.backend.heads == ["burn-scar"] * len(h.backend.counts)
    assert h.backend.thresholds == [0.7] * len(h.backend.counts)
    assert all(c.sensor == "hls" and c.gsd_m == 30 for c in h.backend.contexts)
    assert all(c.options.batch_size == batch_size for c in h.backend.contexts)
    assert all(
        c.acquisition_datetime is not None and c.acquisition_datetime.year == 2024
        for c in h.backend.contexts
    )
    assert sum(len(c.latlons or []) for c in h.backend.contexts) == 5
    assert all(
        -90 < lat < 90 and -180 < lon < 180
        for c in h.backend.contexts
        for lat, lon in c.latlons or []
    )
    record = usage(whirld_home)
    assert record["command"] == "segment" and record["input_type"] == "stac"
    assert record["chip_count"] == 5 and record["model_version"] == "2.0.0"
    assert record["sensor_detected"] == "hls" and record["error"] is None


@pytest.mark.parametrize("failure", [WhirldError, KeyboardInterrupt])
def test_failure_and_interrupt_close_both_owners_and_label_usage(
    whirld_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
) -> None:
    """A later inference failure preserves prior output and closes both owners."""
    h = Harness(monkeypatch, 2, failure=failure)
    output = tmp_path / "mask.tif"
    output.write_bytes(b"prior mask")
    with pytest.raises(failure, match="inference stopped"):
        h.stream.run(
            "item.json",
            model="prithvi-burn-scar",
            output=output,
            options=StreamingSegmentationOptions(batch_size=2),
        )
    assert output.read_bytes() == b"prior mask"
    assert h.reader.closed and h.writer is not None and h.writer.closed
    assert not h.writer.finished and h.writer.indices == [0, 1]
    record = usage(whirld_home)
    assert record["error"] == failure.__name__ and record["chip_count"] == 5


def test_missing_validity_fails_before_backend_and_preserves_output(
    whirld_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalized zeros are never guessed to be valid or invalid source observations."""
    h = Harness(monkeypatch, 2, missing_validity=True)
    output = tmp_path / "mask.tif"
    output.write_bytes(b"prior mask")
    with pytest.raises(InvalidInputError, match="explicit source validity"):
        h.stream.run(
            "item.json",
            model="prithvi-burn-scar",
            output=output,
            options=StreamingSegmentationOptions(batch_size=2),
        )
    assert h.backend.counts == []
    assert h.reader.closed and h.writer is not None and h.writer.closed
    assert not h.writer.finished and output.read_bytes() == b"prior mask"
    assert usage(whirld_home)["error"] == "InvalidInputError"


def test_nonsegmentation_model_rejected_before_installation(
    whirld_home: Path,
    tmp_path: Path,
) -> None:
    """Category mismatch cannot open a source or load an unrelated checkpoint."""
    store = FixtureStore()
    with pytest.raises(UnsupportedOperationError, match="segment"):
        SegmentationStream(store, "cpu").run(
            "item.json",
            model="clay-v1",
            output=tmp_path / "mask.tif",
            options=StreamingSegmentationOptions(),
        )
    assert store.manifest_calls == []
    record = usage(whirld_home)
    assert record["error"] == "UnsupportedOperationError"
    assert record["chip_count"] is None


@pytest.mark.parametrize(
    "source,sensor", [("local.tif", None), ("https://example.invalid/image.tif", None)]
)
def test_local_bbox_or_missing_direct_sensor_fails_before_installation(
    whirld_home: Path,
    tmp_path: Path,
    source: str,
    sensor: str | None,
) -> None:
    """Input-domain failures occur before manifest or source acquisition."""
    store = FixtureStore()
    with pytest.raises(InvalidInputError):
        SegmentationStream(store, "cpu").run(
            source,
            model="prithvi-burn-scar",
            sensor=sensor,
            output=tmp_path / "mask.tif",
            options=StreamingSegmentationOptions(
                bbox=(-1.0, 50.0, 0.0, 51.0) if source == "local.tif" else None
            ),
        )
    assert store.manifest_calls == [] and not (tmp_path / "mask.tif").exists()


@pytest.mark.parametrize("direct", [False, True])
def test_reader_factory_forwards_native_grid_cache_and_credential_policy(
    whirld_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    direct: bool,
) -> None:
    """Source adapters receive explicit policies without opening resources early."""
    from whirld.core.registry import ModelEntry
    from whirld.io.input import RasterInput

    store = FixtureStore()
    stream = SegmentationStream(store, "cpu")
    entry = store.registry.get("prithvi-burn-scar")
    options = StreamingSegmentationOptions(
        batch_size=3, cache_mb=7, bbox=(-1.0, 50.0, 0.0, 51.0)
    )
    cog = CogOptions(scale=1.0) if direct else None
    location = (
        "https://example.invalid/scene.tif"
        if direct
        else "https://example.invalid/item.json"
    )
    source = RasterInput.resolve(location, cog)
    reader = FixtureReader(3)
    calls = []

    def make_reader(
        url: str,
        *,
        entry: ModelEntry,
        chip_size: int,
        batch_size: int,
        cache_mb: int,
        sensor: str | None,
        crs: str | None,
        bbox: tuple[float, float, float, float] | None,
        token: str | None,
        token_origins: tuple[str, ...],
        cog: CogOptions | None = None,
    ) -> RasterWindowReader:
        """Assert every constructor field at the resource adapter boundary."""
        assert url == location and entry.name == "prithvi-burn-scar"
        assert chip_size == 512 and batch_size == 3 and cache_mb == 7
        assert sensor == "hls" and crs == "EPSG:32630" and bbox == options.bbox
        assert token == "fixture-token" and token_origins == ("https://assets.invalid",)
        assert cog == source.cog
        calls.append(url)
        return cast(RasterWindowReader, reader)

    target = "CogWindowReader" if direct else "StacWindowReader"
    monkeypatch.setattr(segmentation_stream, target, make_reader)
    actual = stream._reader(
        source,
        entry,
        options,
        "hls",
        "EPSG:32630",
        "fixture-token",
        ("https://assets.invalid",),
    )
    assert actual is reader and calls == [location] and not reader.opened


def test_backend_load_failure_closes_reader_before_writer_creation(
    whirld_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed model initialization cannot leak the already-open source."""
    h = Harness(monkeypatch, 2)

    def fail(*args: object, **kwargs: object) -> ModelBackend:
        """Model loading fails after source metadata resolution."""
        raise WhirldError("model stopped")

    monkeypatch.setattr(segmentation_stream, "load_backend", fail)
    with pytest.raises(WhirldError, match="model stopped"):
        h.stream.run(
            "item.json",
            model="prithvi-burn-scar",
            output=tmp_path / "mask.tif",
            options=StreamingSegmentationOptions(),
        )
    assert h.reader.closed and h.writer is None
    assert usage(whirld_home)["error"] == "WhirldError"


def test_writer_publication_failure_closes_owners_and_records_error(
    whirld_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinator propagates finish failures without reporting a completed result."""
    h = Harness(monkeypatch, 2)
    output = tmp_path / "mask.tif"
    output.write_bytes(b"prior mask")

    def fail(writer: FixtureWriter) -> Path:
        """Fail publication after every batch traversed output staging."""
        assert writer.indices == list(range(5))
        raise WhirldError("publication stopped")

    monkeypatch.setattr(FixtureWriter, "finish", fail)
    with pytest.raises(WhirldError, match="publication stopped"):
        h.stream.run(
            "item.json",
            model="prithvi-burn-scar",
            output=output,
            options=StreamingSegmentationOptions(),
        )
    assert h.reader.closed and h.writer is not None and h.writer.closed
    assert output.read_bytes() == b"prior mask" and not h.writer.finished
    assert usage(whirld_home)["error"] == "WhirldError"
