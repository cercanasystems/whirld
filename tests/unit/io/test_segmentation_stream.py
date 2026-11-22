"""Windowed mask output preserves geometry, validity and previous destinations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from pydantic import ValidationError
from rasterio.env import get_gdal_config

from whirld.core.chips import ChipMeta
from whirld.errors import InvalidInputError, WhirldError
from whirld.io.segmentation_stream import (
    SegmentationMetadata,
    SegmentationStreamWriter,
)


def metadata(**overrides: Any) -> SegmentationMetadata:
    """Build a non-square 7x5 grid tiled with four native 4x4 chips."""
    return SegmentationMetadata.model_validate(
        {
            "crs": "EPSG:32630",
            "transform": (10.0, 0.0, 500000.0, 0.0, -10.0, 4500000.0),
            "width": 7,
            "height": 5,
            "chip_size_px": 4,
            "classes": 2,
        }
        | overrides
    )


def chips(meta: SegmentationMetadata | None = None) -> list[ChipMeta]:
    """Derive complete padded chip geometry from the declared affine grid."""
    meta = meta or metadata()
    tile = meta.chip_size_px
    rows = (meta.height + tile - 1) // tile
    cols = (meta.width + tile - 1) // tile
    transform = rasterio.Affine(*meta.transform)
    result = []
    for row in range(rows):
        for col in range(cols):
            x0, y0 = transform * (col * tile, row * tile)
            x1, y1 = transform * ((col + 1) * tile, (row + 1) * tile)
            result.append(
                ChipMeta(
                    index=row * cols + col,
                    row=row,
                    col=col,
                    bbox=[x0, y1, x1, y0],
                )
            )
    return result


def batch() -> tuple[np.ndarray, np.ndarray]:
    """Return binary tiles with invalid source cells and false edge padding."""
    masks = np.zeros((4, 4, 4), dtype=np.uint8)
    masks[0, 1:3, 1:3] = 1
    masks[1:] = 1
    valid = np.ones_like(masks, dtype=bool)
    valid[0, 0, 0] = False
    valid[1, :, 3:] = False
    valid[2:, 1:, :] = False
    valid[3, :, 3:] = False
    return masks, valid


def assert_no_staging(path: Path) -> None:
    """Require all owned temporary datasets and directories to be removed."""
    assert not list(path.parent.glob(".whirld-segment-*"))
    assert not list(path.parent.glob("*.aux.xml"))
    assert not list(path.parent.glob("*.msk"))


def test_partial_batches_crop_edges_and_preserve_class_zero(tmp_path: Path) -> None:
    """Pixel bytes, nodata, CRS/affine and counts exactly match eager assembly."""
    meta = metadata()
    predictions, valid = batch()
    records = chips(meta)
    expected = np.zeros((5, 7), dtype=np.uint8)
    for mask, keep, record in zip(predictions, valid, records, strict=True):
        row, col = record.row * 4, record.col * 4
        height, width = min(4, 5 - row), min(4, 7 - col)
        expected[row : row + height, col : col + width] = np.where(
            keep[:height, :width], mask[:height, :width], 255
        )
    path = tmp_path / "mask.exact-name"
    with SegmentationStreamWriter(path, chip_count=4, metadata=meta) as writer:
        assert get_gdal_config("GDAL_CACHEMAX") == 8 * 1024 * 1024
        writer.write(predictions[:3], records[:3], valid[:3])
        writer.write(predictions[3:], records[3:], valid[3:])
        assert writer.finish() == path
        assert writer.finish() == path
        assert writer.valid_pixel_count == int(np.count_nonzero(expected != 255))
        assert writer.positive_pixel_count == int(
            np.count_nonzero((expected > 0) & (expected != 255))
        )
    with rasterio.open(path) as source:
        np.testing.assert_array_equal(source.read(1), expected)
        assert source.shape == (5, 7)
        assert source.crs == rasterio.crs.CRS.from_epsg(32630)
        assert source.transform == rasterio.Affine(*meta.transform)
        assert source.nodata == 255
        assert source.is_tiled
        assert source.compression.name == "lzw"
        assert source.read_masks(1)[0, 1] == 255  # Valid class zero remains valid.
        assert source.read_masks(1)[0, 0] == 0
    assert_no_staging(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"width": 0},
        {"height": -1},
        {"chip_size_px": 0},
        {"classes": 0},
        {"classes": 256},
        {"classes": True},
        {"crs": "not-a-crs"},
        {"nodata": 0},
        {"transform": (10.0, 0.1, 0.0, 0.0, -10.0, 0.0)},
        {"transform": (10.0, 0.0, 0.0, 0.1, -10.0, 0.0)},
        {"transform": (10.0, 0.0, 0.0, 0.0, 10.0, 0.0)},
        {"transform": (0.0, 0.0, 0.0, 0.0, -10.0, 0.0)},
        {"transform": (10.0, 0.0, float("nan"), 0.0, -10.0, 0.0)},
    ],
)
def test_invalid_metadata_fails_before_output(overrides: dict[str, Any]) -> None:
    """Reject invalid file contracts at the typed schema boundary."""
    with pytest.raises(ValidationError):
        metadata(**overrides)


@pytest.mark.parametrize("count", [0, -1, 3, 5, True, 4.0])
def test_declared_chip_count_must_cover_grid(tmp_path: Path, count: Any) -> None:
    """Total chip count cannot under-cover, exceed or coerce the native grid."""
    with pytest.raises(InvalidInputError):
        SegmentationStreamWriter(
            tmp_path / "mask.tif", chip_count=count, metadata=metadata()
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "invalid",
    [
        "shape",
        "float",
        "bool",
        "negative",
        "class",
        "valid-shape",
        "valid-type",
        "order",
        "row",
        "bbox",
        "nan-bbox",
        "excess",
    ],
)
def test_invalid_batch_does_not_modify_staging(tmp_path: Path, invalid: str) -> None:
    """An invalid complete batch fails before any pixel or counter mutation."""
    predictions, valid = batch()
    records = chips()
    if invalid == "shape":
        predictions = predictions[:, :3]
    elif invalid == "float":
        predictions = predictions.astype(np.float32)
    elif invalid == "bool":
        predictions = predictions.astype(bool)
    elif invalid == "negative":
        predictions = predictions.astype(np.int32)
        predictions[0, 0, 0] = -1
    elif invalid == "class":
        predictions[0, 0, 0] = 2
    elif invalid == "valid-shape":
        valid = valid[:, :3]
    elif invalid == "valid-type":
        valid = valid.astype(np.uint8)
    elif invalid == "order":
        records[0] = records[0].model_copy(update={"index": 1})
    elif invalid == "row":
        records[0] = records[0].model_copy(update={"row": 1})
    elif invalid == "bbox":
        records[0] = records[0].model_copy(update={"bbox": [1.0, 2.0, 3.0, 4.0]})
    elif invalid == "nan-bbox":
        records[0] = records[0].model_copy(update={"bbox": [float("nan")] * 4})
    else:
        records += [records[-1]]
        predictions = np.concatenate([predictions, predictions[-1:]])
        valid = np.concatenate([valid, valid[-1:]])
    path = tmp_path / "mask.tif"
    with SegmentationStreamWriter(path, chip_count=4, metadata=metadata()) as writer:
        with pytest.raises(InvalidInputError):
            writer.write(predictions, records, valid)
        clean_predictions, clean_valid = batch()
        writer.write(clean_predictions, chips(), clean_valid)
        writer.finish()
    assert_no_staging(path)


@pytest.mark.parametrize("phase", ["before-enter", "after-finish", "after-abort"])
def test_writer_and_counts_require_correct_lifecycle(
    tmp_path: Path, phase: str
) -> None:
    """Writer operations and result summaries are available only at valid stages."""
    path = tmp_path / "mask.tif"
    writer = SegmentationStreamWriter(path, chip_count=4, metadata=metadata())
    predictions, valid = batch()
    if phase == "after-finish":
        with writer:
            writer.write(predictions, chips(), valid)
            writer.finish()
        assert writer.valid_pixel_count > 0
    elif phase == "after-abort":
        with pytest.raises(RuntimeError), writer:
            raise RuntimeError("consumer")
    with pytest.raises(WhirldError):
        writer.write(predictions, chips(), valid)
    if phase != "after-finish":
        with pytest.raises(WhirldError):
            _ = writer.valid_pixel_count
        with pytest.raises(WhirldError):
            _ = writer.positive_pixel_count
        with pytest.raises(WhirldError):
            writer.finish()
    if phase != "before-enter":
        with pytest.raises(WhirldError):
            writer.__enter__()
    assert_no_staging(path)


@pytest.mark.parametrize(
    "reason", ["unfinished", "incomplete", "consumer", "interrupt"]
)
def test_aborted_runs_preserve_old_destination(tmp_path: Path, reason: str) -> None:
    """Incomplete inference and exceptions cannot replace prior output bytes."""
    path = tmp_path / "mask.tif"
    path.write_bytes(b"previous mask")
    predictions, valid = batch()
    expected: type[BaseException] = (
        KeyboardInterrupt if reason == "interrupt" else WhirldError
    )
    if reason == "consumer":
        expected = RuntimeError
    with (
        pytest.raises(expected),
        SegmentationStreamWriter(path, chip_count=4, metadata=metadata()) as writer,
    ):
        writer.write(predictions[:1], chips()[:1], valid[:1])
        if reason == "incomplete":
            writer.finish()
        elif reason == "consumer":
            raise RuntimeError("consumer")
        elif reason == "interrupt":
            raise KeyboardInterrupt()
    assert path.read_bytes() == b"previous mask"
    assert_no_staging(path)


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_atomic_replace_failure_preserves_old_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    """Failure before final replacement leaves old bytes and removes staging."""
    path = tmp_path / "mask.tif"
    path.write_bytes(b"previous mask")
    replace = Path.replace

    def fail(source: Path, target: Path) -> Path:
        """Fail only publication of the owned staged mask."""
        if source.name == "mask.tif" and source.parent.name.startswith(
            ".whirld-segment-"
        ):
            raise failure("publication")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", fail)
    expected = WhirldError if failure is OSError else KeyboardInterrupt
    predictions, valid = batch()
    with (
        pytest.raises(expected),
        SegmentationStreamWriter(path, chip_count=4, metadata=metadata()) as writer,
    ):
        writer.write(predictions, chips(), valid)
        writer.finish()
    assert path.read_bytes() == b"previous mask"
    assert_no_staging(path)


def test_empty_batch_and_all_invalid_masks(tmp_path: Path) -> None:
    """Empty batches are harmless and fully invalid scenes publish all-nodata masks."""
    predictions, valid = batch()
    valid[:] = False
    path = tmp_path / "empty.tif"
    with SegmentationStreamWriter(path, chip_count=4, metadata=metadata()) as writer:
        writer.write(predictions[:0], [], valid[:0])
        writer.write(predictions, chips(), valid)
        writer.finish()
        assert writer.valid_pixel_count == writer.positive_pixel_count == 0
    with rasterio.open(path) as source:
        assert np.all(source.read(1) == 255)
    assert_no_staging(path)


def test_multiclass_indices_preserve_reserved_nodata(tmp_path: Path) -> None:
    """All uint8 classes below255 are legal and invalid cells alone become255."""
    predictions, valid = batch()
    predictions[:] = 254
    path = tmp_path / "multiclass.tif"
    with SegmentationStreamWriter(
        path, chip_count=4, metadata=metadata(classes=255)
    ) as writer:
        writer.write(predictions, chips(), valid)
        writer.finish()
        assert writer.positive_pixel_count == writer.valid_pixel_count == 34
    with rasterio.open(path) as source:
        assert set(np.unique(source.read(1))) == {254, 255}


def test_environment_restores_after_completion_and_failure(tmp_path: Path) -> None:
    """The writer restores its exact caller-owned GDAL cache state on every exit."""
    predictions, valid = batch()
    with rasterio.Env(GDAL_CACHEMAX=3 * 1024 * 1024):
        with SegmentationStreamWriter(
            tmp_path / "ok.tif", chip_count=4, metadata=metadata()
        ) as writer:
            writer.write(predictions, chips(), valid)
            writer.finish()
        assert get_gdal_config("GDAL_CACHEMAX") == 3 * 1024 * 1024
        with (
            pytest.raises(RuntimeError),
            SegmentationStreamWriter(
                tmp_path / "aborted.tif", chip_count=4, metadata=metadata()
            ),
        ):
            raise RuntimeError("consumer")
        assert get_gdal_config("GDAL_CACHEMAX") == 3 * 1024 * 1024


@pytest.mark.parametrize("phase", ["open", "write", "close", "fsync"])
def test_raster_failures_preserve_old_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """Actual dataset operations close even when setup, blocks or flushing fail."""
    path = tmp_path / "mask.tif"
    path.write_bytes(b"previous mask")
    predictions, valid = batch()
    opened: list[Any] = []
    original_open = rasterio.open

    class FaultDataset:
        """Forward owned dataset calls and inject one deterministic I/O failure."""

        def __init__(self, dataset: Any) -> None:
            """Retain the real dataset so its closure can be independently verified."""
            self.dataset = dataset

        def write(self, *args: Any, **kwargs: Any) -> None:
            """Fail mask block writing or forward it to actual GDAL."""
            if phase == "write":
                raise OSError("injected block failure")
            self.dataset.write(*args, **kwargs)

        def close(self) -> None:
            """Always close the real dataset, then optionally report flush failure."""
            self.dataset.close()
            if phase == "close":
                raise OSError("injected flush failure")

    def fault_open(*args: Any, **kwargs: Any) -> FaultDataset:
        """Build a real owned GeoTIFF or fail dataset creation."""
        if phase == "open":
            raise OSError("injected open failure")
        dataset = original_open(*args, **kwargs)
        opened.append(dataset)
        return FaultDataset(dataset)

    def fault_fsync(descriptor: int) -> None:
        """Fail file synchronization before publication."""
        raise OSError("injected fsync failure")

    monkeypatch.setattr(rasterio, "open", fault_open)
    if phase == "fsync":
        monkeypatch.setattr("whirld.io.segmentation_stream.os.fsync", fault_fsync)
    with (
        pytest.raises(WhirldError),
        SegmentationStreamWriter(path, chip_count=4, metadata=metadata()) as writer,
    ):
        writer.write(predictions, chips(), valid)
        writer.finish()
    assert all(dataset.closed for dataset in opened)
    assert path.read_bytes() == b"previous mask"
    assert_no_staging(path)


def test_directory_destination_is_preserved(tmp_path: Path) -> None:
    """A mistaken directory destination fails without replacing its contents."""
    path = tmp_path / "directory.tif"
    path.mkdir()
    (path / "existing.txt").write_text("keep")
    predictions, valid = batch()
    with (
        pytest.raises(WhirldError),
        SegmentationStreamWriter(path, chip_count=4, metadata=metadata()) as writer,
    ):
        writer.write(predictions, chips(), valid)
        writer.finish()
    assert (path / "existing.txt").read_text() == "keep"
    assert_no_staging(path)
