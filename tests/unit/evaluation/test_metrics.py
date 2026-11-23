"""Independent confusion examples and bounded exact-grid raster scoring."""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.env import get_gdal_config
from rasterio.transform import from_origin
from rasterio.windows import Window

from whirld.evaluation.errors import EvaluationError
from whirld.evaluation.metrics import BinaryRasterScorer
from whirld.evaluation.schemas import ConfusionCounts, MetricSummary


def write_raster(
    path: Path,
    pixels: np.ndarray,
    *,
    nodata: int | None,
    crs: str | None = "EPSG:32630",
    transform: Affine | None = None,
    mask: np.ndarray | None = None,
) -> Path:
    """Write an independently specified local raster and optional external mask.

    Args:
        path: Destination under the test's private directory.
        pixels: One-band 2D or source-order multiband 3D integer/float pixels.
        nodata: Explicit nodata code or no nodata declaration.
        crs: Declared CRS, including None for invalid-grid tests.
        transform: Optional affine instead of the default north-up 30 m grid.
        mask: Optional GDAL validity values written to an external sidecar.

    Returns:
        The written TIFF path, with no production scoring logic involved.
    """
    values = pixels[None] if pixels.ndim == 2 else pixels
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=values.shape[2],
            height=values.shape[1],
            count=values.shape[0],
            dtype=values.dtype,
            nodata=nodata,
            crs=crs,
            transform=transform or from_origin(500000, 4500000, 30, 30),
        ) as dataset:
            dataset.write(values)
            if mask is not None:
                dataset.write_mask(mask)
    return path


@pytest.fixture
def counted_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Provide hand-counted TP3/FP3/TN4/FN4, two ignored and four missing pixels."""
    labels = np.array(
        [[1, 1, 1, 0, 0], [1, 0, 0, -1, -1], [1, 0, 1, 0, 1], [0, 0, 1, 1, 0]],
        dtype=np.int16,
    )
    prediction = np.array(
        [
            [1, 0, 255, 1, 0],
            [0, 0, 255, 1, 0],
            [255, 1, 1, 0, 0],
            [0, 1, 1, 0, 255],
        ],
        dtype=np.uint8,
    )
    return (
        write_raster(tmp_path / "prediction.tif", prediction, nodata=255),
        write_raster(tmp_path / "labels.tif", labels, nodata=-1),
    )


@pytest.mark.parametrize("window_size", [1, 2, 3, 256])
def test_hand_counted_scores_do_not_depend_on_window_size(
    counted_pair: tuple[Path, Path], window_size: int
) -> None:
    """Include ignored observations and positive/negative abstentions explicitly."""
    counts = BinaryRasterScorer(window_size).score(*counted_pair)
    assert counts == ConfusionCounts(
        tp=3,
        fp=3,
        tn=4,
        fn=4,
        label_ignored=2,
        prediction_invalid_positive=2,
        prediction_invalid_negative=2,
    )
    assert counts.scored_pixels == 14
    assert counts.label_valid_pixels == 18
    assert counts.total_pixels == 20
    metrics = MetricSummary.from_counts(counts)
    assert metrics.precision == 0.5
    assert metrics.recall == pytest.approx(3 / 7)
    assert metrics.f1 == pytest.approx(6 / 13)
    assert metrics.iou == 0.3
    assert metrics.accuracy == 0.5
    assert metrics.coverage == pytest.approx(7 / 9)


def test_masks_are_honored_even_when_inherited_environment_hides_sidecars(
    counted_pair: tuple[Path, Path],
) -> None:
    """Label-mask holes ignore observations; prediction holes remain abstentions."""
    prediction, labels = counted_pair
    for path, point in ((prediction, (0, 0)), (labels, (0, 3))):
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
            with rasterio.open(path, "r+") as dataset:
                mask = np.full(dataset.shape, 255, dtype=np.uint8)
                mask[point] = 0
                dataset.write_mask(mask)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        counts = BinaryRasterScorer().score(prediction, labels)
        assert get_gdal_config("GDAL_DISABLE_READDIR_ON_OPEN") == "EMPTY_DIR"
    assert counts == ConfusionCounts(
        tp=2,
        fp=2,
        tn=4,
        fn=4,
        label_ignored=3,
        prediction_invalid_positive=3,
        prediction_invalid_negative=2,
    )


@pytest.mark.parametrize(
    "counts,expected",
    [
        (
            ConfusionCounts(tn=5),
            {
                "precision": None,
                "recall": None,
                "f1": None,
                "iou": None,
                "accuracy": 1.0,
                "coverage": 1.0,
            },
        ),
        (
            ConfusionCounts(fp=3),
            {
                "precision": 0.0,
                "recall": None,
                "f1": 0.0,
                "iou": 0.0,
                "accuracy": 0.0,
                "coverage": 1.0,
            },
        ),
        (
            ConfusionCounts(fn=3),
            {
                "precision": None,
                "recall": 0.0,
                "f1": 0.0,
                "iou": 0.0,
                "accuracy": 0.0,
                "coverage": 1.0,
            },
        ),
        (
            ConfusionCounts(prediction_invalid_positive=2),
            {
                "precision": None,
                "recall": None,
                "f1": None,
                "iou": None,
                "accuracy": None,
                "coverage": 0.0,
            },
        ),
        (
            ConfusionCounts(label_ignored=4),
            {
                "precision": None,
                "recall": None,
                "f1": None,
                "iou": None,
                "accuracy": None,
                "coverage": None,
            },
        ),
    ],
)
def test_zero_denominators_are_json_null(
    counts: ConfusionCounts, expected: dict[str, float | None]
) -> None:
    """Undefined positive-class ratios remain distinct from zero/perfect scores."""
    payload = MetricSummary.from_counts(counts).model_dump_json()
    assert "NaN" not in payload and "Infinity" not in payload
    assert json.loads(payload) == expected


def test_aggregate_scores_come_from_combined_counts() -> None:
    """A split's micro IoU differs from the unweighted mean of per-case IoUs."""
    first = ConfusionCounts(tp=1)
    second = ConfusionCounts(fp=9)
    combined = ConfusionCounts.combine((first, second))
    assert MetricSummary.from_counts(combined).iou == 0.1
    assert combined == ConfusionCounts(tp=1, fp=9)
    assert ConfusionCounts.combine(()) == ConfusionCounts()


@pytest.mark.parametrize("invalid", [0, -1, 257, True, 1.5])
def test_invalid_window_size_fails_before_io(invalid: Any) -> None:
    """Reject ambiguous or excessively large window settings at construction."""
    with pytest.raises(EvaluationError, match="window_size"):
        BinaryRasterScorer(invalid)


@pytest.mark.parametrize(
    "change",
    ["shift", "shape", "crs", "missing_crs", "rotation", "geographic", "south_up"],
)
def test_grid_mismatch_is_rejected_without_resampling(
    tmp_path: Path, change: str
) -> None:
    """Even equal-shaped data cannot silently bypass geospatial registration."""
    prediction = write_raster(
        tmp_path / "prediction.tif", np.zeros((3, 4), dtype=np.uint8), nodata=255
    )
    transform = from_origin(500000, 4500000, 30, 30)
    crs: str | None = "EPSG:32630"
    shape = (3, 4)
    if change == "shift":
        transform = transform @ Affine.translation(0.5, 0)
    elif change == "shape":
        shape = (4, 4)
    elif change == "crs":
        crs = "EPSG:32631"
    elif change == "missing_crs":
        crs = None
    elif change == "rotation":
        transform = transform @ Affine.rotation(2)
    elif change == "geographic":
        crs = "EPSG:4326"
    elif change == "south_up":
        transform = Affine(30, 0, 500000, 0, 30, 4500000)
    labels = write_raster(
        tmp_path / "labels.tif",
        np.zeros(shape, dtype=np.int16),
        nodata=-1,
        crs=crs,
        transform=transform,
    )
    with pytest.raises(EvaluationError, match="grid|CRS|dimensions"):
        BinaryRasterScorer().score(prediction, labels)


@pytest.mark.parametrize(
    "change",
    [
        "prediction_float",
        "prediction_nodata",
        "prediction_bands",
        "label_float",
        "label_bands",
    ],
)
def test_storage_contract_is_checked_before_scoring(
    tmp_path: Path, change: str
) -> None:
    """Binary class data cannot be silently rounded or selected from many bands."""
    prediction_pixels = np.zeros((2, 3), dtype=np.uint8)
    label_pixels = np.zeros((2, 3), dtype=np.int16)
    if change == "prediction_float":
        prediction_pixels = prediction_pixels.astype(np.float32)
    elif change == "prediction_bands":
        prediction_pixels = np.stack((prediction_pixels, prediction_pixels))
    elif change == "label_float":
        label_pixels = label_pixels.astype(np.float32)
    elif change == "label_bands":
        label_pixels = np.stack((label_pixels, label_pixels))
    prediction = write_raster(
        tmp_path / "prediction.tif",
        prediction_pixels,
        nodata=None if change == "prediction_nodata" else 255,
    )
    labels = write_raster(tmp_path / "labels.tif", label_pixels, nodata=-1)
    with pytest.raises(EvaluationError, match="single|integer|uint8|nodata"):
        BinaryRasterScorer().score(prediction, labels)


@pytest.mark.parametrize("source", ["prediction", "label"])
def test_unknown_unmasked_classes_fail_even_outside_scoring_domain(
    tmp_path: Path, source: str
) -> None:
    """Ignored labels cannot conceal malformed available prediction classes."""
    predicted = np.array([[1, 2 if source == "prediction" else 0]], dtype=np.uint8)
    actual = np.array([[1, -1 if source == "prediction" else 2]], dtype=np.int16)
    prediction = write_raster(tmp_path / "prediction.tif", predicted, nodata=255)
    labels = write_raster(tmp_path / "labels.tif", actual, nodata=-1)
    with pytest.raises(EvaluationError, match="unexpected unmasked class"):
        BinaryRasterScorer().score(prediction, labels)


@pytest.mark.parametrize("empty", ["labels", "predictions"])
def test_empty_scoring_domain_is_an_error(tmp_path: Path, empty: str) -> None:
    """Do not publish an apparently successful case with undefined observations."""
    prediction = write_raster(
        tmp_path / "prediction.tif",
        np.full((2, 3), 255 if empty == "predictions" else 0, dtype=np.uint8),
        nodata=255,
    )
    labels = write_raster(
        tmp_path / "labels.tif",
        np.full((2, 3), -1 if empty == "labels" else 0, dtype=np.int16),
        nodata=-1,
    )
    with pytest.raises(EvaluationError, match="no valid"):
        BinaryRasterScorer().score(prediction, labels)


def test_label_ignore_value_does_not_require_a_nodata_tag(tmp_path: Path) -> None:
    """The explicit -1 label policy also applies to a metadata-free label TIFF."""
    prediction = write_raster(
        tmp_path / "prediction.tif", np.array([[1, 0]], dtype=np.uint8), nodata=255
    )
    labels = write_raster(
        tmp_path / "labels.tif", np.array([[1, -1]], dtype=np.int16), nodata=None
    )
    assert BinaryRasterScorer().score(prediction, labels) == ConfusionCounts(
        tp=1, label_ignored=1
    )


@pytest.mark.parametrize("interrupt", [False, True])
def test_window_reads_are_bounded_and_resources_restore_on_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: bool
) -> None:
    """Every data/mask read has a bounded window; exceptions close both files."""
    pixels = np.zeros((513, 777), dtype=np.uint8)
    prediction = write_raster(tmp_path / "prediction.tif", pixels, nodata=255)
    labels = write_raster(tmp_path / "labels.tif", pixels.astype(np.int16), nodata=-1)
    original_open = rasterio.open
    datasets: list[Any] = []
    windows: list[Window] = []

    class ObservedReader:
        """Delegate real GDAL reads while observing requests and interruptions."""

        def __init__(self, dataset: Any) -> None:
            """Retain the underlying resource until context exit."""
            self.dataset = dataset

        def __getattr__(self, name: str) -> Any:
            """Delegate metadata to the actual on-disk TIFF reader."""
            return getattr(self.dataset, name)

        def __enter__(self) -> ObservedReader:
            """Expose this observer under the normal dataset context protocol."""
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            """Close the actual dataset after normal, failing or interrupted reads."""
            self.dataset.close()

        def read(self, index: int, *, window: Window) -> np.ndarray:
            """Require a data window and optionally interrupt the first read."""
            assert 0 < window.width <= 256 and 0 < window.height <= 256
            windows.append(window)
            if interrupt:
                raise KeyboardInterrupt()
            return self.dataset.read(index, window=window)

        def read_masks(self, index: int, *, window: Window) -> np.ndarray:
            """Require mask reads to obey the same bound as data windows."""
            assert 0 < window.width <= 256 and 0 < window.height <= 256
            windows.append(window)
            return self.dataset.read_masks(index, window=window)

    def observed_open(*args: Any, **kwargs: Any) -> ObservedReader:
        """Observe only scoring opens after test fixture construction."""
        assert kwargs["driver"] == "GTiff"
        dataset = original_open(*args, **kwargs)
        datasets.append(dataset)
        return ObservedReader(dataset)

    monkeypatch.setattr(rasterio, "open", observed_open)
    with rasterio.Env(GDAL_CACHEMAX=23 * 1024**2):
        if interrupt:
            with pytest.raises(KeyboardInterrupt):
                BinaryRasterScorer().score(prediction, labels)
        else:
            counts = BinaryRasterScorer().score(prediction, labels)
            assert counts == ConfusionCounts(tn=513 * 777)
            assert len(windows) == 4 * 3 * 4
        assert get_gdal_config("GDAL_CACHEMAX") == 23 * 1024**2
    assert windows and len(datasets) == 2
    assert all(dataset.closed for dataset in datasets)


def test_only_regular_gtiff_files_are_opened(tmp_path: Path) -> None:
    """Missing paths, directories and VRT wrappers cannot enter the scorer."""
    valid = write_raster(
        tmp_path / "valid.tif", np.zeros((2, 3), dtype=np.uint8), nodata=255
    )
    wrapper = tmp_path / "wrapper.tif"
    wrapper.write_text(
        '<VRTDataset rasterXSize="3" rasterYSize="2">'
        '<VRTRasterBand dataType="Byte" band="1"/></VRTDataset>'
    )
    for invalid in (tmp_path / "missing.tif", tmp_path, wrapper, Path("/vsimem/test")):
        with pytest.raises(EvaluationError):
            BinaryRasterScorer().score(invalid, valid)
