"""Exact-grid binary raster scores accumulated in bounded windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioError
from rasterio.windows import Window

from ..errors import InvalidInputError
from ..io.raster import validate_metric_grid
from .errors import EvaluationError
from .schemas import ConfusionCounts


class _BinaryCountsAccumulator:
    """Own only integer totals while consuming independent raster windows."""

    def __init__(self) -> None:
        """Initialize zero counts without retaining raster samples."""
        self.counts = ConfusionCounts()

    def add(
        self,
        prediction: np.ndarray,
        labels: np.ndarray,
        prediction_mask: np.ndarray,
        label_mask: np.ndarray,
    ) -> None:
        """Validate one window's classes and add its observed counts.

        Args:
            prediction: Integer binary predictions or nodata 255.
            labels: Integer labels with -1 ignored and 0/1 valid.
            prediction_mask: GDAL validity values; zero is a missing prediction.
            label_mask: GDAL validity values; zero is an ignored observation.

        Raises:
            EvaluationError: An unmasked prediction or label has an unknown class.
        """
        prediction_valid = (prediction_mask != 0) & (prediction != 255)
        label_valid = (label_mask != 0) & (labels != -1)
        if np.any(prediction_valid & ~np.isin(prediction, (0, 1))):
            raise EvaluationError("Prediction contains an unexpected unmasked class.")
        if np.any(label_valid & ~np.isin(labels, (0, 1))):
            raise EvaluationError("Labels contain an unexpected unmasked class.")
        scored = label_valid & prediction_valid
        positive = labels == 1
        predicted_positive = prediction == 1
        current = ConfusionCounts(
            tp=int(np.count_nonzero(scored & positive & predicted_positive)),
            fp=int(np.count_nonzero(scored & ~positive & predicted_positive)),
            tn=int(np.count_nonzero(scored & ~positive & ~predicted_positive)),
            fn=int(np.count_nonzero(scored & positive & ~predicted_positive)),
            label_ignored=int(np.count_nonzero(~label_valid)),
            prediction_invalid_positive=int(
                np.count_nonzero(label_valid & positive & ~prediction_valid)
            ),
            prediction_invalid_negative=int(
                np.count_nonzero(label_valid & ~positive & ~prediction_valid)
            ),
        )
        self.counts = ConfusionCounts.combine((self.counts, current))


class BinaryRasterScorer:
    """Compare binary prediction and label TIFFs without resizing or full reads.

    The scored domain contains valid 0/1 labels and available 0/1 predictions.
    Missing predictions over valid labels are retained as separate positive and
    negative counts, so conditional scores can always be interpreted with coverage.
    Explicit TIFF masks/nodata and label -1 values identify ignored observations.
    """

    def __init__(self, window_size: int = 256) -> None:
        """Bind a bounded square read dimension without opening files.

        Args:
            window_size: Maximum window width/height in pixels, from 1 through 256.

        Raises:
            EvaluationError: Size is not a strict integer in the supported range.
        """
        if (
            isinstance(window_size, bool)
            or not isinstance(window_size, int)
            or not 1 <= window_size <= 256
        ):
            raise EvaluationError(
                "Evaluation window_size must be an integer in [1, 256]."
            )
        self._window_size = window_size

    @staticmethod
    def _local_file(path: Path) -> Path:
        """Bind a regular local file before GDAL interprets its location.

        Args:
            path: Existing local prediction or label artifact.

        Returns:
            Absolute regular-file path; GDAL virtual/subdataset strings fail first.

        Raises:
            EvaluationError: Path does not resolve to an existing regular file.
        """
        try:
            bound = path.resolve(strict=True)
            if not bound.is_file():
                raise EvaluationError("Evaluation raster must be a regular local file.")
            return bound
        except (OSError, ValueError, RuntimeError):
            raise EvaluationError(
                "Evaluation raster must be an existing regular local file."
            ) from None

    @staticmethod
    def _validate_grids(
        prediction: rasterio.DatasetReader, labels: rasterio.DatasetReader
    ) -> None:
        """Require binary storage and identical projected north-up pixel identities.

        Args:
            prediction: Open uint8 single-band mask with nodata 255.
            labels: Open single-band integer labels with the same exact grid.

        Raises:
            EvaluationError: Storage, CRS, geometry or grid alignment is invalid.
        """
        if prediction.count != 1 or labels.count != 1:
            raise EvaluationError("Evaluation rasters must each contain a single band.")
        if np.dtype(prediction.dtypes[0]) != np.dtype("uint8"):
            raise EvaluationError("Prediction dtype must be uint8.")
        if prediction.nodata != 255:
            raise EvaluationError("Prediction nodata must be 255.")
        if np.dtype(labels.dtypes[0]).kind not in "iu":
            raise EvaluationError("Labels must use an integer dtype.")
        if prediction.crs is None or labels.crs is None:
            raise EvaluationError("Evaluation rasters must each declare a CRS.")
        try:
            validate_metric_grid(prediction.crs.to_string(), prediction.transform)
            validate_metric_grid(labels.crs.to_string(), labels.transform)
        except InvalidInputError as exc:
            raise EvaluationError(f"Invalid evaluation grid: {exc}") from exc
        if (
            prediction.width != labels.width
            or prediction.height != labels.height
            or prediction.crs != labels.crs
            or prediction.transform != labels.transform
        ):
            raise EvaluationError(
                "Prediction and label dimensions, CRS and affine must match exactly."
            )

    def score(self, prediction: Path, label: Path) -> ConfusionCounts:
        """Accumulate counts over aligned GTiff artifacts in fixed-size windows.

        Args:
            prediction: Binary uint8 GTiff whose missing pixels use nodata 255.
            label: Integer GTiff with 0/1 labels and optional ignored -1 values.

        Returns:
            Integer confusion, ignored-label and missing-prediction totals.

        Raises:
            EvaluationError: Artifacts, geometry, classes or evaluation domain are
                invalid. No label resampling or silent empty-domain score occurs.
            KeyboardInterrupt: Propagates after datasets and GDAL settings close.
        """
        prediction_path = self._local_file(prediction)
        label_path = self._local_file(label)
        try:
            with (
                rasterio.Env(
                    GDAL_CACHEMAX=8 * 1024**2,
                    GDAL_DISABLE_READDIR_ON_OPEN="FALSE",
                ),
                rasterio.open(prediction_path, driver="GTiff") as predicted,
                rasterio.open(label_path, driver="GTiff") as labels,
            ):
                self._validate_grids(predicted, labels)
                accumulator = _BinaryCountsAccumulator()
                for row in range(0, predicted.height, self._window_size):
                    for column in range(0, predicted.width, self._window_size):
                        window = Window(
                            column,
                            row,
                            min(self._window_size, predicted.width - column),
                            min(self._window_size, predicted.height - row),
                        )
                        accumulator.add(
                            predicted.read(1, window=window),
                            labels.read(1, window=window),
                            predicted.read_masks(1, window=window),
                            labels.read_masks(1, window=window),
                        )
                counts = accumulator.counts
                if counts.label_valid_pixels == 0:
                    raise EvaluationError("Evaluation case contains no valid labels.")
                if counts.scored_pixels == 0:
                    raise EvaluationError(
                        "Evaluation case contains no valid predictions."
                    )
                return counts
        except (OSError, RasterioError) as exc:
            raise EvaluationError(
                "Could not read evaluation rasters as GeoTIFF."
            ) from exc
