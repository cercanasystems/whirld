"""Bounded windowed segmentation output with owned atomic GeoTIFF publication."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import TracebackType
from typing import Literal

import numpy as np
import rasterio
from pydantic import BaseModel, ConfigDict, Field, field_validator
from rasterio.windows import Window

from ..core.chips import ChipMeta
from ..errors import InvalidInputError, WhirldError
from ..logging_setup import get_logger

_log = get_logger("io.segmentation_stream")


class SegmentationMetadata(BaseModel):
    """Immutable native-grid mask contract, excluding nodata from valid classes.

    Invalid dimensions, class counts, CRS or non-north-up/nonfinite affine values
    raise Pydantic validation errors before any file is opened. Transform values
    use Affine order (a, b, c, d, e, f), not GDAL geotransform order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    crs: str
    transform: tuple[float, float, float, float, float, float]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    chip_size_px: int = Field(gt=0)
    classes: int = Field(ge=1, le=255)
    nodata: Literal[255] = 255

    @field_validator("crs")
    @classmethod
    def valid_crs(cls, value: str) -> str:
        """Require a parseable target CRS without changing its declared identity."""
        try:
            rasterio.crs.CRS.from_user_input(value)
        except (rasterio.errors.CRSError, ValueError) as exc:
            raise ValueError("segmentation CRS must be valid") from exc
        return value

    @field_validator("transform")
    @classmethod
    def valid_transform(
        cls, value: tuple[float, float, float, float, float, float]
    ) -> tuple[float, float, float, float, float, float]:
        """Require finite north-up geometry for exact native-window assembly."""
        a, b, _, d, e, _ = value
        if not all(math.isfinite(part) for part in value) or not (
            a > 0 and e < 0 and b == 0 and d == 0
        ):
            raise ValueError("segmentation transform must be finite and north-up")
        return value


class SegmentationStreamWriter:
    """Own a staged mask dataset and publish only a complete native-grid output.

    Args:
        output_path: Exact GeoTIFF destination filename, replaced after success.
        chip_count: Total expected row-major, non-overlapping native-size chips.
        metadata: Typed dimensions, affine, CRS, class and nodata contract.

    Only the supplied mask batch, one cropped tile and a fixed 8 MiB GDAL cache
    are resident. Invalid source cells become 255; valid zero remains class zero.
    Partial edge tiles are cropped to exact output dimensions. Normal errors and
    interrupts clean temporary files and preserve an existing destination. Call
    finish explicitly before leaving a successful context. Concurrent writers must
    use different destinations; no power-loss durability transaction is promised.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        chip_count: int,
        metadata: SegmentationMetadata,
    ) -> None:
        """Validate total coverage and bind the destination without creating files.

        Raises:
            InvalidInputError: chip_count does not match native grid coverage.
        """
        self._metadata = metadata
        self._n_cols = (
            metadata.width + metadata.chip_size_px - 1
        ) // metadata.chip_size_px
        n_rows = (metadata.height + metadata.chip_size_px - 1) // metadata.chip_size_px
        if (
            isinstance(chip_count, bool)
            or not isinstance(chip_count, int)
            or chip_count != self._n_cols * n_rows
        ):
            raise InvalidInputError(
                "Segmentation chip_count must cover the native grid."
            )
        self._output_path = Path(output_path).absolute()
        self._chip_count = chip_count
        self._written = 0
        self._valid_count = 0
        self._positive_count = 0
        self._entered = False
        self._finished = False
        self._stage: Path | None = None
        self._dataset: rasterio.io.DatasetWriter | None = None
        self._resources = ExitStack()

    @property
    def output_path(self) -> Path:
        """Return the exact bound destination, without adding a filename suffix."""
        return self._output_path

    @property
    def valid_pixel_count(self) -> int:
        """Return valid source pixels after successful publication.

        Raises:
            WhirldError: No completed output has been published.
        """
        if not self._finished:
            raise WhirldError("Segmentation output has not been finished.")
        return self._valid_count

    @property
    def positive_pixel_count(self) -> int:
        """Return valid pixels with a nonzero class after successful publication.

        Raises:
            WhirldError: No completed output has been published.
        """
        if not self._finished:
            raise WhirldError("Segmentation output has not been finished.")
        return self._positive_count

    def __enter__(self) -> SegmentationStreamWriter:
        """Create the owned temporary GeoTIFF and fixed-cache environment.

        Returns:
            This writer ready for bounded mask batches.

        Raises:
            WhirldError: Writer was reused or its staged dataset cannot open.
        """
        if self._entered:
            raise WhirldError("A segmentation stream writer cannot be reused.")
        self._entered = True
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._stage = Path(
                tempfile.mkdtemp(prefix=".whirld-segment-", dir=self.output_path.parent)
            )
            self._resources.enter_context(
                rasterio.Env(GDAL_CACHEMAX=8 * 1024 * 1024, GDAL_PAM_ENABLED=False)
            )
            meta = self._metadata
            self._dataset = rasterio.open(
                self._stage / "mask.tif",
                "w",
                driver="GTiff",
                width=meta.width,
                height=meta.height,
                count=1,
                dtype="uint8",
                crs=meta.crs,
                transform=rasterio.Affine(*meta.transform),
                nodata=meta.nodata,
                tiled=True,
                blockxsize=256,
                blockysize=256,
                compress="LZW",
                BIGTIFF="IF_SAFER",
            )
        except BaseException as exc:
            self._cleanup()
            if isinstance(exc, (OSError, rasterio.errors.RasterioError)):
                raise WhirldError(
                    "Could not open streaming segmentation output."
                ) from exc
            raise
        return self

    def _active(self) -> rasterio.io.DatasetWriter:
        """Require an active unfinished dataset before writing or publication.

        Raises:
            WhirldError: Writer is unopened, closed or already finished.
        """
        if self._dataset is None or self._finished:
            raise WhirldError("Segmentation stream writer is not open for writing.")
        return self._dataset

    def _validate_batch(
        self, masks: np.ndarray, chips: list[ChipMeta], valid_masks: np.ndarray
    ) -> None:
        """Validate the complete batch before changing any staged raster window.

        Args:
            masks: Integer class predictions shaped (batch, native_tile, native_tile).
            chips: Consecutive row-major chip geometry matching the native grid.
            valid_masks: Boolean source-validity cells with the same mask shape.

        Raises:
            InvalidInputError: Shape, values, validity, ordering or geometry is invalid.
        """
        tile = self._metadata.chip_size_px
        shape = (len(chips), tile, tile)
        if masks.shape != shape or masks.dtype.kind not in "iu":
            raise InvalidInputError(
                "Segmentation masks must be integer native-size tiles."
            )
        if valid_masks.shape != shape or valid_masks.dtype.kind != "b":
            raise InvalidInputError(
                "Segmentation validity must be boolean native-size tiles."
            )
        if np.any(masks < 0) or np.any(masks >= self._metadata.classes):
            raise InvalidInputError(
                "Segmentation predictions contain an invalid class."
            )
        if self._written + len(chips) > self._chip_count:
            raise InvalidInputError("Segmentation exceeds the declared chip count.")
        transform = rasterio.Affine(*self._metadata.transform)
        for offset, chip in enumerate(chips):
            index = self._written + offset
            row, col = divmod(index, self._n_cols)
            if (chip.index, chip.row, chip.col) != (index, row, col):
                raise InvalidInputError(
                    "Segmentation chips must follow the native grid order."
                )
            x0, y0 = transform * (col * tile, row * tile)
            x1, y1 = transform * ((col + 1) * tile, (row + 1) * tile)
            expected = (x0, y1, x1, y0)
            if len(chip.bbox) != 4 or any(
                not math.isfinite(value)
                or not math.isclose(value, wanted, rel_tol=1e-10, abs_tol=1e-7)
                for value, wanted in zip(chip.bbox, expected, strict=True)
            ):
                raise InvalidInputError(
                    "Segmentation chip geometry must match the native grid."
                )

    def write(
        self, masks: np.ndarray, chips: list[ChipMeta], valid_masks: np.ndarray
    ) -> None:
        """Write cropped tile windows with source-invalid cells set to nodata255.

        Args:
            masks: Bounded batch of model class indices on the padded native grid.
            chips: Consecutive global chip geometry records for the mask rows.
            valid_masks: Boolean validity before normalized padding and model inference.

        Raises:
            InvalidInputError: Batch values or geometry violate the mask contract.
            WhirldError: The writer is inactive or a staged write fails.
        """
        dataset = self._active()
        self._validate_batch(masks, chips, valid_masks)
        tile = self._metadata.chip_size_px
        try:
            for mask, chip, valid in zip(masks, chips, valid_masks, strict=True):
                row, col = chip.row * tile, chip.col * tile
                height = min(tile, self._metadata.height - row)
                width = min(tile, self._metadata.width - col)
                keep = valid[:height, :width]
                cropped = np.where(keep, mask[:height, :width], self._metadata.nodata)
                dataset.write(
                    cropped.astype(np.uint8, copy=False),
                    1,
                    window=Window(col, row, width, height),
                )
                self._valid_count += int(np.count_nonzero(keep))
                self._positive_count += int(
                    np.count_nonzero(keep & (mask[:height, :width] > 0))
                )
                self._written += 1
        except (OSError, rasterio.errors.RasterioError) as exc:
            raise WhirldError("Could not write streaming segmentation output.") from exc

    def finish(self) -> Path:
        """Close, flush and atomically publish the complete single GeoTIFF.

        Returns:
            Published mask path. Repeated successful calls return the same path.

        Raises:
            InvalidInputError: Fewer than the expected chips have been written.
            WhirldError: Dataset closing, flushing or replacement fails.
        """
        if self._finished:
            return self.output_path
        self._active()
        if self._written != self._chip_count:
            raise InvalidInputError(
                f"Segmentation output incomplete: {self._written} of "
                f"{self._chip_count} chips written."
            )
        assert self._stage is not None
        staged = self._stage / "mask.tif"
        try:
            self._close_dataset()
            with staged.open("rb") as source:
                os.fsync(source.fileno())
            staged.replace(self.output_path)
        except (OSError, rasterio.errors.RasterioError) as exc:
            raise WhirldError(
                "Could not finalize streaming segmentation output."
            ) from exc
        self._finished = True
        self._cleanup()
        _log.info("Wrote %s", self.output_path.name)
        return self.output_path

    def _close_dataset(self) -> None:
        """Close the owned raster handle and clear its active writer reference.

        Raises:
            RasterioError: Dataset closing or pending block flushing fails.
        """
        dataset = self._dataset
        self._dataset = None
        if dataset is not None:
            dataset.close()

    def _cleanup(self) -> None:
        """Release owned raster/environment resources without masking active errors."""
        try:
            self._close_dataset()
        except (OSError, rasterio.errors.RasterioError):
            _log.debug("Could not close segmentation output.", exc_info=True)
        try:
            self._resources.close()
        except (OSError, rasterio.errors.RasterioError):
            _log.debug("Could not close segmentation environment.", exc_info=True)
        if self._stage is not None:
            shutil.rmtree(self._stage, ignore_errors=True)
            self._stage = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Discard unfinished files and preserve active exceptions.

        Args:
            exc_type: Exception class leaving the context or None on normal exit.
            exc: Active exception instance, never suppressed.
            traceback: Original traceback, never suppressed.

        Raises:
            WhirldError: A normal context exit omitted successful publication.
        """
        self._cleanup()
        if exc_type is None and not self._finished:
            raise WhirldError("Segmentation stream was not finished; output discarded.")
