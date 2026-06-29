"""Chipping a translated raster into fixed-size tiles and tracking geometry.

Inference runs on fixed-size square chips. This module tiles a translated band
stack into ``chip_size`` tiles (optionally overlapping), pads edge tiles to full
size with the contract's nodata fill (PRD section 7.2 step 8), and records each
chip's georeferenced bounding box so outputs can be written back in the source
CRS.
"""

from __future__ import annotations

import numpy as np
import rasterio
from pydantic import BaseModel

from ..errors import WhirldError
from ..logging_setup import get_logger
from .contract import TranslatedRaster

_log = get_logger("core.chips")


class ChipMeta(BaseModel):
    """Georeferencing metadata for a single chip.

    Attributes:
        index: Zero-based chip index in row-major order.
        row: Chip row in the tile grid.
        col: Chip column in the tile grid.
        bbox: Bounding box ``[minx, miny, maxx, maxy]`` in the source CRS.
    """

    index: int
    row: int
    col: int
    bbox: list[float]


class ChipSet(BaseModel):
    """A batch of chips plus per-chip geometry.

    The pixel data is intentionally kept out of this Pydantic model (it is a
    large numpy array); callers receive it alongside via :func:`chip_raster`.

    Attributes:
        chips: Per-chip georeferencing metadata.
        chip_size_px: Tile edge length in pixels.
        n_rows: Number of tile rows.
        n_cols: Number of tile columns.
    """

    chips: list[ChipMeta]
    chip_size_px: int
    n_rows: int
    n_cols: int


def chip_raster(
    translated: TranslatedRaster,
    chip_size: int,
    *,
    overlap: int = 0,
    nodata_fill: float = 0.0,
) -> tuple[np.ndarray, ChipSet]:
    """Tile a translated raster into chips with georeferenced metadata.

    Args:
        translated: The normalized band stack to chip.
        chip_size: Tile edge length in pixels.
        overlap: Overlap between adjacent tiles in pixels.
        nodata_fill: Fill value for padding partial edge tiles.

    Returns:
        A tuple ``(array, chipset)`` where ``array`` has shape
        ``(n_chips, bands, chip_size, chip_size)`` (float32) and ``chipset``
        carries per-chip geometry.

    Raises:
        WhirldError: ``chip_size`` is non-positive or ``overlap`` is invalid.
    """
    if chip_size <= 0:
        raise WhirldError(f"chip_size must be positive, got {chip_size}.")
    if overlap < 0 or overlap >= chip_size:
        raise WhirldError(
            f"overlap must satisfy 0 <= overlap < chip_size; got {overlap} "
            f"for chip_size {chip_size}."
        )

    bands, height, width = translated.data.shape
    step = chip_size - overlap
    row_starts = list(range(0, max(height, 1), step))
    col_starts = list(range(0, max(width, 1), step))

    transform = translated.transform
    arrays: list[np.ndarray] = []
    metas: list[ChipMeta] = []
    index = 0
    for r, row0 in enumerate(row_starts):
        for c, col0 in enumerate(col_starts):
            tile = np.full(
                (bands, chip_size, chip_size),
                fill_value=nodata_fill,
                dtype=np.float32,
            )
            row1 = min(row0 + chip_size, height)
            col1 = min(col0 + chip_size, width)
            tile[:, : row1 - row0, : col1 - col0] = translated.data[
                :, row0:row1, col0:col1
            ]
            arrays.append(tile)
            metas.append(
                ChipMeta(
                    index=index,
                    row=r,
                    col=c,
                    bbox=_chip_bbox(transform, row0, col0, chip_size),
                )
            )
            index += 1

    stacked = (
        np.stack(arrays, axis=0)
        if arrays
        else np.empty((0, bands, chip_size, chip_size), dtype=np.float32)
    )
    chipset = ChipSet(
        chips=metas,
        chip_size_px=chip_size,
        n_rows=len(row_starts),
        n_cols=len(col_starts),
    )
    _log.debug(
        "Chipped into %d tiles (%dx%d grid).",
        len(metas),
        len(row_starts),
        len(col_starts),
    )
    return stacked, chipset


def _chip_bbox(
    transform: rasterio.Affine,
    row0: int,
    col0: int,
    chip_size: int,
) -> list[float]:
    """Compute a chip's bounding box in CRS coordinates.

    Args:
        transform: Affine transform of the translated raster.
        row0: Top pixel row of the chip.
        col0: Left pixel column of the chip.
        chip_size: Tile edge length in pixels.

    Returns:
        ``[minx, miny, maxx, maxy]`` in the source CRS.
    """
    x0, y0 = transform * (col0, row0)
    x1, y1 = transform * (col0 + chip_size, row0 + chip_size)
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def reassemble_mask(
    masks: np.ndarray,
    chipset: ChipSet,
    height: int,
    width: int,
) -> np.ndarray:
    """Stitch per-chip masks back into a full-extent mask (inverse of chipping).

    Each chip's ``(chip_size, chip_size)`` mask is placed at its pixel offset and
    edge padding is cropped to the original ``(height, width)``. Assumes
    non-overlapping tiles (segmentation chips the scene with ``overlap=0``).

    Args:
        masks: Per-chip masks aligned to ``chipset.chips``, shape
            ``(n_chips, chip_size, chip_size)``.
        chipset: The chip layout produced by :func:`chip_raster`.
        height: Target (pre-padding) raster height in pixels.
        width: Target raster width in pixels.

    Returns:
        The reassembled mask, shape ``(height, width)``, same dtype as ``masks``.

    Raises:
        WhirldError: ``masks`` count does not match ``chipset``.
    """
    if masks.shape[0] != len(chipset.chips):
        raise WhirldError(
            f"reassemble_mask: {masks.shape[0]} masks for "
            f"{len(chipset.chips)} chips."
        )
    tile = chipset.chip_size_px
    out = np.zeros((height, width), dtype=masks.dtype)
    for chip, mask in zip(chipset.chips, masks, strict=True):
        row0, col0 = chip.row * tile, chip.col * tile
        row1, col1 = min(row0 + tile, height), min(col0 + tile, width)
        out[row0:row1, col0:col1] = mask[: row1 - row0, : col1 - col0]
    return out
