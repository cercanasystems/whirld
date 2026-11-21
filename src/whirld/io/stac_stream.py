"""Windowed STAC chip reads with bounded buffers and explicit resource ownership."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
from types import TracebackType
from typing import Any, Protocol

import numpy as np
import rasterio
from pydantic import BaseModel, ConfigDict
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from ..core.chips import ChipMeta, _chip_bbox
from ..core.contract import translate
from ..core.inference_metadata import AcquisitionMetadata
from ..core.options import StreamingEmbeddingOptions
from ..core.registry import ModelEntry
from ..errors import InvalidInputError
from .radiometry import StacBand
from .stac import (
    StacWarpPolicy,
    _band_source,
    _build_source,
    _fetch_item,
    _gdal_path,
    _Grid,
    _item_id,
    _match_asset,
    _output_grid,
    _resolve_sensor,
)
from .stac_auth import StacAccessPolicy


class ChipBatch(BaseModel):
    """A bounded normalized pixel batch and its global row-major chip metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    pixels: np.ndarray
    chips: list[ChipMeta]


class _ChipLayout(BaseModel):
    """Own a lazy chip layout without allocating scene-sized metadata lists."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    width: int
    height: int
    transform: rasterio.Affine
    chip_size: int
    overlap: int

    @property
    def step(self) -> int:
        """Pixel advance between neighboring chip origins."""
        return self.chip_size - self.overlap

    @property
    def n_rows(self) -> int:
        """Count rows including the padded final row."""
        return (self.height + self.step - 1) // self.step

    @property
    def n_cols(self) -> int:
        """Count columns including the padded final column."""
        return (self.width + self.step - 1) // self.step

    @property
    def count(self) -> int:
        """Total chips computed from dimensions without materializing the layout."""
        return self.n_rows * self.n_cols

    def chips(self) -> Iterator[ChipMeta]:
        """Generate global geometry without retaining completed chip records.

        Yields:
            Row-major chip indices and full padded bounds in the scene CRS.
        """
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                yield ChipMeta(
                    index=row * self.n_cols + col,
                    row=row,
                    col=col,
                    bbox=_chip_bbox(
                        self.transform, row * self.step, col * self.step, self.chip_size
                    ),
                )

    def window(self, chip: ChipMeta) -> Window:
        """Locate the valid pixel region for a chip from this layout.

        Args:
            chip: Metadata yielded by this layout's chips iterator.

        Returns:
            A target-grid window clipped to scene dimensions. Padding is applied
            after reading and normalization, outside this method.
        """
        row, col = chip.row * self.step, chip.col * self.step
        return Window(
            col,
            row,
            min(self.chip_size, self.width - col),
            min(self.chip_size, self.height - row),
        )


class _WarpedRaster(Protocol):
    """Typed subset of Rasterio's warped read interface used by the adapter."""

    def read(self, index: int, *, window: Window, masked: bool) -> np.ma.MaskedArray:
        """Describe the Rasterio operation used at this adapter boundary.

        Args:
            index: One-based source band number.
            window: Target-grid pixel window to read.
            masked: Whether to retain the warped validity mask; callers use True.

        Returns:
            A masked pixel array; Rasterio read errors propagate to the adapter.
        """
        ...


class _BandWindowReader:
    """Own one selected band's source, mask adapter, and full-grid warped view."""

    def __init__(
        self,
        band: StacBand,
        grid: _Grid,
        access: StacAccessPolicy,
        warp_mb: int,
        crs_override: str | None,
    ) -> None:
        """Bind one band without opening datasets or allocating scene pixels.

        Args:
            band: Selected asset, band index, and radiometry/nodata metadata.
            grid: Common destination grid shared by every scene band.
            access: Credential policy used around opens and lazy pixel reads.
            warp_mb: Fixed memory budget for this band's GDAL warp buffers.
            crs_override: Source CRS assignment when the asset declares none.
        """
        self._band = band
        self._grid = grid
        self._access = access
        self._warp_mb = warp_mb
        self._crs_override = crs_override
        self._resources = ExitStack()
        self._vrt: _WarpedRaster | None = None
        self._index = 1

    def __enter__(self) -> _BandWindowReader:
        """Open source and warped datasets under the asset's credential policy.

        Returns:
            This reader, owning all handles until context exit.

        Raises:
            InvalidInputError: Band index, CRS, or streaming grid is unsupported.
            rasterio.errors.RasterioError: GDAL cannot open an asset or warped view.
                Partially acquired resources are closed before propagation.
        """
        try:
            with self._access.asset_environment(self._band.href):
                source = self._resources.enter_context(
                    rasterio.open(_gdal_path(self._band.href), driver="GTiff")
                )
                if self._band.index > source.count:
                    raise InvalidInputError(
                        f"STAC band index {self._band.index} exceeds asset band count "
                        f"{source.count}."
                    )
                options: dict[str, Any] = {}
                if self._band.radiometry.nodata is not None:
                    options["src_nodata"] = self._band.radiometry.nodata_value
                if source.crs is None:
                    if self._crs_override is None:
                        raise InvalidInputError(
                            "STAC asset has no CRS; supply a CRS override."
                        )
                    options["src_crs"] = self._crs_override
                options.update(
                    StacWarpPolicy(source, self._grid, self._crs_override).options(
                        require_stable=True
                    )
                )
                selected, self._index = self._resources.enter_context(
                    _band_source(source, self._band)
                )
                self._vrt = self._resources.enter_context(
                    WarpedVRT(
                        selected,
                        crs=self._grid.crs,
                        transform=self._grid.transform,
                        width=self._grid.width,
                        height=self._grid.height,
                        resampling=Resampling.bilinear,
                        dtype="float32",
                        nodata=float("nan"),
                        warp_mem_limit=self._warp_mb,
                        **options,
                    )
                )
        except BaseException:
            self._resources.close()
            raise
        return self

    def read(self, window: Window) -> np.ndarray:
        """Read a clipped target-grid window under the asset's credential policy.

        Args:
            window: Requested target-grid pixel rectangle.

        Returns:
            A float32 band array with missing/nonfinite observations set to NaN.

        Raises:
            InvalidInputError: The reader is unopened or already closed.
            rasterio.errors.RasterioError: GDAL cannot read the asset window.
        """
        if self._vrt is None:
            raise InvalidInputError("Open the STAC band reader before reading pixels.")
        with self._access.asset_environment(self._band.href):
            values = (
                self._vrt.read(self._index, window=window, masked=True)
                .filled(np.nan)
                .astype(np.float32)
            )
        values[~np.isfinite(values)] = np.nan
        return values

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close warped and source datasets in reverse acquisition order."""
        self._resources.close()
        self._vrt = None


class StacWindowReader:
    """Yield normalized chip batches from a STAC item without whole-scene arrays.

    Args:
        url: Local path, file URL, or HTTP(S) URL of one STAC item.
        entry: Model contract defining band order, normalization, and output grid.
        chip_size: Padded square chip size in target-grid pixels.
        overlap: Pixel overlap; must be smaller than chip_size.
        batch_size: Maximum chips held in each returned batch.
        sensor: Sensor override; otherwise resolved from the item.
        crs: Source CRS override for assets without their own CRS.
        bbox: Optional EPSG:4326 window, matching the ordinary STAC reader.
        token: Bearer credential scoped to authorized origins.
        token_origins: Additional explicitly authorized asset origins.
        cache_mb: MiB budget for each of the GDAL block and curl caches. Per-file
            VSI cache and per-band warp budgets are divided over selected bands,
            with a 1 MiB warp minimum per band. VRT source handles, model/batch
            buffers, and source decoder blocks add bounded working memory.

    Use one consumer inside a with block. The reader owns its GDAL environments
    and datasets, and closes them on normal completion and failures. No full-scene
    pixel, chip, or metadata array is created. The fixed target grid matches the
    eager reader; windows preserve interpolation across chip boundaries.
    """

    def __init__(
        self,
        url: str,
        *,
        entry: ModelEntry,
        chip_size: int,
        overlap: int = 0,
        batch_size: int = 8,
        sensor: str | None = None,
        crs: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        token: str | None = None,
        token_origins: tuple[str, ...] = (),
        cache_mb: int = 32,
    ) -> None:
        """Validate limits and bind the contract/policy without reading pixels."""
        self._options = StreamingEmbeddingOptions.from_arguments(
            chip_size=chip_size,
            overlap=overlap,
            batch_size=batch_size,
            bbox=bbox,
            cache_mb=cache_mb,
        )
        self._url = str(url)
        self._entry = entry
        self._sensor_override = sensor
        self._crs_override = crs
        self._access = StacAccessPolicy(self._url, token, token_origins)
        self._resources: ExitStack | None = None
        self._grid: _Grid | None = None
        self._layout: _ChipLayout | None = None
        self._sensor = ""
        self._metadata = AcquisitionMetadata()
        self._selected: list[StacBand] = []
        self._readers: list[_BandWindowReader] = []

    def __enter__(self) -> StacWindowReader:
        """Resolve item geometry and open credential-scoped warped band readers.

        Returns:
            This reader with scene properties and typed acquisition metadata ready.

        Raises:
            InvalidInputError: Already open, invalid geometry/assets, or GDAL failure.
            NetworkError: The item fetch fails.
            UnsupportedSensorError: No supported sensor or required band resolves.
                All partially opened resources are closed before propagation.
        """
        if self._resources is not None:
            raise InvalidInputError("STAC window reader is already open.")
        self._grid = None
        self._layout = None
        self._metadata = AcquisitionMetadata()
        resources = ExitStack()
        self._resources = resources
        try:
            item = _fetch_item(self._url, self._access.token_for(self._url))
            self._sensor = _resolve_sensor(item, self._entry, self._sensor_override)
            contract = self._entry.band_contract.sensors[self._sensor]
            assets = item.get("assets") or {}
            if not assets:
                raise InvalidInputError(
                    f"STAC item '{_item_id(item)}' declares no assets."
                )
            self._selected = [
                _match_asset(assets, band, alias, item)
                for band, alias in zip(contract.bands, contract.aliases, strict=True)
            ]
            cache_bytes = self._options.cache_mb * 1024 * 1024
            resources.enter_context(
                rasterio.Env(
                    GDAL_CACHEMAX=cache_bytes,
                    VSI_CACHE_SIZE=str(max(1, cache_bytes // len(self._selected))),
                    CPL_VSIL_CURL_CACHE_SIZE=str(cache_bytes),
                )
            )
            with self._access.asset_environment(self._selected[0].href):
                with rasterio.open(
                    _gdal_path(self._selected[0].href), driver="GTiff"
                ) as source:
                    crs = (
                        source.crs.to_string()
                        if source.crs is not None
                        else self._crs_override
                    )
                    if crs is None:
                        raise InvalidInputError(
                            "STAC asset has no CRS; supply a CRS override."
                        )
                    grid = _output_grid(
                        source, crs, self.resolution_m, self._options.bbox
                    )
            self._grid = grid
            assert self._options.chip_size is not None
            self._layout = _ChipLayout(
                width=grid.width,
                height=grid.height,
                transform=grid.transform,
                chip_size=self._options.chip_size,
                overlap=self._options.overlap,
            )
            acquired = (item.get("properties") or {}).get("datetime")
            self._metadata = AcquisitionMetadata(
                timestamp=str(acquired) if acquired else None
            )
            self._readers = [
                resources.enter_context(
                    _BandWindowReader(
                        band,
                        grid,
                        self._access,
                        max(1, self._options.cache_mb // len(self._selected)),
                        self._crs_override,
                    )
                )
                for band in self._selected
            ]
        except BaseException as exc:
            resources.close()
            self._resources = None
            self._readers = []
            if isinstance(exc, rasterio.errors.RasterioError):
                raise InvalidInputError("Could not open STAC raster assets.") from exc
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close all datasets and restore enclosing GDAL settings."""
        if self._resources is not None:
            self._resources.close()
            self._resources = None
        self._readers = []

    def _require_layout(self) -> _ChipLayout:
        """Return resolved geometry or reject access before context entry."""
        if self._layout is None:
            raise InvalidInputError("Enter the STAC window reader before use.")
        return self._layout

    @property
    def sensor(self) -> str:
        """Resolved sensor key after opening."""
        self._require_layout()
        return self._sensor

    @property
    def crs(self) -> str:
        """Common output-grid CRS after opening."""
        self._require_layout()
        assert self._grid is not None
        return self._grid.crs

    @property
    def metadata(self) -> AcquisitionMetadata:
        """Return immutable acquisition metadata after source resolution.

        Raises:
            InvalidInputError: The reader has not resolved its scene yet.
        """
        self._require_layout()
        return self._metadata

    @property
    def chip_count(self) -> int:
        """Total chip count without constructing a scene-sized list."""
        return self._require_layout().count

    @property
    def chip_size(self) -> int:
        """Padded chip width and height in pixels."""
        assert self._options.chip_size is not None
        return self._options.chip_size

    @property
    def resolution_m(self) -> float:
        """Contract target resolution in meters."""
        return float(self._entry.band_contract.target_resolution_m)

    @property
    def n_rows(self) -> int:
        """Number of rows in the global chip layout."""
        return self._require_layout().n_rows

    @property
    def n_cols(self) -> int:
        """Number of columns in the global chip layout."""
        return self._require_layout().n_cols

    def _normalized_window(self, window: Window) -> np.ndarray:
        """Read and normalize one clipped tile through the shared band contract.

        Args:
            window: In-bounds rectangle on the common output grid.

        Returns:
            A bounded float32 band stack before edge padding, with missing values
            replaced by the model contract's normalized nodata fill.

        Raises:
            InvalidInputError: Raster reads or radiometric conversion fail.
            UnsupportedSensorError: Required bands or normalization are incompatible.
        """
        assert self._grid is not None
        bands = np.empty(
            (len(self._selected), int(window.height), int(window.width)),
            dtype=np.float32,
        )
        try:
            for index, reader in enumerate(self._readers):
                bands[index] = reader.read(window)
        except rasterio.errors.RasterioError as exc:
            raise InvalidInputError("Could not read a STAC asset window.") from exc
        grid = _Grid(
            self.crs,
            window_transform(window, self._grid.transform),
            int(window.width),
            int(window.height),
        )
        # The eager source builder remains the single band/radiometry adapter.
        source = _build_source(
            bands,
            grid,
            self._entry.band_contract.sensors[self.sensor].bands,
            self.sensor,
            {},
            self._selected,
        )
        return translate(source, self._entry, self.sensor).data

    def batches(self) -> Iterator[ChipBatch]:
        """Yield normalized float32 batches with global chip coordinates.

        Yields:
            At most batch_size padded chips and their row-major geometry. The
            consumer must release each batch to maintain bounded working memory.

        Raises:
            InvalidInputError: The reader is not open, or a source read fails.
        """
        if self._resources is None:
            raise InvalidInputError("STAC window reader must be open while iterating.")
        layout = self._require_layout()
        chips = layout.chips()
        for start in range(0, layout.count, self._options.batch_size):
            if self._resources is None:
                raise InvalidInputError(
                    "STAC window reader was closed during iteration."
                )
            count = min(self._options.batch_size, layout.count - start)
            pixels = np.full(
                (count, len(self._selected), self.chip_size, self.chip_size),
                self._entry.band_contract.nodata_fill,
                dtype=np.float32,
            )
            metadata = []
            for index in range(count):
                chip = next(chips)
                window = layout.window(chip)
                pixels[index, :, : int(window.height), : int(window.width)] = (
                    self._normalized_window(window)
                )
                metadata.append(chip)
            yield ChipBatch(pixels=pixels, chips=metadata)
            del pixels, metadata
