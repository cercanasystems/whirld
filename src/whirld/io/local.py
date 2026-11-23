"""Local-only metadata selection and bounded windows with eager raster semantics."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Literal
from xml.etree.ElementTree import Element, SubElement, tostring

import rasterio
from rasterio.errors import RasterioError

from ..core.contract import _select_band_indices
from ..core.inference_metadata import AcquisitionMetadata
from ..core.registry import ModelEntry
from ..core.sensor import detect_sensor
from ..errors import InvalidInputError
from .radiometry import RasterBand
from .raster import RasterHeader, validate_metric_grid
from .scene import RasterScene
from .stac import _band_source, _Grid
from .stac_auth import StacAccessPolicy
from .stac_stream import RasterWindowReader, _BandWindowReader


class LocalRasterAccessPolicy(StacAccessPolicy):
    """Keep headers isolated while permitting local mask/PAM sidecar discovery."""

    @contextmanager
    def asset_environment(self, href: str) -> Iterator[None]:
        """Scope file reads to local sidecar-aware GDAL settings.

        Args:
            href: Bound absolute filesystem location selected by the local adapter.

        Yields:
            A credential-cleared environment that restores caller options on exit.
            Local .msk/.aux.xml files remain visible as with eager read_raster.
        """
        with (
            super().asset_environment(href),
            rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="FALSE"),
        ):
            yield


class LocalSceneSource:
    """Own local file identity and header-only sensor/band resolution.

    Local TIFF scale/offset defaults and nonidentity values are intentionally not
    interpreted as radiometry, matching the existing local eager reader. Source
    nodata and masks are honored later by the shared selected-band window reader.
    """

    def __init__(self, path: str | Path, crs: str | None = None) -> None:
        """Bind an absolute existing file before GDAL interprets any input string.

        Args:
            path: Filesystem path, already normalized from any file URI by routing.
            crs: Assignment used only when the dataset does not declare a CRS.

        Raises:
            InvalidInputError: Path is missing, not a regular file or invalid.
                GDAL virtual filesystem and subdataset strings are never opened.
        """
        try:
            bound = Path(path).resolve(strict=True)
            if not bound.is_file():
                raise InvalidInputError("Local streaming input must be a regular file.")
        except (OSError, ValueError, RuntimeError):
            raise InvalidInputError(
                "Local streaming input must be an existing file."
            ) from None
        self._path = bound
        self._crs_override = crs

    @property
    def path(self) -> Path:
        """Return the bound absolute file identity, stable across cwd changes."""
        return self._path

    @property
    def kind(self) -> Literal["geotiff"]:
        """Identify the adapter as an ordinary local GeoTIFF source."""
        return "geotiff"

    def resolve(
        self, entry: ModelEntry, sensor: str | None, access: StacAccessPolicy
    ) -> RasterScene:
        """Inspect a GTiff header and apply unchanged local sensor/band algorithms.

        Args:
            entry: Model contract supplying sensor, native names and normalization.
            sensor: Optional explicit sensor; otherwise use eager detection rules.
            access: Local sidecar-aware policy guarding this metadata open.

        Returns:
            Selected model-order band references with unset radiometry transforms.
            No source pixels are read or represented by placeholder arrays.

        Raises:
            InvalidInputError: File is not GTiff, lacks CRS, or cannot be opened.
            UnsupportedSensorError: Sensor identity or required bands are invalid.
        """
        try:
            with (
                access.asset_environment(str(self._path)),
                rasterio.open(self._path, driver="GTiff") as source,
            ):
                crs = source.crs.to_string() if source.crs else self._crs_override
                if crs is None:
                    raise InvalidInputError(
                        "Local raster has no CRS; supply a CRS override."
                    )
                header = RasterHeader(
                    width=source.width,
                    height=source.height,
                    crs=crs,
                    transform=source.transform,
                    band_descriptions=tuple(source.descriptions),
                    tags=dict(source.tags()),
                )
                resolved = detect_sensor(header, entry, sensor)
                contract = entry.band_contract.sensors[resolved]
                selected = _select_band_indices(header, contract.bands)
                acquired = header.tags.get("TIFFTAG_DATETIME")
                return RasterScene(
                    sensor=resolved,
                    band_names=tuple(contract.bands),
                    kind="geotiff",
                    bands=tuple(
                        RasterBand(href=str(self._path), index=index + 1)
                        for index in selected
                    ),
                    metadata=AcquisitionMetadata(timestamp=acquired or None),
                )
        except RasterioError:
            raise InvalidInputError(
                "Could not open local streaming input as GeoTIFF."
            ) from None


class _LocalBandWindowReader(_BandWindowReader):
    """Exclude nonfinite stored samples lazily before GDAL interpolation."""

    def _warp_options(self, source: rasterio.DatasetReader) -> dict[str, float]:
        """Match eager full-grid sampling scales when ceil padding clips edges.

        Args:
            source: GTiff header with full native source dimensions.

        Returns:
            GDAL sampling scales derived from the full destination dimensions,
            clamped to one for bilinear upsampling. Eager reproject derives its
            downsampling kernel from these edge-clipped full-scene dimensions,
            rather than the physical resolution ratio used by STAC. Fixing them
            once prevents per-window kernel estimates and interpolation seams.
        """
        return {
            "XSCALE": min(1.0, self._grid.width / source.width),
            "YSCALE": min(1.0, self._grid.height / source.height),
        }

    @contextmanager
    def _band_source(self, source: rasterio.DatasetReader) -> Iterator[tuple[Any, int]]:
        """Layer small VRT descriptors over selected data without copying pixels.

        Args:
            source: Open GTiff whose masks and nodata use eager local semantics.

        Yields:
            A float32 selected-band dataset with nodata and both infinities
            converted to NaN before bilinear interpolation. Descriptor handles
            close in reverse order when the surrounding band reader exits.
        """
        with ExitStack() as resources:
            selected, index = resources.enter_context(_band_source(source, self._band))
            for excluded in (
                selected.nodatavals[index - 1],
                float("inf"),
                -float("inf"),
            ):
                if excluded is None:
                    continue
                root = Element(
                    "VRTDataset",
                    rasterXSize=str(source.width),
                    rasterYSize=str(source.height),
                )
                if source.crs:
                    SubElement(root, "SRS").text = source.crs.to_wkt()
                SubElement(root, "GeoTransform").text = ",".join(
                    str(value) for value in source.transform.to_gdal()
                )
                target = SubElement(root, "VRTRasterBand", dataType="Float32", band="1")
                SubElement(target, "NoDataValue").text = "nan"
                reference = SubElement(target, "ComplexSource")
                SubElement(reference, "SourceFilename", relativeToVRT="0").text = (
                    selected.name
                )
                SubElement(reference, "SourceBand").text = str(index)
                SubElement(reference, "NODATA").text = str(excluded)
                memory = resources.enter_context(rasterio.MemoryFile(tostring(root)))
                selected = resources.enter_context(memory.open())
                index = 1
            yield selected, index


class LocalWindowReader(RasterWindowReader):
    """Reuse owned windows with local eager sensor, sidecar and grid behavior."""

    def __init__(
        self,
        path: str | Path,
        *,
        entry: ModelEntry,
        chip_size: int,
        overlap: int = 0,
        batch_size: int = 8,
        sensor: str | None = None,
        crs: str | None = None,
        cache_mb: int = 32,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Bind local input and bounded execution without reading scene pixels.

        Args:
            path: Existing filesystem file; the top-level driver must be GTiff.
            entry: Model spectral/spatial contract.
            chip_size: Positive padded chip dimension.
            overlap: Neighbor overlap smaller than chip_size.
            batch_size: Maximum prepared chips retained together.
            sensor: Optional explicit sensor, following local eager precedence.
            crs: CRS assignment when metadata has none.
            cache_mb: Fixed per-cache raster memory budget in MiB.
            bbox: Unsupported for local streaming; crop first or use a STAC item.

        Raises:
            InvalidInputError: Path, crop request or execution settings are invalid.
        """
        if bbox is not None:
            raise InvalidInputError(
                "Local streaming does not support bbox; crop first or use STAC."
            )
        source = LocalSceneSource(path, crs)
        super().__init__(
            str(source.path),
            entry=entry,
            chip_size=chip_size,
            overlap=overlap,
            batch_size=batch_size,
            sensor=sensor,
            crs=crs,
            cache_mb=cache_mb,
            source=source,
            access=LocalRasterAccessPolicy(str(source.path)),
        )

    def _band_reader(self, band: RasterBand, grid: _Grid) -> _BandWindowReader:
        """Own local pre-warp finite filtering under the shared memory budget.

        Args:
            band: Selected local band and unset local radiometry transforms.
            grid: Complete destination grid, shared by every chip.

        Returns:
            An unopened resource owner that sanitizes samples before warping.
        """
        return _LocalBandWindowReader(
            band,
            grid,
            self._access,
            max(1, self._options.cache_mb // len(self._selected)),
            self._crs_override,
        )

    def _target_grid(self, source: rasterio.DatasetReader, crs: str) -> _Grid:
        """Match eager tolerance, affine preservation and ceil-based resampling.

        Args:
            source: Open local reference header under the local access policy.
            crs: Declared source CRS or explicit missing-CRS assignment.

        Returns:
            Original grid within eager's 0.5% per-axis tolerance, otherwise a
            target-resolution grid covering the full source extent.

        Raises:
            InvalidInputError: Grid is not north-up/projected in meters.
        """
        source_x, source_y = validate_metric_grid(crs, source.transform)
        target = self.resolution_m
        if all(abs(size - target) <= 0.005 * target for size in (source_x, source_y)):
            return _Grid(crs, source.transform, source.width, source.height)
        transform = rasterio.Affine(
            target,
            source.transform.b,
            source.transform.c,
            source.transform.d,
            -target,
            source.transform.f,
        )
        return _Grid(
            crs,
            transform,
            max(1, math.ceil(source.width * source_x / target)),
            max(1, math.ceil(source.height * source_y / target)),
        )
