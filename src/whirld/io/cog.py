"""Direct HTTP(S) GeoTIFF metadata resolution and shared eager/windowed reads."""

from __future__ import annotations

import math
from typing import Literal
from urllib.parse import urlsplit

import rasterio
from pydantic import ValidationError

from ..core.inference_metadata import AcquisitionMetadata
from ..core.options import CogOptions
from ..core.registry import ModelEntry, SensorContract
from ..errors import InvalidInputError, UnsupportedSensorError
from .diagnostics import RasterDiagnostics
from .radiometry import BandRadiometry, RasterBand
from .raster import RasterSource
from .scene import RasterScene
from .stac import _gdal_path
from .stac_auth import StacAccessPolicy, http_origin
from .stac_stream import RasterWindowReader


class CogSceneSource:
    """Own direct-raster metadata interpretation without inventing a STAC item.

    Args:
        url: HTTP(S) raster URL, including any signed query string.
        options: Explicit source-position names and reflectance transform overrides.

    Bare identity TIFF transforms preserve the established canonical-DN convention;
    nonidentity TIFF transforms or explicit overrides declare physical reflectance.
    """

    def __init__(self, url: str, options: CogOptions | None = None) -> None:
        """Bind a validated remote location and typed options without reading it.

        Raises:
            InvalidInputError: URL is not HTTP(S), contains user information or a
                fragment, or cannot be parsed. Diagnostics omit the supplied URL.
        """
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in {"http", "https"} and not parsed.fragment
            if not valid or http_origin(url) is None:
                raise InvalidInputError("Direct COG input requires an HTTP(S) URL.")
        except (ValueError, InvalidInputError):
            raise InvalidInputError(
                "Direct COG input requires an HTTP(S) URL without user information "
                "or a fragment."
            ) from None
        self._url = url
        self._options = options or CogOptions()

    @property
    def kind(self) -> Literal["cog"]:
        """Identify this adapter as a direct COG source."""
        return "cog"

    def resolve(
        self, entry: ModelEntry, sensor: str | None, access: StacAccessPolicy
    ) -> RasterScene:
        """Read only header metadata and select source bands in model order.

        Args:
            entry: Model contract defining required native band names and aliases.
            sensor: Required explicit sensor key, validated against the model.
            access: Origin-scoped policy used even for the initial header open.

        Returns:
            Typed band references, reflectance transforms and acquisition metadata.
            No pixel array is allocated and no whole-file download is performed.

        Raises:
            UnsupportedSensorError: Sensor is absent or unsupported.
            InvalidInputError: Band mapping/transforms are invalid or the GTiff
                header cannot be read. Error chains never expose the signed URL.
        """
        if sensor is None:
            raise UnsupportedSensorError(
                "Direct COG input requires an explicit sensor."
            )
        if sensor not in entry.band_contract.sensors:
            raise UnsupportedSensorError(
                f"Sensor '{sensor}' is not supported by {entry.name}."
            )
        contract = entry.band_contract.sensors[sensor]
        try:
            with (
                RasterDiagnostics(access.token_for(self._url)),
                access.asset_environment(self._url),
                rasterio.open(_gdal_path(self._url), driver="GTiff") as source,
            ):
                self._validate_lengths(source.count)
                names = self._options.bands or tuple(source.descriptions)
                indices = self._indices(names, contract)
                bands = tuple(
                    RasterBand(
                        href=self._url,
                        index=index + 1,
                        radiometry=self._radiometry(
                            index, source.scales[index], source.offsets[index]
                        ),
                    )
                    for index in indices
                )
                acquired = source.tags().get("TIFFTAG_DATETIME")
                return RasterScene(
                    sensor=sensor,
                    bands=bands,
                    band_names=tuple(contract.bands),
                    metadata=AcquisitionMetadata(timestamp=acquired or None),
                    kind="cog",
                )
        except rasterio.errors.RasterioError:
            raise InvalidInputError("Could not read the remote COG header.") from None
        except ValidationError:
            raise InvalidInputError("Invalid remote COG radiometry metadata.") from None

    def _validate_lengths(self, count: int) -> None:
        """Require source-position arrays to describe every stored band.

        Args:
            count: Source header band count.

        Raises:
            InvalidInputError: Names or per-band transform values have another size.
        """
        for name in ("bands", "scale", "offset"):
            value = getattr(self._options, name)
            if isinstance(value, tuple) and len(value) != count:
                raise InvalidInputError(
                    f"COG {name} must contain one value per source band ({count})."
                )

    @staticmethod
    def _indices(names: tuple[str | None, ...], contract: SensorContract) -> list[int]:
        """Resolve native names or aliases without positional spectral guessing.

        Args:
            names: Complete source-order descriptions or explicit replacements.
            contract: Selected model sensor's required native names and aliases.

        Returns:
            Zero-based source indices in canonical model order.

        Raises:
            InvalidInputError: Names are incomplete, duplicated, missing or ambiguous.
        """
        if not names or any(name is None or not name.strip() for name in names):
            raise InvalidInputError(
                "Direct COG bands require complete names; provide explicit bands."
            )
        normalized = [str(name).strip().casefold() for name in names]
        if len(set(normalized)) != len(normalized):
            raise InvalidInputError("Direct COG band names must be unique.")
        selected = []
        for native, alias in zip(contract.bands, contract.aliases, strict=True):
            accepted = {native.casefold(), alias.casefold()}
            matches = [i for i, name in enumerate(normalized) if name in accepted]
            if len(matches) != 1:
                raise InvalidInputError(
                    f"Direct COG must identify exactly one band for '{native}'."
                )
            selected.append(matches[0])
        if len(set(selected)) != len(selected):
            raise InvalidInputError("Direct COG band mapping is ambiguous.")
        return selected

    def _radiometry(self, index: int, scale: float, offset: float) -> BandRadiometry:
        """Combine explicit per-field overrides with TIFF reflectance metadata.

        Args:
            index: Zero-based source band position.
            scale: TIFF scale, including Rasterio's implicit identity default.
            offset: TIFF offset, including Rasterio's implicit zero default.

        Returns:
            An omitted transform for untouched identity defaults, otherwise an
            explicit stored-to-reflectance transform. TIFF masks/nodata remain
            authoritative because no nodata override is synthesized here.

        Raises:
            InvalidInputError: Effective scale/offset is not finite.
        """
        explicit = self._options.scale is not None or self._options.offset is not None
        if self._options.scale is not None:
            value = self._options.scale
            scale = value[index] if isinstance(value, tuple) else value
        if self._options.offset is not None:
            value = self._options.offset
            offset = value[index] if isinstance(value, tuple) else value
        if not math.isfinite(scale) or not math.isfinite(offset):
            raise InvalidInputError("COG scale and offset must be finite numbers.")
        if not explicit and scale == 1 and offset == 0:
            return BandRadiometry()
        return BandRadiometry(scale=float(scale), offset=float(offset))


class CogWindowReader(RasterWindowReader):
    """Read direct remote GeoTIFFs through the existing bounded window engine."""

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
        cog: CogOptions | None = None,
    ) -> None:
        """Bind direct source metadata, credentials and bounded execution settings.

        Args:
            url: HTTP(S) GeoTIFF URL; suffixes are not required at this I/O layer.
            entry: Model spectral and spatial contract.
            chip_size: Positive padded chip dimension in target-grid pixels.
            overlap: Adjacent chip overlap, smaller than chip_size.
            batch_size: Maximum number of prepared chips retained together.
            sensor: Required explicit supported sensor key.
            crs: CRS assignment for sources missing georeferencing identity.
            bbox: Optional longitude/latitude crop in EPSG:4326.
            token: Bearer token scoped to the original URL origin.
            token_origins: Additional explicitly authorized direct origins.
            cache_mb: Per-cache memory limit in MiB; per-band budgets are divided.
            cog: Explicit names or scale/offset overrides; otherwise use TIFF metadata.

        Raises:
            InvalidInputError: URL or execution settings are invalid before reads.
        """
        super().__init__(
            url,
            entry=entry,
            chip_size=chip_size,
            overlap=overlap,
            batch_size=batch_size,
            sensor=sensor,
            crs=crs,
            bbox=bbox,
            token=token,
            token_origins=token_origins,
            cache_mb=cache_mb,
            source=CogSceneSource(url, cog),
        )


def read_remote_cog(
    url: str,
    *,
    entry: ModelEntry,
    sensor: str | None = None,
    crs: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    token: str | None = None,
    token_origins: tuple[str, ...] = (),
    cog: CogOptions | None = None,
) -> RasterSource:
    """Read a direct remote GeoTIFF region for eager embed/classify/segment APIs.

    Args:
        url: HTTP(S) raster URL, including signed query parameters if required.
        entry: Model contract specifying spectral order and output resolution.
        sensor: Required explicit sensor key.
        crs: CRS assignment when the source declares none.
        bbox: Optional EPSG:4326 crop; None reads the full source region.
        token: Bearer token scoped to this URL's origin.
        token_origins: Additional authorized direct origins; redirects strip tokens.
        cog: Source-position names and reflectance transform overrides.

    Returns:
        Resident stored-value pixels on the model grid, with metadata for the
        shared translation layer. This eager path intentionally retains a full
        selected-region array; use CogWindowReader for bounded chip batches.

    Raises:
        WhirldError: Source metadata, authorization, geometry or reading fails.
    """
    with CogWindowReader(
        url,
        entry=entry,
        chip_size=entry.band_contract.chip_size_px,
        sensor=sensor,
        crs=crs,
        bbox=bbox,
        token=token,
        token_origins=token_origins,
        cog=cog,
    ) as reader:
        return reader.read_raster()
