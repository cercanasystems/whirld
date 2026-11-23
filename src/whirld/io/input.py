"""Typed input routing shared by public eager and streaming entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field

from ..core.options import CogOptions
from ..core.registry import ModelEntry
from ..errors import InvalidInputError
from .raster import RasterSource, read_raster
from .stac_auth import http_origin


class RasterInput(BaseModel):
    """Resolved input kind and optional COG semantics; keep signed URLs out of repr."""

    model_config = ConfigDict(frozen=True)
    location: str = Field(repr=False)
    kind: Literal["geotiff", "stac", "cog"]
    cog: CogOptions | None = None

    @classmethod
    def resolve(cls, value: str | Path, cog: CogOptions | None = None) -> RasterInput:
        """Classify locations without fetching data or discarding signed queries.

        Args:
            value: Local file, STAC item or remote GeoTIFF URL.
            cog: Explicit COG metadata, also selecting extensionless remote inputs.

        Returns:
            Validated routing identity. Only direct COG URLs require HTTP(S).

        Raises:
            InvalidInputError: URL is malformed or COG options target a local/JSON
                input. Errors never interpolate a potentially credentialed URL.
        """
        if cog is not None and not isinstance(cog, CogOptions):
            raise InvalidInputError("cog must be a CogOptions instance")
        location = str(value)
        try:
            parsed = urlsplit(location)
        except ValueError:
            raise InvalidInputError("Malformed input URL") from None
        remote = parsed.scheme in {"http", "https"}
        suffix = unquote(parsed.path).lower()
        direct = remote and suffix.endswith((".tif", ".tiff"))
        if cog is not None:
            if not remote or suffix.endswith(".json"):
                raise InvalidInputError(
                    "COG options require a direct HTTP(S) GeoTIFF, not local/STAC JSON"
                )
            direct = True
        if parsed.scheme == "file" and suffix.endswith((".tif", ".tiff")):
            if (
                parsed.netloc.lower() not in {"", "localhost"}
                or parsed.query
                or parsed.fragment
                or not parsed.path.startswith("/")
                or "\x00" in unquote(parsed.path)
            ):
                raise InvalidInputError(
                    "Local TIFF file URLs require an absolute path, empty/localhost "
                    "authority and no query or fragment"
                )
            return cls(location=unquote(parsed.path), kind="geotiff")
        if direct:
            http_origin(location)
            if parsed.fragment:
                raise InvalidInputError("Direct COG URLs must not contain a fragment")
            return cls(location=location, kind="cog", cog=cog or CogOptions())
        if remote or parsed.scheme == "file" or location.endswith(".json"):
            return cls(location=location, kind="stac")
        return cls(location=location, kind="geotiff")

    def validate_sensor(self, sensor: str | None) -> None:
        """Reject omitted sensor identity for direct COGs before model/source I/O."""
        if self.kind == "cog" and not sensor:
            raise InvalidInputError("Direct COG input requires an explicit sensor")

    def validate_streaming(
        self, bbox: tuple[float, float, float, float] | None = None
    ) -> None:
        """Reject local streaming options that have no eager-local equivalent.

        Args:
            bbox: Optional read crop. Local eager input ignores this argument;
                initial local streaming rejects it to avoid silent differences.

        Raises:
            InvalidInputError: A local raster is combined with a bbox. Crop it
                beforehand or use a STAC item when geographic windowing is needed.
        """
        if self.kind == "geotiff" and bbox is not None:
            raise InvalidInputError(
                "Local raster streaming does not accept bbox; crop the input "
                "first or reference it from a STAC item."
            )

    def read(
        self,
        *,
        entry: ModelEntry,
        crs: str | None,
        sensor: str | None,
        bbox: tuple[float, float, float, float] | None,
        token: str | None,
    ) -> RasterSource:
        """Read an eager source through the selected adapter and common model grid.

        Args:
            entry: Validated model contract for band/unit selection.
            crs: Missing-CRS assignment; does not reproject unsupported source grids.
            sensor: Explicit sensor for COG, optional inference for STAC/local input.
            bbox: Geographic read window for STAC/COG inputs.
            token: Optional bearer credential scoped to the original remote origin.

        Returns:
            Resident source pixels and metadata for the existing eager pipeline.

        Raises:
            WhirldError: Invalid input metadata, failed reads or authorization.
        """
        self.validate_sensor(sensor)
        if self.kind == "cog":
            from .cog import read_remote_cog

            return read_remote_cog(
                self.location,
                entry=entry,
                crs=crs,
                sensor=sensor,
                bbox=bbox,
                token=token,
                cog=self.cog,
            )
        if self.kind == "stac":
            from .stac import read_stac_item

            return read_stac_item(
                self.location,
                entry=entry,
                crs=crs,
                sensor=sensor,
                bbox=bbox,
                token=token,
            )
        return read_raster(self.location, crs=crs)
