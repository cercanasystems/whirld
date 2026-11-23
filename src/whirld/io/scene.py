"""Typed source resolution shared by STAC and direct-raster window readers."""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..core.inference_metadata import AcquisitionMetadata
from ..core.registry import ModelEntry
from ..errors import InvalidInputError
from .radiometry import RasterBand
from .raster import RasterSource
from .stac import _fetch_item, _Grid, _item_id, _match_asset, _resolve_sensor
from .stac_auth import StacAccessPolicy


class RasterScene(BaseModel):
    """Resolved bands in model order, source identity and acquisition metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    sensor: str
    bands: tuple[RasterBand, ...] = Field(min_length=1)
    band_names: tuple[str, ...] = Field(min_length=1)
    metadata: AcquisitionMetadata = Field(default_factory=AcquisitionMetadata)
    kind: Literal["stac", "cog", "geotiff"]

    def as_raster(self, pixels: np.ndarray, grid: _Grid) -> RasterSource:
        """Adapt a stored-value window to the existing normalization boundary.

        Args:
            pixels: Float32 band stack in this scene's resolved model order.
            grid: CRS/transform and dimensions for the supplied pixel window.

        Returns:
            RasterSource with spectral names, radiometry and acquisition tags.
        """
        tags = {"TIFFTAG_IMAGEDESCRIPTION": f"{self.sensor} ({self.kind.upper()})"}
        if self.metadata.timestamp:
            tags["TIFFTAG_DATETIME"] = self.metadata.timestamp
        return RasterSource(
            data=pixels,
            crs=grid.crs,
            transform=grid.transform,
            band_descriptions=list(self.band_names),
            tags=tags,
            nodata=float("nan"),
            band_radiometry=[band.radiometry for band in self.bands],
        )


class RasterSceneSource(Protocol):
    """Resolve source metadata without materializing scene pixels."""

    @property
    def kind(self) -> Literal["stac", "cog", "geotiff"]:
        """Identify the source for diagnostic messages and usage metadata."""
        ...

    def resolve(
        self, entry: ModelEntry, sensor: str | None, access: StacAccessPolicy
    ) -> RasterScene:
        """Resolve selected bands and metadata under the supplied credential policy.

        Args:
            entry: Model contract defining supported sensors and required bands.
            sensor: Explicit sensor selection or source-specific inference request.
            access: Policy authorizing source opens and metadata fetches.

        Returns:
            Typed selected source metadata without scene-sized pixel arrays.

        Raises:
            WhirldError: Source metadata, sensor or required bands are invalid.
        """
        ...


class StacSceneSource:
    """Own STAC document interpretation at the external JSON adapter boundary."""

    def __init__(self, url: str) -> None:
        """Bind one item URL/path without fetching it."""
        self._url = url

    @property
    def kind(self) -> Literal["stac"]:
        """Identify this adapter as a STAC source."""
        return "stac"

    def resolve(
        self, entry: ModelEntry, sensor: str | None, access: StacAccessPolicy
    ) -> RasterScene:
        """Resolve item asset/radiometry and datetime using established STAC rules.

        Args:
            entry: Model spectral contract.
            sensor: Sensor override; None allows existing item-based detection.
            access: Policy authorizing item retrieval.

        Returns:
            Selected assets in contract order and immutable acquisition metadata.

        Raises:
            InvalidInputError: No assets are declared or metadata is invalid.
            NetworkError: Item retrieval fails.
            UnsupportedSensorError: Sensor or a required band cannot be resolved.
        """
        item = _fetch_item(self._url, access.token_for(self._url))
        resolved_sensor = _resolve_sensor(item, entry, sensor)
        contract = entry.band_contract.sensors[resolved_sensor]
        assets = item.get("assets") or {}
        if not assets:
            raise InvalidInputError(f"STAC item '{_item_id(item)}' declares no assets.")
        bands = tuple(
            _match_asset(assets, band, alias, item)
            for band, alias in zip(contract.bands, contract.aliases, strict=True)
        )
        acquired = (item.get("properties") or {}).get("datetime")
        return RasterScene(
            sensor=resolved_sensor,
            bands=bands,
            band_names=tuple(contract.bands),
            metadata=AcquisitionMetadata(timestamp=str(acquired) if acquired else None),
            kind="stac",
        )
