"""Validated per-band raster radiometry, kept separate from model normalization."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class BandRadiometry(BaseModel):
    """Describe conversion from stored pixels to physical reflectance.

    ``reflectance = stored * scale + offset``. An omitted transform preserves
    legacy canonical-DN input; an explicit scale or offset activates conversion,
    with defaults of one and zero for its missing companion. Nodata is expressed
    in stored pixel units and must be excluded before interpolation/conversion.
    Invalid/nonfinite numeric transforms raise Pydantic validation errors.
    """

    scale: float = Field(default=1.0, strict=True, allow_inf_nan=False)
    offset: float = Field(default=0.0, strict=True, allow_inf_nan=False)
    nodata: (
        Annotated[float, Field(strict=True, allow_inf_nan=False)]
        | Literal["nan", "inf", "-inf"]
        | None
    ) = None

    @field_validator("nodata", mode="before")
    @classmethod
    def _validate_nodata(cls, value: object) -> object:
        """Reject explicit null/bool nodata; omission uses the TIFF mask instead."""
        if value is None or isinstance(value, bool):
            raise ValueError("nodata must be numeric or 'nan', 'inf', '-inf'")
        return value

    @property
    def has_transform(self) -> bool:
        """Whether metadata explicitly declares either part of the affine transform."""
        return bool({"scale", "offset"} & self.model_fields_set)

    @property
    def nodata_value(self) -> float | None:
        """Return a GDAL-compatible nodata value, including special float values."""
        return float(self.nodata) if self.nodata is not None else None


class RasterBand(BaseModel):
    """One selected raster band and its radiometry, aligned to the model contract."""

    href: str = Field(repr=False)
    index: int = Field(default=1, ge=1)
    radiometry: BandRadiometry = Field(default_factory=BandRadiometry)


# Preserve existing STAC imports while the shared reader uses a source-neutral name.
StacBand = RasterBand
