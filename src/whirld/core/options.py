"""Validated inference options shared by Python, CLI, and REST boundaries."""

from __future__ import annotations

import math
from typing import Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ..errors import InvalidInputError


class SpatialOptions(BaseModel):
    """Validate a finite, ordered EPSG:4326 read window when supplied."""

    model_config = ConfigDict(extra="forbid", strict=True)
    bbox: tuple[float, float, float, float] | None = None

    @field_validator("bbox")
    @classmethod
    def valid_bbox(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        """Reject nonfinite, inverted, or out-of-range geographic windows."""
        if value is not None:
            west, south, east, north = value
            if not all(math.isfinite(v) for v in value) or not (
                -180 <= west < east <= 180 and -90 <= south < north <= 90
            ):
                raise ValueError(
                    "bbox must be finite ordered longitude/latitude bounds"
                )
        return value


class EmbeddingOptions(SpatialOptions):
    """Embedding domains; overlap is checked against resolved chip size as well."""

    chip_size: int | None = Field(default=None, gt=0, strict=True)
    overlap: int = Field(default=0, ge=0, strict=True)
    batch_size: int | None = Field(default=None, gt=0, strict=True)
    fmt: Literal["npy", "json"] = "npy"

    @model_validator(mode="after")
    def valid_overlap(self) -> EmbeddingOptions:
        """Ensure adjacent chips advance instead of repeating or stepping backward."""
        if self.chip_size is not None and self.overlap >= self.chip_size:
            raise ValueError("overlap must be less than chip_size")
        return self


class ClassificationOptions(SpatialOptions):
    """Probability threshold and positive number of returned chip scores."""

    top_k: int = Field(default=5, gt=0, strict=True)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)


class SegmentationOptions(SpatialOptions):
    """Finite probability threshold for binary segmentation."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)


OptionsT = TypeVar("OptionsT", bound=BaseModel)


def validate_options(schema: type[OptionsT], **values: object) -> OptionsT:
    """Return validated options, mapping schema errors to a public domain error."""
    try:
        return schema.model_validate(values)
    except ValidationError as exc:
        messages = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'options'}: "
            f"{error['msg']}"
            for error in exc.errors()
        )
        raise InvalidInputError(messages) from exc
