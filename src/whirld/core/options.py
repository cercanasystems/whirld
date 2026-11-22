"""Validated inference options shared by Python, CLI, and REST boundaries."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Annotated, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ..errors import InvalidInputError

OptionsT = TypeVar("OptionsT", bound="RequestOptions")


class RequestOptions(BaseModel):
    """Base for option schemas that own public argument validation and errors."""

    @classmethod
    def from_arguments(cls: type[OptionsT], **values: object) -> OptionsT:
        """Construct typed options, mapping schema failures to InvalidInputError."""
        try:
            return cls.model_validate(values)
        except ValidationError as exc:
            messages = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'options'}: "
                f"{error['msg']}"
                for error in exc.errors()
            )
            raise InvalidInputError(messages) from exc


FiniteValue = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class CogOptions(RequestOptions):
    """Explicit source-position metadata for one remote multiband GeoTIFF.

    ``bands`` overrides TIFF descriptions and must name every source position.
    Scale/offset values may be scalar or per-source-band tuples. Explicit values
    override TIFF transforms; explicit scale 1 declares reflectance, while absent
    identity TIFF values retain canonical DN. Counts are checked against the
    opened TIFF. Constructing invalid values raises Pydantic ValidationError;
    from_arguments maps those failures to InvalidInputError for CLI adapters.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    bands: tuple[str, ...] | None = None
    scale: FiniteValue | tuple[FiniteValue, ...] | None = None
    offset: FiniteValue | tuple[FiniteValue, ...] | None = None

    @field_validator("bands")
    @classmethod
    def validate_bands(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        """Normalize names; reject empty or duplicate source-position declarations."""
        if value is None:
            return None
        names = tuple(name.strip() for name in value)
        if not names or any(not name for name in names):
            raise ValueError("bands must contain nonempty source-position names")
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("bands must have unique names")
        return names

    @field_validator("scale", "offset")
    @classmethod
    def validate_values(
        cls, value: float | tuple[float, ...] | None
    ) -> float | tuple[float, ...] | None:
        """Reject empty per-band metadata tuples before any remote I/O."""
        if isinstance(value, tuple) and not value:
            raise ValueError("per-band values must not be empty")
        return value


class InferenceOptions(RequestOptions):
    """Immutable per-call execution policy, independent of a resident backend.

    An absent batch size inherits the backend's policy. Explicit sizes must be
    positive integers. This object owns batch partitioning so every backend uses
    the same override semantics without changing cached model state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    batch_size: int | None = Field(default=None, gt=0)

    def batches(
        self, count: int, *, fallback: InferenceOptions | None = None
    ) -> Iterator[slice]:
        """Yield ordered, nonoverlapping slices covering all input chips.

        Args:
            count: Number of chips to partition; zero yields no batches.
            fallback: Backend defaults used only when this call has no override.
        """
        size = self.batch_size
        if size is None and fallback is not None:
            size = fallback.batch_size
        size = size if size is not None else max(1, count)
        for start in range(0, count, size):
            yield slice(start, min(start + size, count))


class SpatialOptions(RequestOptions):
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


class StreamingEmbeddingOptions(EmbeddingOptions):
    """Bounded STAC embedding options; the output is always a disk NPY array."""

    batch_size: int = Field(default=8, gt=0, strict=True)
    cache_mb: int = Field(default=32, gt=0, strict=True)
    fmt: Literal["npy"] = "npy"


class ClassificationOptions(SpatialOptions):
    """Probability threshold and positive number of returned chip scores."""

    top_k: int = Field(default=5, gt=0, strict=True)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)


class SegmentationOptions(SpatialOptions):
    """Finite probability threshold for binary segmentation."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
