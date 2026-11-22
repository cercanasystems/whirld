"""Typed CLI adapter for direct COG band and physical-unit declarations."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ...core.options import CogOptions
from ...errors import InvalidInputError


class CogArguments(BaseModel):
    """Own conversion of optional CLI strings into the public COG metadata model."""

    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    bands: str | None = None
    scale: str | None = None
    offset: str | None = None

    @staticmethod
    def _numbers(value: str | None) -> float | tuple[float, ...] | None:
        """Parse a scalar or comma-separated per-band values; reject invalid tokens."""
        if value is None:
            return None
        try:
            numbers = tuple(float(part.strip()) for part in value.split(","))
        except ValueError:
            raise InvalidInputError(
                "COG scale/offset must be numbers separated by commas"
            ) from None
        return numbers[0] if len(numbers) == 1 else numbers

    def options(self) -> CogOptions | None:
        """Return typed metadata only when a COG-specific CLI option was supplied.

        Returns:
            None to preserve automatic routing, or a CogOptions selecting direct
            HTTP(S) input even when its URL path has no TIFF suffix.

        Raises:
            InvalidInputError: Bands are empty/ambiguous or numeric values invalid.
        """
        if not self.enabled and all(
            value is None for value in (self.bands, self.scale, self.offset)
        ):
            return None
        return CogOptions.from_arguments(
            bands=tuple(self.bands.split(",")) if self.bands is not None else None,
            scale=self._numbers(self.scale),
            offset=self._numbers(self.offset),
        )
