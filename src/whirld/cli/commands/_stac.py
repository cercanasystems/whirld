"""Shared CLI helpers for STAC item inputs (``--bbox`` / ``--stac-token``)."""

from __future__ import annotations

from ...config import get_stac_token
from ...errors import InvalidInputError


def parse_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    """Parse a ``--bbox`` string ``"min_lon,min_lat,max_lon,max_lat"`` (EPSG:4326).

    Args:
        value: The raw option string, or ``None``.

    Returns:
        The four floats as a tuple, or ``None`` when unset.

    Raises:
        InvalidInputError: The value is not four comma-separated numbers, or the
            bounds are degenerate.
    """
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise InvalidInputError(
            "--bbox must be 'min_lon,min_lat,max_lon,max_lat' (EPSG:4326)."
        )
    try:
        min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
    except ValueError as exc:
        raise InvalidInputError(f"--bbox values must be numbers: {exc}") from exc
    if min_lon >= max_lon or min_lat >= max_lat:
        raise InvalidInputError(
            "--bbox must have min_lon < max_lon and min_lat < max_lat."
        )
    return (min_lon, min_lat, max_lon, max_lat)


def resolve_stac_token(token: str | None) -> str | None:
    """Return the explicit ``--stac-token`` or fall back to ``WHIRLD_STAC_TOKEN``."""
    return token if token else get_stac_token()
