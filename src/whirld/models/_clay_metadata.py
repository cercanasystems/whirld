"""Clay datacube metadata encoding (time + lat/lon).

These reproduce Clay's **exact** normalization, so the encoder receives the
metadata vectors it was trained with:

* ``normalize_timestamp`` — from ``stacchip.processors.prechip.normalize_timestamp``
* ``normalize_latlon`` — from Clay's ``docs/tutorials/inference.ipynb``

Clay assembles the datacube as
``time = [sin(week), cos(week), sin(hour), cos(hour)]`` and
``latlon = [sin(lat), cos(lat), sin(lon), cos(lon)]`` (``prep_datacube``), with a
single acquisition datetime per scene and a per-chip centroid lat/lon.

Pure math (no torch) so it is cheap and unit-testable in isolation.
"""

from __future__ import annotations

import math
from datetime import datetime


def normalize_timestamp(
    date: datetime,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Encode an acquisition datetime as Clay's ``(week, hour)`` sin/cos pairs.

    Verbatim to ``stacchip``: ``week = isoweek · 2π/52``, ``hour = hour · 2π/24``.

    Args:
        date: Acquisition datetime.

    Returns:
        ``((sin(week), cos(week)), (sin(hour), cos(hour)))``.
    """
    week = date.isocalendar().week * 2 * math.pi / 52
    hour = date.hour * 2 * math.pi / 24
    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


def normalize_latlon(
    lat: float, lon: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Encode a lat/lon (degrees) as Clay's sin/cos pairs.

    Verbatim to Clay's tutorial: radians, then sin/cos of each.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        ``((sin(lat), cos(lat)), (sin(lon), cos(lon)))``.
    """
    lat = lat * math.pi / 180
    lon = lon * math.pi / 180
    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))


def time_vector(date: datetime | None) -> list[float]:
    """Return Clay's 4-element ``time`` vector, or zeros when ``date`` is unknown.

    Args:
        date: Acquisition datetime, or ``None``.

    Returns:
        ``[sin(week), cos(week), sin(hour), cos(hour)]`` (zeros if ``date`` is None).
    """
    if date is None:
        return [0.0, 0.0, 0.0, 0.0]
    (sw, cw), (sh, ch) = normalize_timestamp(date)
    return [sw, cw, sh, ch]


def latlon_vector(latlon: tuple[float, float] | None) -> list[float]:
    """Return Clay's 4-element ``latlon`` vector, or zeros when unknown.

    Args:
        latlon: ``(lat, lon)`` in degrees, or ``None``.

    Returns:
        ``[sin(lat), cos(lat), sin(lon), cos(lon)]`` (zeros if ``latlon`` is None).
    """
    if latlon is None:
        return [0.0, 0.0, 0.0, 0.0]
    (sla, cla), (slo, clo) = normalize_latlon(latlon[0], latlon[1])
    return [sla, cla, slo, clo]
