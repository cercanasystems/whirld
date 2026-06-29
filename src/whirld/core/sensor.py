"""Sensor detection from raster metadata (PRD section 7.1).

Whirld determines which sensor produced a raster so it can apply the correct
band contract. Detection follows a strict precedence:

1. An explicit user override (``--sensor``) — always wins.
2. TIFF tags (``TIFFTAG_IMAGEDESCRIPTION`` / ``TIFFTAG_SOFTWARE``) naming a sensor.
3. Band descriptions matching a sensor's declared native band identifiers.
4. Ground resolution matching a sensor's native resolution.

If none resolve, an :class:`~whirld.errors.UnsupportedSensorError` is raised with
the model's supported sensors and a suggestion to pass ``--sensor``.

Detection is scoped to the candidate sensors declared in the model's band
contract, so "supported by this model" and "detectable" are the same set.
"""

from __future__ import annotations

from ..errors import UnsupportedSensorError
from ..io.raster import RasterSource
from ..logging_setup import get_logger
from .registry import ModelEntry

_log = get_logger("core.sensor")

_RESOLUTION_TOLERANCE = 0.05  # 5% relative tolerance for resolution matching


def _search_tokens(sensor_key: str) -> set[str]:
    """Derive lowercase tokens that identify a sensor in free-text metadata.

    For ``sentinel-2-l2a`` this yields ``{"sentinel-2-l2a", "sentinel-2",
    "sentinel2"}``; for ``landsat-8-l2`` it yields ``{"landsat-8-l2",
    "landsat-8", "landsat8"}``.

    Args:
        sensor_key: Registry sensor identifier.

    Returns:
        A set of normalized tokens to look for in tags.
    """
    key = sensor_key.lower()
    parts = key.split("-")
    tokens = {key}
    if len(parts) >= 2:
        family = f"{parts[0]}-{parts[1]}"
        tokens.add(family)
        tokens.add(family.replace("-", ""))
    return tokens


def detect_sensor(
    raster: RasterSource,
    entry: ModelEntry,
    override: str | None = None,
) -> str:
    """Detect the sensor that produced ``raster`` for the given model.

    Args:
        raster: The loaded input raster.
        entry: The model whose band contract scopes the candidate sensors.
        override: Explicit sensor key supplied by the user; wins if valid.

    Returns:
        The detected sensor key (a key of ``entry.band_contract.sensors``).

    Raises:
        UnsupportedSensorError: The override is unsupported, or detection fails.
    """
    candidates = entry.band_contract.sensors
    supported = ", ".join(sorted(candidates))

    if override is not None:
        if override not in candidates:
            raise UnsupportedSensorError(
                f"Sensor '{override}' is not supported by {entry.name}.\n"
                f"       {entry.name} supports: {supported}"
            )
        _log.info("Sensor '%s' set explicitly via override.", override)
        return override

    detected = (
        _detect_from_tags(raster, candidates)
        or _detect_from_band_descriptions(raster, candidates)
        or _detect_from_resolution(raster, candidates)
    )
    if detected is None:
        raise UnsupportedSensorError(
            f"Could not determine the sensor for this input.\n"
            f"       {entry.name} supports: {supported}\n"
            f"       If your file is one of these sensors with non-standard "
            f"metadata,\n"
            f"       use --sensor to specify it explicitly."
        )
    _log.info("Detected sensor '%s'.", detected)
    return detected


def _detect_from_tags(
    raster: RasterSource, candidates: dict[str, object]
) -> str | None:
    """Match a sensor by scanning IMAGEDESCRIPTION/SOFTWARE TIFF tags."""
    haystack = " ".join(
        str(raster.tags.get(tag, ""))
        for tag in ("TIFFTAG_IMAGEDESCRIPTION", "TIFFTAG_SOFTWARE")
    ).lower()
    if not haystack.strip():
        return None
    for sensor_key in candidates:
        if any(token in haystack for token in _search_tokens(sensor_key)):
            return sensor_key
    return None


def _detect_from_band_descriptions(
    raster: RasterSource, candidates: dict[str, object]
) -> str | None:
    """Match a sensor when the raster's band descriptions equal its native bands.

    Requires an exact set match between the raster's (case-insensitive) band
    descriptions and a sensor's declared ``bands`` so that, e.g., a Sentinel-2
    scene is not mistaken for Landsat.
    """
    present = {d.strip().lower() for d in raster.band_descriptions if d}
    if not present:
        return None
    for sensor_key, contract in candidates.items():
        native = {b.lower() for b in contract.bands}  # type: ignore[attr-defined]
        if native.issubset(present):
            return sensor_key
    return None


def _detect_from_resolution(
    raster: RasterSource, candidates: dict[str, object]
) -> str | None:
    """Match a sensor by native ground resolution within a small tolerance.

    Only resolves when exactly one candidate matches, to avoid ambiguity between
    sensors that share a resolution (e.g. both Landsat 8 and 9 at 30 m).
    """
    res_x, _ = raster.resolution_m
    matches = []
    for sensor_key, contract in candidates.items():
        native = float(contract.native_resolution_m)  # type: ignore[attr-defined]
        if abs(res_x - native) <= _RESOLUTION_TOLERANCE * native:
            matches.append(sensor_key)
    if len(matches) == 1:
        return matches[0]
    return None
