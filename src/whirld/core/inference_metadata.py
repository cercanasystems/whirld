"""Shared, stateless acquisition metadata and runtime-estimate transformations."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict

from ..logging_setup import get_logger
from .chips import ChipSet
from .fetch import Manifest

_log = get_logger("core.inference_metadata")

# Rough per-chip CPU wall-clock estimate (seconds) for the runtime warning. A
# conservative order-of-magnitude figure — labeled an estimate, never a guarantee.
_CPU_SECONDS_PER_CHIP = 1.0


def warn_cpu(
    model: str,
    manifest: Manifest,
    device: str,
    chip_count: int,
    no_warnings: bool,
) -> None:
    """Warn when running a real full-precision model on CPU (PRD section 12.2).

    Fires only for real (Hugging Face) full-precision weights on CPU. The numpy
    reference backend and quantized variants are silent, as is ``--no-warnings``.

    Args:
        model: Model identifier.
        manifest: The model's local manifest.
        device: Resolved inference device.
        chip_count: Number of chips to process (drives the estimate).
        no_warnings: Suppress the warning entirely.
    """
    if no_warnings or device != "cpu":
        return
    if manifest.source_type != "huggingface" or manifest.quantized:
        return
    est_s = chip_count * _CPU_SECONDS_PER_CHIP
    est = f"~{est_s:.0f}s" if est_s < 90 else f"~{est_s / 60:.0f} min"
    _log.warning(
        "Running '%s' full-precision on CPU (%d chips); this may be slow "
        "(estimate %s). Pass --device mps/cuda to speed it up if available. "
        "(int8 quantization is not yet available.) Suppress with --no-warnings.",
        model,
        chip_count,
        est,
    )


def chip_latlons(chipset: ChipSet, crs: str) -> list[tuple[float, float]] | None:
    """Reproject each chip's bbox centroid to ``(lat, lon)`` in EPSG:4326.

    Args:
        chipset: The chip set whose chips carry source-CRS bounding boxes.
        crs: The source CRS string.

    Returns:
        Per-chip ``(lat, lon)`` aligned to ``chipset.chips``, or ``None`` if the
        reprojection fails (e.g. an unusable CRS) — callers fall back to zeros.
    """
    try:
        from rasterio.warp import transform as warp_transform  # noqa: PLC0415

        xs = [(c.bbox[0] + c.bbox[2]) / 2.0 for c in chipset.chips]
        ys = [(c.bbox[1] + c.bbox[3]) / 2.0 for c in chipset.chips]
        if not xs:
            return []
        lons, lats = warp_transform(crs, "EPSG:4326", xs, ys)
        return [(float(lat), float(lon)) for lat, lon in zip(lats, lons, strict=True)]
    except Exception as exc:  # unusable CRS, projection error, etc.
        _log.debug("Could not derive chip lat/lon (crs=%s): %s", crs, exc)
        return None


def resolve_datetime(override: str | None, tags: dict[str, str]) -> datetime | None:
    """Resolve a scene acquisition datetime from an override or TIFF tag.

    Precedence: explicit ISO-8601 ``override`` › ``TIFFTAG_DATETIME`` (TIFF or
    STAC ISO format) › ``None``. Naive values are treated as UTC; offsets are
    normalized to UTC before model metadata is encoded.

    Args:
        override: Optional ISO-8601 datetime string.
        tags: Dataset tags (may contain ``TIFFTAG_DATETIME``).

    Returns:
        A parsed :class:`datetime.datetime`, or ``None`` if neither source parses.
    """
    return AcquisitionMetadata(timestamp=tags.get("TIFFTAG_DATETIME")).resolve(override)


class AcquisitionMetadata(BaseModel):
    """Immutable scene acquisition metadata passed from readers to inference.

    Attributes:
        timestamp: Original ISO-8601 or TIFF timestamp, if supplied by the source.
            Keep the original string so invalid source values remain optional and
            an explicit valid override can still take precedence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    timestamp: str | None = None

    def resolve(self, override: str | None = None) -> datetime | None:
        """Resolve acquisition time using the established UTC/fallback policy.

        Args:
            override: Optional ISO-8601 or TIFF timestamp; tried before the source.

        Returns:
            The first parseable value normalized to UTC, or None if both are
            absent or malformed. Naive timestamps are interpreted as UTC.

        Invalid timestamp strings are ignored with a debug log, matching the
        existing eager API. This method does not mutate source metadata.
        """
        for value in (override, self.timestamp):
            if not value:
                continue
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                try:
                    parsed = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    _log.debug("Could not parse acquisition datetime; ignoring.")
                    continue
            # TIFF timestamps carry no zone; consistently interpret them as UTC.
            return (
                parsed.replace(tzinfo=timezone.utc)
                if parsed.tzinfo is None
                else parsed.astimezone(timezone.utc)
            )
        return None
