"""STAC item input — assemble a :class:`RasterSource` from a STAC item URL.

Whirld accepts a single **STAC item** (one co-registered scene) as an alternative
to a local GeoTIFF (PRD section 9). This module fetches the item JSON, selects only
the band assets the model's contract needs, and reads them with **COG range
requests** (rasterio's ``/vsicurl/`` over the bundled GDAL), aligning every band
onto one grid at the contract's target resolution. The result is an ordinary
:class:`~whirld.io.raster.RasterSource` that feeds the existing
translate -> chip -> infer pipeline unchanged.

Scope: a single item, not a multi-item search/mosaic. The item JSON is fetched with
the standard library (no extra dependency); assets are read through rasterio. A
bearer token (``--stac-token`` / ``WHIRLD_STAC_TOKEN``) is sent on both the item
fetch and the ``/vsicurl/`` asset reads for gated endpoints.

The reader stamps two things into the assembled raster so the rest of the pipeline
behaves identically to the local path: the native band names as band descriptions
(so selection-by-alias matches by name) and the item's acquisition ``datetime`` as
``TIFFTAG_DATETIME`` (so metadata-conditioned models such as Clay get real scene
time for free).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ..errors import InvalidInputError, NetworkError, UnsupportedSensorError
from ..logging_setup import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.registry import ModelEntry
    from .raster import RasterSource

_log = get_logger("io.stac")

# Substring hints (platform / constellation / collection / id) -> our sensor keys,
# checked in order. Used only when the caller does not pass an explicit --sensor.
_PLATFORM_SENSORS: tuple[tuple[str, str], ...] = (
    ("sentinel-2", "sentinel-2-l2a"),
    ("sentinel2", "sentinel-2-l2a"),
    ("landsat-9", "landsat-9-l2"),
    ("landsat-09", "landsat-9-l2"),
    ("landsat-8", "landsat-8-l2"),
    ("landsat-08", "landsat-8-l2"),
    ("landsat", "landsat-9-l2"),
)

# Warn when an unbounded (no --bbox) read would allocate more than this (bytes).
_LARGE_READ_WARN_BYTES = 1_500_000_000

_HTTP_TIMEOUT_S = 30


def read_stac_item(
    url: str,
    *,
    entry: ModelEntry,
    sensor: str | None = None,
    crs: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    token: str | None = None,
) -> RasterSource:
    """Read a STAC item URL into a :class:`RasterSource`.

    Args:
        url: STAC item URL (``https://…/item.json``) or a local item path/``file://``.
        entry: The model entry whose band contract scopes the needed bands + sensors.
        sensor: Explicit sensor key; ``None`` infers it from the item properties.
        crs: CRS to assign if an asset declares none (rare; assets normally carry one).
        bbox: Optional ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326 to window
            the read (the COG range-request fast path); ``None`` reads the full item.
        token: Bearer token for gated endpoints (item fetch + asset reads).

    Returns:
        The assembled raster (bands in the sensor's contract order, native band names
        as descriptions, item datetime carried in ``tags``).

    Raises:
        InvalidInputError: The item cannot be fetched/parsed or has no usable assets.
        UnsupportedSensorError: The sensor cannot be resolved, or a required band has
            no matching asset.
        NetworkError: The item fetch fails for network reasons.
    """
    import numpy as np

    item = _fetch_item(url, token)
    resolved_sensor = _resolve_sensor(item, entry, sensor)
    contract = entry.band_contract.sensors[resolved_sensor]
    target_res = float(entry.band_contract.target_resolution_m)

    assets = item.get("assets") or {}
    if not assets:
        raise InvalidInputError(
            f"STAC item '{_item_id(item)}' declares no assets.\n"
            f"       Whirld needs band assets to read."
        )

    hrefs = [
        _match_asset(assets, band, alias, item)
        for band, alias in zip(contract.bands, contract.aliases, strict=True)
    ]

    with _gdal_env(token):
        import rasterio

        first_path = _gdal_path(hrefs[0])
        with rasterio.open(first_path) as ref:
            out_crs = ref.crs.to_string() if ref.crs is not None else crs
            if out_crs is None:
                raise InvalidInputError(
                    f"STAC asset '{hrefs[0]}' has no CRS and none was supplied.\n"
                    f"       Pass --crs to assign one."
                )
            grid = _output_grid(ref, out_crs, target_res, bbox)

        _warn_if_large(grid, len(hrefs))
        bands = [_read_band(_gdal_path(h), grid) for h in hrefs]

    data = np.stack(bands, axis=0)
    _log.info(
        "Read STAC item '%s': %d bands, %dx%d @ %.1f m, sensor=%s",
        _item_id(item),
        data.shape[0],
        grid.width,
        grid.height,
        target_res,
        resolved_sensor,
    )
    return _build_source(data, grid, contract.bands, resolved_sensor, item)


def _fetch_item(url: str, token: str | None) -> dict[str, Any]:
    """Fetch and parse the STAC item JSON (local path/``file://`` or http(s))."""
    parsed = urlparse(str(url))
    try:
        if parsed.scheme in ("", "file"):
            local = parsed.path if parsed.scheme == "file" else str(url)
            with open(local, encoding="utf-8") as handle:
                raw = handle.read()
        else:
            headers = {"Accept": "application/json"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            request = Request(str(url), headers=headers)
            with urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:
                raw = response.read().decode("utf-8")
    except OSError as exc:
        raise NetworkError(f"Could not fetch STAC item '{url}'.\n       {exc}") from exc

    try:
        item = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidInputError(
            f"STAC item '{url}' is not valid JSON.\n       {exc}"
        ) from exc
    if not isinstance(item, dict) or item.get("type") != "Feature":
        raise InvalidInputError(
            f"'{url}' is not a STAC item (expected a GeoJSON Feature)."
        )
    return item


def _resolve_sensor(
    item: dict[str, Any], entry: ModelEntry, override: str | None
) -> str:
    """Resolve the sensor: explicit override, else infer from item properties."""
    candidates = entry.band_contract.sensors
    supported = ", ".join(sorted(candidates))
    if override is not None:
        if override not in candidates:
            raise UnsupportedSensorError(
                f"Sensor '{override}' is not supported by {entry.name}.\n"
                f"       {entry.name} supports: {supported}"
            )
        return override

    props = item.get("properties") or {}
    haystack = " ".join(
        str(props.get(key, ""))
        for key in ("platform", "constellation", "mission", "collection")
    )
    haystack = f"{haystack} {item.get('collection', '')} {item.get('id', '')}".lower()
    for hint, mapped in _PLATFORM_SENSORS:
        if hint in haystack and mapped in candidates:
            _log.info("Inferred sensor '%s' from STAC item metadata.", mapped)
            return mapped

    raise UnsupportedSensorError(
        f"Could not infer the sensor for STAC item '{_item_id(item)}'.\n"
        f"       Pass --sensor explicitly. {entry.name} supports: {supported}"
    )


def _match_asset(
    assets: dict[str, Any], band: str, alias: str, item: dict[str, Any]
) -> str:
    """Resolve a contract band to an asset href (native id -> eo:bands -> alias)."""
    band_l, alias_l = band.lower(), alias.lower()

    # Tier 1: asset key equals the native band id (e.g. "B04").
    for key, asset in assets.items():
        if key.lower() == band_l and asset.get("href"):
            return str(asset["href"])

    # Tier 2: an asset whose eo:bands names the native band id.
    # Tier 3: asset key or eo:bands common_name equals the spectral alias (e.g. "red").
    alias_match: str | None = None
    for key, asset in assets.items():
        href = asset.get("href")
        if not href:
            continue
        for eo in asset.get("eo:bands", []) or []:
            if str(eo.get("name", "")).lower() == band_l:
                return str(href)
            if str(eo.get("common_name", "")).lower() == alias_l:
                alias_match = alias_match or str(href)
        if key.lower() == alias_l:
            alias_match = alias_match or str(href)
    if alias_match:
        return alias_match

    available = ", ".join(sorted(assets)) or "(none)"
    raise UnsupportedSensorError(
        f"STAC item '{_item_id(item)}' has no asset for band '{band}' "
        f"(alias '{alias}').\n       Available assets: {available}"
    )


class _Grid:
    """A target read grid: output CRS, affine transform, and pixel dimensions."""

    def __init__(self, crs: str, transform: Any, width: int, height: int) -> None:
        self.crs = crs
        self.transform = transform
        self.width = width
        self.height = height


def _output_grid(
    ref: Any,
    out_crs: str,
    target_res: float,
    bbox: tuple[float, float, float, float] | None,
) -> _Grid:
    """Compute the common output grid from the reference asset + optional bbox."""
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    if bbox is not None:
        left, bottom, right, top = transform_bounds("EPSG:4326", out_crs, *bbox)
    else:
        left, bottom, right, top = ref.bounds

    width = max(1, int(round((right - left) / target_res)))
    height = max(1, int(round((top - bottom) / target_res)))
    transform = from_origin(left, top, target_res, target_res)
    return _Grid(crs=out_crs, transform=transform, width=width, height=height)


def _read_band(gdal_path: str, grid: _Grid) -> Any:
    """Read one asset onto the common grid via a WarpedVRT (range-request reads)."""
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    try:
        with (
            rasterio.open(gdal_path) as src,
            WarpedVRT(
                src,
                crs=grid.crs,
                transform=grid.transform,
                width=grid.width,
                height=grid.height,
                resampling=Resampling.bilinear,
            ) as vrt,
        ):
            return vrt.read(1).astype(np.float32)
    except rasterio.errors.RasterioError as exc:
        raise InvalidInputError(
            f"Could not read STAC asset '{gdal_path}'.\n"
            f"       {exc}\n"
            f"       Ensure it is a readable (ideally cloud-optimized) GeoTIFF."
        ) from exc


def _build_source(
    data: Any, grid: _Grid, bands: list[str], sensor: str, item: dict[str, Any]
) -> RasterSource:
    """Wrap the stacked array + metadata in a :class:`RasterSource`."""
    from .raster import RasterSource

    tags: dict[str, str] = {"TIFFTAG_IMAGEDESCRIPTION": f"{sensor} (STAC)"}
    acquired = (item.get("properties") or {}).get("datetime")
    if acquired:
        tags["TIFFTAG_DATETIME"] = str(acquired)

    return RasterSource(
        data=data,
        crs=grid.crs,
        transform=grid.transform,
        band_descriptions=list(bands),
        tags=tags,
        nodata=None,
    )


@contextmanager
def _gdal_env(token: str | None) -> Iterator[None]:
    """Configure GDAL for efficient ``/vsicurl/`` reads (and bearer auth)."""
    import rasterio

    options: dict[str, Any] = {
        # Don't list the bucket/dir on open — slow and often 403 on object stores.
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "VSI_CACHE": True,
    }
    if token:
        options["GDAL_HTTP_HEADERS"] = f"Authorization: Bearer {token}"
    with rasterio.Env(**options):
        yield


def _gdal_path(href: str) -> str:
    """Translate a STAC asset href into a GDAL-readable path."""
    parsed = urlparse(href)
    if parsed.scheme in ("http", "https"):
        return f"/vsicurl/{href}"
    if parsed.scheme == "s3":
        return f"/vsis3/{parsed.netloc}{parsed.path}"
    if parsed.scheme == "file":
        return parsed.path
    return href


def _item_id(item: dict[str, Any]) -> str:
    """Best-effort human label for an item in messages."""
    return str(item.get("id", "<unknown>"))


def _warn_if_large(grid: _Grid, band_count: int) -> None:
    """Warn when an unbounded read would allocate a very large array."""
    nbytes = grid.width * grid.height * band_count * 4
    if nbytes > _LARGE_READ_WARN_BYTES:
        _log.warning(
            "STAC read is large (%.1f GB for %d bands at %dx%d). "
            "Pass --bbox to window the read.",
            nbytes / 1e9,
            band_count,
            grid.width,
            grid.height,
        )
