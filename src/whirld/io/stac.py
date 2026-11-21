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
bearer token (``--stac-token`` / ``WHIRLD_STAC_TOKEN``) is scoped to the original
item origin. Additional asset origins require explicit ``token_origins`` consent;
asset redirects do not receive authorization.

The reader stamps two things into the assembled raster so the rest of the pipeline
behaves identically to the local path: the native band names as band descriptions
(so selection-by-alias matches by name) and the item's acquisition ``datetime`` as
``TIFFTAG_DATETIME`` (so metadata-conditioned models such as Clay get real scene
time for free).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from xml.etree.ElementTree import Element, SubElement, tostring

from pydantic import ValidationError

from ..errors import InvalidInputError, NetworkError, UnsupportedSensorError
from ..logging_setup import get_logger
from .radiometry import BandRadiometry, StacBand
from .stac_auth import StacAccessPolicy, http_origin

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


class _OriginRedirectHandler(HTTPRedirectHandler):
    """Preserve bearer authorization only across same-origin item redirects."""

    def redirect_request(
        self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Request | None:
        """Remove credentials before urllib follows a cross-origin redirect."""
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and http_origin(req.full_url) != http_origin(newurl):
            redirected.remove_header("Authorization")
        return redirected


def read_stac_item(
    url: str,
    *,
    entry: ModelEntry,
    sensor: str | None = None,
    crs: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    token: str | None = None,
    token_origins: tuple[str, ...] = (),
) -> RasterSource:
    """Read a STAC item URL into a :class:`RasterSource`.

    Args:
        url: STAC item URL (``https://…/item.json``) or a local item path/``file://``.
        entry: The model entry whose band contract scopes the needed bands + sensors.
        sensor: Explicit sensor key; ``None`` infers it from the item properties.
        crs: CRS to assign if an asset declares none (rare; assets normally carry one).
        bbox: Optional ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326 to window
            the read (the COG range-request fast path); ``None`` reads the full item.
        token: Bearer token scoped to the original HTTP(S) item origin.
        token_origins: Additional HTTP(S) origins explicitly authorized to receive
            this token on direct asset reads. Asset redirects never carry it.

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

    access = StacAccessPolicy(url, token, token_origins)
    item = _fetch_item(url, access.token_for(url))
    resolved_sensor = _resolve_sensor(item, entry, sensor)
    contract = entry.band_contract.sensors[resolved_sensor]
    target_res = float(entry.band_contract.target_resolution_m)

    assets = item.get("assets") or {}
    if not assets:
        raise InvalidInputError(
            f"STAC item '{_item_id(item)}' declares no assets.\n"
            f"       Whirld needs band assets to read."
        )

    selected = [
        _match_asset(assets, band, alias, item)
        for band, alias in zip(contract.bands, contract.aliases, strict=True)
    ]

    import rasterio

    with access.asset_environment(selected[0].href):
        first_path = _gdal_path(selected[0].href)
        with rasterio.open(first_path, driver="GTiff") as ref:
            out_crs = ref.crs.to_string() if ref.crs is not None else crs
            if out_crs is None:
                raise InvalidInputError(
                    f"STAC asset '{selected[0].href}' has no CRS.\n"
                    f"       Pass --crs to assign one."
                )
            grid = _output_grid(ref, out_crs, target_res, bbox)

    _warn_if_large(grid, len(selected))
    bands = []
    for band in selected:
        with access.asset_environment(band.href):
            bands.append(_read_band(band, grid))

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
    return _build_source(data, grid, contract.bands, resolved_sensor, item, selected)


def _fetch_item(url: str, token: str | None) -> dict[str, Any]:
    """Fetch and parse the STAC item JSON (local path/``file://`` or http(s))."""
    parsed = urlparse(str(url))
    try:
        if parsed.scheme in ("", "file"):
            local = unquote(parsed.path) if parsed.scheme == "file" else str(url)
            with open(local, encoding="utf-8") as handle:
                raw = handle.read()
        elif parsed.scheme in ("http", "https"):
            http_origin(url)
            headers = {"Accept": "application/json"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            request = Request(str(url), headers=headers)
            with build_opener(_OriginRedirectHandler()).open(
                request, timeout=_HTTP_TIMEOUT_S
            ) as response:
                raw = response.read().decode("utf-8")
        else:
            raise InvalidInputError(
                "STAC items must use HTTP(S), file URLs, or local paths."
            )
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
) -> StacBand:
    """Resolve a native band/alias to an asset, band index, and typed radiometry.

    Native asset keys take precedence, then band names, then spectral aliases.
    Multiband assets require a matching band name/common name to avoid silently
    reading band one. Malformed selected metadata raises InvalidInputError.
    """
    band_l, alias_l = band.lower(), alias.lower()

    # Tier 1: asset key equals the native band id (e.g. "B04").
    for key, asset in assets.items():
        if key.lower() == band_l and asset.get("href"):
            descriptions = _band_descriptions(asset)
            for index, eo in enumerate(descriptions):
                if str(eo.get("name", "")).lower() == band_l:
                    return _asset_band(asset, index)
            if len(descriptions) > 1:
                raise InvalidInputError(f"Ambiguous multiband asset for '{band}'.")
            return _asset_band(asset, 0)

    # Tier 2: an asset whose eo:bands names the native band id.
    # Tier 3: asset key or eo:bands common_name equals the spectral alias (e.g. "red").
    alias_match: tuple[dict[str, Any], int] | None = None
    for key, asset in assets.items():
        href = asset.get("href")
        if not href:
            continue
        descriptions = _band_descriptions(asset)
        for index, eo in enumerate(descriptions):
            if str(eo.get("name", "")).lower() == band_l:
                return _asset_band(asset, index)
            common = eo.get("common_name", eo.get("eo:common_name", ""))
            if str(common).lower() == alias_l:
                alias_match = alias_match or (asset, index)
        if key.lower() == alias_l and len(descriptions) <= 1:
            alias_match = alias_match or (asset, 0)
    if alias_match:
        return _asset_band(*alias_match)

    available = ", ".join(sorted(assets)) or "(none)"
    raise UnsupportedSensorError(
        f"STAC item '{_item_id(item)}' has no asset for band '{band}' "
        f"(alias '{alias}').\n       Available assets: {available}"
    )


def _band_descriptions(asset: dict[str, Any]) -> list[dict[str, Any]]:
    """Read STAC 1.0 EO bands or STAC 1.1 common bands, validating array shape."""
    key = "bands" if "bands" in asset else "eo:bands"
    if key not in asset:
        return []
    bands = asset[key]
    if (
        not isinstance(bands, list)
        or not bands
        or not all(isinstance(band, dict) for band in bands)
    ):
        raise InvalidInputError(f"Asset '{key}' must be a nonempty array of objects.")
    return bands


def _asset_band(asset: dict[str, Any], index: int) -> StacBand:
    """Validate the selected band's radiometry without guessing missing units.

    Supports raster v1 ``raster:bands`` and v2 ``bands`` with namespaced scale/
    offset. Both forms together are rejected as ambiguous. Missing transform
    fields retain legacy DN behavior; explicit invalid values raise a domain error.
    """
    metadata: dict[str, Any] = {}
    descriptions = _band_descriptions(asset)
    if "raster:bands" in asset:
        bands = asset["raster:bands"]
        if (
            not isinstance(bands, list)
            or not bands
            or not all(isinstance(band, dict) for band in bands)
            or index >= len(bands)
            or (descriptions and len(bands) != len(descriptions))
            or (not descriptions and len(bands) != 1)
        ):
            raise InvalidInputError("Asset raster:bands must align with named bands.")
        if "bands" in asset:
            raise InvalidInputError("Use either raster:bands or bands radiometry.")
        metadata = bands[index]
    elif "bands" in asset:
        band = descriptions[index]
        for source, target in (
            ("raster:scale", "scale"),
            ("raster:offset", "offset"),
            ("nodata", "nodata"),
        ):
            if source in band:
                metadata[target] = band[source]
            elif source in asset:
                metadata[target] = asset[source]
    else:
        # Raster v2 permits asset-level defaults even for unnamed single bands.
        for source, target in (
            ("raster:scale", "scale"),
            ("raster:offset", "offset"),
            ("nodata", "nodata"),
        ):
            if source in asset:
                metadata[target] = asset[source]
    try:
        return StacBand(
            href=asset["href"],
            index=index + 1,
            radiometry=BandRadiometry.model_validate(metadata),
        )
    except ValidationError as exc:
        # Avoid echoing the whole asset (which could contain signed URLs).
        raise InvalidInputError(
            f"Invalid STAC radiometry for asset band {index + 1}: "
            "scale/offset must be finite numbers; nodata must be numeric "
            "or 'nan', 'inf', '-inf'."
        ) from exc


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

    from .raster import validate_metric_grid

    validate_metric_grid(out_crs, ref.transform)

    if bbox is not None:
        left, bottom, right, top = transform_bounds("EPSG:4326", out_crs, *bbox)
    else:
        left, bottom, right, top = ref.bounds

    width = max(1, math.ceil((right - left) / target_res))
    height = max(1, math.ceil((top - bottom) / target_res))
    transform = from_origin(left, top, target_res, target_res)
    return _Grid(crs=out_crs, transform=transform, width=width, height=height)


def _read_band(band: StacBand, grid: _Grid) -> Any:
    """Read stored values onto the common grid, excluding nodata from interpolation.

    STAC nodata overrides the TIFF sentinel when supplied. Otherwise TIFF nodata
    and masks apply. Destination gaps use NaN, retaining missing observations
    until the model contract fills them after normalization.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    gdal_path = _gdal_path(band.href)
    options: dict[str, Any] = {}
    if band.radiometry.nodata is not None:
        options["src_nodata"] = band.radiometry.nodata_value
    try:
        with rasterio.open(gdal_path, driver="GTiff") as src:
            if band.index > src.count:
                raise InvalidInputError(
                    f"STAC band index {band.index} exceeds asset band count "
                    f"{src.count}."
                )
            with (
                _band_source(src, band) as (source, index),
                WarpedVRT(
                    source,
                    crs=grid.crs,
                    transform=grid.transform,
                    width=grid.width,
                    height=grid.height,
                    resampling=Resampling.bilinear,
                    dtype="float32",
                    nodata=float("nan"),
                    **options,
                ) as vrt,
            ):
                values = vrt.read(index, masked=True).filled(np.nan).astype(np.float32)
                values[~np.isfinite(values)] = np.nan
                return values
    except rasterio.errors.RasterioError as exc:
        raise InvalidInputError(
            f"Could not read STAC asset '{gdal_path}'.\n"
            f"       {exc}\n"
            f"       Ensure it is a readable (ideally cloud-optimized) GeoTIFF."
        ) from exc


@contextmanager
def _band_source(source: Any, band: StacBand) -> Iterator[tuple[Any, int]]:
    """Isolate the selected band and combine its explicit mask with nodata.

    GDAL otherwise ignores per-dataset masks when a source nodata value is set.
    Multiband warps can also treat selected-band nodata as valid when another
    band is valid, so those sources must be reduced to a single band first.
    This tiny VRT references the original pixels/mask lazily, retaining COG range
    reads. Generated nodata masks are deliberately excluded: STAC may override
    the TIFF sentinel. XML escaping protects paths including signed URL queries.

    Args:
        source: Open rasterio source dataset.
        band: Selected source band and its optional STAC nodata override.

    Yields:
        An open dataset and data band index, valid for the context's lifetime.
    """
    from rasterio.enums import MaskFlags
    from rasterio.io import MemoryFile

    flags = source.mask_flag_enums[band.index - 1]
    nodata = band.radiometry.nodata_value
    if nodata is None:
        nodata = source.nodatavals[band.index - 1]
    explicit_mask = bool({MaskFlags.per_dataset, MaskFlags.alpha}.intersection(flags))
    if source.count == 1 and not explicit_mask:
        yield source, band.index
        return
    if nodata is None:
        nodata = float("nan")

    root = Element(
        "VRTDataset", rasterXSize=str(source.width), rasterYSize=str(source.height)
    )
    if source.crs:
        SubElement(root, "SRS").text = source.crs.to_wkt()
    SubElement(root, "GeoTransform").text = ",".join(
        str(value) for value in source.transform.to_gdal()
    )
    target = SubElement(root, "VRTRasterBand", dataType="Float32", band="1")
    SubElement(target, "NoDataValue").text = str(nodata)
    reference = SubElement(target, "ComplexSource" if explicit_mask else "SimpleSource")
    SubElement(reference, "SourceFilename", relativeToVRT="0").text = source.name
    SubElement(reference, "SourceBand").text = str(band.index)
    if explicit_mask:
        SubElement(reference, "UseMaskBand").text = "true"
    with MemoryFile(tostring(root)) as memory, memory.open() as dataset:
        yield dataset, 1


def _build_source(
    data: Any,
    grid: _Grid,
    bands: list[str],
    sensor: str,
    item: dict[str, Any],
    selected: list[StacBand],
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
        nodata=float("nan"),
        band_radiometry=[band.radiometry for band in selected],
    )


def _gdal_path(href: str) -> str:
    """Translate a STAC asset href into a GDAL-readable path."""
    parsed = urlparse(href)
    if parsed.scheme in ("http", "https"):
        return f"/vsicurl/{href}"
    if parsed.scheme == "s3":
        return f"/vsis3/{parsed.netloc}{parsed.path}"
    if parsed.scheme == "file":
        return unquote(parsed.path)
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
