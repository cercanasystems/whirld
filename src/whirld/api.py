"""High-level orchestration for Whirld's public operations.

This module wires the building blocks (registry -> manifest -> raster -> sensor
detection -> band-contract translation -> chipping -> backend inference -> output)
into the single-call operations the CLI, the Python API, and the REST server share.

There is exactly one pipeline. :func:`embed` resolves a backend per call (one-shot
CLI/library use), while :func:`embed_raster` pulls a **warm** backend from a
:class:`~whirld.core.session.ModelSession` (server use). Both feed the same
private orchestrator, so behavior — including the ``usage.jsonl`` record on success
and failure — is identical regardless of entry point.

Keeping the orchestration here (rather than in ``__init__``) preserves the lazy
top-level import promise: ``import whirld`` does not import numpy, rasterio, or
torch — those are pulled in only when an operation actually runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .core import contract as contract_mod
from .core import sensor as sensor_mod
from .core.chips import ChipMeta, chip_raster, reassemble_mask
from .core.fetch import Manifest, load_manifest
from .core.fetch import pull as _pull
from .core.registry import ModelEntry, Registry
from .errors import InvalidInputError, WhirldError
from .io import output as output_mod
from .io.raster import RasterSource, read_raster
from .logging_setup import configure_logging, get_logger, record_usage
from .models.base import InferenceContext, ModelBackend, detect_device
from .models.loader import load_backend

_log = get_logger("api")

# A resolver yields the four things the pipeline needs, given a resolved device.
_Resolved = tuple[ModelEntry, Manifest, ModelBackend, RasterSource]
_Resolver = Callable[[str], _Resolved]


@dataclass
class EmbedResult:
    """Result of an embed operation.

    Attributes:
        embeddings: Array of shape ``(n_chips, embed_dim)``, float32.
        chips: Per-chip georeferencing metadata.
        meta: The metadata sidecar dictionary.
        model: Model identifier used.
        sensor: Detected/declared sensor key.
        device: Inference device used.
        output_path: Path the embeddings were written to, or ``None``.
        meta_path: Path the metadata sidecar was written to, or ``None``.
    """

    embeddings: np.ndarray
    chips: list[ChipMeta]
    meta: dict[str, Any]
    model: str
    sensor: str
    device: str
    output_path: Path | None = None
    meta_path: Path | None = None


@dataclass
class ClassifyResult:
    """Result of a classify operation.

    Attributes:
        feature_collection: GeoJSON FeatureCollection (thresholded + top-k).
        scores: Full per-chip, per-query probabilities, shape
            ``(n_chips, n_queries)``, float32.
        chips: Per-chip georeferencing metadata (all chips, unfiltered).
        model: Model identifier used.
        sensor: Detected/declared sensor key.
        device: Inference device used.
        query: The primary (first) query.
        queries: All queries scored, in order.
        output_path: Path the GeoJSON was written to, or ``None``.
    """

    feature_collection: dict[str, Any]
    scores: np.ndarray
    chips: list[ChipMeta]
    model: str
    sensor: str
    device: str
    query: str
    queries: list[str] = field(default_factory=list)
    output_path: Path | None = None


@dataclass
class SegmentResult:
    """Result of a segment operation.

    Attributes:
        mask: The reassembled class-index mask ``(height, width)``, uint8.
        model: Resolved model identifier.
        head: Task head name, if supplied.
        sensor: Detected/declared sensor key.
        device: Inference device used.
        classes: Number of mask classes.
        output_path: Path the mask GeoTIFF was written to, or ``None``.
    """

    mask: np.ndarray
    model: str
    head: str | None
    sensor: str
    device: str
    classes: int
    crs: str = ""
    transform: Any = None
    output_path: Path | None = None


def pull(
    name: str,
    *,
    force: bool = False,
    quantize: str | None = None,
) -> Manifest:
    """Download/materialize, verify, and cache a model.

    Args:
        name: Model identifier (e.g. ``clay-v1``).
        force: Re-acquire even if already cached.
        quantize: Quantized variant (deferred — raises if requested).

    Returns:
        The written :class:`Manifest`.
    """
    configure_logging()
    return _pull(name, force=force, quantize=quantize)


def embed(
    input: str | Path,
    *,
    model: str,
    output: str | Path | None = None,
    fmt: str = "npy",
    chip_size: int | None = None,
    overlap: int = 0,
    device: str | None = None,
    sensor: str | None = None,
    crs: str | None = None,
    batch_size: int | None = None,
    datetime: str | None = None,
    no_warnings: bool = False,
    write: bool = True,
    bbox: tuple[float, float, float, float] | None = None,
    stac_token: str | None = None,
) -> EmbedResult:
    """Generate embeddings for a raster input (one-shot).

    Loads the model backend for this call. For repeated calls that keep a model
    resident, use :func:`embed_raster` with a
    :class:`~whirld.core.session.ModelSession`.

    Args:
        input: Local GeoTIFF path or a STAC item URL (``https://…/item.json``).
        model: Model identifier (e.g. ``clay-v1``).
        output: Output path; defaults to ``<input_stem>_embeddings.npy``.
        fmt: Output format, ``npy`` or ``json``.
        chip_size: Override chip size in pixels; defaults to the model's.
        overlap: Chip overlap in pixels.
        device: ``cuda``, ``mps``, ``cpu``, or ``None`` for auto.
        sensor: Explicit sensor override; ``None`` to auto-detect.
        crs: CRS to assign when the input has none (e.g. ``EPSG:32630``).
        batch_size: Inference batch size; ``None`` keeps the backend default.
        datetime: Acquisition datetime (ISO-8601) for metadata-conditioned models
            such as Clay; overrides any ``TIFFTAG_DATETIME`` tag. For STAC inputs the
            item's own ``datetime`` is used automatically when this is ``None``.
        no_warnings: Suppress the CPU full-precision runtime warning.
        write: If true, write outputs to disk.
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` EPSG:4326 window for STAC
            inputs (COG range-request read); ignored for local files.
        stac_token: Bearer token for gated STAC endpoints.

    Returns:
        An :class:`EmbedResult`.

    Raises:
        WhirldError: Any domain error (model not installed, unsupported sensor,
            invalid input, etc.).
    """

    def resolve(resolved_device: str) -> _Resolved:
        entry = Registry().get(model)
        manifest = load_manifest(model)
        raster = _read_input(
            input, entry=entry, crs=crs, sensor=sensor, bbox=bbox, stac_token=stac_token
        )
        backend = load_backend(entry, manifest, resolved_device)
        return entry, manifest, backend, raster

    def compute(p: _Prepared) -> EmbedResult:
        if batch_size is not None and hasattr(p.backend, "_batch_size"):
            p.backend._batch_size = batch_size
        return _embed_compute(
            p, output=output, fmt=fmt, write=write, default_output_basis=input
        )

    return _run(
        model,
        command="embed",
        resolve=resolve,
        chip_size=chip_size,
        overlap=overlap,
        sensor=sensor,
        device=device,
        compute=compute,
        acquisition_datetime=datetime,
        no_warnings=no_warnings,
        input_type="stac" if _is_stac_input(input) else "geotiff",
    )


def embed_raster(
    raster: RasterSource,
    *,
    model: str,
    session: Any,
    fmt: str = "npy",
    chip_size: int | None = None,
    overlap: int = 0,
    device: str | None = None,
    sensor: str | None = None,
) -> EmbedResult:
    """Generate embeddings for an already-loaded raster using a warm session.

    Used by the REST server: the model backend is pulled from the session cache
    (loaded once, kept resident) and outputs are never written to disk.

    Args:
        raster: A loaded :class:`~whirld.io.raster.RasterSource`.
        model: Model identifier.
        session: A :class:`~whirld.core.session.ModelSession`.
        fmt: Output format hint carried into the result metadata.
        chip_size: Override chip size in pixels.
        overlap: Chip overlap in pixels.
        device: Device override; defaults to the session device.
        sensor: Explicit sensor override; ``None`` to auto-detect.

    Returns:
        An :class:`EmbedResult` (``output_path``/``meta_path`` are ``None``).
    """

    def resolve(resolved_device: str) -> _Resolved:
        loaded = session.get(model, resolved_device)
        return loaded.entry, loaded.manifest, loaded.backend, raster

    def compute(p: _Prepared) -> EmbedResult:
        return _embed_compute(
            p, output=None, fmt=fmt, write=False, default_output_basis=None
        )

    return _run(
        model,
        command="embed",
        resolve=resolve,
        chip_size=chip_size,
        overlap=overlap,
        sensor=sensor,
        # Default to the session's device so warm-preloaded models are reused
        # (rather than re-auto-detecting and loading a second copy).
        device=device if device is not None else session.device,
        compute=compute,
        no_warnings=True,  # server: no per-request CPU warning
    )


def classify(
    input: str | Path,
    *,
    model: str,
    query: str | list[str],
    top_k: int = 5,
    threshold: float = 0.0,
    output: str | Path | None = None,
    device: str | None = None,
    sensor: str | None = None,
    crs: str | None = None,
    no_warnings: bool = False,
    bbox: tuple[float, float, float, float] | None = None,
    stac_token: str | None = None,
) -> ClassifyResult:
    """Classify a raster against one or more text queries (zero-shot, one-shot load).

    Args:
        input: Local GeoTIFF path or a STAC item URL (``https://…/item.json``).
        model: Model identifier (e.g. ``remoteclip``).
        query: One query or a list of queries (the first is primary; scores are
            softmax probabilities across the queries).
        top_k: Keep only the top-k chips by the primary query's score.
        threshold: Drop chips whose primary-query score is below this value.
        output: GeoJSON output path; ``None`` to skip writing (caller serializes).
        device: ``cuda``, ``mps``, ``cpu``, or ``None`` for auto.
        sensor: Explicit sensor override; ``None`` to auto-detect.
        crs: CRS to assign when the input has none (e.g. ``EPSG:32630``).
        no_warnings: Suppress the CPU full-precision runtime warning.
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` EPSG:4326 window for STAC
            inputs; ignored for local files.
        stac_token: Bearer token for gated STAC endpoints.

    Returns:
        A :class:`ClassifyResult`.

    Raises:
        WhirldError: Any domain error (model not installed, unsupported sensor,
            empty query, invalid input, etc.).
    """
    queries = _as_queries(query)

    def resolve(resolved_device: str) -> _Resolved:
        entry = Registry().get(model)
        manifest = load_manifest(model)
        raster = _read_input(
            input, entry=entry, crs=crs, sensor=sensor, bbox=bbox, stac_token=stac_token
        )
        backend = load_backend(entry, manifest, resolved_device)
        return entry, manifest, backend, raster

    def compute(p: _Prepared) -> ClassifyResult:
        return _classify_compute(
            p, queries=queries, top_k=top_k, threshold=threshold, output=output
        )

    return _run(
        model,
        command="classify",
        resolve=resolve,
        chip_size=None,
        overlap=0,
        sensor=sensor,
        device=device,
        compute=compute,
        no_warnings=no_warnings,
        input_type="stac" if _is_stac_input(input) else "geotiff",
    )


def classify_raster(
    raster: RasterSource,
    *,
    model: str,
    session: Any,
    query: str | list[str],
    top_k: int = 5,
    threshold: float = 0.0,
    device: str | None = None,
    sensor: str | None = None,
) -> ClassifyResult:
    """Classify an already-loaded raster using a warm session (server path).

    Args:
        raster: A loaded :class:`~whirld.io.raster.RasterSource`.
        model: Model identifier.
        session: A :class:`~whirld.core.session.ModelSession`.
        query: One query or a list of queries (the first is primary).
        top_k: Keep only the top-k chips by the primary query's score.
        threshold: Drop chips whose primary-query score is below this value.
        device: Device override; defaults to the session device.
        sensor: Explicit sensor override; ``None`` to auto-detect.

    Returns:
        A :class:`ClassifyResult` (``output_path`` is ``None``).
    """
    queries = _as_queries(query)

    def resolve(resolved_device: str) -> _Resolved:
        loaded = session.get(model, resolved_device)
        return loaded.entry, loaded.manifest, loaded.backend, raster

    def compute(p: _Prepared) -> ClassifyResult:
        return _classify_compute(
            p, queries=queries, top_k=top_k, threshold=threshold, output=None
        )

    return _run(
        model,
        command="classify",
        resolve=resolve,
        chip_size=None,
        overlap=0,
        sensor=sensor,
        # Default to the session's device so warm-preloaded models are reused.
        device=device if device is not None else session.device,
        compute=compute,
        no_warnings=True,  # server logs separately; no per-request CPU warning
    )


def _as_queries(query: str | list[str]) -> list[str]:
    """Normalize a query argument (str or list) to a non-empty list of strings."""
    queries = [query] if isinstance(query, str) else list(query)
    queries = [q for q in queries if q and q.strip()]
    if not queries:
        raise WhirldError("classify requires at least one non-empty --query.")
    return queries


def _resolve_segment_model(model: str, head: str | None) -> str:
    """Resolve a segmentation model, honoring the PRD's ``--head`` ergonomics.

    ``prithvi-eo-2 --head <h>`` maps to the real per-head model ``prithvi-<h>``;
    other model names are used directly.

    Args:
        model: Requested model identifier.
        head: Task head, required only for the ``prithvi-eo-2`` alias.

    Returns:
        The resolved registry model name.

    Raises:
        WhirldError: The ``prithvi-eo-2`` alias is used without a ``--head``.
    """
    if model == "prithvi-eo-2":
        if not head:
            raise WhirldError(
                "Model 'prithvi-eo-2' requires --head (e.g. burn-scar, flood)."
            )
        return f"prithvi-{head}"
    return model


def segment(
    input: str | Path,
    *,
    model: str,
    head: str | None = None,
    output: str | Path | None = None,
    device: str | None = None,
    sensor: str | None = None,
    threshold: float = 0.5,
    crs: str | None = None,
    no_warnings: bool = False,
    bbox: tuple[float, float, float, float] | None = None,
    stac_token: str | None = None,
) -> SegmentResult:
    """Run per-pixel segmentation on a raster, writing a single-band mask GeoTIFF.

    Args:
        input: Local GeoTIFF path or a STAC item URL (``https://…/item.json``).
        model: Model identifier (e.g. ``prithvi-burn-scar``, or ``prithvi-eo-2``
            with ``head``).
        head: Task head (``burn-scar`` / ``flood``); required for ``prithvi-eo-2``.
        output: Output GeoTIFF path; defaults to ``<input_stem>_<model>.tif``.
        device: ``cuda``, ``mps``, ``cpu``, or ``None`` for auto.
        sensor: Explicit sensor override; ``None`` to auto-detect.
        threshold: Binary mask threshold on the positive class; ``0.5`` ≡ argmax.
        crs: CRS to assign when the input has none.
        no_warnings: Suppress the CPU full-precision runtime warning.
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` EPSG:4326 window for STAC
            inputs; ignored for local files.
        stac_token: Bearer token for gated STAC endpoints.

    Returns:
        A :class:`SegmentResult`.

    Raises:
        WhirldError: Any domain error (model not installed, unsupported sensor,
            invalid input, etc.).
    """
    resolved_model = _resolve_segment_model(model, head)

    def resolve(resolved_device: str) -> _Resolved:
        entry = Registry().get(resolved_model)
        manifest = load_manifest(resolved_model)
        raster = _read_input(
            input, entry=entry, crs=crs, sensor=sensor, bbox=bbox, stac_token=stac_token
        )
        backend = load_backend(entry, manifest, resolved_device)
        return entry, manifest, backend, raster

    def compute(p: _Prepared) -> SegmentResult:
        return _segment_compute(
            p, head=head, threshold=threshold, output=output, default_basis=input
        )

    return _run(
        resolved_model,
        command="segment",
        resolve=resolve,
        chip_size=None,
        overlap=0,
        sensor=sensor,
        device=device,
        compute=compute,
        no_warnings=no_warnings,
        input_type="stac" if _is_stac_input(input) else "geotiff",
    )


def segment_raster(
    raster: RasterSource,
    *,
    model: str,
    session: Any,
    head: str | None = None,
    threshold: float = 0.5,
    device: str | None = None,
    sensor: str | None = None,
) -> SegmentResult:
    """Segment an already-loaded raster using a warm session (server path).

    Args:
        raster: A loaded :class:`~whirld.io.raster.RasterSource`.
        model: Model identifier.
        session: A :class:`~whirld.core.session.ModelSession`.
        head: Task head; required for the ``prithvi-eo-2`` alias.
        threshold: Binary mask threshold on the positive class.
        device: Device override; defaults to the session device.
        sensor: Explicit sensor override; ``None`` to auto-detect.

    Returns:
        A :class:`SegmentResult` (``output_path`` is ``None``).
    """
    resolved_model = _resolve_segment_model(model, head)

    def resolve(resolved_device: str) -> _Resolved:
        loaded = session.get(resolved_model, resolved_device)
        return loaded.entry, loaded.manifest, loaded.backend, raster

    def compute(p: _Prepared) -> SegmentResult:
        return _segment_compute(
            p, head=head, threshold=threshold, output=None, default_basis=None
        )

    return _run(
        resolved_model,
        command="segment",
        resolve=resolve,
        chip_size=None,
        overlap=0,
        sensor=sensor,
        device=device if device is not None else session.device,
        compute=compute,
        no_warnings=True,
    )


@dataclass
class _Prepared:
    """The shared inputs every per-model compute step receives."""

    entry: ModelEntry
    manifest: Manifest
    backend: ModelBackend
    translated: Any
    chips_array: np.ndarray
    chipset: Any
    context: InferenceContext
    chip_size: int
    device: str


def _run(
    model: str,
    *,
    command: str,
    resolve: _Resolver,
    chip_size: int | None,
    overlap: int,
    sensor: str | None,
    device: str | None,
    compute: Callable[[_Prepared], Any],
    acquisition_datetime: str | None = None,
    no_warnings: bool = False,
    input_type: str = "geotiff",
) -> Any:
    """Run the shared pipeline and record usage on success and failure.

    Resolves the backend + raster, detects the sensor, translates and chips, builds
    the inference context (including per-chip lat/lon and acquisition datetime for
    metadata-conditioned models), then hands a :class:`_Prepared` to ``compute`` for
    the model-specific step. A ``usage.jsonl`` record is written for both outcomes.

    Args:
        model: Model identifier (for usage records on early failure).
        command: Usage-record command label (``embed`` / ``classify``).
        resolve: Callable resolving ``(entry, manifest, backend, raster)``.
        chip_size: Chip size override, or ``None`` for the model default.
        overlap: Chip overlap in pixels.
        sensor: Sensor override, or ``None`` to auto-detect.
        device: Device override, or ``None`` for auto.
        compute: The model-specific step producing the result object.
        acquisition_datetime: ISO-8601 datetime override for scene acquisition
            time; falls back to the raster's ``TIFFTAG_DATETIME`` tag.
        no_warnings: Suppress the CPU full-precision runtime warning.
        input_type: Usage-record input kind (``geotiff`` or ``stac``).

    Returns:
        Whatever ``compute`` returns.

    Raises:
        WhirldError: Any domain error encountered in the pipeline.
    """
    configure_logging()
    started = time.monotonic()
    resolved_device = detect_device(device)
    detected_sensor: str | None = None
    chip_count: int | None = None
    manifest_version = ""

    try:
        entry, manifest, backend, raster = resolve(resolved_device)
        manifest_version = manifest.version

        detected_sensor = sensor_mod.detect_sensor(raster, entry, override=sensor)
        translated = contract_mod.translate(raster, entry, detected_sensor)

        size = chip_size or entry.band_contract.chip_size_px
        chips_array, chipset = chip_raster(
            translated,
            size,
            overlap=overlap,
            nodata_fill=entry.band_contract.nodata_fill,
        )
        chip_count = len(chipset.chips)
        _maybe_warn_cpu(entry.name, manifest, resolved_device, chip_count, no_warnings)

        sensor_contract = entry.band_contract.sensors[detected_sensor]
        context = InferenceContext(
            sensor=detected_sensor,
            gsd_m=translated.target_resolution_m,
            wavelengths=sensor_contract.wavelengths,
            latlons=_chip_latlons(chipset, translated.crs),
            acquisition_datetime=_resolve_datetime(acquisition_datetime, raster.tags),
        )

        result = compute(
            _Prepared(
                entry=entry,
                manifest=manifest,
                backend=backend,
                translated=translated,
                chips_array=chips_array,
                chipset=chipset,
                context=context,
                chip_size=size,
                device=resolved_device,
            )
        )

        duration_ms = int((time.monotonic() - started) * 1000)
        record_usage(
            command=command,
            model=entry.name,
            model_version=manifest.version,
            input_type=input_type,
            sensor_detected=detected_sensor,
            chip_count=chip_count,
            device=resolved_device,
            quantized=manifest.quantized,
            duration_ms=duration_ms,
            error=None,
        )
        _log.info(
            "%s: %d chips, sensor=%s, device=%s, %d ms",
            command,
            chip_count,
            detected_sensor,
            resolved_device,
            duration_ms,
        )
        return result
    except WhirldError as exc:
        duration_ms = int((time.monotonic() - started) * 1000)
        record_usage(
            command=command,
            model=model,
            model_version=manifest_version,
            input_type=input_type,
            sensor_detected=detected_sensor,
            chip_count=chip_count,
            device=resolved_device,
            quantized=False,
            duration_ms=duration_ms,
            error=type(exc).__name__,
        )
        raise


def _embed_compute(
    p: _Prepared,
    *,
    output: str | Path | None,
    fmt: str,
    write: bool,
    default_output_basis: str | Path | None,
) -> EmbedResult:
    """Embed step: run the backend, build meta, optionally write npy + sidecar."""
    embeddings = p.backend.embed(p.chips_array, p.context)
    meta = output_mod.build_meta(
        model=p.entry.name,
        model_version=p.manifest.version,
        crs=p.translated.crs,
        embed_dim=p.backend.embed_dim,
        chip_size_px=p.chip_size,
        resolution_m=p.translated.target_resolution_m,
        chipset=p.chipset,
    )
    out_path: Path | None = None
    meta_path: Path | None = None
    if write:
        target = Path(output) if output else _default_output_path(default_output_basis)
        out_path, meta_path = output_mod.write_embeddings(
            embeddings, meta, target, fmt=fmt
        )
    return EmbedResult(
        embeddings=embeddings,
        chips=p.chipset.chips,
        meta=meta,
        model=p.entry.name,
        sensor=p.context.sensor,
        device=p.device,
        output_path=out_path,
        meta_path=meta_path,
    )


def _classify_compute(
    p: _Prepared,
    *,
    queries: list[str],
    top_k: int,
    threshold: float,
    output: str | Path | None,
) -> ClassifyResult:
    """Classify step: score chips, build GeoJSON (threshold + top-k), maybe write.

    Scores are ``(n_chips, n_queries)`` softmax probabilities; the primary (first)
    query drives the ``score`` property, the threshold, and the top-k ranking.
    """
    scores = p.backend.classify(p.chips_array, queries, p.context)
    feature_collection = output_mod.build_feature_collection(
        scores,
        p.chipset,
        queries=queries,
        model=p.entry.name,
        model_version=p.manifest.version,
        crs=p.translated.crs,
    )
    features = [
        f
        for f in feature_collection["features"]
        if f["properties"]["score"] >= threshold
    ]
    features.sort(key=lambda f: f["properties"]["score"], reverse=True)
    if top_k and top_k > 0:
        features = features[:top_k]
    feature_collection["features"] = features

    out_path: Path | None = None
    if output is not None:
        out_path = output_mod.write_geojson(feature_collection, output)
    return ClassifyResult(
        feature_collection=feature_collection,
        scores=scores,
        chips=p.chipset.chips,
        model=p.entry.name,
        sensor=p.context.sensor,
        device=p.device,
        query=queries[0],
        queries=queries,
        output_path=out_path,
    )


def _segment_compute(
    p: _Prepared,
    *,
    head: str | None,
    threshold: float,
    output: str | Path | None,
    default_basis: str | Path | None,
) -> SegmentResult:
    """Segment step: per-chip masks → reassemble to full extent → mask GeoTIFF."""
    masks = p.backend.segment(p.chips_array, head, threshold, p.context)
    height, width = p.translated.data.shape[1], p.translated.data.shape[2]
    mask = reassemble_mask(masks, p.chipset, height, width)

    out_path: Path | None = None
    if output is not None:
        target: Path = Path(output)
    elif default_basis is not None:
        basis = Path(default_basis)
        target = basis.with_name(f"{basis.stem}_{p.entry.name}.tif")
    else:
        target = None  # server path: caller serializes from the mask
    if target is not None:
        out_path = output_mod.write_mask_geotiff(
            mask, p.translated.transform, p.translated.crs, target
        )
    return SegmentResult(
        mask=mask,
        model=p.entry.name,
        head=head,
        sensor=p.context.sensor,
        device=p.device,
        classes=getattr(p.backend, "classes", p.entry.output.classes or 2),
        crs=p.translated.crs,
        transform=p.translated.transform,
        output_path=out_path,
    )


def _default_output_path(input_path: str | Path | None) -> Path:
    """Derive the default embeddings output path from the input stem.

    Args:
        input_path: The input raster path.

    Returns:
        ``<input_stem>_embeddings.npy`` in the input's directory.

    Raises:
        InvalidInputError: No basis path is available to derive a name from.
    """
    if input_path is None:
        raise InvalidInputError("Cannot derive a default output path without input.")
    p = Path(input_path)
    return p.with_name(f"{p.stem}_embeddings.npy")


def _looks_like_url(value: str | Path) -> bool:
    """Return whether a value looks like an HTTP(S) URL (i.e. a STAC input)."""
    text = str(value)
    return text.startswith("http://") or text.startswith("https://")


def _is_stac_input(value: str | Path) -> bool:
    """Return whether the input is a STAC item (URL, ``file://``, or ``.json`` path)."""
    text = str(value)
    return (
        _looks_like_url(value) or text.startswith("file://") or text.endswith(".json")
    )


def _read_input(
    value: str | Path,
    *,
    entry: ModelEntry,
    crs: str | None,
    sensor: str | None,
    bbox: tuple[float, float, float, float] | None,
    stac_token: str | None,
) -> RasterSource:
    """Load the input into a :class:`RasterSource` — local GeoTIFF or STAC item URL.

    A STAC item URL/``.json`` path is read via :func:`whirld.io.stac.read_stac_item`
    (lazy-imported so the dependency only loads on the STAC path); anything else is a
    local GeoTIFF read with :func:`whirld.io.raster.read_raster`.
    """
    if _is_stac_input(value):
        from .io.stac import read_stac_item

        return read_stac_item(
            str(value),
            entry=entry,
            sensor=sensor,
            crs=crs,
            bbox=bbox,
            token=stac_token,
        )
    return read_raster(value, crs=crs)


# Rough per-chip CPU wall-clock estimate (seconds) for the runtime warning. A
# conservative order-of-magnitude figure — labeled an estimate, never a guarantee.
_CPU_SECONDS_PER_CHIP = 1.0


def _maybe_warn_cpu(
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


def _chip_latlons(chipset: Any, crs: str) -> list[tuple[float, float]] | None:
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


def _resolve_datetime(override: str | None, tags: dict[str, str]) -> datetime | None:
    """Resolve a scene acquisition datetime from an override or TIFF tag.

    Precedence: explicit ISO-8601 ``override`` › ``TIFFTAG_DATETIME``
    (``"YYYY:MM:DD HH:MM:SS"``) › ``None``.

    Args:
        override: Optional ISO-8601 datetime string.
        tags: Dataset tags (may contain ``TIFFTAG_DATETIME``).

    Returns:
        A parsed :class:`datetime.datetime`, or ``None`` if neither source parses.
    """
    if override:
        try:
            return datetime.fromisoformat(override.replace("Z", "+00:00"))
        except ValueError:
            _log.warning("Could not parse --datetime '%s'; ignoring.", override)
    tag = tags.get("TIFFTAG_DATETIME")
    if tag:
        try:
            return datetime.strptime(tag, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            _log.debug("Could not parse TIFFTAG_DATETIME '%s'; ignoring.", tag)
    return None
