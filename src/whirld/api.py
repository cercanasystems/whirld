"""High-level orchestration for Whirld's public operations.

This module wires the building blocks (registry -> manifest -> raster -> sensor
detection -> band-contract translation -> chipping -> backend inference -> output)
into the single-call operations the CLI, the Python API, and the REST server share.

The in-memory pipeline serves :func:`embed` (one-shot CLI/library use) and
:func:`embed_raster` (a warm ModelSession for server use). :func:`embed_to_disk`
uses an owned streaming coordinator with bounded raster windows and incremental
outputs. Both paths share band translation, chip geometry, model interfaces,
acquisition metadata rules, and usage-record conventions.

Keeping the orchestration here (rather than in ``__init__``) preserves the lazy
top-level import promise: ``import whirld`` does not import numpy, rasterio, or
torch — those are pulled in only when an operation actually runs.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np

from .core import contract as contract_mod
from .core import sensor as sensor_mod
from .core.chips import ChipMeta, chip_raster, reassemble_mask
from .core.classification_stream import ClassificationStream, DiskClassifyResult
from .core.embedding_stream import DiskEmbedResult, EmbeddingStream
from .core.fetch import Manifest, ModelStore
from .core.inference_metadata import (
    chip_latlons as _chip_latlons,
)
from .core.inference_metadata import (
    resolve_datetime as _resolve_datetime,
)
from .core.inference_metadata import (
    warn_cpu as _maybe_warn_cpu,
)
from .core.options import (
    ClassificationOptions,
    CogOptions,
    EmbeddingOptions,
    InferenceOptions,
    SegmentationOptions,
    StreamingClassificationOptions,
    StreamingEmbeddingOptions,
    StreamingSegmentationOptions,
)
from .core.registry import ModelEntry
from .core.segmentation_stream import DiskSegmentResult, SegmentationStream
from .errors import InvalidInputError, WhirldError
from .io import output as output_mod
from .io.input import RasterInput
from .io.raster import RasterSource
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
    return ModelStore().pull(name, force=force, quantize=quantize)


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
    cog: CogOptions | None = None,
) -> EmbedResult:
    """Generate embeddings for a raster input (one-shot).

    Loads the model backend for this call. For repeated calls that keep a model
    resident, use :func:`embed_raster` with a
    :class:`~whirld.core.session.ModelSession`.

    Args:
        input: Local GeoTIFF, STAC item, or direct HTTP(S) .tif/.tiff URL.
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
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` EPSG:4326 window for STAC/COG
            inputs (COG range-request read); ignored for local files.
        stac_token: Bearer token for the original STAC/COG origin.
        cog: Optional direct COG band/scale/offset metadata; also selects
            extensionless HTTP(S) TIFF URLs. Direct COG input requires sensor.

    Returns:
        An :class:`EmbedResult`.

    Raises:
        WhirldError: Any domain error (model not installed, unsupported sensor,
            invalid input, etc.).
    """

    source = RasterInput.resolve(input, cog)
    source.validate_sensor(sensor)
    EmbeddingOptions.from_arguments(
        chip_size=chip_size,
        overlap=overlap,
        batch_size=batch_size,
        fmt=fmt,
        bbox=bbox,
    )

    def resolve(resolved_device: str) -> _Resolved:
        store = ModelStore()
        entry = store.registry.get(model)
        manifest = store.load_manifest(model)
        raster = _read_input(
            input,
            entry=entry,
            crs=crs,
            sensor=sensor,
            bbox=bbox,
            stac_token=stac_token,
            cog=cog,
        )
        backend = load_backend(entry, manifest, resolved_device, store=store)
        return entry, manifest, backend, raster

    def compute(p: _Prepared) -> EmbedResult:
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
        input_type=source.kind,
        inference_options=InferenceOptions(batch_size=batch_size),
    )


def embed_to_disk(
    input: str | Path,
    *,
    model: str,
    output: str | Path | None = None,
    fmt: str = "npy",
    chip_size: int | None = None,
    overlap: int = 0,
    batch_size: int = 8,
    cache_mb: int = 32,
    device: str | None = None,
    sensor: str | None = None,
    crs: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    stac_token: str | None = None,
    cog: CogOptions | None = None,
    token_origins: tuple[str, ...] = (),
    datetime: str | None = None,
    no_warnings: bool = False,
) -> DiskEmbedResult:
    """Stream local, STAC or direct COG embeddings to disk in bounded batches.

    Args:
        input: Local GeoTIFF path/file URL, STAC item or direct HTTP(S) COG URL.
        model: Installed embedding model identifier.
        output: NPY destination; defaults to the existing input-based filename.
        fmt: Must be ``npy``; metadata is streamed to the usual JSON sidecar.
        chip_size: Model chip size override in pixels.
        overlap: Overlap between adjacent chips, in pixels.
        batch_size: Maximum chips read and inferred together (default eight).
        cache_mb: GDAL block cache budget in MiB; other I/O caches are bounded.
        device: Inference device override, or automatic selection.
        sensor: Required for direct COGs; otherwise inferred from source metadata.
        crs: CRS to assign to an asset when absent.
        bbox: Optional EPSG:4326 read window for STAC/remote COGs; rejected locally.
        stac_token: Bearer token scoped to the input's origin.
        cog: Optional COG band/scale/offset metadata; also selects extensionless
            HTTP(S) TIFF URLs. Direct COG input requires sensor.
        token_origins: Additional asset origins authorized to receive that token.
        datetime: Acquisition datetime override, interpreted as UTC when naive.
        no_warnings: Suppress the real-model CPU runtime estimate.

    Returns:
        Paths and counts only. Use ``numpy.load(result.output_path, mmap_mode='r')``
        for subsequent bounded reads. Ordinary ``embed()`` still returns arrays.

    Raises:
        InvalidInputError: Unsupported input/output or invalid execution options.
        WhirldError: Registry, model, source, inference, or disk processing fails.
    """
    source = RasterInput.resolve(input, cog)
    source.validate_sensor(sensor)
    options = StreamingEmbeddingOptions.from_arguments(
        chip_size=chip_size,
        overlap=overlap,
        batch_size=batch_size,
        cache_mb=cache_mb,
        bbox=bbox,
        fmt=fmt,
    )
    source.validate_streaming(bbox)
    target = Path(output) if output is not None else _default_output_path(input)
    return EmbeddingStream(ModelStore(), device).run(
        str(input),
        model=model,
        output=target,
        options=options,
        sensor=sensor,
        crs=crs,
        token=stac_token,
        cog=cog,
        token_origins=token_origins,
        acquisition_datetime=datetime,
        no_warnings=no_warnings,
    )


def embed_raster(
    raster: RasterSource,
    *,
    model: str,
    session: Any,
    fmt: str = "npy",
    chip_size: int | None = None,
    overlap: int = 0,
    batch_size: int | None = None,
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
        batch_size: Optional per-call batch size; never changes the resident backend.
        device: Device override; defaults to the session device.
        sensor: Explicit sensor override; ``None`` to auto-detect.

    Returns:
        An :class:`EmbedResult` (``output_path``/``meta_path`` are ``None``).
    """

    EmbeddingOptions.from_arguments(
        chip_size=chip_size,
        overlap=overlap,
        batch_size=batch_size,
        fmt=fmt,
    )

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
        inference_options=InferenceOptions(batch_size=batch_size),
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
    cog: CogOptions | None = None,
) -> ClassifyResult:
    """Classify a raster against one or more text queries (zero-shot, one-shot load).

    Args:
        input: Local GeoTIFF, STAC item, or direct HTTP(S) .tif/.tiff URL.
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
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` EPSG:4326 window for STAC/COG
            inputs; ignored for local files.
        stac_token: Bearer token for the original STAC/COG origin.
        cog: Optional direct COG band/scale/offset metadata; also selects
            extensionless HTTP(S) TIFF URLs. Direct COG input requires sensor.

    Returns:
        A :class:`ClassifyResult`.

    Raises:
        WhirldError: Any domain error (model not installed, unsupported sensor,
            empty query, invalid input, etc.).
    """
    source = RasterInput.resolve(input, cog)
    source.validate_sensor(sensor)
    ClassificationOptions.from_arguments(top_k=top_k, threshold=threshold, bbox=bbox)
    queries = _as_queries(query)

    def resolve(resolved_device: str) -> _Resolved:
        store = ModelStore()
        entry = store.registry.get(model)
        manifest = store.load_manifest(model)
        raster = _read_input(
            input,
            entry=entry,
            crs=crs,
            sensor=sensor,
            bbox=bbox,
            stac_token=stac_token,
            cog=cog,
        )
        backend = load_backend(entry, manifest, resolved_device, store=store)
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
        input_type=source.kind,
    )


def classify_to_disk(
    input: str | Path,
    *,
    model: str,
    query: str | list[str],
    output: str | Path,
    top_k: int | None = 5,
    threshold: float = 0.0,
    batch_size: int = 8,
    cache_mb: int = 32,
    device: str | None = None,
    sensor: str | None = None,
    crs: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    stac_token: str | None = None,
    token_origins: tuple[str, ...] = (),
    cog: CogOptions | None = None,
    no_warnings: bool = False,
) -> DiskClassifyResult:
    """Classify bounded raster windows and publish globally ranked GeoJSON on disk.

    Args:
        input: Local GeoTIFF path/file URL, STAC item or direct HTTP(S) COG URL.
        model: Installed classification model identifier, for example remoteclip.
        query: One or more text queries; the first identifies the primary score.
        output: Required GeoJSON destination, preserved if processing fails.
        top_k: Positive maximum matches; None emits all threshold-passing chips.
        threshold: Inclusive primary-score cutoff in [0, 1].
        batch_size: Maximum prepared chips per model call; defaults to eight.
        cache_mb: Bounded raster cache budget in MiB; defaults to 32.
        device: Inference device override; None selects an available device.
        sensor: Required for direct COGs; otherwise inferred from source metadata.
        crs: CRS assignment for sources whose CRS is missing.
        bbox: Optional EPSG:4326 read window for STAC/remote COGs; rejected locally.
        stac_token: Bearer token scoped to the original input origin.
        token_origins: Additional direct asset origins explicitly authorized.
        cog: Optional direct COG band and radiometry overrides.
        no_warnings: Suppress the real-model CPU runtime estimate.

    Returns:
        Output path, counts and model/sensor/device identity. Scores and geometry
        remain on disk. Results sort by descending primary score, breaking ties
        by chip index. Disk-backed sorting requires O(scene) temporary space.

    Raises:
        InvalidInputError: Invalid options, queries, input kind or output path.
        WhirldError: Source, installation, inference or disk processing fails.
    """
    source = RasterInput.resolve(input, cog)
    source.validate_sensor(sensor)
    source.validate_streaming(bbox)
    if output is None:
        raise InvalidInputError("Streaming classification requires an output path.")
    options = StreamingClassificationOptions.from_arguments(
        top_k=top_k,
        threshold=threshold,
        batch_size=batch_size,
        cache_mb=cache_mb,
        bbox=bbox,
    )
    queries = _as_queries(query)
    return ClassificationStream(ModelStore(), device).run(
        str(input),
        model=model,
        queries=queries,
        output=Path(output),
        options=options,
        sensor=sensor,
        crs=crs,
        cog=cog,
        token=stac_token,
        token_origins=token_origins,
        no_warnings=no_warnings,
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
    ClassificationOptions.from_arguments(top_k=top_k, threshold=threshold)
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
    if not isinstance(query, (str, list)) or (
        isinstance(query, list) and any(not isinstance(q, str) for q in query)
    ):
        raise InvalidInputError("query must be a string or a list of strings.")
    queries = [query] if isinstance(query, str) else list(query)
    queries = [q for q in queries if q and q.strip()]
    if not queries:
        raise InvalidInputError("classify requires at least one non-empty --query.")
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
    cog: CogOptions | None = None,
) -> SegmentResult:
    """Run per-pixel segmentation on a raster, writing a single-band mask GeoTIFF.

    Args:
        input: Local GeoTIFF, STAC item, or direct HTTP(S) .tif/.tiff URL.
        model: Model identifier (e.g. ``prithvi-burn-scar``, or ``prithvi-eo-2``
            with ``head``).
        head: Task head (``burn-scar`` / ``flood``); required for ``prithvi-eo-2``.
        output: Output GeoTIFF path; defaults to ``<input_stem>_<model>.tif``.
        device: ``cuda``, ``mps``, ``cpu``, or ``None`` for auto.
        sensor: Explicit sensor override; ``None`` to auto-detect.
        threshold: Binary mask threshold on the positive class; ``0.5`` ≡ argmax.
        crs: CRS to assign when the input has none.
        no_warnings: Suppress the CPU full-precision runtime warning.
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` EPSG:4326 window for STAC/COG
            inputs; ignored for local files.
        stac_token: Bearer token for the original STAC/COG origin.
        cog: Optional direct COG band/scale/offset metadata; also selects
            extensionless HTTP(S) TIFF URLs. Direct COG input requires sensor.

    Returns:
        A :class:`SegmentResult`.

    Raises:
        WhirldError: Any domain error (model not installed, unsupported sensor,
            invalid input, etc.).
    """
    source = RasterInput.resolve(input, cog)
    source.validate_sensor(sensor)
    SegmentationOptions.from_arguments(threshold=threshold, bbox=bbox)
    resolved_model = _resolve_segment_model(model, head)

    def resolve(resolved_device: str) -> _Resolved:
        store = ModelStore()
        entry = store.registry.get(resolved_model)
        manifest = store.load_manifest(resolved_model)
        raster = _read_input(
            input,
            entry=entry,
            crs=crs,
            sensor=sensor,
            bbox=bbox,
            stac_token=stac_token,
            cog=cog,
        )
        backend = load_backend(entry, manifest, resolved_device, store=store)
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
        input_type=source.kind,
    )


def segment_to_disk(
    input: str | Path,
    *,
    model: str,
    output: str | Path,
    head: str | None = None,
    threshold: float = 0.5,
    batch_size: int = 4,
    cache_mb: int = 32,
    device: str | None = None,
    sensor: str | None = None,
    crs: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    stac_token: str | None = None,
    token_origins: tuple[str, ...] = (),
    cog: CogOptions | None = None,
    no_warnings: bool = False,
) -> DiskSegmentResult:
    """Segment bounded native tiles and publish a mask GeoTIFF incrementally.

    Args:
        input: Local GeoTIFF path/file URL, STAC item or direct HTTP(S) COG URL.
        model: Installed segmentation model or prithvi-eo-2 task alias.
        output: Required single-band GeoTIFF destination.
        head: Task head for the prithvi-eo-2 alias, or backend-specific hint.
        threshold: Positive-class probability threshold; default matches eager.
        batch_size: Maximum prepared native chips per call (default four).
        cache_mb: Bounded raster cache budget in MiB (default 32).
        device: Inference device override; None selects an available device.
        sensor: Required for direct COGs; otherwise inferred from source metadata.
        crs: CRS assignment for sources whose identity is missing.
        bbox: Optional EPSG:4326 read window for STAC/remote COGs; rejected locally.
        stac_token: Bearer credential scoped to the original input origin.
        token_origins: Additional explicitly authorized direct asset origins.
        cog: Direct source band and radiometry overrides.
        no_warnings: Suppress the real-model CPU runtime estimate.

    Returns:
        Output path, grid, dimensions, class/pixel counts and model identity.
        No mask array or scene-sized chip list is retained. Invalid source pixels
        become nodata255; padded chip edges are cropped to the target extent.

    Raises:
        InvalidInputError: Invalid options, input kind, sensor or output path.
        WhirldError: Source/model/inference/output processing fails. Prior output
            survives incomplete processing; successful publication is atomic.
    """
    source = RasterInput.resolve(input, cog)
    source.validate_sensor(sensor)
    source.validate_streaming(bbox)
    if output is None:
        raise InvalidInputError("Streaming segmentation requires an output path.")
    options = StreamingSegmentationOptions.from_arguments(
        threshold=threshold,
        batch_size=batch_size,
        cache_mb=cache_mb,
        bbox=bbox,
    )
    return SegmentationStream(ModelStore(), device).run(
        str(input),
        model=_resolve_segment_model(model, head),
        output=Path(output),
        options=options,
        head=head,
        sensor=sensor,
        crs=crs,
        cog=cog,
        token=stac_token,
        token_origins=token_origins,
        no_warnings=no_warnings,
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
    SegmentationOptions.from_arguments(threshold=threshold)
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
    inference_options: InferenceOptions | None = None,
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
        inference_options: Immutable execution policy passed to this call's backend.

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

        size = chip_size if chip_size is not None else entry.band_contract.chip_size_px
        EmbeddingOptions.from_arguments(chip_size=size, overlap=overlap)
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
            options=inference_options or InferenceOptions(),
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
        embed_dim=embeddings.shape[1],
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
    if p.translated.valid_mask is not None:
        mask[~p.translated.valid_mask] = 255

    out_path: Path | None = None
    if output is not None:
        target: Path | None = Path(output)
    elif default_basis is not None:
        basis = _output_basis(default_basis)
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
        ``<input_stem>_embeddings.npy`` next to local inputs, or in the current
        directory for HTTP(S) inputs. URL credentials and queries are discarded.

    Raises:
        InvalidInputError: No basis path is available to derive a name from.
    """
    if input_path is None:
        raise InvalidInputError("Cannot derive a default output path without input.")
    p = _output_basis(input_path)
    return p.with_name(f"{p.stem}_embeddings.npy")


def _output_basis(input_path: str | Path) -> Path:
    """Use local paths directly and derive safe basenames from remote URLs."""
    value = str(input_path)
    parsed = urlsplit(value)
    if parsed.scheme in {"http", "https"}:
        name = unquote(parsed.path.rsplit("/", 1)[-1])
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".")
        return Path(name or "scene")
    if parsed.scheme == "file":
        if parsed.netloc.lower() not in {"", "localhost"}:
            raise InvalidInputError("file URLs must refer to the local host.")
        return Path(unquote(parsed.path))
    return Path(input_path)


def _is_stac_input(value: str | Path) -> bool:
    """Compatibility query for STAC routing, excluding direct .tif/.tiff URLs."""
    return RasterInput.resolve(value).kind == "stac"


def _read_input(
    value: str | Path,
    *,
    entry: ModelEntry,
    crs: str | None,
    sensor: str | None,
    bbox: tuple[float, float, float, float] | None,
    stac_token: str | None,
    cog: CogOptions | None = None,
) -> RasterSource:
    """Read local, STAC or direct COG input using its owned source adapter.

    Args:
        value: Input path or HTTP(S) URL; signed queries are preserved for reads.
        entry: Model band/unit contract.
        crs: Assign a CRS only when absent.
        sensor: Explicit sensor for COG; optional for existing local/STAC paths.
        bbox: Geographic read window for remote/STAC sources.
        stac_token: Bearer credential scoped to the input origin.
        cog: Optional direct COG metadata/extensionless selection.

    Returns:
        Resident source pixels with geometry, acquisition and radiometry metadata.

    Raises:
        WhirldError: Input selection, metadata validation or source reads fail.
    """
    return RasterInput.resolve(value, cog).read(
        entry=entry, crs=crs, sensor=sensor, bbox=bbox, token=stac_token
    )
