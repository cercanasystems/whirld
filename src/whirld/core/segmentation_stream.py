"""Resource-owned segmentation over bounded raster windows and GeoTIFF writes."""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..errors import InvalidInputError
from ..io.cog import CogWindowReader
from ..io.input import RasterInput
from ..io.segmentation_stream import SegmentationMetadata, SegmentationStreamWriter
from ..io.stac_stream import RasterWindowReader, StacWindowReader
from ..logging_setup import configure_logging, record_usage
from ..models.base import InferenceContext, detect_device
from ..models.loader import load_backend
from .chips import ChipSet
from .fetch import ModelStore
from .inference_metadata import chip_latlons, warn_cpu
from .options import CogOptions, InferenceOptions, StreamingSegmentationOptions
from .registry import ModelEntry


class DiskSegmentResult(BaseModel):
    """Published mask identity and counts without resident arrays or chip lists."""

    model_config = ConfigDict(frozen=True)
    output_path: Path
    width: int
    height: int
    chip_count: int
    classes: int
    valid_pixel_count: int
    positive_pixel_count: int
    model: str
    head: str | None
    sensor: str
    device: str
    crs: str
    transform: tuple[float, float, float, float, float, float]


class SegmentationStream:
    """Own a model run and its bounded source/output lifetimes.

    Args:
        store: Registry/cache identity shared by manifest and backend loading.
        device: Device override; None chooses an available device.
    """

    def __init__(self, store: ModelStore, device: str | None = None) -> None:
        """Bind installation identity; defer device selection until task validation."""
        self._store = store
        self._device = device if device is not None else "auto"

    def _reader(
        self,
        source: RasterInput,
        entry: ModelEntry,
        options: StreamingSegmentationOptions,
        sensor: str | None,
        crs: str | None,
        token: str | None,
        token_origins: tuple[str, ...],
    ) -> RasterWindowReader:
        """Construct an unopened native-chip reader with explicit source policies.

        Args:
            source: Validated local/STAC/COG identity and COG metadata.
            entry: Segmentation model's spectral and spatial contract.
            options: Typed memory, threshold and bbox settings.
            sensor: Explicit selection or local/STAC inference request.
            crs: Missing-CRS assignment without reprojection.
            token: Origin-scoped bearer credential.
            token_origins: Additional authorized direct asset origins.

        Returns:
            Resource-owning reader using zero overlap and native chip size.
        """
        if source.kind == "geotiff":
            from ..io.local import LocalWindowReader

            return LocalWindowReader(
                source.location,
                entry=entry,
                chip_size=entry.band_contract.chip_size_px,
                batch_size=options.batch_size,
                cache_mb=options.cache_mb,
                sensor=sensor,
                crs=crs,
                bbox=options.bbox,
            )
        if source.kind == "cog":
            return CogWindowReader(
                source.location,
                entry=entry,
                chip_size=entry.band_contract.chip_size_px,
                batch_size=options.batch_size,
                cache_mb=options.cache_mb,
                sensor=sensor,
                crs=crs,
                bbox=options.bbox,
                token=token,
                token_origins=token_origins,
                cog=source.cog,
            )
        return StacWindowReader(
            source.location,
            entry=entry,
            chip_size=entry.band_contract.chip_size_px,
            batch_size=options.batch_size,
            cache_mb=options.cache_mb,
            sensor=sensor,
            crs=crs,
            bbox=options.bbox,
            token=token,
            token_origins=token_origins,
        )

    def run(
        self,
        url: str,
        *,
        model: str,
        output: Path,
        options: StreamingSegmentationOptions,
        head: str | None = None,
        sensor: str | None = None,
        crs: str | None = None,
        cog: CogOptions | None = None,
        token: str | None = None,
        token_origins: tuple[str, ...] = (),
        no_warnings: bool = False,
    ) -> DiskSegmentResult:
        """Infer native non-overlapping tiles and atomically publish a mask.

        Args:
            url: Local GeoTIFF, STAC item or direct HTTP(S) GeoTIFF URL.
            model: Resolved installed segmentation model identity.
            output: Required final GeoTIFF destination.
            options: Validated batch/cache, bbox and threshold settings.
            head: Task head passed to the backend; model identifies its weights.
            sensor: Sensor override; required for direct COGs.
            crs: Missing source CRS assignment.
            cog: Explicit direct raster spectral/radiometry metadata.
            token: Credential scoped to the input's original origin.
            token_origins: Additional explicitly authorized direct asset origins.
            no_warnings: Suppress the real-model CPU runtime estimate.

        Returns:
            Output dimensions, grid, counts and model identity, without mask arrays.

        Raises:
            InvalidInputError: Unsupported input/model or invalid batch output.
            WhirldError: Source, model, inference or output processing fails.
            KeyboardInterrupt: Propagates after closing source and staged output.
        """
        source = RasterInput.resolve(url, cog)
        source.validate_sensor(sensor)
        source.validate_streaming(options.bbox)
        configure_logging()
        started = time.monotonic()
        version = ""
        quantized = False
        resolved_sensor: str | None = None
        count: int | None = None
        error: str | None = None
        try:
            entry = self._store.registry.get(model)
            entry.require_operation("segment")
            self._device = detect_device(self._device)
            manifest = self._store.load_manifest(model)
            version, quantized = manifest.version, manifest.quantized
            with self._reader(
                source, entry, options, sensor, crs, token, token_origins
            ) as reader:
                resolved_sensor, count = reader.sensor, reader.chip_count
                warn_cpu(model, manifest, self._device, count, no_warnings)
                backend = load_backend(entry, manifest, self._device, store=self._store)
                affine = reader.transform
                transform = (affine.a, affine.b, affine.c, affine.d, affine.e, affine.f)
                metadata = SegmentationMetadata(
                    crs=reader.crs,
                    transform=transform,
                    width=reader.width,
                    height=reader.height,
                    chip_size_px=reader.chip_size,
                    classes=getattr(backend, "classes", entry.output.classes or 2),
                )
                acquired = reader.metadata.resolve()
                wavelengths = entry.band_contract.sensors[reader.sensor].wavelengths
                with SegmentationStreamWriter(
                    output, chip_count=count, metadata=metadata
                ) as writer:
                    for batch in reader.batches(include_validity=True):
                        if batch.valid_masks is None:
                            raise InvalidInputError(
                                "Segmentation windows require explicit source validity."
                            )
                        chipset = ChipSet(
                            chips=batch.chips,
                            chip_size_px=reader.chip_size,
                            n_rows=reader.n_rows,
                            n_cols=reader.n_cols,
                        )
                        context = InferenceContext(
                            sensor=reader.sensor,
                            gsd_m=reader.resolution_m,
                            wavelengths=wavelengths,
                            latlons=chip_latlons(chipset, reader.crs),
                            acquisition_datetime=acquired,
                            options=InferenceOptions(batch_size=options.batch_size),
                        )
                        masks = backend.segment(
                            batch.pixels, head, options.threshold, context
                        )
                        writer.write(masks, batch.chips, batch.valid_masks)
                        del masks, batch, chipset, context
                    path = writer.finish()
                    return DiskSegmentResult(
                        output_path=path,
                        width=reader.width,
                        height=reader.height,
                        chip_count=count,
                        classes=metadata.classes,
                        valid_pixel_count=writer.valid_pixel_count,
                        positive_pixel_count=writer.positive_pixel_count,
                        model=entry.name,
                        head=head,
                        sensor=reader.sensor,
                        device=self._device,
                        crs=reader.crs,
                        transform=transform,
                    )
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            record_usage(
                command="segment",
                model=model,
                model_version=version,
                input_type=source.kind,
                sensor_detected=resolved_sensor,
                chip_count=count,
                device=self._device,
                quantized=quantized,
                duration_ms=int((time.monotonic() - started) * 1000),
                error=error,
            )
