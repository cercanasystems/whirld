"""Resource-owned bounded classification with globally ranked disk output."""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..errors import InvalidInputError
from ..io.classification_stream import (
    ClassificationMetadata,
    ClassificationStreamWriter,
)
from ..io.cog import CogWindowReader
from ..io.input import RasterInput
from ..io.stac_stream import RasterWindowReader, StacWindowReader
from ..logging_setup import configure_logging, record_usage
from ..models.base import InferenceContext, detect_device
from ..models.loader import load_backend
from .chips import ChipSet
from .fetch import ModelStore
from .inference_metadata import chip_latlons, warn_cpu
from .options import (
    ClassificationQueries,
    CogOptions,
    InferenceOptions,
    StreamingClassificationOptions,
)
from .registry import ModelEntry


class DiskClassifyResult(BaseModel):
    """Published classification summary without scene arrays or feature lists."""

    model_config = ConfigDict(frozen=True)
    output_path: Path
    chip_count: int
    feature_count: int
    model: str
    sensor: str
    device: str
    query: str
    queries: tuple[str, ...]


class ClassificationStream:
    """Own one store and backend run over bounded source batches.

    Args:
        store: Registry/cache owner shared by manifest resolution and loading.
        device: Inference device override; None selects the available default.

    A run owns its raster reader and staged writer through publication. Completed
    scores and global ranking live on disk instead of accumulating in memory.
    """

    def __init__(self, store: ModelStore, device: str | None = None) -> None:
        """Bind installation identity and select one device for subsequent runs."""
        self._store = store
        self._device = detect_device(device)

    def _reader(
        self,
        source: RasterInput,
        entry: ModelEntry,
        options: StreamingClassificationOptions,
        sensor: str | None,
        crs: str | None,
        token: str | None,
        token_origins: tuple[str, ...],
    ) -> RasterWindowReader:
        """Bind a source-specific window reader without opening source resources.

        Args:
            source: Validated STAC/direct-COG location and metadata overrides.
            entry: Classification contract defining native chip size.
            options: Validated memory and source-window limits.
            sensor: Explicit selection, required for direct rasters.
            crs: CRS assignment for sources declaring none.
            token: Bearer credential scoped to authorized origins.
            token_origins: Additional explicitly authorized direct origins.

        Returns:
            Unopened reader configured with native-size non-overlapping chips.
        """
        if source.kind == "cog":
            return CogWindowReader(
                source.location,
                entry=entry,
                chip_size=entry.band_contract.chip_size_px,
                batch_size=options.batch_size,
                sensor=sensor,
                crs=crs,
                bbox=options.bbox,
                token=token,
                token_origins=token_origins,
                cache_mb=options.cache_mb,
                cog=source.cog,
            )
        return StacWindowReader(
            source.location,
            entry=entry,
            chip_size=entry.band_contract.chip_size_px,
            batch_size=options.batch_size,
            sensor=sensor,
            crs=crs,
            bbox=options.bbox,
            token=token,
            token_origins=token_origins,
            cache_mb=options.cache_mb,
        )

    def run(
        self,
        url: str,
        *,
        model: str,
        queries: list[str],
        output: Path,
        options: StreamingClassificationOptions,
        sensor: str | None = None,
        crs: str | None = None,
        cog: CogOptions | None = None,
        token: str | None = None,
        token_origins: tuple[str, ...] = (),
        no_warnings: bool = False,
    ) -> DiskClassifyResult:
        """Score native-size source windows and publish globally ranked GeoJSON.

        Args:
            url: STAC item URL/path or direct HTTP(S) GeoTIFF URL.
            model: Installed classification model identifier.
            queries: Ordered prompts; first drives threshold and ranking.
            output: Destination GeoJSON, replaced only after successful completion.
            options: Immutable batch/cache, bbox, top-k and threshold settings.
            sensor: Sensor override; required for direct COG input.
            crs: Missing-CRS assignment without general source reprojection.
            cog: Explicit direct raster bands or radiometry overrides.
            token: Bearer credential scoped to the original source origin.
            token_origins: Additional authorized direct origins.
            no_warnings: Suppress a full-precision real-model CPU estimate.

        Returns:
            Paths, counts and model/query provenance only. Scored chips, GeoJSON
            features and complete source arrays are never retained in the result.

        Raises:
            InvalidInputError: Input/options/prompts/model category are invalid.
            WhirldError: Source, model, inference or disk publication fails.
            KeyboardInterrupt: Interruption propagates after owned resources close.

        Ranking matches eager classification: descending primary score with
        original chip order for ties. Existing output survives incomplete runs.
        """
        source = RasterInput.resolve(url, cog)
        source.validate_sensor(sensor)
        if source.kind == "geotiff":
            raise InvalidInputError(
                "Streaming classification requires STAC or direct remote COG input."
            )
        queries = ClassificationQueries.from_arguments(queries=queries).queries
        configure_logging()
        started = time.monotonic()
        version = ""
        quantized = False
        resolved_sensor: str | None = None
        count: int | None = None
        error: str | None = None
        try:
            entry = self._store.registry.get(model)
            if entry.category != "classification":
                raise InvalidInputError("Streaming requires a classification model.")
            manifest = self._store.load_manifest(model)
            version, quantized = manifest.version, manifest.quantized
            with self._reader(
                source, entry, options, sensor, crs, token, token_origins
            ) as reader:
                resolved_sensor, count = reader.sensor, reader.chip_count
                warn_cpu(model, manifest, self._device, count, no_warnings)
                backend = load_backend(entry, manifest, self._device, store=self._store)
                metadata = ClassificationMetadata(
                    model=entry.name,
                    model_version=version,
                    crs=reader.crs,
                    queries=tuple(queries),
                )
                acquired = reader.metadata.resolve()
                wavelengths = entry.band_contract.sensors[reader.sensor].wavelengths
                with ClassificationStreamWriter(
                    output,
                    chip_count=count,
                    metadata=metadata,
                    top_k=options.top_k,
                    threshold=options.threshold,
                ) as writer:
                    for batch in reader.batches():
                        chipset = ChipSet(
                            chips=batch.chips,
                            chip_size_px=entry.band_contract.chip_size_px,
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
                        scores = backend.classify(batch.pixels, queries, context)
                        writer.write(scores, batch.chips)
                        del scores, batch, chipset, context
                    path = writer.finish()
                    features = writer.feature_count
                return DiskClassifyResult(
                    output_path=path,
                    chip_count=count,
                    feature_count=features,
                    model=entry.name,
                    sensor=reader.sensor,
                    device=self._device,
                    query=queries[0],
                    queries=tuple(queries),
                )
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            record_usage(
                command="classify",
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
