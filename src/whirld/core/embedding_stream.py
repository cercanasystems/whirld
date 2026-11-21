"""Resource-owned, bounded-memory STAC embedding execution."""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..errors import InvalidInputError
from ..io.embedding_stream import EmbeddingMetadata, EmbeddingStreamWriter
from ..io.stac_stream import StacWindowReader
from ..logging_setup import configure_logging, record_usage
from ..models.base import InferenceContext, detect_device
from ..models.loader import load_backend
from .chips import ChipSet
from .fetch import ModelStore
from .inference_metadata import chip_latlons, warn_cpu
from .options import InferenceOptions, StreamingEmbeddingOptions


class DiskEmbedResult(BaseModel):
    """Completed disk output summary, without resident scene arrays or chip lists."""

    model_config = ConfigDict(frozen=True)
    output_path: Path
    meta_path: Path
    chip_count: int
    embed_dim: int
    model: str
    sensor: str
    device: str


class EmbeddingStream:
    """Own a store and execute one backend over bounded reader/writer batches.

    Args:
        store: Registry/cache identity shared by model resolution and loading.
        device: Inference device; None chooses the available default.

    Each run owns its reader and staged output resources until publication. No
    scene-sized pixels, embeddings, or chip metadata are retained by this object.
    """

    def __init__(self, store: ModelStore, device: str | None = None) -> None:
        """Bind installation identity and resolve the inference device once."""
        self._store = store
        self._device = detect_device(device)

    def run(
        self,
        url: str,
        *,
        model: str,
        output: Path,
        options: StreamingEmbeddingOptions,
        sensor: str | None = None,
        crs: str | None = None,
        token: str | None = None,
        token_origins: tuple[str, ...] = (),
        acquisition_datetime: str | None = None,
        no_warnings: bool = False,
    ) -> DiskEmbedResult:
        """Stream a STAC item's embeddings into a compatible NPY/JSON pair.

        Args:
            url: HTTP(S) item URL, file URL, or local STAC JSON path.
            model: Installed embedding model identifier in this stream's store.
            output: NPY destination; a missing .npy suffix is appended.
            options: Validated chip, overlap, batch, bbox, and cache configuration.
            sensor: Sensor override; otherwise infer it from item metadata.
            crs: Source CRS assignment for assets whose CRS is absent.
            token: Bearer credential authorized for the original item origin.
            token_origins: Additional direct asset origins authorized for token.
            acquisition_datetime: Timestamp override, tried before source metadata.
            no_warnings: Suppress the real-model CPU runtime estimate.

        Returns:
            Published output paths, chip count, embedding width, and model/sensor/
            device identifiers. No scene-sized arrays or metadata lists are retained.

        Raises:
            InvalidInputError: The model cannot embed or options are invalid.
            WhirldError: Installation, source, inference, or output fails.

        Failed runs clean staging files; previously published outputs survive
        inference failures. Usage records contain only non-sensitive summaries.
        """
        configure_logging()
        started = time.monotonic()
        version = ""
        quantized = False
        resolved_sensor: str | None = None
        count: int | None = None
        error: str | None = None
        try:
            entry = self._store.registry.get(model)
            if entry.category != "embedding" or entry.output.embed_dim is None:
                raise InvalidInputError("Streaming requires an embedding model.")
            manifest = self._store.load_manifest(model)
            version, quantized = manifest.version, manifest.quantized
            size = options.chip_size or entry.band_contract.chip_size_px
            options = StreamingEmbeddingOptions.from_arguments(
                **(options.model_dump() | {"chip_size": size})
            )
            with StacWindowReader(
                url,
                entry=entry,
                chip_size=size,
                overlap=options.overlap,
                batch_size=options.batch_size,
                sensor=sensor,
                crs=crs,
                bbox=options.bbox,
                token=token,
                token_origins=token_origins,
                cache_mb=options.cache_mb,
            ) as reader:
                resolved_sensor, count = reader.sensor, reader.chip_count
                warn_cpu(model, manifest, self._device, count, no_warnings)
                backend = load_backend(entry, manifest, self._device, store=self._store)
                acquired = reader.metadata.resolve(acquisition_datetime)
                wavelengths = entry.band_contract.sensors[reader.sensor].wavelengths
                metadata = EmbeddingMetadata(
                    model=entry.name,
                    model_version=version,
                    crs=reader.crs,
                    embed_dim=entry.output.embed_dim,
                    chip_size_px=size,
                    resolution_m=reader.resolution_m,
                )
                with EmbeddingStreamWriter(
                    output, chip_count=count, metadata=metadata
                ) as writer:
                    for batch in reader.batches():
                        chipset = ChipSet(
                            chips=batch.chips,
                            chip_size_px=size,
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
                        embeddings = backend.embed(batch.pixels, context)
                        writer.write(embeddings, batch.chips)
                        del embeddings, batch, chipset, context
                    data_path, meta_path = writer.finish()
                return DiskEmbedResult(
                    output_path=data_path,
                    meta_path=meta_path,
                    chip_count=count,
                    embed_dim=metadata.embed_dim,
                    model=entry.name,
                    sensor=reader.sensor,
                    device=self._device,
                )
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            record_usage(
                command="embed",
                model=model,
                model_version=version,
                input_type="stac",
                sensor_detected=resolved_sensor,
                chip_count=count,
                device=self._device,
                quantized=quantized,
                duration_ms=int((time.monotonic() - started) * 1000),
                error=error,
            )
