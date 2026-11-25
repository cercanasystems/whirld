"""Application-scoped inference service with an explicit model-store lifetime."""

from __future__ import annotations

from anyio import Semaphore, to_thread
from fastapi.responses import Response

from .. import api
from .._version import __version__
from ..core.session import ModelSession
from ..errors import WhirldError
from ..io.raster import read_raster_from_bytes
from ..models.providers import Operation
from .responses import (
    classification_response,
    embedding_response,
    segmentation_response,
)
from .schemas import (
    ClassifyRequest,
    EmbedRequest,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    SegmentRequest,
)


class InferenceService:
    """Own a warm session, its worker admission policy, and inference operations.

    Args:
        session: Explicit session whose registry and installation store are used
            for every operation. The service clears it when closed.

    One operation at a time materializes uploads, decodes rasters, uses a shared
    backend, and serializes its result. Waiting uploads remain spooled. Blocking
    model/status work executes off-loop; health reads an immutable cache snapshot.
    """

    def __init__(self, session: ModelSession) -> None:
        """Bind the owned session and create this service's inference admission slot."""
        self._session = session
        self._slots = Semaphore(1)
        self._closed = False

    @property
    def session(self) -> ModelSession:
        """Return the service's explicitly bound model session."""
        return self._session

    def _ensure_open(self) -> None:
        """Reject operations after resources have been released."""
        if self._closed:
            raise WhirldError("Inference service is closed.")

    async def start(self, preload: list[str] | None = None) -> None:
        """Preload requested models using this service's session and worker slot."""
        async with self._slots:
            self._ensure_open()
            if preload:
                await to_thread.run_sync(self._session.preload, preload)

    async def aclose(self) -> None:
        """Wait for active inference, then release resident models exactly once."""
        async with self._slots:
            if not self._closed:
                self._closed = True
                await to_thread.run_sync(self._session.clear)

    def health(self) -> HealthResponse:
        """Return a nonblocking snapshot without acquiring model-loading locks."""
        return HealthResponse(
            status="ok" if not self._closed else "closed",
            device=self._session.device,
            models_loaded=self._session.loaded,
            version=__version__,
        )

    async def models(self) -> ModelsResponse:
        """Inspect this store's installation state without blocking HTTP handling."""
        self._ensure_open()
        return await to_thread.run_sync(self._model_status)

    def _model_status(self) -> ModelsResponse:
        """Build registry/install metadata using the session's own dependencies."""
        store = self._session.store
        loaded = set(self._session.loaded)
        infos = []
        for name in store.registry.available():
            entry = store.registry.get(name)
            infos.append(
                ModelInfo(
                    name=entry.name,
                    version=entry.version,
                    category=entry.category,
                    display_name=entry.display_name,
                    sensors=entry.supported_sensors(),
                    installed=store.is_installed(name),
                    loaded=name in loaded,
                )
            )
        return ModelsResponse(models=infos)

    async def _require_operation(
        self, model: str, operation: Operation, *, head: str | None = None
    ) -> None:
        """Check a task off-loop before reading or decoding an upload.

        Args:
            model: Model identifier or the supported segmentation alias.
            operation: Requested public task.
            head: Optional segmentation head used to resolve the existing alias.

        Raises:
            WhirldError: The model is unknown or cannot perform the task.
        """
        resolved = (
            api._resolve_segment_model(model, head) if operation == "segment" else model
        )
        await to_thread.run_sync(self._session.require_operation, resolved, operation)

    async def embed(self, options: EmbedRequest) -> Response:
        """Materialize, embed, and serialize an upload within this service's bound."""
        async with self._slots:
            self._ensure_open()
            await self._require_operation(options.model, "embed")
            data = await options.file.read()
            return await to_thread.run_sync(self._embed, data, options)

    def _embed(self, data: bytes, options: EmbedRequest) -> Response:
        """Run the embedding pipeline against the bound warm session."""
        raster = read_raster_from_bytes(data, label=options.file.filename or "upload")
        result = api.embed_raster(
            raster,
            model=options.model,
            session=self._session,
            fmt="npy",
            chip_size=options.chip_size,
            overlap=options.overlap,
            sensor=options.sensor,
            batch_size=options.batch_size,
        )
        return embedding_response(result, options.format)

    async def classify(self, options: ClassifyRequest) -> Response:
        """Materialize and classify an upload while serializing shared model access."""
        async with self._slots:
            self._ensure_open()
            await self._require_operation(options.model, "classify")
            data = await options.file.read()
            return await to_thread.run_sync(self._classify, data, options)

    def _classify(self, data: bytes, options: ClassifyRequest) -> Response:
        """Run classification and GeoJSON serialization in the worker."""
        raster = read_raster_from_bytes(data, label=options.file.filename or "upload")
        result = api.classify_raster(
            raster,
            model=options.model,
            session=self._session,
            query=options.query,
            top_k=options.top_k,
            threshold=options.threshold,
            sensor=options.sensor,
        )
        return classification_response(result)

    async def segment(self, options: SegmentRequest) -> Response:
        """Materialize and segment an upload while serializing shared model access."""
        async with self._slots:
            self._ensure_open()
            await self._require_operation(options.model, "segment", head=options.head)
            data = await options.file.read()
            return await to_thread.run_sync(self._segment, data, options)

    def _segment(self, data: bytes, options: SegmentRequest) -> Response:
        """Run segmentation and GeoTIFF serialization in the worker."""
        raster = read_raster_from_bytes(data, label=options.file.filename or "upload")
        result = api.segment_raster(
            raster,
            model=options.model,
            session=self._session,
            head=options.head,
            threshold=options.threshold,
            sensor=options.sensor,
        )
        return segmentation_response(result)
