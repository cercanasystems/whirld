"""FastAPI application factory for ``whirld serve`` (PRD sections 5.8, 6).

``create_app`` builds an app that holds a single warm
:class:`~whirld.core.session.ModelSession` in ``app.state``. Models named at
startup are loaded immediately; any other installed model loads on first request
and stays resident. Domain errors (:class:`~whirld.errors.WhirldError`) are
translated to appropriate HTTP status codes by a single exception handler.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .._version import __version__
from ..core.session import ModelSession
from ..errors import (
    ChecksumMismatchError,
    InvalidInputError,
    ModelNotFoundError,
    ModelNotInstalledError,
    NetworkError,
    UnsupportedSensorError,
    WhirldError,
)
from ..logging_setup import configure_logging, get_logger
from .routes import classify, embed, health, models, segment

_log = get_logger("server.app")

# Map domain error types to HTTP status codes.
_STATUS_BY_ERROR: dict[type[WhirldError], int] = {
    ModelNotFoundError: 404,
    ModelNotInstalledError: 404,
    UnsupportedSensorError: 422,
    InvalidInputError: 422,
    NetworkError: 502,
    ChecksumMismatchError: 500,
}


def _status_for(exc: WhirldError) -> int:
    """Return the HTTP status code for a domain error (default 500)."""
    return _STATUS_BY_ERROR.get(type(exc), 500)


def create_app(
    device: str | None = None,
    preload: list[str] | None = None,
) -> FastAPI:
    """Build the Whirld FastAPI application.

    Args:
        device: Inference device for the session; ``None`` for auto-detect.
        preload: Model names to load into memory at startup; ``None`` for none.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Create the warm session and preload requested models at startup."""
        session = ModelSession(device=device)
        if preload:
            _log.info("Preloading models at startup: %s", ", ".join(preload))
            session.preload(preload)
        app.state.session = session
        yield
        session.clear()

    app = FastAPI(
        title="Whirld",
        version=__version__,
        summary="Local-first geospatial foundation models.",
        lifespan=lifespan,
    )

    @app.exception_handler(WhirldError)
    async def _whirld_error_handler(
        _request: Request, exc: WhirldError
    ) -> JSONResponse:
        """Translate domain errors into JSON responses with mapped status codes."""
        status = _status_for(exc)
        _log.info("%s -> HTTP %d", type(exc).__name__, status)
        return JSONResponse(
            status_code=status,
            content={"error": type(exc).__name__, "detail": exc.message},
        )

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(embed.router)
    app.include_router(segment.router)
    app.include_router(classify.router)
    return app
