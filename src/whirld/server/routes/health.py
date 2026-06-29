"""``GET /health`` — server status, device, and loaded models (PRD section 6.1)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..._version import __version__
from ...core.session import ModelSession
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Report server status, the active device, and resident models.

    Args:
        request: The incoming request (carries the warm session in app state).

    Returns:
        A :class:`HealthResponse`.
    """
    session: ModelSession = request.app.state.session
    return HealthResponse(
        status="ok",
        device=session.device,
        models_loaded=session.loaded,
        version=__version__,
    )
