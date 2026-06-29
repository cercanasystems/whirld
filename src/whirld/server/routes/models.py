"""``GET /models`` — list installed models with metadata (PRD section 6.2)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ...core.fetch import is_installed
from ...core.registry import Registry
from ...core.session import ModelSession
from ..schemas import ModelInfo, ModelsResponse

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
async def list_models(request: Request) -> ModelsResponse:
    """Return every registry model with install/loaded status and metadata.

    Args:
        request: The incoming request (carries the warm session in app state).

    Returns:
        A :class:`ModelsResponse`.
    """
    session: ModelSession = request.app.state.session
    registry = Registry()
    loaded = set(session.loaded)

    infos: list[ModelInfo] = []
    for name in registry.available():
        entry = registry.get(name)
        infos.append(
            ModelInfo(
                name=entry.name,
                version=entry.version,
                category=entry.category,
                display_name=entry.display_name,
                sensors=entry.supported_sensors(),
                installed=is_installed(name),
                loaded=name in loaded,
            )
        )
    return ModelsResponse(models=infos)
