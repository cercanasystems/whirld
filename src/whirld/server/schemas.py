"""Pydantic response/request schemas for the REST API (PRD section 6).

Typed schemas keep raw dicts out of the route boundary and give FastAPI an
accurate OpenAPI description.
"""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response body for ``GET /health`` (PRD section 6.1).

    Attributes:
        status: Always ``ok`` when the server is responding.
        device: The server's resolved inference device.
        models_loaded: Names of models currently resident in memory.
        version: Whirld version string.
    """

    status: str
    device: str
    models_loaded: list[str]
    version: str


class ModelInfo(BaseModel):
    """Summary metadata for one installed model (``GET /models``).

    Attributes:
        name: Model identifier.
        version: Installed model version.
        category: ``embedding`` / ``segmentation`` / ``classification``.
        display_name: Human-readable name.
        sensors: Supported sensor keys.
        installed: Whether the model has been pulled locally.
        loaded: Whether the model is currently resident in memory.
    """

    name: str
    version: str
    category: str
    display_name: str
    sensors: list[str]
    installed: bool
    loaded: bool


class ModelsResponse(BaseModel):
    """Response body for ``GET /models``.

    Attributes:
        models: One :class:`ModelInfo` per registry model.
    """

    models: list[ModelInfo]


class ErrorResponse(BaseModel):
    """Standard error envelope returned for handled failures.

    Attributes:
        error: Exception class name (e.g. ``ModelNotInstalledError``).
        detail: Actionable, human-readable message.
    """

    error: str
    detail: str
