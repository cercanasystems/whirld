"""Pydantic response/request schemas for the REST API (PRD section 6).

Typed schemas keep raw dicts out of the route boundary and give FastAPI an
accurate OpenAPI description.
"""

from __future__ import annotations

from typing import Literal

from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RasterRequest(BaseModel):
    """Multipart raster upload and model selection shared by inference routes."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    file: UploadFile
    model: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9.-]*$")
    sensor: str | None = None


class EmbedRequest(RasterRequest):
    """Embedding options exposed directly in multipart OpenAPI schemas."""

    chip_size: int | None = Field(default=None, gt=0)
    overlap: int = Field(default=0, ge=0)
    batch_size: int | None = Field(default=None, gt=0)
    format: Literal["npz", "npy", "json"] = "npz"

    @model_validator(mode="after")
    def valid_overlap(self) -> EmbedRequest:
        """Reject an overlap that would prevent forward chip traversal."""
        if self.chip_size is not None and self.overlap >= self.chip_size:
            raise ValueError("overlap must be less than chip_size")
        return self


class ClassifyRequest(RasterRequest):
    """Classification queries and bounded probability/ranking options."""

    query: list[str] = Field(min_length=1)
    top_k: int = Field(default=5, gt=0)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("query")
    @classmethod
    def nonempty_queries(cls, values: list[str]) -> list[str]:
        """Require meaningful text for every submitted query."""
        if any(not value.strip() for value in values):
            raise ValueError("query entries must not be empty")
        return values


class SegmentRequest(RasterRequest):
    """Segmentation head selection and finite probability threshold."""

    head: str | None = None
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)


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
