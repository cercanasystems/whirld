"""Typed probability and footprint validation shared by demo and example."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from ..errors import InvalidInputError

PROMPTS = (
    "a satellite image of farmland",
    "a satellite image of a forest",
    "a satellite image of water",
    "a satellite image of an urban area",
)
COLORS = ("#e8b339", "#2ca25f", "#3182bd", "#d95f8d")


class ScoreGeometry(BaseModel):
    """RFC 7946 polygon geometry from Whirld's public classification output."""

    type: Literal["Polygon"]
    coordinates: list[list[list[float]]]


class ScoreProperties(BaseModel):
    """Typed public per-chip probabilities and source-grid indices."""

    score: float
    query: str
    scores: dict[str, float]
    model: str
    chip_index: int
    row: int
    col: int


class ScoreFeature(BaseModel):
    """One classified chip and its WGS84 footprint."""

    type: Literal["Feature"]
    geometry: ScoreGeometry
    properties: ScoreProperties


class ScoreCollection(BaseModel):
    """Typed public GeoJSON contract, preserving model and query provenance."""

    type: Literal["FeatureCollection"]
    source_crs: str
    model: str
    model_version: str
    whirld_version: str
    query: str
    queries: list[str]
    features: list[ScoreFeature]

    def validate_probabilities(self) -> None:
        """Check score normalization, prompt identity and geographic coordinates.

        Raises:
            InvalidInputError: Empty output, mismatched prompts, invalid probability
                vectors, duplicate chips, or invalid WGS84 polygon coordinates.
        """
        if (
            not self.features
            or not self.queries
            or self.query not in self.queries
            or len(set(self.queries)) != len(self.queries)
        ):
            raise InvalidInputError("Classification needs unique prompts and chips.")
        indices = [feature.properties.chip_index for feature in self.features]
        if len(indices) != len(set(indices)):
            raise InvalidInputError("Classification contains duplicate chips.")
        for feature in self.features:
            properties = feature.properties
            if set(properties.scores) != set(self.queries):
                raise InvalidInputError("Scores do not match the requested prompts.")
            scores = np.asarray([properties.scores[prompt] for prompt in self.queries])
            if (
                not np.isfinite(scores).all()
                or np.any(scores < 0)
                or np.any(scores > 1)
                or not np.isclose(scores.sum(), 1.0, atol=1e-5, rtol=0)
            ):
                raise InvalidInputError("Prompt scores must be finite probabilities.")
            if (
                properties.query != self.query
                or properties.score != properties.scores[self.query]
            ):
                raise InvalidInputError("Primary score disagrees with prompt scores.")
            if not feature.geometry.coordinates:
                raise InvalidInputError("Classification footprint is empty.")
            for ring in feature.geometry.coordinates:
                points = np.asarray(ring)
                if (
                    points.ndim != 2
                    or points.shape[1] != 2
                    or len(points) < 4
                    or not np.isfinite(points).all()
                    or not np.array_equal(points[0], points[-1])
                    or np.any(np.abs(points[:, 0]) > 180)
                    or np.any(np.abs(points[:, 1]) > 90)
                ):
                    raise InvalidInputError("Classification footprint is not WGS84.")
