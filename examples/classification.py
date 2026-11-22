"""Compare competing landscape prompts using real RemoteCLIP scores."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from whirld.errors import InvalidInputError

PROMPTS = (
    "a satellite image of farmland",
    "a satellite image of a forest",
    "a satellite image of water",
    "a satellite image of an urban area",
)
COLORS = ("#e8b339", "#2ca25f", "#3182bd", "#d95f8d")
EXPECTED_CHIPS = 16


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


class ClassificationOptions(BaseModel):
    """Validated user settings for the small CPU classification example."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    output_dir: Path
    threads: int = Field(default=2, gt=0)


class ClassificationExample:
    """Own real RemoteCLIP inference, score checks and preview generation."""

    def __init__(self, options: ClassificationOptions) -> None:
        """Bind validated settings without writing files or loading weights."""
        self.options = options

    def run(self) -> None:
        """Score all scene chips against fixed prompts and persist viewable outputs.

        Raises:
            WhirldError: Missing weights, remote I/O or inference fails.
            InvalidInputError: Probabilities or output geometry are invalid.
            OSError: An output file cannot be created.
        """
        import logging
        import time

        import whirld
        from examples.common import (
            SENTINEL_SCENE,
            ExampleRuntime,
            OverlayPolygon,
            Preview,
            RunMetrics,
        )

        runtime = ExampleRuntime(self.options.output_dir, threads=self.options.threads)
        runtime.prepare("remoteclip")
        started = time.monotonic()
        output = self.options.output_dir
        result = whirld.classify(
            SENTINEL_SCENE.item_url,
            model="remoteclip",
            query=list(PROMPTS),
            top_k=EXPECTED_CHIPS,
            threshold=0.0,
            bbox=SENTINEL_SCENE.bbox,
            sensor=SENTINEL_SCENE.sensor,
            device="cpu",
            no_warnings=True,
            output=output / "scores.geojson",
        )
        collection = ScoreCollection.model_validate(result.feature_collection)
        collection.validate_probabilities()
        if (
            len(result.chips) != EXPECTED_CHIPS
            or len(collection.features) != EXPECTED_CHIPS
        ):
            raise InvalidInputError(
                "Pinned region must produce exactly 16 chips and retain every score."
            )
        polygons = []
        for feature in collection.features:
            scores = feature.properties.scores
            winner = max(range(len(PROMPTS)), key=lambda index: scores[PROMPTS[index]])
            prompt = PROMPTS[winner]
            polygons.append(
                OverlayPolygon(
                    coordinates=feature.geometry.coordinates[0],
                    label=(
                        f"chip {feature.properties.chip_index}: "
                        f"{prompt.removeprefix('a satellite image of ')} "
                        f"{scores[prompt]:.2f}"
                    ),
                    color=COLORS[winner],
                )
            )
        Preview.write(
            output / "preview.html",
            title="RemoteCLIP competing landscape prompts",
            scene=SENTINEL_SCENE,
            polygons=polygons,
            summary=(
                "Colors show the highest-scoring prompt per chip: gold farmland, "
                "green forest, blue water, pink urban. Scores are normalized only "
                "over these four prompts, not calibrated land-cover confidence. "
                "Sentinel-2 reflectance differs from training RGB imagery."
            ),
        )
        (output / "metrics.json").write_text(
            RunMetrics.capture(started).model_dump_json(indent=2)
        )
        (output / "scene.json").write_text(SENTINEL_SCENE.model_dump_json(indent=2))
        logging.getLogger(__name__).info("Open %s", output / "preview.html")


def main() -> int:
    """Parse settings and run real RemoteCLIP inference on the pinned scene.

    Returns:
        Zero after successful artifact generation.

    Raises:
        SystemExit: Arguments are invalid or help is requested.
        WhirldError: Setup or inference fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    ClassificationExample(
        ClassificationOptions(output_dir=args.output_dir, threads=args.threads)
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
