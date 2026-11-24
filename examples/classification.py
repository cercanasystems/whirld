"""Compare competing landscape prompts using real RemoteCLIP scores."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from whirld.demo.scores import (
    COLORS as COLORS,
)
from whirld.demo.scores import (
    PROMPTS as PROMPTS,
)
from whirld.demo.scores import (
    ScoreCollection as ScoreCollection,
)
from whirld.demo.scores import (
    ScoreFeature as ScoreFeature,
)
from whirld.demo.scores import (
    ScoreGeometry as ScoreGeometry,
)
from whirld.demo.scores import (
    ScoreProperties as ScoreProperties,
)
from whirld.errors import InvalidInputError

EXPECTED_CHIPS = 16


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
