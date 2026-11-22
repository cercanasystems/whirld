"""Find similar Sentinel-2 landscape chips with real Clay v1.5 embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from rasterio.warp import transform_geom

from whirld.core.chips import ChipMeta
from whirld.errors import InvalidInputError
from whirld.io.embedding_stream import EmbeddingMetadata


class SimilarityMatch(BaseModel):
    """One ranked non-reference chip; cosine similarity is not a probability."""

    chip_index: int
    rank: int
    cosine_similarity: float


class SimilarityIndex:
    """Own a small example's normalized embeddings and deterministic ranking."""

    def __init__(self, embeddings: np.ndarray) -> None:
        """Validate and normalize an embedding matrix without changing its input.

        Args:
            embeddings: Finite floating-point matrix with at least two chip rows.

        Raises:
            InvalidInputError: Shape, finite values or vector norms are invalid.
        """
        if embeddings.ndim != 2 or len(embeddings) < 2 or not embeddings.shape[1]:
            raise InvalidInputError("Similarity needs at least two embedding rows.")
        if not np.isfinite(embeddings).all():
            raise InvalidInputError("Embeddings must contain only finite values.")
        values = embeddings.astype(np.float64)
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            raise InvalidInputError("Cannot rank zero-length or overflowing vectors.")
        self._normalized = values / norms[:, None]

    def rank(self, reference: int, limit: int = 5) -> list[SimilarityMatch]:
        """Rank other chips by cosine score, breaking exact ties by chip index.

        Args:
            reference: Zero-based row index to use as the query chip.
            limit: Positive maximum number of non-reference results.

        Returns:
            Descending scores with stable ties, excluding the query chip itself.

        Raises:
            InvalidInputError: Reference is out of bounds or limit is not positive.
        """
        if not 0 <= reference < len(self._normalized) or limit < 1:
            raise InvalidInputError("Reference index or result limit is invalid.")
        scores = np.clip(self._normalized @ self._normalized[reference], -1, 1)
        indices = np.argsort(-scores, kind="stable")
        selected = [int(index) for index in indices if index != reference][:limit]
        return [
            SimilarityMatch(
                chip_index=index, rank=rank, cosine_similarity=scores[index]
            )
            for rank, index in enumerate(selected, start=1)
        ]


class ChipMetadata(EmbeddingMetadata):
    """Embedding sidecar with its ordered projected chip footprints."""

    chips: list[ChipMeta]


class MatchGeometry(BaseModel):
    """RFC 7946 polygon with longitude/latitude coordinates."""

    type: Literal["Polygon"] = "Polygon"
    coordinates: list[list[list[float]]]


class MatchFeature(BaseModel):
    """A georeferenced result with rank and cosine score."""

    type: Literal["Feature"] = "Feature"
    geometry: MatchGeometry
    properties: SimilarityMatch


class MatchCollection(BaseModel):
    """Georeferenced matches; the reference is identified without ranking itself."""

    type: Literal["FeatureCollection"] = "FeatureCollection"
    model: str = "clay-v1.5"
    reference_chip: int
    source_crs: str
    features: list[MatchFeature]

    @classmethod
    def from_matches(
        cls, metadata: ChipMetadata, reference: int, matches: list[SimilarityMatch]
    ) -> MatchCollection:
        """Transform each projected chip footprint to GeoJSON longitude/latitude.

        Args:
            metadata: Actual embedding sidecar carrying source CRS and chip bounds.
            reference: Query row index, recorded as provenance.
            matches: Ranked indices returned by the similarity index.

        Returns:
            Typed RFC 7946 feature collection ready for JSON serialization.

        Raises:
            InvalidInputError: Chip order or match indices disagree with metadata.
            rasterio.errors.RasterioError: Coordinate transformation fails.
        """
        if [chip.index for chip in metadata.chips] != list(range(len(metadata.chips))):
            raise InvalidInputError("Embedding chip metadata must be in row order.")
        if not 0 <= reference < len(metadata.chips):
            raise InvalidInputError("Reference chip is absent from metadata.")
        features = []
        for match in matches:
            if not 0 <= match.chip_index < len(metadata.chips):
                raise InvalidInputError("Ranked chip is absent from metadata.")
            left, bottom, right, top = metadata.chips[match.chip_index].bbox
            geometry = transform_geom(
                metadata.crs,
                "EPSG:4326",
                {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [left, bottom],
                            [right, bottom],
                            [right, top],
                            [left, top],
                            [left, bottom],
                        ]
                    ],
                },
            )
            features.append(
                MatchFeature(
                    geometry=MatchGeometry.model_validate(geometry), properties=match
                )
            )
        return cls(reference_chip=reference, source_crs=metadata.crs, features=features)


class SimilarityOptions(BaseModel):
    """Validated user settings for the small CPU similarity example."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    output_dir: Path
    reference: int = Field(default=0, ge=0)
    top_k: int = Field(default=5, gt=0)
    threads: int = Field(default=2, gt=0)


class SimilarityExample:
    """Own real-model inference, ranking, geospatial output and preview creation."""

    def __init__(self, options: SimilarityOptions) -> None:
        """Bind validated settings; no downloads or file writes occur yet."""
        self.options = options

    def run(self) -> None:
        """Execute the pinned scene workflow and persist results in a new directory.

        Raises:
            WhirldError: Model installation, remote reads or inference fail.
            InvalidInputError: Embeddings or chip metadata fail validation.
            OSError: Outputs cannot be created or read.
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
        runtime.prepare("clay-v1.5")
        started = time.monotonic()
        output = self.options.output_dir
        result = whirld.embed_to_disk(
            SENTINEL_SCENE.item_url,
            model="clay-v1.5",
            output=output / "embeddings.npy",
            bbox=SENTINEL_SCENE.bbox,
            sensor=SENTINEL_SCENE.sensor,
            chip_size=256,
            batch_size=1,
            device="cpu",
            no_warnings=True,
        )
        embeddings = np.load(result.output_path, mmap_mode="r", allow_pickle=False)
        metadata = ChipMetadata.model_validate_json(result.meta_path.read_text())
        if embeddings.shape != (len(metadata.chips), 1024):
            raise InvalidInputError(
                "Real Clay output shape and chip metadata disagree."
            )
        matches = SimilarityIndex(embeddings).rank(
            self.options.reference, self.options.top_k
        )
        collection = MatchCollection.from_matches(
            metadata, self.options.reference, matches
        )
        (output / "matches.geojson").write_text(collection.model_dump_json(indent=2))
        query = MatchCollection.from_matches(
            metadata,
            self.options.reference,
            [
                SimilarityMatch(
                    chip_index=self.options.reference, rank=0, cosine_similarity=1.0
                )
            ],
        ).features[0]
        polygons = [
            OverlayPolygon(
                coordinates=feature.geometry.coordinates[0],
                label=(
                    f"#{feature.properties.rank} chip {feature.properties.chip_index}: "
                    f"{feature.properties.cosine_similarity:.3f}"
                ),
                color="#ffb000",
            )
            for feature in collection.features
        ]
        polygons.append(
            OverlayPolygon(
                coordinates=query.geometry.coordinates[0],
                label=f"Reference chip {self.options.reference}",
                color="#ffffff",
            )
        )
        Preview.write(
            output / "preview.html",
            title="Clay landscape similarity",
            scene=SENTINEL_SCENE,
            polygons=polygons,
            summary=(
                "Cosine similarity ranks other chips against the white reference. "
                "The query is excluded from matches. Scores are not probabilities "
                "or proof that two places share a land-cover class."
            ),
        )
        metrics = RunMetrics.capture(started)
        (output / "metrics.json").write_text(metrics.model_dump_json(indent=2))
        (output / "scene.json").write_text(SENTINEL_SCENE.model_dump_json(indent=2))
        logging.getLogger(__name__).info("Open %s", output / "preview.html")


def main() -> int:
    """Parse user settings and run the real CPU workflow.

    Returns:
        Zero after artifacts are written successfully.

    Raises:
        SystemExit: Arguments are invalid or help is requested.
        WhirldError: Setup or real inference fails; the cause remains visible.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    SimilarityExample(
        SimilarityOptions(
            output_dir=args.output_dir,
            reference=args.reference,
            top_k=args.top_k,
            threads=args.threads,
        )
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
