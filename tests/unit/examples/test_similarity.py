"""Hermetic spatial and ranking checks for the real Clay example."""

from __future__ import annotations

import numpy as np
import pytest

from examples.similarity import ChipMetadata, MatchCollection, SimilarityIndex
from whirld.core.chips import ChipMeta
from whirld.errors import InvalidInputError


def test_ranking_excludes_reference_and_breaks_ties_by_index() -> None:
    """Nearest chips rank by direction, not length; exact ties preserve chip order."""
    vectors = np.array([[1, 0], [2, 0], [0, 1], [-1, 0], [3, 0]], dtype=float)
    matches = SimilarityIndex(vectors).rank(reference=0, limit=20)
    assert [match.chip_index for match in matches] == [1, 4, 2, 3]
    assert [match.rank for match in matches] == [1, 2, 3, 4]
    assert [match.cosine_similarity for match in matches] == [1.0, 1.0, 0.0, -1.0]
    np.testing.assert_array_equal(vectors[1], [2, 0])


@pytest.mark.parametrize(
    "vectors",
    [
        np.array([[0, 0], [1, 2]]),
        np.array([[np.nan, 0], [1, 2]]),
        np.array([[np.inf, 0], [1, 2]]),
        np.array([[1, 2]]),
        np.ones((2, 0)),
    ],
)
def test_invalid_embeddings_are_rejected(vectors: np.ndarray) -> None:
    """Zero/nonfinite vectors and unusable matrices cannot yield plausible scores."""
    with pytest.raises(InvalidInputError):
        SimilarityIndex(vectors)


@pytest.mark.parametrize("reference,limit", [(-1, 1), (2, 1), (0, 0)])
def test_invalid_ranking_request_is_rejected(reference: int, limit: int) -> None:
    """Reject absent reference chips and empty result limits."""
    with pytest.raises(InvalidInputError):
        SimilarityIndex(np.eye(2)).rank(reference, limit)


def test_matches_use_wgs84_and_retain_the_right_chip() -> None:
    """Projected bounds become longitude/latitude while preserving selected IDs."""
    metadata = ChipMetadata(
        model="clay-v1.5",
        model_version="1.5.0",
        crs="EPSG:32610",
        embed_dim=1024,
        chip_size_px=256,
        resolution_m=10.0,
        chips=[
            ChipMeta(index=0, row=0, col=0, bbox=[500000, 4200000, 502560, 4202560]),
            ChipMeta(index=1, row=0, col=1, bbox=[502560, 4200000, 505120, 4202560]),
        ],
    )
    matches = SimilarityIndex(np.array([[1, 0], [1, 1]])).rank(0)
    collection = MatchCollection.from_matches(metadata, 0, matches)
    feature = collection.features[0]
    assert feature.properties.chip_index == 1
    assert collection.reference_chip == 0
    ring = feature.geometry.coordinates[0]
    assert ring[0] == ring[-1]
    assert -123 < ring[0][0] < -122
    assert 37 < ring[0][1] < 39
