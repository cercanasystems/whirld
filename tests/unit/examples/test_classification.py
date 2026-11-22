"""Hermetic probability and spatial checks for the real RemoteCLIP example."""

from __future__ import annotations

import pytest

from examples.classification import (
    EXPECTED_CHIPS,
    ScoreCollection,
    ScoreFeature,
    ScoreGeometry,
    ScoreProperties,
)
from whirld.core.options import ClassificationOptions
from whirld.errors import InvalidInputError


def _collection() -> ScoreCollection:
    """Build a valid two-prompt classification artifact without a model runtime."""
    return ScoreCollection(
        type="FeatureCollection",
        source_crs="EPSG:32610",
        model="remoteclip",
        model_version="1.0.0",
        whirld_version="0.1.0",
        query="forest",
        queries=["forest", "water"],
        features=[
            ScoreFeature(
                type="Feature",
                geometry=ScoreGeometry(
                    type="Polygon",
                    coordinates=[[[-122, 38], [-121, 38], [-121, 39], [-122, 38]]],
                ),
                properties=ScoreProperties(
                    score=0.7,
                    query="forest",
                    scores={"forest": 0.7, "water": 0.3},
                    model="remoteclip",
                    chip_index=0,
                    row=0,
                    col=0,
                ),
            )
        ],
    )


def test_valid_relative_prompt_scores_pass() -> None:
    """Finite normalized probabilities and geographic polygons pass inspection."""
    _collection().validate_probabilities()


def test_pinned_chip_count_satisfies_public_classification_contract() -> None:
    """The example's all-chip request must pass public positive-top-k validation."""
    options = ClassificationOptions.from_arguments(top_k=EXPECTED_CHIPS, threshold=0.0)
    assert options.top_k == 16
    with pytest.raises(InvalidInputError):
        ClassificationOptions.from_arguments(top_k=0)


@pytest.mark.parametrize(
    "scores",
    [
        {"forest": 0.7, "water": 0.7},
        {"forest": float("nan"), "water": 0.3},
        {"forest": 1.1, "water": -0.1},
        {"forest": 1.0},
    ],
)
def test_invalid_scores_are_rejected(scores: dict[str, float]) -> None:
    """Reject unnormalized, nonfinite, negative, or missing prompt scores."""
    collection = _collection()
    collection.features[0].properties.scores = scores
    with pytest.raises(InvalidInputError):
        collection.validate_probabilities()


@pytest.mark.parametrize("failure", ["projected", "unclosed", "duplicate", "primary"])
def test_invalid_output_contract_is_rejected(failure: str) -> None:
    """Detect coordinates, chip duplication and primary-score inconsistencies."""
    collection = _collection()
    if failure == "projected":
        collection.features[0].geometry.coordinates[0][0][0] = 500000
    elif failure == "unclosed":
        collection.features[0].geometry.coordinates[0][-1] = [-121, 39]
    elif failure == "duplicate":
        collection.features.append(collection.features[0])
    else:
        collection.features[0].properties.score = 0.6
    with pytest.raises(InvalidInputError):
        collection.validate_probabilities()
