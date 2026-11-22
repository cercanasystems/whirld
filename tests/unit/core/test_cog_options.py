"""Public direct COG metadata validation before remote reads."""

import pytest
from pydantic import ValidationError

from whirld.core.options import CogOptions


@pytest.mark.parametrize(
    "options",
    [
        {"bands": ()},
        {"bands": ("B02", "b02")},
        {"bands": ("B02", " ")},
        {"scale": float("nan")},
        {"scale": True},
        {"offset": (1.0, float("inf"))},
        {"offset": ()},
    ],
)
def test_invalid_metadata_fails_before_open(options: dict[str, object]) -> None:
    """Require finite transforms and unique complete source-name declarations."""
    with pytest.raises(ValidationError):
        CogOptions(**options)
