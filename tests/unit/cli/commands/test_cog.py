"""Direct COG CLI metadata conversion tests."""

import pytest

from whirld.cli.commands._cog import CogArguments
from whirld.core.options import CogOptions
from whirld.errors import InvalidInputError


def test_cli_values_match_python_schema() -> None:
    """Scalar/per-band CLI values preserve explicit identity reflectance intent."""
    assert CogArguments().options() is None
    result = CogArguments(bands="B03, B02", scale="1", offset="-0.1,0").options()
    assert result == CogOptions(bands=("B03", "B02"), scale=1.0, offset=(-0.1, 0.0))
    with pytest.raises(InvalidInputError):
        CogArguments(scale="1,wat").options()
    with pytest.raises(InvalidInputError):
        CogArguments(offset="nan").options()
