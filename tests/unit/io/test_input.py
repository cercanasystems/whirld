"""Input dispatch and early validation without filesystem or remote I/O."""

from pathlib import Path

import pytest

from whirld.core.options import CogOptions
from whirld.errors import InvalidInputError
from whirld.io.input import RasterInput


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        ("https://example.org/image.TIFF?signature=private", "cog"),
        ("https://example.org/image.tif?filename=item.json", "cog"),
        ("https://example.org/items/scene", "stac"),
        ("https://example.org/item.json?asset=scene.tif", "stac"),
        ("item.json", "stac"),
        ("file:///tmp/item.json", "stac"),
        ("local.tif", "geotiff"),
    ],
)
def test_input_kind_preserves_url(value: str, kind: str) -> None:
    """Route by path without losing query authentication or exposing it in repr."""
    source = RasterInput.resolve(value)
    assert source.kind == kind
    assert source.location == value
    assert "signature=private" not in repr(source)


def test_extensionless_cog_requires_explicit_sensor() -> None:
    """Explicit metadata selects an extensionless endpoint without guessing sensor."""
    source = RasterInput.resolve("https://example.org/download", CogOptions(scale=1.0))
    assert source.kind == "cog"
    with pytest.raises(InvalidInputError, match="sensor"):
        source.validate_sensor(None)
    source.validate_sensor("sentinel-2-l2a")


@pytest.mark.parametrize(
    "value",
    [
        "scene.tif",
        "file:///tmp/scene.tif",
        "ftp://example.org/scene.tif",
        "https://example.org/item.json",
    ],
)
def test_cog_metadata_is_not_ignored_for_other_inputs(value: str) -> None:
    """Local files and explicit STAC JSON cannot silently discard COG overrides."""
    with pytest.raises(InvalidInputError, match="COG options"):
        RasterInput.resolve(value, CogOptions())


@pytest.mark.parametrize(
    "value",
    [
        "https://user:secret@example.org/scene.tif",
        "https://example.org:bad/scene.tif",
        "https:///scene.tif",
        "https://example.org/scene.tif#secret",
    ],
)
def test_direct_url_errors_are_sanitized(value: str) -> None:
    """Reject malformed URLs without reflecting credentials in the domain error."""
    with pytest.raises(InvalidInputError) as error:
        RasterInput.resolve(value)
    assert value not in str(error.value)
    assert "secret" not in str(error.value)


def test_local_path_object_retains_existing_dispatch(tmp_path: Path) -> None:
    """A pathlib TIFF stays a local raster and requires no remote sensor option."""
    source = RasterInput.resolve(tmp_path / "scene.tif")
    assert source.kind == "geotiff"
    source.validate_sensor(None)
