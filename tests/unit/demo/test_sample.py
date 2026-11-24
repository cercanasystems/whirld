"""Pinned packaged raster bytes, geometry and sidecar isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pydantic import ValidationError

from tests.fixtures.demo import make_demo_resource
from whirld.core.fetch import sha256_file
from whirld.demo.sample import BundledDemoSample, DemoSampleSpec
from whirld.errors import InvalidInputError


@pytest.fixture
def packaged_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Substitute only the importlib-resources directory for a generated TIFF."""
    resource = tmp_path / "resource"
    make_demo_resource(resource)
    monkeypatch.setattr("whirld.demo.sample.files", lambda package: resource)
    return resource


def test_sample_copy_preserves_pinned_bytes_and_valid_zeros(
    tmp_path: Path, packaged_sample: Path
) -> None:
    """Materialize exactly the packaged TIFF without losing zero-valued observations."""
    output = tmp_path / "output"
    output.mkdir()
    before = (packaged_sample / "data/starter-rgb.tif").read_bytes()
    spec, path = BundledDemoSample().prepare(output)
    assert path.read_bytes() == before
    assert sha256_file(path) == spec.sha256
    with rasterio.open(path) as source:
        assert source.read(1)[0, 0] == 0
        assert source.read_masks(1)[0, 0] == 255
        assert source.read_masks(1)[1, 1] == 0


@pytest.mark.parametrize("corruption", ["bytes", "size"])
def test_changed_packaged_artifact_fails_before_copy(
    tmp_path: Path, packaged_sample: Path, corruption: str
) -> None:
    """A valid-looking TIFF cannot silently change under the recorded file identity."""
    output = tmp_path / "output"
    output.mkdir()
    image = packaged_sample / "data/starter-rgb.tif"
    data = image.read_bytes()
    image.write_bytes(data + b"extra" if corruption == "size" else b"X" + data[1:])
    with pytest.raises(InvalidInputError, match="checksum"):
        BundledDemoSample().prepare(output)
    assert list(output.iterdir()) == []


def test_corrupt_metadata_has_actionable_domain_error(packaged_sample: Path) -> None:
    """Malformed or unavailable resource schemas fail as ordinary Whirld errors."""
    (packaged_sample / "data/sample.json").write_text("{broken")
    with pytest.raises(InvalidInputError, match="metadata"):
        BundledDemoSample.load()


@pytest.mark.parametrize(
    "field,value",
    [
        ("file", "../elsewhere.tif"),
        ("width", 449),
        ("height", 0),
        ("dtype", "uint8"),
        ("units", "reflectance"),
        ("nodata", 0),
        ("sha256", "bad"),
        ("size_bytes", True),
        ("band_names", ["B02", "B03", "B04"]),
    ],
)
def test_schema_rejects_ambiguous_sample_semantics(
    packaged_sample: Path, field: str, value: object
) -> None:
    """Manifest fields enforce the one supported first-run source contract."""
    fields = json.loads((packaged_sample / "data/sample.json").read_text())
    fields[field] = value
    with pytest.raises(ValidationError):
        DemoSampleSpec.model_validate(fields)


def test_existing_sample_destination_is_not_overwritten(
    tmp_path: Path, packaged_sample: Path
) -> None:
    """The materializer retains exclusive output ownership even when called directly."""
    output = tmp_path / "output"
    output.mkdir()
    target = output / "starter-rgb.tif"
    target.write_bytes(b"prior")
    with pytest.raises(FileExistsError):
        BundledDemoSample().prepare(output)
    assert target.read_bytes() == b"prior"


def test_external_mask_and_pam_do_not_change_pinned_interpretation(
    tmp_path: Path, packaged_sample: Path
) -> None:
    """Unpinned source companions are not copied or consulted for demo metadata."""
    image = packaged_sample / "data/starter-rgb.tif"
    before = sha256_file(image)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        with rasterio.open(image, "r+") as dataset:
            dataset.write_mask(np.zeros((448, 448), dtype=np.uint8))
    assert sha256_file(image) == before
    output = tmp_path / "output"
    output.mkdir()
    _, copied = BundledDemoSample().prepare(output)
    with rasterio.open(copied) as dataset:
        assert dataset.read_masks(1)[0, 0] == 255
    assert not (output / "starter-rgb.tif.msk").exists()


@pytest.mark.parametrize("change", ["bands", "crs", "nodata"])
def test_header_changes_fail_even_when_artifact_pin_is_updated(
    tmp_path: Path, packaged_sample: Path, change: str
) -> None:
    """Metadata-layer regressions cannot be hidden by accepting a different digest."""
    image = packaged_sample / "data/starter-rgb.tif"
    with rasterio.open(image, "r+") as dataset:
        if change == "bands":
            dataset.descriptions = ("B02", "B03", "B04")
        elif change == "crs":
            dataset.crs = "EPSG:32611"
        else:
            dataset.nodata = 0
    fields = json.loads((packaged_sample / "data/sample.json").read_text())
    fields.update(sha256=sha256_file(image), size_bytes=image.stat().st_size)
    (packaged_sample / "data/sample.json").write_text(json.dumps(fields))
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(InvalidInputError, match="header"):
        BundledDemoSample().prepare(output)
