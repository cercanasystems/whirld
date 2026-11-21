"""Regression coverage for public option domains, scene time, and URL outputs."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from whirld import api
from whirld.errors import InvalidInputError
from whirld.models._clay_metadata import time_vector
from whirld.models.base import InferenceContext


@pytest.mark.parametrize(
    "options",
    [
        {"chip_size": 0},
        {"chip_size": -1},
        {"chip_size": True},
        {"overlap": -1},
        {"chip_size": 16, "overlap": 16},
        {"batch_size": 0},
        {"batch_size": -1},
        {"fmt": "nonsense"},
        {"bbox": (0, 0, float("nan"), 1)},
        {"bbox": (1, 0, 0, 1)},
        {"bbox": (-181, 0, 0, 1)},
        {"bbox": (0, 0, 1, 91)},
    ],
)
def test_invalid_embedding_options_fail_before_loading(
    options: dict[str, object],
) -> None:
    """Bad arguments fail without looking up a model or opening an input."""
    with pytest.raises(InvalidInputError):
        api.embed("missing.tif", model="does-not-exist", **options)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_probability_thresholds_are_validated(threshold: float) -> None:
    """Both score-based entry points reject non-probability thresholds."""
    with pytest.raises(InvalidInputError):
        api.classify("missing.tif", model="missing", query="water", threshold=threshold)
    with pytest.raises(InvalidInputError):
        api.segment("missing.tif", model="missing", threshold=threshold)


def test_invalid_top_k_is_rejected() -> None:
    """Zero results is an invalid top-k request rather than an empty success."""
    with pytest.raises(InvalidInputError):
        api.classify("missing.tif", model="missing", query="water", top_k=0)


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            "https://user:secret@example.test/catalog/scene.json?token=secret",
            "scene_embeddings.npy",
        ),
        ("https://example.test/", "scene_embeddings.npy"),
        ("https://example.test/%2e%2e%2fscene.json", "_scene_embeddings.npy"),
        ("file:///tmp/satellite%20scene.json", "/tmp/satellite scene_embeddings.npy"),
        ("/tmp/scene.tif", "/tmp/scene_embeddings.npy"),
    ],
)
def test_default_output_locations(value: str, expected: str) -> None:
    """URL defaults are safe local basenames; local paths stay adjacent."""
    assert api._default_output_path(value) == Path(expected)


def test_url_embedding_writes_in_current_directory(
    whirld_home: Path,
    s2_tif: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise actual default output writing without fetching a remote scene."""
    from whirld.io.raster import read_raster

    api.pull("clay-v1")
    raster = read_raster(s2_tif)
    monkeypatch.setattr(api, "_read_input", lambda *args, **kwargs: raster)
    monkeypatch.chdir(tmp_path)
    result = api.embed(
        "https://example.test/catalog/scene.json?secret=token", model="clay-v1"
    )
    assert result.output_path == Path("scene_embeddings.npy")
    assert result.output_path.exists()
    assert not (tmp_path / "https:").exists()


def test_stac_iso_timestamp_and_tiff_timestamp_agree() -> None:
    """TIFF-format and STAC ISO values preserve the same known UTC instant."""
    expected = datetime(2024, 6, 1, 10, tzinfo=timezone.utc)
    for tag in (
        "2024-06-01T10:00:00Z",
        "2024:06:01 10:00:00",
        "2024-06-01T06:00:00-04:00",
    ):
        actual = api._resolve_datetime(None, {"TIFFTAG_DATETIME": tag})
        assert actual == expected
        assert actual.tzinfo == timezone.utc
        np.testing.assert_allclose(time_vector(actual), time_vector(expected))


def test_equivalent_instants_have_identical_clay_vectors() -> None:
    """Timezone offsets cannot shift hour or ISO week in model metadata."""
    utc = datetime.fromisoformat("2024-01-01T00:00:00+00:00")
    offset = datetime.fromisoformat("2023-12-31T19:00:00-05:00")
    assert time_vector(utc) == time_vector(offset)
    assert api._resolve_datetime(offset.isoformat(), {}) == utc


def test_scene_time_reaches_backend(
    whirld_home: Path,
    s2_tif: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared orchestration passes parsed scene acquisition time to inference."""
    from whirld.io.raster import read_raster
    from whirld.models.clay import ClayBackend

    api.pull("clay-v1")
    raster = read_raster(s2_tif)
    raster.tags["TIFFTAG_DATETIME"] = "2024-06-01T06:00:00-04:00"
    monkeypatch.setattr(api, "_read_input", lambda *args, **kwargs: raster)
    seen = []
    original = ClayBackend.embed

    def capture(
        self: Any, chips: np.ndarray, context: InferenceContext | None = None
    ) -> np.ndarray:
        """Observe metadata at the actual inference boundary."""
        assert context is not None
        seen.append(context.acquisition_datetime)
        return original(self, chips, context)

    monkeypatch.setattr(ClayBackend, "embed", capture)
    api.embed(s2_tif, model="clay-v1", write=False)
    assert seen == [datetime(2024, 6, 1, 10, tzinfo=timezone.utc)]


@pytest.mark.parametrize(
    "args",
    [
        ["embed", "--chip-size", "0"],
        ["embed", "--overlap", "-1"],
        ["embed", "--batch-size", "0"],
        ["embed", "--format", "unknown"],
        ["classify", "--query", "water", "--threshold", "nan"],
        ["segment", "--threshold", "1.1"],
    ],
)
def test_cli_invalid_domains_use_documented_exit(
    whirld_home: Path, args: list[str]
) -> None:
    """CLI users get the invalid-input exit code rather than a Python exception."""
    from typer.testing import CliRunner

    from whirld.cli import app

    result = CliRunner().invoke(app, [*args, "missing.tif", "--model", "missing"])
    assert result.exit_code == 7, result.output


@pytest.mark.parametrize("fail", [False, True])
def test_warm_batch_override_does_not_leak_to_next_request(
    whirld_home: Path, s2_tif: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    """Each request carries its own batch policy on success and model failure."""
    from whirld.core.session import ModelSession
    from whirld.errors import WhirldError
    from whirld.io.raster import read_raster

    api.pull("clay-v1")
    session = ModelSession(device="cpu")
    backend = session.get("clay-v1").backend
    original = backend.embed
    observed: list[int | None] = []

    def capture(chips: np.ndarray, context: InferenceContext | None) -> np.ndarray:
        """Check the request's active batch size at the model boundary."""
        assert context is not None
        observed.append(context.options.batch_size)
        if fail and len(observed) == 1:
            raise WhirldError("simulated model failure")
        return original(chips, context)

    monkeypatch.setattr(backend, "embed", capture)
    if fail:
        with pytest.raises(WhirldError, match="simulated model failure"):
            api.embed_raster(
                read_raster(s2_tif), model="clay-v1", session=session, batch_size=2
            )
    else:
        api.embed_raster(
            read_raster(s2_tif), model="clay-v1", session=session, batch_size=2
        )
    api.embed_raster(read_raster(s2_tif), model="clay-v1", session=session)
    assert observed == [2, None]
