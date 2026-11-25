"""Unsupported provider tasks fail consistently before expensive execution."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, NoReturn, cast

import pytest
from click import unstyle
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile
from typer.testing import CliRunner

from whirld import api
from whirld.cli import app
from whirld.core import classification_stream, embedding_stream, segmentation_stream
from whirld.core import session as session_module
from whirld.core.fetch import ModelStore
from whirld.core.session import ModelSession
from whirld.errors import UnsupportedOperationError
from whirld.io.raster import RasterSource
from whirld.models.providers import Operation
from whirld.server import inference
from whirld.server.app import create_app
from whirld.server.inference import InferenceService

_MODELS: tuple[tuple[str, Operation], ...] = (
    ("clay-v1", "embed"),
    ("clay-v1.5", "embed"),
    ("remoteclip", "classify"),
    ("remoteclip-rn50", "classify"),
    ("remoteclip-vit-l14", "classify"),
    ("prithvi-burn-scar", "segment"),
    ("prithvi-flood", "segment"),
)
_OPERATIONS: tuple[Operation, ...] = ("embed", "classify", "segment")
_UNSUPPORTED = tuple(
    (model, operation, supported)
    for model, supported in _MODELS
    for operation in _OPERATIONS
    if operation != supported
)


@pytest.fixture
def provider_session(whirld_home: Path) -> ModelSession:
    """Create an isolated CPU session without pulling or loading any model."""
    return ModelSession(device="cpu")


@pytest.fixture
def forbid_execution(
    provider_session: ModelSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make device probing, installation reads, raster work and loading fail."""

    def unexpected(*args: object, **kwargs: object) -> NoReturn:
        """Expose any expensive work attempted before the capability error."""
        raise AssertionError("unsupported task reached expensive execution")

    async def unexpected_upload(*args: object, **kwargs: object) -> NoReturn:
        """Ensure unsupported REST requests keep their uploads unmaterialized."""
        unexpected()

    for module in (
        api,
        session_module,
        embedding_stream,
        classification_stream,
        segmentation_stream,
    ):
        monkeypatch.setattr(module, "detect_device", unexpected)
        monkeypatch.setattr(module, "load_backend", unexpected)
    monkeypatch.setattr(ModelStore, "load_manifest", unexpected)
    monkeypatch.setattr(ModelStore, "installation_lease", unexpected)
    monkeypatch.setattr(api, "_read_input", unexpected)
    monkeypatch.setattr(inference, "read_raster_from_bytes", unexpected)
    monkeypatch.setattr(UploadFile, "read", unexpected_upload)


@pytest.mark.parametrize(("model", "operation", "supported"), _UNSUPPORTED)
@pytest.mark.parametrize("mode", ("eager", "stream", "session", "cli", "rest"))
def test_unsupported_task_fails_before_expensive_work(
    tmp_path: Path,
    provider_session: ModelSession,
    forbid_execution: None,
    model: str,
    operation: Operation,
    supported: Operation,
    mode: Literal["eager", "stream", "session", "cli", "rest"],
) -> None:
    """Every bundled model rejects both other tasks with one domain contract."""
    expected = UnsupportedOperationError(
        model, operation, supported_operations=(supported,)
    )
    source, output = tmp_path / "unread.tif", tmp_path / "output"
    if mode == "rest":
        service = InferenceService(provider_session)
        with TestClient(create_app(service=service)) as client:
            response = client.post(
                f"/{operation}",
                files={"file": ("unread.tif", b"must not decode", "image/tiff")},
                data=(
                    {"model": model, "query": "forest"}
                    if operation == "classify"
                    else {"model": model}
                ),
            )
        assert response.status_code == 422, response.text
        assert response.json() == {
            "error": "UnsupportedOperationError",
            "detail": str(expected),
        }
    elif mode == "cli":
        arguments = [operation, str(source), "--model", model, "--output", str(output)]
        if operation == "classify":
            arguments += ["--query", "forest"]
        result = CliRunner().invoke(app, arguments)
        assert result.exit_code == 7, result.output
        assert "".join(str(expected).split()) in "".join(unstyle(result.output).split())
        assert "Traceback" not in result.output
    else:
        call: Callable[[], object]
        if mode == "eager":
            calls = {
                "embed": lambda: api.embed(source, model=model, write=False),
                "classify": lambda: api.classify(source, model=model, query="forest"),
                "segment": lambda: api.segment(source, model=model, output=output),
            }
        elif mode == "stream":
            calls = {
                "embed": lambda: api.embed_to_disk(source, model=model, output=output),
                "classify": lambda: api.classify_to_disk(
                    source, model=model, query="forest", output=output
                ),
                "segment": lambda: api.segment_to_disk(
                    source, model=model, output=output
                ),
            }
        else:
            unread = cast(RasterSource, object())
            calls = {
                "embed": lambda: api.embed_raster(
                    unread, model=model, session=provider_session
                ),
                "classify": lambda: api.classify_raster(
                    unread, model=model, query="forest", session=provider_session
                ),
                "segment": lambda: api.segment_raster(
                    unread, model=model, session=provider_session
                ),
            }
        call = calls[operation]
        with pytest.raises(UnsupportedOperationError) as caught:
            call()
        assert str(caught.value) == str(expected)
        assert caught.value.exit_code == 7
    assert not output.exists()
    assert provider_session.loaded == []


def test_session_get_checks_operation_before_device_or_installation(
    provider_session: ModelSession, forbid_execution: None
) -> None:
    """Direct session consumers receive the same inexpensive capability guard."""
    with pytest.raises(UnsupportedOperationError):
        provider_session.get("clay-v1", "auto", operation="classify")


def test_session_validates_the_actual_resident_entry(
    pulled_clay: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registry edits cannot authorize a task on a different cached provider."""
    session = ModelSession(device="cpu")
    resident = session.get(pulled_clay)
    replacement = session.store.registry.get("remoteclip").model_copy(
        update={"name": pulled_clay}
    )

    def current_entry(name: str) -> object:
        """Expose a changed registry while retaining the original resident model."""
        assert name == pulled_clay
        return replacement

    def unexpected_probe(*args: object, **kwargs: object) -> NoReturn:
        """Capability validation must precede a new automatic device selection."""
        raise AssertionError("unsupported resident task probed a device")

    monkeypatch.setattr(session.store.registry, "get", current_entry)
    monkeypatch.setattr(session_module, "detect_device", unexpected_probe)
    with pytest.raises(UnsupportedOperationError):
        session.get(pulled_clay, "auto", operation="classify")
    assert session.get(pulled_clay, operation="embed") is resident
