"""Verify independent application services and their explicit store dependencies."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

import pytest
from fastapi.testclient import TestClient

from whirld import api
from whirld.config import WhirldPaths
from whirld.core.fetch import ModelStore
from whirld.core.session import ModelSession
from whirld.errors import WhirldError
from whirld.server.app import create_app
from whirld.server.dependencies import get_inference_service
from whirld.server.inference import InferenceService


def _service(home: Path, *, installed: bool) -> InferenceService:
    """Construct a service bound to an explicit cache, independent of environment."""
    store = ModelStore(WhirldPaths(home))
    if installed:
        store.pull("clay-v1")
    return InferenceService(ModelSession(device="cpu", store=store))


def test_services_keep_store_identity_and_routes_honor_dependency_override(
    whirld_home: Path, tmp_path: Path, s2_tif: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registry/status/inference follow the injected store after environment changes."""
    first = _service(tmp_path / "first", installed=True)
    second = _service(tmp_path / "second", installed=False)
    monkeypatch.setenv("WHIRLD_HOME", str(tmp_path / "unrelated"))
    app = create_app(service=first, preload=["clay-v1"])
    payload = s2_tif.read_bytes()
    with TestClient(app) as client, TestClient(create_app(service=second)):
        assert client.get("/health").json()["models_loaded"] == ["clay-v1"]
        models = {
            model["name"]: model for model in client.get("/models").json()["models"]
        }
        assert models["clay-v1"]["installed"] is True
        assert first.session.store.paths.home == tmp_path / "first"

        app.dependency_overrides[get_inference_service] = lambda: second
        assert client.get("/health").json()["models_loaded"] == []
        models = {
            model["name"]: model for model in client.get("/models").json()["models"]
        }
        assert models["clay-v1"]["installed"] is False
        rejected = client.post(
            "/embed",
            files={"file": ("scene.tif", payload, "image/tiff")},
            data={"model": "clay-v1"},
        )
        assert rejected.status_code == 404

        app.dependency_overrides.clear()
        accepted = client.post(
            "/embed",
            files={"file": ("scene.tif", payload, "image/tiff")},
            data={"model": "clay-v1"},
        )
        assert accepted.status_code == 200
    assert first.session.loaded == second.session.loaded == []
    assert first.health().status == second.health().status == "closed"


def test_separate_services_do_not_share_inference_admission(
    whirld_home: Path, tmp_path: Path, s2_tif: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blocking one service does not serialize an unrelated service's inference."""
    first = _service(tmp_path / "first", installed=True)
    second = _service(tmp_path / "second", installed=True)
    entered, release = Event(), Event()
    original = api.embed_raster

    def block_first(*args: Any, **kwargs: Any) -> api.EmbedResult:
        """Hold only the first session at the actual model-pipeline boundary."""
        if kwargs["session"] is first.session:
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "embed_raster", block_first)
    payload = s2_tif.read_bytes()
    with (
        TestClient(create_app(service=first)) as first_client,
        TestClient(create_app(service=second)) as second_client,
        ThreadPoolExecutor(2) as workers,
    ):

        def post(client: TestClient) -> Any:
            return client.post(
                "/embed",
                files={"file": ("scene.tif", payload, "image/tiff")},
                data={"model": "clay-v1"},
            )

        pending = workers.submit(post, first_client)
        try:
            assert entered.wait(2)
            assert (
                workers.submit(post, second_client).result(timeout=2).status_code == 200
            )
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200


def test_failed_preload_releases_models(whirld_home: Path, tmp_path: Path) -> None:
    """Application startup failure clears any models loaded before that failure."""
    store = ModelStore(WhirldPaths(tmp_path / "partial-start"))
    store.pull("clay-v1")

    class FailingSession(ModelSession):
        def preload(self, models: list[str], device: str | None = None) -> None:
            """Load one model, then simulate failure of another startup load."""
            super().preload(models, device)
            raise WhirldError("preload failed")

    session = FailingSession(device="cpu", store=store)
    service = InferenceService(session)
    with pytest.raises(WhirldError, match="preload failed"):
        with TestClient(create_app(service=service, preload=["clay-v1"])):
            pytest.fail("failed startup was accepted")
    assert session.loaded == []
    assert service.health().status == "closed"
