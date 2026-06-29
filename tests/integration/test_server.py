"""Integration tests for the REST API via FastAPI's TestClient."""

from __future__ import annotations

import base64
import io
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from whirld.core.fetch import pull
from whirld.server.app import create_app


@pytest.fixture
def client(whirld_home: Path) -> Iterator[TestClient]:
    """A TestClient against an app with clay-v1 pulled and preloaded."""
    pull("clay-v1")
    app = create_app(device="cpu", preload=["clay-v1"])
    with TestClient(app) as test_client:
        yield test_client


def _upload(client: TestClient, tif: Path, **data: object):
    """POST a GeoTIFF to /embed as multipart form-data."""
    with open(tif, "rb") as handle:
        return client.post(
            "/embed",
            files={"file": ("scene.tif", handle, "image/tiff")},
            data={"model": "clay-v1", **data},
        )


def test_health(client: TestClient) -> None:
    """GET /health reports status, device, version, and loaded models."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"
    assert body["version"]
    assert "clay-v1" in body["models_loaded"]


def test_models(client: TestClient) -> None:
    """GET /models lists clay-v1 as installed and loaded."""
    resp = client.get("/models")
    assert resp.status_code == 200
    models = {m["name"]: m for m in resp.json()["models"]}
    assert "clay-v1" in models
    assert models["clay-v1"]["installed"] is True
    assert models["clay-v1"]["loaded"] is True
    assert "sentinel-2-l2a" in models["clay-v1"]["sensors"]


def test_embed_npy(client: TestClient, s2_tif: Path) -> None:
    """POST /embed returns npy bytes plus a chip-metadata header."""
    resp = _upload(client, s2_tif)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/octet-stream"
    arr = np.load(io.BytesIO(resp.content))
    assert arr.shape[1] == 512
    meta = json.loads(base64.b64decode(resp.headers["X-Whirld-Chips-Meta"]))
    assert meta["model"] == "clay-v1"
    assert len(meta["chips"]) == arr.shape[0]
    assert meta["chips"][0]["bbox"][0] == 320000.0


def test_embed_json(client: TestClient, s2_tif: Path) -> None:
    """POST /embed?format=json inlines the embeddings array."""
    resp = _upload(client, s2_tif, format="json")
    assert resp.status_code == 200
    body = resp.json()
    assert "embeddings" in body
    assert len(body["embeddings"]) == len(body["chips"])


def test_embed_unknown_model(client: TestClient, s2_tif: Path) -> None:
    """An unknown model maps to HTTP 404 with the error envelope."""
    with open(s2_tif, "rb") as handle:
        resp = client.post(
            "/embed",
            files={"file": ("scene.tif", handle, "image/tiff")},
            data={"model": "no-such-model"},
        )
    assert resp.status_code == 404
    assert resp.json()["error"] == "ModelNotFoundError"


def test_embed_unsupported_sensor(client: TestClient, s2_tif: Path) -> None:
    """An unsupported --sensor override maps to HTTP 422."""
    resp = _upload(client, s2_tif, sensor="spot-6")
    assert resp.status_code == 422
    assert resp.json()["error"] == "UnsupportedSensorError"


def test_embed_not_installed(whirld_home: Path, s2_tif: Path) -> None:
    """Embedding a known-but-unpulled model maps to HTTP 404."""
    app = create_app(device="cpu")  # nothing pulled or preloaded
    with TestClient(app) as client, open(s2_tif, "rb") as handle:
        resp = client.post(
            "/embed",
            files={"file": ("scene.tif", handle, "image/tiff")},
            data={"model": "clay-v1"},
        )
    assert resp.status_code == 404
    assert resp.json()["error"] == "ModelNotInstalledError"


def test_embed_stac_json_deferred(client: TestClient) -> None:
    """A JSON (STAC) body is rejected with HTTP 400 (deferred)."""
    resp = client.post("/embed", json={"model": "clay-v1", "input": "https://x/i.json"})
    assert resp.status_code == 400


def test_embed_missing_file(client: TestClient) -> None:
    """A multipart request without a file is rejected with HTTP 422."""
    resp = client.post("/embed", data={"model": "clay-v1"})
    assert resp.status_code == 422


def test_segment_missing_model(client: TestClient, s2_tif: Path) -> None:
    """POST /segment without a model is rejected with 422 (route is now real)."""
    with open(s2_tif, "rb") as handle:
        resp = client.post(
            "/segment",
            files={"file": ("scene.tif", handle, "image/tiff")},
        )
    assert resp.status_code == 422


def test_classify_missing_query(client: TestClient, s2_tif: Path) -> None:
    """POST /classify without a query is rejected with 422 (route is now real)."""
    with open(s2_tif, "rb") as handle:
        resp = client.post(
            "/classify",
            files={"file": ("scene.tif", handle, "image/tiff")},
            data={"model": "remoteclip"},
        )
    assert resp.status_code == 422
