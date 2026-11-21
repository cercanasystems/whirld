"""Regression checks for request domains, HTTP interoperability, and concurrency."""

from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPResponse
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from whirld.server.app import create_app


@pytest.mark.parametrize(
    "route,fields",
    [
        ("embed", {"chip_size": "-1"}),
        ("embed", {"chip_size": "0"}),
        ("embed", {"chip_size": "2.5"}),
        ("embed", {"batch_size": "0"}),
        ("embed", {"overlap": "-1"}),
        ("embed", {"chip_size": "10", "overlap": "10"}),
        ("embed", {"format": "garbage"}),
        ("embed", {"model": " "}),
        ("embed", {"model": "../outside"}),
        ("classify", {"query": "forest", "top_k": "0"}),
        ("classify", {"query": "forest", "threshold": "nan"}),
        ("classify", {"query": "forest", "threshold": "-0.1"}),
        ("classify", {"query": " "}),
        ("segment", {"threshold": "inf"}),
        ("segment", {"threshold": "1.1"}),
    ],
)
def test_invalid_request_domains_fail_before_raster_read(
    whirld_home: Path, route: str, fields: dict[str, str]
) -> None:
    """Malformed options return 422 even if the supplied raster bytes are invalid."""
    with TestClient(create_app(device="cpu")) as client:
        response = client.post(
            f"/{route}",
            files={"file": ("bad.tif", b"invalid", "image/tiff")},
            data={"model": "clay-v1", **fields},
        )
    assert response.status_code == 422, response.text
    assert "detail" in response.json()


def test_openapi_describes_actual_multipart_domains(whirld_home: Path) -> None:
    """OpenAPI advertises uploaded files, formats, and constrained numeric fields."""
    schema = create_app(device="cpu").openapi()
    request = schema["paths"]["/embed"]["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]
    name = request["$ref"].rsplit("/", 1)[1]
    fields = schema["components"]["schemas"][name]["properties"]
    assert (
        fields["file"].get("format") == "binary"
        or fields["file"].get("contentMediaType") == "application/octet-stream"
    )
    assert fields["format"]["enum"] == ["npz", "npy", "json"]
    assert fields["format"]["default"] == "npz"
    assert fields["overlap"]["minimum"] == 0
    assert fields["chip_size"]["anyOf"][0]["exclusiveMinimum"] == 0


@pytest.mark.parametrize("route", ["embed", "classify", "segment"])
def test_health_remains_responsive_and_inference_is_serialized(
    whirld_home: Path, s2_tif: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    """A blocked backend does not stall health, and two requests never overlap."""
    from rasterio.transform import from_origin

    entered, release = Event(), Event()
    lock = Lock()
    active = 0
    maximum = 0

    def blocking_backend(*args: Any, **kwargs: Any) -> Any:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        entered.set()
        try:
            assert release.wait(5), "test did not release inference"
            return SimpleNamespace(
                meta={"chips": []},
                embeddings=np.zeros((0, 2), dtype="float32"),
                feature_collection={"type": "FeatureCollection", "features": []},
                mask=np.zeros((2, 2), dtype="uint8"),
                crs="EPSG:32630",
                transform=from_origin(500000, 4500000, 10, 10),
            )
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(f"whirld.api.{route}_raster", blocking_backend)
    payload = s2_tif.read_bytes()
    fields = {"model": "clay-v1"}
    if route == "classify":
        fields["query"] = "forest"
    with (
        TestClient(create_app(device="cpu")) as client,
        ThreadPoolExecutor(3) as workers,
    ):

        def post() -> Any:
            return client.post(
                f"/{route}",
                files={"file": ("scene.tif", payload, "image/tiff")},
                data=fields,
            )

        first = workers.submit(post)
        try:
            assert entered.wait(2)
            second = workers.submit(post)
            health = workers.submit(client.get, "/health")
            assert health.result(timeout=1).status_code == 200
        finally:
            release.set()
        assert first.result(timeout=5).status_code == 200
        assert second.result(timeout=5).status_code == 200
    assert maximum == 1


def test_large_chip_metadata_passes_standard_http_parser(
    whirld_home: Path, s2_tif: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 600-chip reproduction carries metadata in a body, not oversized headers."""
    meta = {
        "chips": [
            {
                "row": i,
                "col": 0,
                "bbox": [320000, 5000000, 322240, 5002240],
                "description": "x" * 100,
            }
            for i in range(600)
        ]
    }
    assert len(json.dumps(meta)) > 65536
    monkeypatch.setattr(
        "whirld.api.embed_raster",
        lambda *args, **kwargs: SimpleNamespace(
            meta=meta, embeddings=np.zeros((600, 2), dtype="float32")
        ),
    )
    with TestClient(create_app(device="cpu")) as client:
        response = client.post(
            "/embed",
            files={"file": ("scene.tif", s2_tif.read_bytes(), "image/tiff")},
            data={"model": "clay-v1"},
        )
    assert response.status_code == 200
    wire = b"HTTP/1.1 200 OK\r\n" + b"".join(
        name + b": " + value + b"\r\n" for name, value in response.headers.raw
    )
    wire += b"\r\n" + response.content

    class Socket:
        def makefile(self, mode: str) -> io.BytesIO:
            return io.BytesIO(wire)

    parsed = HTTPResponse(Socket())
    parsed.begin()
    with ZipFile(io.BytesIO(parsed.read())) as archive:
        assert json.loads(archive.read("metadata.json")) == meta
        assert np.load(io.BytesIO(archive.read("embeddings.npy"))).shape == (600, 2)
    assert "X-Whirld-Chips-Meta" not in response.headers


def test_model_status_lock_wait_does_not_block_health(
    whirld_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installation-state checks may wait for pulls without blocking the event loop."""
    entered, release = Event(), Event()

    def blocked_status(name: str) -> bool:
        entered.set()
        assert release.wait(5)
        return False

    monkeypatch.setattr("whirld.server.routes.models.is_installed", blocked_status)
    with (
        TestClient(create_app(device="cpu")) as client,
        ThreadPoolExecutor(2) as workers,
    ):
        status = workers.submit(client.get, "/models")
        try:
            assert entered.wait(2)
            assert (
                workers.submit(client.get, "/health").result(timeout=1).status_code
                == 200
            )
        finally:
            release.set()
        assert status.result(timeout=5).status_code == 200
