"""Credential isolation over real local HTTP and GDAL range requests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from whirld.core.registry import Registry
from whirld.errors import InvalidInputError
from whirld.io import stac

Route = tuple[int, dict[str, str], bytes]


@contextmanager
def _server(
    routes: dict[str, Route], seen: list[tuple[str, str | None]]
) -> Iterator[str]:
    """Serve controlled files and redirects, recording credentials on each request."""

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            self._respond(False)

        def do_GET(self) -> None:
            self._respond(True)

        def _respond(self, body: bool) -> None:
            seen.append((self.path, self.headers.get("Authorization")))
            code, headers, content = routes.get(self.path, (404, {}, b""))
            selected = content
            byte_range = self.headers.get("Range")
            if code == 200 and byte_range:
                first, last = byte_range.removeprefix("bytes=").split("-")
                start, end = int(first), min(int(last), len(content) - 1)
                selected = content[start : end + 1]
                code = 206
                headers = {
                    **headers,
                    "Content-Range": f"bytes {start}-{end}/{len(content)}",
                }
            self.send_response(code)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(selected)))
            self.end_headers()
            if body:
                self.wfile.write(selected)

        def log_message(self, format: str, *args: object) -> None:
            """Keep expected HTTP fixture traffic out of pytest output."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _item(asset: str = "") -> bytes:
    """Build a six-band Sentinel-2 item with a configurable source asset."""
    return json.dumps(
        {
            "type": "Feature",
            "id": "sentinel-2-test",
            "properties": {},
            "assets": {
                name: {"href": asset}
                for name in ("B02", "B03", "B04", "B08", "B11", "B12")
            },
        }
    ).encode()


def _tiff() -> bytes:
    """Return a small georeferenced TIFF suitable for GDAL range reads."""
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            width=10,
            height=10,
            count=1,
            dtype="uint16",
            crs="EPSG:32630",
            transform=from_origin(500000, 4500000, 10, 10),
        ) as ds:
            ds.write(np.full((1, 10, 10), 100, dtype="uint16"))
        return mem.read()


def test_item_redirect_strips_auth_across_ports() -> None:
    """A same-host redirect to another port is a different credential origin."""
    first: list[tuple[str, str | None]] = []
    second: list[tuple[str, str | None]] = []
    routes: dict[str, Route] = {}
    with _server({"/item": (200, {}, _item())}, second) as other:
        routes["/redirect"] = (302, {"Location": other + "/item"}, b"")
        with _server(routes, first) as origin:
            stac._fetch_item(origin + "/redirect", "test-token")
    assert first == [("/redirect", "Bearer test-token")]
    assert second == [("/item", None)]


def test_item_same_origin_redirect_retains_auth() -> None:
    """A redirect within the original origin retains legitimate authorization."""
    seen: list[tuple[str, str | None]] = []
    routes = {
        "/redirect": (302, {"Location": "/item"}, b""),
        "/item": (200, {}, _item()),
    }
    with _server(routes, seen) as origin:
        stac._fetch_item(origin + "/redirect", "test-token")
    assert seen == [("/redirect", "Bearer test-token"), ("/item", "Bearer test-token")]


@pytest.mark.parametrize("explicit", [False, True])
def test_asset_other_origin_requires_explicit_authorization(
    whirld_home: Path, explicit: bool
) -> None:
    """Assets on another port receive no item token unless explicitly authorized."""
    first: list[tuple[str, str | None]] = []
    second: list[tuple[str, str | None]] = []
    with _server({"/band.tif": (200, {}, _tiff())}, second) as other:
        with _server({"/item": (200, {}, _item(other + "/band.tif"))}, first) as origin:
            result = stac.read_stac_item(
                origin + "/item",
                entry=Registry().get("clay-v1"),
                token="test-token",
                token_origins=(other,) if explicit else (),
            )
    assert result.data.shape == (6, 10, 10)
    assert first == [("/item", "Bearer test-token")]
    assert second
    assert all(
        auth == ("Bearer test-token" if explicit else None) for _, auth in second
    )


def test_gdal_asset_redirect_does_not_forward_token(whirld_home: Path) -> None:
    """GDAL strips authorization even on same-host, cross-port asset redirects."""
    first: list[tuple[str, str | None]] = []
    second: list[tuple[str, str | None]] = []
    routes: dict[str, Route] = {}
    with _server({"/band.tif": (200, {}, _tiff())}, second) as other:
        routes["/band.tif"] = (302, {"Location": other + "/band.tif"}, b"")
        with _server(routes, first) as origin:
            routes["/item"] = (200, {}, _item(origin + "/band.tif"))
            result = stac.read_stac_item(
                origin + "/item", entry=Registry().get("clay-v1"), token="test-token"
            )
    assert np.all(result.data == 100)
    assert any(
        path == "/band.tif" and auth == "Bearer test-token" for path, auth in first
    )
    assert second and all(auth is None for _, auth in second)


def test_old_gdal_rejects_authenticated_asset_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GDAL versions without redirect credential control must fail closed."""
    monkeypatch.setattr(rasterio, "__gdal_version__", "3.9.0")
    with pytest.raises(InvalidInputError, match="GDAL >=3.10"):
        with stac._gdal_env("test-token"):
            pytest.fail("unsafe authenticated read was allowed")
