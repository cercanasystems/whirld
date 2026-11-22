"""Direct remote COG range, radiometry, credential and memory integration checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import TracebackType
from urllib.parse import urlsplit

import numpy as np
import pytest
import rasterio
from pydantic import BaseModel
from rasterio.shutil import copy as copy_raster
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from whirld.core.registry import Registry

BANDS = ("B12", "B11", "B08", "B04", "B03", "B02")


class CogScene:
    """Own a block-generated true multiband COG and equivalent STAC document."""

    def __init__(self, directory: Path, size: int = 300, *, named: bool = True) -> None:
        """Bind fixture dimensions and source-order band descriptions."""
        self.directory = directory
        self.size = size
        self.named = named
        self.path = directory / "scene.tif"

    def create(self) -> Path:
        """Write bounded blocks and package a COG before any measured worker.

        Returns:
            A six-band uint16 COG with nodata, an internal mask and acquisition time.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        raw = self.directory / "raw.tif"
        with rasterio.Env(GDAL_CACHEMAX=8 * 1024 * 1024, GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(
                raw,
                "w",
                driver="GTiff",
                width=self.size,
                height=self.size,
                count=6,
                dtype="uint16",
                crs="EPSG:32630",
                transform=from_origin(500000, 4500000, 10, 10),
                nodata=0,
                tiled=True,
                blockxsize=128,
                blockysize=128,
            ) as source:
                rng = np.random.default_rng(412)
                for _, window in source.block_windows(1):
                    pixels = rng.integers(
                        1100,
                        9000,
                        size=(6, int(window.height), int(window.width)),
                        dtype=np.uint16,
                    )
                    if window.col_off == 0 and window.row_off == 0:
                        pixels[:, 2:5, 3:8] = 0
                    source.write(pixels, window=window)
                    mask = np.full(pixels.shape[1:], 255, dtype=np.uint8)
                    if window.col_off == 0 and window.row_off == 0:
                        mask[10:15, 12:18] = 0
                    source.write_mask(mask, window=window)
                if self.named:
                    source.descriptions = BANDS
                source.update_tags(TIFFTAG_DATETIME="2024:06:01 12:00:00")
            copy_raster(
                raw,
                self.path,
                driver="COG",
                BLOCKSIZE=128,
                COMPRESS="NONE",
                OVERVIEWS="NONE",
            )
        raw.unlink()
        return self.path

    def document(
        self, url: str, *, transformed: bool = False, per_band: bool = False
    ) -> bytes:
        """Serialize equivalent STAC metadata at the fixture's external boundary."""
        radiometry = [
            {
                "nodata": 0,
                **(
                    {
                        "scale": 0.0001 * (index + 1 if per_band else 1),
                        "offset": -0.1 * (index + 1 if per_band else 1),
                    }
                    if transformed
                    else {}
                ),
            }
            for index, _ in enumerate(BANDS)
        ]
        return json.dumps(
            {
                "type": "Feature",
                "id": "sentinel-2-cog-test",
                "properties": {"datetime": "2024-06-01T12:00:00Z"},
                "assets": {
                    "image": {
                        "href": url,
                        "eo:bands": [{"name": name} for name in BANDS],
                        "raster:bands": radiometry,
                    }
                },
            }
        ).encode()


class RequestRecord(BaseModel):
    """Observed HTTP request and body bytes; credentials are synthetic test tokens."""

    method: str
    path: str
    authorization: str | None
    byte_range: str | None
    byte_count: int


class CogServer:
    """Own a real localhost HTTP range server and its request observations."""

    def __init__(
        self,
        scene: CogScene,
        *,
        token: str | None = None,
        redirect: str | None = None,
        transformed: bool = False,
        per_band: bool = False,
    ) -> None:
        """Bind the file, optional required token, redirect and STAC unit convention."""
        self.scene = scene
        self.token = token
        self.redirect = redirect
        self.transformed = transformed
        self.per_band = per_band
        self.records: list[RequestRecord] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            """Serve test artifacts with actual byte ranges and redirects."""

            def do_HEAD(self) -> None:
                """Respond with headers without transmitting file content."""
                self.respond(False)

            def do_GET(self) -> None:
                """Respond with the requested file bytes or small STAC document."""
                self.respond(True)

            def respond(self, body: bool) -> None:
                """Enforce fixture auth and serve only explicitly registered paths."""
                path = urlsplit(self.path).path
                authorization = self.headers.get("Authorization")
                byte_range = self.headers.get("Range")
                record = RequestRecord(
                    method=self.command,
                    path=self.path,
                    authorization=authorization,
                    byte_range=byte_range,
                    byte_count=0,
                )
                owner.records.append(record)
                if path == "/redirect.tif" and owner.redirect:
                    self.send_response(302)
                    self.send_header("Location", owner.redirect)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if owner.token and authorization != f"Bearer {owner.token}":
                    self.send_error(401)
                    return
                if path == "/item.json":
                    document = owner.scene.document(
                        owner.url,
                        transformed=owner.transformed,
                        per_band=owner.per_band,
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(document)))
                    self.end_headers()
                    if body:
                        self.wfile.write(document)
                    return
                if path not in {"/scene.TIF", "/download"}:
                    self.send_error(404)
                    return
                size = owner.scene.path.stat().st_size
                start, end = 0, size - 1
                if byte_range:
                    first, last = byte_range.removeprefix("bytes=").split("-")
                    start, end = int(first), min(int(last or size - 1), size - 1)
                self.send_response(206 if byte_range else 200)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Type", "image/tiff")
                self.send_header("Content-Length", str(end - start + 1))
                if byte_range:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if body:
                    with owner.scene.path.open("rb") as source:
                        source.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = source.read(min(65536, remaining))
                            self.wfile.write(chunk)
                            record.byte_count += len(chunk)
                            remaining -= len(chunk)

            def log_message(self, format: str, *args: object) -> None:
                """Keep fixture HTTP traffic out of ordinary test output."""

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    @property
    def origin(self) -> str:
        """Return this server's unique HTTP origin."""
        return f"http://127.0.0.1:{self.server.server_port}"

    @property
    def url(self) -> str:
        """Return a mixed-case TIFF URL to exercise suffix routing."""
        return self.origin + "/scene.TIF"

    def __enter__(self) -> CogServer:
        """Start serving the fixture and return its resource owner."""
        self.thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the socket and wait for the server thread on every exit path."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


@pytest.mark.parametrize(
    ("transformed", "per_band"), [(False, False), (True, False), (True, True)]
)
def test_direct_windows_match_stac_units_masks_and_geometry(
    whirld_home: Path,
    tmp_path: Path,
    transformed: bool,
    per_band: bool,
) -> None:
    """Direct COG and equivalent STAC retain radiometry, masks, order and edge chips."""
    from whirld.core.options import CogOptions
    from whirld.io.cog import CogWindowReader
    from whirld.io.stac_stream import StacWindowReader

    scene = CogScene(tmp_path / "scene")
    scene.create()
    entry = Registry().get("clay-v1")
    options = CogOptions(scale=0.0001, offset=-0.1) if transformed else None
    if per_band:
        options = CogOptions(
            scale=tuple(0.0001 * i for i in range(1, 7)),
            offset=tuple(-0.1 * i for i in range(1, 7)),
        )
    bbox = transform_bounds("EPSG:32630", "EPSG:4326", 500015, 4497045, 502965, 4499985)
    with CogServer(scene, transformed=transformed, per_band=per_band) as server:
        with (
            CogWindowReader(
                server.url,
                entry=entry,
                sensor="sentinel-2-l2a",
                chip_size=128,
                overlap=17,
                batch_size=2,
                bbox=bbox,
                cog=options,
            ) as direct,
            StacWindowReader(
                server.origin + "/item.json",
                entry=entry,
                chip_size=128,
                overlap=17,
                batch_size=2,
                bbox=bbox,
            ) as stac,
        ):
            assert direct.metadata.resolve() == stac.metadata.resolve()
            assert direct.chip_count == stac.chip_count
            for actual, expected in zip(direct.batches(), stac.batches(), strict=True):
                np.testing.assert_allclose(
                    actual.pixels, expected.pixels, rtol=1e-6, atol=1e-6
                )
                assert actual.chips == expected.chips


def test_direct_bbox_reads_ranges_and_keeps_bearer_on_lazy_reads(
    whirld_home: Path,
    tmp_path: Path,
) -> None:
    """Small bbox reads only range subsets, preserving auth on every data request."""
    from whirld.io.cog import CogWindowReader

    scene = CogScene(tmp_path / "large", 2048)
    scene.create()
    bbox = transform_bounds("EPSG:32630", "EPSG:4326", 500640, 4498720, 501280, 4499360)
    with CogServer(scene, token="fixture-token") as server:
        with CogWindowReader(
            server.url + "?signature=fixture",
            entry=Registry().get("clay-v1"),
            sensor="sentinel-2-l2a",
            chip_size=32,
            batch_size=2,
            bbox=bbox,
            token="fixture-token",
        ) as reader:
            assert sum(len(batch.chips) for batch in reader.batches()) >= 4
        gets = [record for record in server.records if record.method == "GET"]
        assert gets and all(record.byte_range for record in gets)
        assert all(record.authorization == "Bearer fixture-token" for record in gets)
        assert all("signature=fixture" in record.path for record in gets)
        assert (
            sum(record.byte_count for record in gets) < scene.path.stat().st_size // 2
        )


def test_direct_redirect_does_not_forward_bearer(
    whirld_home: Path,
    tmp_path: Path,
) -> None:
    """Direct input credentials never reach a different redirect destination."""
    from whirld.io.cog import CogWindowReader

    scene = CogScene(tmp_path / "redirect")
    scene.create()
    with (
        CogServer(scene) as destination,
        CogServer(scene, redirect=destination.url) as origin,
    ):
        with CogWindowReader(
            origin.origin + "/redirect.tif",
            entry=Registry().get("clay-v1"),
            sensor="sentinel-2-l2a",
            chip_size=128,
            token="fixture-token",
        ) as reader:
            assert next(reader.batches()).pixels.shape[1] == 6
        assert origin.records
        assert any(r.authorization == "Bearer fixture-token" for r in origin.records)
        assert destination.records
        assert all(r.authorization is None for r in destination.records)


class MemoryResult(BaseModel):
    """A fresh direct-URL worker's output count and process peak bytes."""

    chips: int
    rss: int


def test_direct_streaming_memory_plateaus_with_scene_growth(tmp_path: Path) -> None:
    """Real HTTP direct COG reads do not accumulate whole-scene arrays or metadata."""
    worker = """
import json,resource,sys
from pathlib import Path
import whirld
whirld.pull("clay-v1")
result=whirld.embed_to_disk(sys.argv[1], model="clay-v1", sensor="sentinel-2-l2a",
 output=Path(sys.argv[2]), chip_size=128, batch_size=4, device="cpu")
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
sys.stdout.write(json.dumps({"chips":result.chip_count,
 "rss":int(peak if sys.platform=="darwin" else peak*1024)}))
"""
    results = []
    for size in (512, 2048, 4096):
        scene = CogScene(tmp_path / str(size), size)
        scene.create()
        with CogServer(scene) as server:
            process = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    worker,
                    server.url,
                    str(scene.directory / "output.npy"),
                ],
                env={
                    **os.environ,
                    "WHIRLD_HOME": str(scene.directory / "home"),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
                text=True,
                capture_output=True,
                timeout=120,
                check=True,
            )
            results.append(MemoryResult.model_validate_json(process.stdout))
    assert results[1].chips == 16 * results[0].chips
    assert results[2].chips == 4 * results[1].chips
    assert results[1].rss - results[0].rss < 64 * 1024 * 1024, results
    assert results[2].rss - results[1].rss < 32 * 1024 * 1024, results
