"""Real HTTP range transfers and process RSS bounds for streaming embeddings."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import numpy as np
from pydantic import BaseModel
from rasterio.warp import transform_bounds

import whirld
from scripts.benchmark_streaming import BenchmarkScene, measure_scene


class Transfer(BaseModel):
    """An observed asset response body, excluding HEAD/item traffic."""

    byte_range: str | None
    byte_count: int


@contextmanager
def _serve_cog(scene: BenchmarkScene, transfers: list[Transfer]) -> Iterator[str]:
    """Serve a file-backed COG with byte ranges, recording bytes actually sent."""

    class Handler(BaseHTTPRequestHandler):
        """Minimal single-file range server; no directory listings or other files."""

        def do_HEAD(self) -> None:
            """Return asset length without transferring pixels."""
            self._respond(False)

        def do_GET(self) -> None:
            """Return the item or the requested asset byte range."""
            self._respond(True)

        def _respond(self, body: bool) -> None:
            """Stream one controlled response and record the requested byte span."""
            if self.path == "/item.json":
                document = scene.document(
                    f"http://127.0.0.1:{self.server.server_port}/band.tif"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(document)))
                self.end_headers()
                if body:
                    self.wfile.write(document)
                return
            if self.path != "/band.tif":
                self.send_error(404)
                return
            size = scene.asset_path.stat().st_size
            byte_range = self.headers.get("Range")
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
                with scene.asset_path.open("rb") as source:
                    source.seek(start)
                    payload = source.read(end - start + 1)
                self.wfile.write(payload)
                transfers.append(
                    Transfer(byte_range=byte_range, byte_count=len(payload))
                )

        def log_message(self, format: str, *args: object) -> None:
            """Suppress expected fixture request logs."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/item.json"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_streaming_bbox_reads_http_ranges_not_whole_cog(
    tmp_path: Path, pulled_clay: str
) -> None:
    """A small remote bbox produces valid output using a fraction of COG bytes."""
    scene = BenchmarkScene(tmp_path / "http", 2048)
    scene.create()
    bbox = transform_bounds("EPSG:32630", "EPSG:4326", 500640, 4498720, 501280, 4499360)
    transfers: list[Transfer] = []
    with _serve_cog(scene, transfers) as url:
        result = whirld.embed_to_disk(
            url,
            model=pulled_clay,
            output=tmp_path / "remote.npy",
            bbox=bbox,
            chip_size=128,
            batch_size=2,
            device="cpu",
        )
    assert result.chip_count == 1
    assert np.load(result.output_path).shape == (1, 512)
    assert len(json.loads(result.meta_path.read_text())["chips"]) == 1
    assert transfers and all(transfer.byte_range for transfer in transfers)
    assert sum(transfer.byte_count for transfer in transfers) < (
        scene.asset_path.stat().st_size // 2
    )


def test_streaming_peak_rss_does_not_scale_with_scene_area(tmp_path: Path) -> None:
    """Growing scenes fill fixed caches, then RSS plateaus as output keeps growing."""
    small = BenchmarkScene(tmp_path / "small", 512)
    large = BenchmarkScene(tmp_path / "large", 2048)
    larger = BenchmarkScene(tmp_path / "larger", 4096)
    small.create()
    large.create()
    larger.create()
    first = measure_scene(small)
    second = measure_scene(large)
    third = measure_scene(larger)
    assert second.chip_count == 16 * first.chip_count
    assert second.output_bytes > 12 * first.output_bytes
    # Fixed GDAL/warp caches and allocator warmup can increase RSS before plateau.
    # A full six-band 2048² float32 scene alone would add 96 MiB, before copies.
    assert second.peak_rss_bytes - first.peak_rss_bytes < 64 * 1024 * 1024, (
        first.model_dump(),
        second.model_dump(),
    )
    assert third.chip_count == 4 * second.chip_count
    assert third.output_bytes > 3 * second.output_bytes
    assert third.peak_rss_bytes - second.peak_rss_bytes < 32 * 1024 * 1024, (
        second.model_dump(),
        third.model_dump(),
    )
