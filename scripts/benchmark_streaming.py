"""Measure streaming peak RSS in fresh processes using block-written local COGs.

Run with ``.venv/bin/python scripts/benchmark_streaming.py``. Fixture creation is
outside measured workers. The default 512/2048/4096-pixel scenes span a 64-fold
area difference while chip size, batch size, model, and source block size stay fixed.
The benchmark uses the offline reference encoder; it does not measure GPU memory
or real-checkpoint model allocations. JSON results are written to standard output.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel

_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")


class StreamingMeasurement(BaseModel):
    """One fresh process's completed inference and peak resident memory."""

    scene_size: int
    chip_size: int
    batch_size: int
    chip_count: int
    output_bytes: int
    peak_rss_bytes: int
    elapsed_seconds: float


class BenchmarkScene:
    """Own an isolated synthetic STAC scene and its block-written COG assets."""

    def __init__(self, directory: Path, size: int) -> None:
        """Bind a directory and square scene size without allocating its pixels.

        Args:
            directory: Isolated directory for source assets and worker outputs.
            size: Positive scene width and height in pixels; CLI input validation
                occurs before construction.
        """
        self.directory = directory
        self.size = size
        self.item_path = directory / "item.json"
        self.asset_path = directory / "band.tif"

    def create(self) -> Path:
        """Write deterministic bounded-size blocks, then package a real COG.

        All six synthetic bands share one source asset. The pipeline still reads,
        normalizes, and embeds six channels; source file/block layout stays fixed
        across benchmark sizes.

        Returns:
            The local STAC JSON path referencing the completed COG.

        Raises:
            OSError: A fixture file cannot be created, written, or removed.
            rasterio.errors.RasterioError: Raster creation or COG packaging fails.
        """
        import numpy as np
        import rasterio
        from rasterio.shutil import copy as copy_raster
        from rasterio.transform import from_origin

        self.directory.mkdir(parents=True, exist_ok=True)
        raw_path = self.directory / "source.tif"
        with rasterio.Env(GDAL_CACHEMAX=16 * 1024 * 1024):
            with rasterio.open(
                raw_path,
                "w",
                driver="GTiff",
                width=self.size,
                height=self.size,
                count=1,
                dtype="uint16",
                crs="EPSG:32630",
                transform=from_origin(500000, 4500000, 10, 10),
                tiled=True,
                blockxsize=128,
                blockysize=128,
            ) as dataset:
                rng = np.random.default_rng(18)
                for _, window in dataset.block_windows(1):
                    pixels = rng.integers(
                        1,
                        10000,
                        size=(int(window.height), int(window.width)),
                        dtype=np.uint16,
                    )
                    dataset.write(pixels, 1, window=window)
            copy_raster(
                raw_path,
                self.asset_path,
                driver="COG",
                BLOCKSIZE=128,
                COMPRESS="NONE",
                OVERVIEWS="NONE",
            )
        raw_path.unlink()
        self.item_path.write_text(
            self.document(self.asset_path.as_uri()), encoding="utf-8"
        )
        return self.item_path

    def document(self, asset_href: str) -> str:
        """Serialize a synthetic item using the chosen local or HTTP asset href.

        Args:
            asset_href: Asset URL shared by the six synthetic band declarations.

        Returns:
            A STAC item JSON string with fixed acquisition metadata and band order.
        """
        return json.dumps(
            {
                "type": "Feature",
                "id": f"sentinel-2-stream-{self.size}",
                "properties": {"datetime": "2024-06-01T12:00:00Z"},
                "assets": {name: {"href": asset_href} for name in _BANDS},
            }
        )


def measure_scene(
    scene: BenchmarkScene, *, chip_size: int = 128, batch_size: int = 4
) -> StreamingMeasurement:
    """Measure an existing scene in a fresh interpreter with an isolated cache.

    Fixture construction must occur before this call, outside the measured worker.

    Args:
        scene: Existing synthetic STAC scene and its isolated output directory.
        chip_size: Positive square chip edge length in pixels.
        batch_size: Positive maximum chips processed together.

    Returns:
        Validated output counts, elapsed time, and process peak RSS in bytes.

    Raises:
        subprocess.CalledProcessError: The worker fails; captured output identifies
            its inference or fixture error.
        subprocess.TimeoutExpired: The worker exceeds the 120-second limit.
        pydantic.ValidationError: Worker output does not match the report schema.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(scene.item_path),
            "--scene-size",
            str(scene.size),
            "--chip-size",
            str(chip_size),
            "--batch-size",
            str(batch_size),
        ],
        env={
            **os.environ,
            "WHIRLD_HOME": str(scene.directory / "whirld-home"),
            "HF_HOME": str(scene.directory / "hf-home"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return StreamingMeasurement.model_validate_json(result.stdout)


def _worker(
    item: Path, scene_size: int, chip_size: int, batch_size: int
) -> StreamingMeasurement:
    """Run actual reference inference and report process-lifetime peak RSS.

    Args:
        item: Existing local STAC item whose asset creation is already complete.
        scene_size: Scene width/height used to label the measurement.
        chip_size: Positive square inference chip edge length in pixels.
        batch_size: Positive maximum number of chips per inference batch.

    Returns:
        A measurement including model setup, inference, and output publication.

    Raises:
        WhirldError: Model installation, reading, inference, or publication fails.
        OSError: Completed output sizes cannot be read from the filesystem.
    """
    import resource

    import whirld

    started = time.monotonic()
    whirld.pull("clay-v1")
    result = whirld.embed_to_disk(
        item,
        model="clay-v1",
        output=item.parent / "embeddings.npy",
        chip_size=chip_size,
        batch_size=batch_size,
        device="cpu",
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes; Linux and other supported Unix runners report KiB.
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    return StreamingMeasurement(
        scene_size=scene_size,
        chip_size=chip_size,
        batch_size=batch_size,
        chip_count=result.chip_count,
        output_bytes=result.output_path.stat().st_size
        + result.meta_path.stat().st_size,
        peak_rss_bytes=peak_bytes,
        elapsed_seconds=time.monotonic() - started,
    )


def main() -> int:
    """Parse command-line options and emit measurement JSON to standard output.

    Normal mode builds fixtures before starting fresh measured workers. Internal
    worker mode processes one existing fixture in the current process.

    Returns:
        Zero when the requested measurements and JSON output complete.

    Raises:
        SystemExit: Help is requested or command-line values are invalid.
        subprocess.SubprocessError: A measured worker fails or times out.
        OSError: Fixture creation or report output fails.
        WhirldError: Inference fails in internal worker mode.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 2048, 4096])
    parser.add_argument("--chip-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--scene-size", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        result = _worker(args.worker, args.scene_size, args.chip_size, args.batch_size)
        sys.stdout.write(result.model_dump_json() + "\n")
        return 0
    if min(args.sizes) <= 0 or args.chip_size <= 0 or args.batch_size <= 0:
        parser.error("scene, chip, and batch sizes must be positive")
    measurements = []
    with tempfile.TemporaryDirectory(prefix="whirld-stream-rss-") as temporary:
        for size in args.sizes:
            scene = BenchmarkScene(Path(temporary) / str(size), size)
            scene.create()
            measurements.append(
                measure_scene(
                    scene, chip_size=args.chip_size, batch_size=args.batch_size
                ).model_dump()
            )
    sys.stdout.write(json.dumps(measurements, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
