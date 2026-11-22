"""Shared CPU setup, measurement and georeferenced previews for runnable examples."""

from __future__ import annotations

import base64
import html
import importlib.metadata
import logging
import os
import platform
import resource
import sys
import time
import warnings
from importlib.resources import files
from pathlib import Path

import numpy as np
from affine import Affine
from pydantic import BaseModel, ConfigDict
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.warp import transform

from whirld.core.contract import translate
from whirld.core.fetch import ModelStore, sha256_file
from whirld.core.registry import ModelEntry, Registry
from whirld.errors import InvalidInputError
from whirld.io.stac import read_stac_item


class SceneSpec(BaseModel):
    """Pinned public item, geographic subset, input contract and attribution."""

    model_config = ConfigDict(frozen=True)
    item_url: str
    bbox: tuple[float, float, float, float]
    sensor: str = "sentinel-2-l2a"
    attribution: str
    item_id: str = ""


SENTINEL_SCENE = SceneSpec(
    item_id="S2A_10SFH_20240722_0_L2A",
    item_url=(
        "https://earth-search.aws.element84.com/v1/collections/"
        "sentinel-2-l2a/items/S2A_10SFH_20240722_0_L2A"
    ),
    bbox=(-121.57, 38.53, -121.48, 38.60),
    attribution="Contains modified Copernicus Sentinel data 2024, via Element 84.",
)


class RunMetrics(BaseModel):
    """Wall time from the caller's start and process-lifetime peak CPU RSS."""

    elapsed_seconds: float
    peak_rss_bytes: int

    @classmethod
    def capture(cls, started: float) -> RunMetrics:
        """Capture time/RSS after a workflow; RSS includes imports and model loading.

        Args:
            started: Monotonic timestamp after setup and before data preparation.

        Returns:
            Total workflow seconds and process peak bytes, not GPU allocations.
        """
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return cls(
            elapsed_seconds=time.monotonic() - started,
            peak_rss_bytes=int(rss if sys.platform == "darwin" else rss * 1024),
        )


class ExampleRuntime:
    """Own an exclusive output directory and explicit CPU setup for one example."""

    def __init__(self, output_dir: Path, threads: int = 2) -> None:
        """Bind output and thread settings; reject nonpositive thread counts."""
        if threads < 1:
            raise InvalidInputError("threads must be positive")
        self.output_dir = output_dir
        self.threads = threads

    def prepare(self, model: str) -> None:
        """Check installed weights, configure CPU threads and reserve a fresh output.

        Args:
            model: Registry identifier already installed using whirld pull.

        Raises:
            ModelNotInstalledError: The model is absent or its manifest is invalid.
            InvalidInputError: Registry or checkpoint differs from the curated pin.
            FileExistsError: Output exists; previous example results are preserved.
            ImportError: Required torch extra is missing.
        """
        import torch

        logging.basicConfig(level=os.environ.get("WHIRLD_LOG_LEVEL", "INFO").upper())
        store = ModelStore()
        entry = store.registry.get(model)
        manifest = store.load_manifest(model)
        bundled = ModelEntry.from_registry_file(
            Path(str(files("whirld").joinpath(f"registry_data/models/{model}.yaml")))
        )
        if entry != bundled or not manifest.matches_entry(entry):
            raise InvalidInputError(f"Re-pull {model}: expected bundled model registry")
        weights = store.paths.model_file(model, manifest.weights_file)
        if sha256_file(weights) != entry.distribution.sha256:
            raise InvalidInputError(f"Re-pull {model}: checkpoint checksum differs")
        torch.set_num_threads(self.threads)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        provenance = ModelProvenance(
            model=model,
            revision=entry.source.revision or "",
            checkpoint_sha256=manifest.sha256,
            python_version=platform.python_version(),
            platform=platform.platform(),
            torch_version=importlib.metadata.version("torch"),
            threads=self.threads,
        )
        (self.output_dir / "model.json").write_text(
            provenance.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )


class ModelProvenance(BaseModel):
    """Verified curated weights and CPU runtime identity for one recorded run."""

    model: str
    revision: str
    checkpoint_sha256: str
    python_version: str
    platform: str
    torch_version: str
    threads: int
    device: str = "cpu"


class OverlayPolygon(BaseModel):
    """One labeled WGS84 polygon for the preview's vector overlay."""

    coordinates: list[list[float]]
    label: str
    color: str = "#f9c74f"


class RasterImage(BaseModel):
    """Display-only RGB pixels on their original projected affine grid."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    rgb: np.ndarray
    transform: Affine
    crs: str


class Preview:
    """Render a self-contained HTML/SVG map over actual raster pixels."""

    @staticmethod
    def stretch(pixels: np.ndarray) -> np.ndarray:
        """Stretch three reflectance bands for display, without changing model input.

        Args:
            pixels: Float CHW RGB image; nonfinite values become black.

        Returns:
            uint8 CHW RGB stretched independently from the 2nd to 98th percentile.

        Raises:
            InvalidInputError: Input does not have three bands and two spatial axes.
        """
        if pixels.ndim != 3 or pixels.shape[0] != 3:
            raise InvalidInputError("Preview requires three RGB bands")
        out = np.zeros(pixels.shape, dtype=np.uint8)
        for i, band in enumerate(pixels):
            valid = np.isfinite(band)
            if not valid.any():
                continue
            low, high = np.percentile(band[valid], [2, 98])
            if high <= low:
                continue
            scaled = np.clip((np.where(valid, band, low) - low) / (high - low), 0, 1)
            out[i] = (scaled * 255).astype(np.uint8)
        return out

    @classmethod
    def write(
        cls,
        path: Path,
        title: str,
        scene: SceneSpec,
        polygons: list[OverlayPolygon],
        summary: str,
    ) -> None:
        """Read a small Sentinel RGB backdrop and render a standalone geographic map.

        Args:
            path: HTML destination; an RGB PNG is written alongside it.
            title: Human-readable workflow title.
            scene: Pinned STAC region and attribution.
            polygons: Labeled WGS84 footprint rings to project onto the image.
            summary: Plain-text result interpretation, escaped before HTML output.

        Raises:
            WhirldError: STAC selection, reading or radiometry conversion fails.
            OSError: Preview artifacts cannot be written.
        """
        entry = Registry().get("remoteclip")
        source = read_stac_item(
            scene.item_url, entry=entry, sensor=scene.sensor, bbox=scene.bbox
        )
        normalized = translate(source, entry, scene.sensor)
        norm = entry.band_contract.normalization
        reflectance = normalized.data * np.asarray(norm.std)[:, None, None]
        reflectance += np.asarray(norm.mean)[:, None, None]
        raster = RasterImage(
            rgb=cls.stretch(reflectance),
            transform=normalized.transform,
            crs=normalized.crs,
        )
        cls.write_raster(
            path, title, raster, polygons, summary + " " + scene.attribution
        )

    @classmethod
    def write_raster(
        cls,
        path: Path,
        title: str,
        raster: RasterImage,
        polygons: list[OverlayPolygon],
        summary: str,
    ) -> None:
        """Write an inline-image SVG map and companion PNG from an existing raster.

        Args:
            path: HTML destination inside the example's output directory.
            title: Display title, escaped as text.
            raster: uint8 CHW RGB image with projected grid metadata.
            polygons: WGS84 rings; labels/colors are escaped before serialization.
            summary: Plain text with attribution and interpretation limits.

        Raises:
            InvalidInputError: Pixel array is not nonempty uint8 RGB.
            OSError: Files cannot be written.
            rasterio.errors.RasterioError: Image encoding or reprojection fails.
        """
        rgb = raster.rgb
        if rgb.ndim != 3 or rgb.shape[0] != 3 or rgb.dtype != np.uint8 or not rgb.size:
            raise InvalidInputError("Preview raster must be nonempty uint8 CHW RGB")
        height, width = rgb.shape[1:]
        with warnings.catch_warnings(), MemoryFile() as memory:
            # Display PNGs intentionally lack georeferencing; spatial results
            # remain in GeoJSON/GeoTIFF and the SVG uses the real affine transform.
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with memory.open(
                driver="PNG", width=width, height=height, count=3, dtype="uint8"
            ) as dataset:
                dataset.write(rgb)
            png = memory.read()
        path.with_suffix(".png").write_bytes(png)
        uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        overlays = []
        inverse = ~raster.transform
        for polygon in polygons:
            if not polygon.coordinates:
                continue
            xs, ys = transform(
                "EPSG:4326",
                raster.crs,
                [point[0] for point in polygon.coordinates],
                [point[1] for point in polygon.coordinates],
            )
            points = [inverse @ (x, y) for x, y in zip(xs, ys, strict=True)]
            coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            color = html.escape(polygon.color, quote=True)
            label = html.escape(polygon.label)
            left = max(0.0, min(x for x, _ in points))
            right = min(float(width), max(x for x, _ in points))
            top = max(0.0, min(y for _, y in points))
            bottom = min(float(height), max(y for _, y in points))
            if right <= left or bottom <= top:
                continue
            # Keep labels inside the visible part of padded edge chips. Splitting
            # the chip ID from its result avoids clipping long class names.
            lines = polygon.label.split(": ", 1)
            font_size = min(
                14.0,
                (bottom - top) / (len(lines) + 1),
                max(1.0, right - left - 8) / (0.65 * max(map(len, lines), default=1)),
            )
            x, y = (left + right) / 2, (top + bottom) / 2
            spans = "".join(
                f'<tspan x="{x:.2f}" '
                f'y="{y + (i - (len(lines) - 1) / 2) * font_size * 1.2:.2f}">'
                f"{html.escape(line)}</tspan>"
                for i, line in enumerate(lines)
            )
            overlays.append(
                f'<polygon points="{coordinates}" fill="{color}" fill-opacity=".16" '
                f'stroke="{color}" stroke-width="2"><title>{label}</title></polygon>'
                f'<text text-anchor="middle" dominant-baseline="middle" '
                f'fill="white" stroke="#111" stroke-width="3" paint-order="stroke" '
                f'font-size="{font_size:.2f}">{spans}</text>'
            )
        text = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
body{{margin:0;background:#111827;color:#e5e7eb;font:16px system-ui,sans-serif}}
main{{max-width:1100px;margin:36px auto;padding:0 24px}}h1{{font-size:30px}}
p{{line-height:1.6;max-width:1000px}}svg{{width:100%;background:#080f1d}}
small{{color:#aebaca}}code{{color:#a7f3d0}}
</style><main><small>WHIRLD · REAL-WORLD EXAMPLES</small>
<h1>{html.escape(title)}</h1><p>{html.escape(summary)}</p>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img"
aria-label="{html.escape(title, quote=True)}">
<image href="{uri}" width="{width}" height="{height}"/>{''.join(overlays)}</svg>
<p><small>North up · display-only percentile stretch.
Hover over a footprint for its label. Full georeferencing is in the output files.
Padded chip footprints are clipped at the image boundary in this preview.</small></p>
</main></html>"""
        path.write_text(text, encoding="utf-8")
