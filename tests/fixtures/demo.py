"""Small generated raster resources for installed-demo contract checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from whirld.demo.sample import DemoSampleSpec


def make_demo_resource(directory: Path) -> DemoSampleSpec:
    """Write a complete standalone RGB fixture and its exact immutable metadata."""
    data = directory / "data"
    data.mkdir(parents=True)
    image = data / "starter-rgb.tif"
    pixels = np.full((3, 448, 448), 2000, dtype=np.float32)
    pixels[:, :224, :224] = 0
    pixels[:, :224, 224:] = 1000
    pixels[:, 224:, :224] = 4000
    pixels[:, 224:, 224:] = 8000
    pixels[:, 1:3, 1:4] = -9999
    transform = from_origin(500000, 4300000, 10, 10)
    with rasterio.open(
        image,
        "w",
        driver="GTiff",
        width=448,
        height=448,
        count=3,
        dtype="float32",
        crs="EPSG:32610",
        transform=transform,
        nodata=-9999,
        tiled=True,
        blockxsize=128,
        blockysize=128,
        compress="lzw",
    ) as source:
        source.write(pixels)
        source.descriptions = ("B04", "B03", "B02")
    bbox = transform_bounds("EPSG:32610", "EPSG:4326", 500000, 4295520, 504480, 4300000)
    spec = DemoSampleSpec(
        id="fixture",
        sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        size_bytes=image.stat().st_size,
        crs="EPSG:32610",
        transform=tuple(transform)[:6],
        source_item_url="https://example.invalid/item.json",
        source_item_id="fixture",
        source_bbox=bbox,
        source_datetime="2024-07-22T18:00:00Z",
        source_crop_window=(0, 0, 448, 448),
        attribution="Generated synthetic fixture; no accuracy claim.",
        license_url="https://example.invalid/terms",
        preparation="Known canonical-DN quadrants with valid zeros and nodata.",
    )
    (data / "sample.json").write_text(spec.model_dump_json(indent=2))
    return spec
