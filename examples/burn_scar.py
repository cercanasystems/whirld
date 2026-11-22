"""Run real Prithvi burn-scar segmentation on a pinned official HLS sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import rasterio
from pydantic import BaseModel, ConfigDict, Field
from rasterio.warp import transform_bounds

import whirld
from examples.common import ExampleRuntime, Preview, RasterImage, RunMetrics
from whirld.errors import InvalidInputError

_LOG = logging.getLogger(__name__)
REVISION = "a3f2c410e45b8ac7417976614528a872f024d831"
SAMPLE_NAME = "subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif"
SAMPLE_URL = (
    "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars/"
    f"resolve/{REVISION}/examples/{SAMPLE_NAME}"
)
SAMPLE_SHA256 = "13bc592a5e569d837bd8bb3524bb0d2f28418830bcc7b0750e74033078f8b17e"
SAMPLE_BYTES = 6295168
BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")
ATTRIBUTION = (
    "Imagery: HLS Burn Scars, Phillips et al. (2023), "
    "https://doi.org/10.57967/hf/0956, CC BY 4.0 "
    "(https://creativecommons.org/licenses/by/4.0/). "
    "Display modified by RGB stretching and, in the prediction view, an overlay. "
    "Model: IBM/NASA Prithvi EO 2.0 BurnScars, Apache 2.0."
)


class BurnScarOptions(BaseModel):
    """Validated output destination and CPU parallelism for the fixed example."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    output_dir: Path
    threads: int = Field(default=2, strict=True, gt=0)


class BurnScarReport(BaseModel):
    """Reproducible source identity, spatial checks and actual resource results."""

    source_url: str = SAMPLE_URL
    source_sha256: str = SAMPLE_SHA256
    source_bytes: int = SAMPLE_BYTES
    model: str = "prithvi-burn-scar"
    device: str = "cpu"
    mask_path: Path
    preview_path: Path
    imagery_path: Path
    width: int
    height: int
    crs: str
    burned_pixels: int
    valid_pixels: int
    burned_fraction: float
    metrics: RunMetrics


class HlsSample:
    """Own retrieval and explicit reflectance metadata for the official sample."""

    def prepare(self, output_dir: Path) -> tuple[Path, Path]:
        """Download a checked artifact and write its local STAC input adapter.

        Args:
            output_dir: Existing new run directory, owned by ExampleRuntime.

        Returns:
            Original unchanged imagery and a local STAC item naming all six bands.

        Raises:
            InvalidInputError: Size, digest or raster contract differs from the pin.
            OSError: Download or local output fails; partial download is removed.
        """
        image_path = output_dir / SAMPLE_NAME
        partial = image_path.with_suffix(".part")
        digest = hashlib.sha256()
        received = 0
        try:
            with urlopen(SAMPLE_URL, timeout=60) as response, partial.open("xb") as out:
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > SAMPLE_BYTES:
                        raise InvalidInputError("HLS sample exceeds pinned byte size")
                    digest.update(block)
                    out.write(block)
            if received != SAMPLE_BYTES or digest.hexdigest() != SAMPLE_SHA256:
                raise InvalidInputError("HLS sample size or SHA-256 differs from pin")
            partial.rename(image_path)
        finally:
            partial.unlink(missing_ok=True)
        item_path = output_dir / "hls-item.json"
        self.write_item(image_path, item_path)
        return image_path, item_path

    @staticmethod
    def write_item(image_path: Path, item_path: Path) -> None:
        """Adapt already-scaled HLS reflectance without modifying source pixels.

        Args:
            image_path: Downloaded six-band north-up 30 m reflectance raster.
            item_path: Destination local STAC item consumed by Whirld.

        Raises:
            InvalidInputError: Source geometry/bands/data do not match this example.
            OSError: Input or output cannot be opened.

        Dataset documentation fixes the band order. Scale 1 and offset 0 tell
        Whirld these floats are reflectance, avoiding a second 0.0001 scaling.
        """
        with rasterio.open(image_path) as source:
            if (
                source.count != 6
                or (source.width, source.height) != (512, 512)
                or source.crs is None
                or not source.crs.is_projected
                or source.transform.a != 30
                or source.transform.e != -30
                or source.transform.b != 0
                or source.transform.d != 0
                or any(dtype != "float32" for dtype in source.dtypes)
            ):
                raise InvalidInputError("HLS sample raster contract differs from pin")
            west, south, east, north = transform_bounds(
                source.crs, "EPSG:4326", *source.bounds
            )
        # Raw mappings remain at the STAC serialization adapter boundary.
        item = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "stac_extensions": [
                "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
                "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
            ],
            "bbox": [west, south, east, north],
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [west, south],
                        [east, south],
                        [east, north],
                        [west, north],
                        [west, south],
                    ]
                ],
            },
            "links": [],
            "id": "HLS.S30.T10SEH.2018190.v1.4",
            "properties": {
                "datetime": None,
                "start_datetime": "2018-07-09T00:00:00Z",
                "end_datetime": "2018-07-09T23:59:59Z",
            },
            "assets": {
                "reflectance": {
                    "href": image_path.resolve().as_uri(),
                    "type": "image/tiff; application=geotiff",
                    "roles": ["data"],
                    "eo:bands": [{"name": band} for band in BANDS],
                    "raster:bands": [
                        {"scale": 1.0, "offset": 0.0, "nodata": 0.0} for _ in BANDS
                    ],
                }
            },
        }
        item_path.write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")


class BurnScarExample:
    """Own source preparation, real inference, validation and local previews."""

    def __init__(self, options: BurnScarOptions) -> None:
        """Bind validated run options without reading data or loading a model."""
        self.options = options

    @staticmethod
    def validate_mask(image_path: Path, mask_path: Path) -> tuple[int, int]:
        """Require a georeferenced binary mask on the unchanged sample grid.

        Args:
            image_path: Original pinned sample raster.
            mask_path: Completed model mask GeoTIFF.

        Returns:
            Counts of predicted burned and valid pixels; neither implies accuracy.

        Raises:
            InvalidInputError: Output classes, dimensions or georeferencing differ.
            OSError: Raster artifacts cannot be read.
        """
        with rasterio.open(image_path) as source, rasterio.open(mask_path) as target:
            if (
                source.crs != target.crs
                or not source.transform.almost_equals(target.transform)
                or source.shape != target.shape
                or target.count != 1
                or target.dtypes != ("uint8",)
                or target.nodata != 255
            ):
                raise InvalidInputError("Burn mask does not preserve the source grid")
            mask = target.read(1)
            valid = target.read_masks(1) != 0
            expected_valid = (source.read_masks() != 0).all(axis=0)
            if not np.array_equal(valid, expected_valid):
                raise InvalidInputError("Burn mask validity differs from source")
            if not valid.any() or not np.isin(mask[valid], [0, 1]).all():
                raise InvalidInputError("Burn mask has invalid or missing classes")
        return int(np.count_nonzero((mask == 1) & valid)), int(valid.sum())

    @staticmethod
    def preview_images(
        image_path: Path, mask_path: Path
    ) -> tuple[RasterImage, RasterImage]:
        """Create original RGB and orange prediction overlay on the source grid.

        Args:
            image_path: HLS reflectance sample; channels 3/2/1 are RGB.
            mask_path: Validated binary prediction aligned with the sample.

        Returns:
            Original and blended RGB rasters, using identical display stretch.

        Raises:
            OSError: Raster inputs cannot be read.
        """
        with rasterio.open(image_path) as source:
            pixels = source.read([3, 2, 1])
            rgb = Preview.stretch(pixels)
            original = RasterImage(
                rgb=rgb, transform=source.transform, crs=str(source.crs)
            )
        with rasterio.open(mask_path) as mask_source:
            burned = mask_source.read(1) == 1
        overlay = rgb.copy()
        color = np.array([255, 110, 20], dtype=np.float32).reshape(3, 1)
        overlay[:, burned] = (rgb[:, burned] * 0.5 + color * 0.5).astype(np.uint8)
        blended = RasterImage(
            rgb=overlay, transform=original.transform, crs=original.crs
        )
        return original, blended

    def run(self) -> BurnScarReport:
        """Run the small CPU example and persist mask, previews and JSON evidence.

        Returns:
            Source provenance, independently checked output counts, and resource use.

        Raises:
            WhirldError: Model installation, source contract or inference fails.
            OSError: Download or artifact publication fails.
        """
        runtime = ExampleRuntime(self.options.output_dir, threads=self.options.threads)
        runtime.prepare("prithvi-burn-scar")
        started = time.monotonic()
        image_path, item_path = HlsSample().prepare(self.options.output_dir)
        mask_path = self.options.output_dir / "burn-mask.tif"
        whirld.segment(
            item_path,
            model="prithvi-burn-scar",
            sensor="hls",
            device="cpu",
            output=mask_path,
            no_warnings=True,
        )
        burned, valid = self.validate_mask(image_path, mask_path)
        original, blended = self.preview_images(image_path, mask_path)
        imagery_path = self.options.output_dir / "imagery.html"
        preview_path = self.options.output_dir / "preview.html"
        Preview.write_raster(
            imagery_path,
            "HLS burn-scar example — source imagery",
            original,
            [],
            "HLS S30 T10SEH, 2018-07-09. RGB uses a 2–98% display stretch. "
            "See preview.html for the model prediction overlay. " + ATTRIBUTION,
        )
        Preview.write_raster(
            preview_path,
            "Prithvi burn-scar prediction",
            blended,
            [],
            f"Orange: predicted burned area ({burned / valid:.1%} of valid pixels). "
            "This is model output, not a validated fire perimeter. "
            "See imagery.html for the source image without the overlay. " + ATTRIBUTION,
        )
        report = BurnScarReport(
            mask_path=mask_path,
            preview_path=preview_path,
            imagery_path=imagery_path,
            width=original.rgb.shape[2],
            height=original.rgb.shape[1],
            crs=original.crs,
            burned_pixels=burned,
            valid_pixels=valid,
            burned_fraction=burned / valid,
            metrics=RunMetrics.capture(started),
        )
        (self.options.output_dir / "report.json").write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        _LOG.info("Burn-scar example complete: %s", preview_path)
        return report


def main() -> int:
    """Parse output/thread settings and execute the example; return zero on success.

    Raises:
        SystemExit: Required CLI arguments are missing or help is requested.
        WhirldError: Model/source/inference validation fails with an actionable error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    BurnScarExample(
        BurnScarOptions(output_dir=args.output_dir.resolve(), threads=args.threads)
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
