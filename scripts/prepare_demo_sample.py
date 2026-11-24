"""Rebuild the attributed bundled RGB sample from the documented Sentinel-2 ROI."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from whirld import config
from whirld.core.registry import ModelEntry
from whirld.errors import InvalidInputError
from whirld.io.stac import read_stac_item

ITEM_ID = "S2A_10SFH_20240722_0_L2A"
ITEM_URL = (
    "https://earth-search.aws.element84.com/v1/collections/"
    f"sentinel-2-l2a/items/{ITEM_ID}"
)
SOURCE_BBOX = (-121.57, 38.53, -121.48, 38.60)
BANDS = ("B04", "B03", "B02")
NODATA = -9999.0


class DemoSamplePreparation:
    """Own one explicit source read and canonical-DN crop; no model weights needed."""

    def run(self, output_dir: Path) -> None:
        """Write a fresh standalone sample and exact artifact/provenance manifest.

        Args:
            output_dir: New directory; existing artifacts are never overwritten.

        Raises:
            InvalidInputError: Source names, radiometry, shape or time are invalid.
            WhirldError: Public item/asset reads fail.
            OSError: A new output directory or artifact cannot be written.

        Regeneration may change compressed TIFF bytes across GDAL versions; the
        committed manifest identifies the exact distributed sample, not every
        future response from the public source.
        """
        entry = ModelEntry.from_registry_file(
            config.bundled_registry_dir() / "models/remoteclip-rn50.yaml"
        )
        source = read_stac_item(
            ITEM_URL, entry=entry, sensor="sentinel-2-l2a", bbox=SOURCE_BBOX
        )
        if (
            tuple(source.band_descriptions) != BANDS
            or source.band_radiometry is None
            or source.width < 448
            or source.height < 448
        ):
            raise InvalidInputError(
                "Starter source no longer matches its RGB contract."
            )
        col, row = (source.width - 448) // 2, (source.height - 448) // 2
        window = Window(col, row, 448, 448)
        pixels = source.data[:, row : row + 448, col : col + 448].astype(np.float64)
        transforms: list[str] = []
        for index, band in enumerate(source.band_radiometry):
            if not band.has_transform:
                raise InvalidInputError(
                    "Starter source must declare physical radiometry."
                )
            pixels[index] = (pixels[index] * band.scale + band.offset) / 0.0001
            transforms.append(f"{BANDS[index]}:scale={band.scale},offset={band.offset}")
        valid = np.isfinite(pixels).all(axis=0)
        pixels[:, ~valid] = NODATA
        pixels = pixels.astype(np.float32)
        if not np.isfinite(pixels).all() or not valid.any():
            raise InvalidInputError("Starter source has no finite RGB observations.")
        acquired = source.tags.get("TIFFTAG_DATETIME")
        if not acquired:
            raise InvalidInputError(
                "Starter source is missing its acquisition datetime."
            )
        output_dir.mkdir(parents=True, exist_ok=False)
        image = output_dir / "starter-rgb.tif"
        transform = rasterio.windows.transform(window, source.transform)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(
                image,
                "w",
                driver="GTiff",
                width=448,
                height=448,
                count=3,
                dtype="float32",
                crs=source.crs,
                transform=transform,
                nodata=NODATA,
                tiled=True,
                blockxsize=128,
                blockysize=128,
                compress="deflate",
                predictor=3,
            ) as target:
                target.write(pixels)
                target.write_mask(valid.astype(np.uint8) * 255)
                target.descriptions = BANDS
                target.update_tags(
                    TIFFTAG_DATETIME=datetime.fromisoformat(
                        acquired.replace("Z", "+00:00")
                    ).strftime("%Y:%m:%d %H:%M:%S"),
                    TIFFTAG_IMAGEDESCRIPTION=(
                        "Contains modified Copernicus Sentinel data 2024"
                    ),
                )
        # This mapping is the JSON serialization adapter for the sample manifest.
        manifest = {
            "schema_version": 1,
            "id": "sentinel2-sacramento-rgb-20240722",
            "file": image.name,
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "size_bytes": image.stat().st_size,
            "width": 448,
            "height": 448,
            "crs": source.crs,
            "transform": list(transform)[:6],
            "band_names": BANDS,
            "sensor": "sentinel-2-l2a",
            "dtype": "float32",
            "units": "canonical_dn",
            "nodata": NODATA,
            "source_item_url": ITEM_URL,
            "source_item_id": ITEM_ID,
            "source_bbox": SOURCE_BBOX,
            "source_datetime": acquired,
            "source_crop_window": (col, row, 448, 448),
            "attribution": (
                "Contains modified Copernicus Sentinel data 2024, via Element 84. "
                "Cropped RGB bands converted to canonical DN; display uses a "
                "percentile stretch."
            ),
            "license_url": "https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice",
            "preparation": (
                "read_stac_item on the stated ROI, target 10 m; central 448x448 "
                "window; canonical_DN=(stored*scale+offset)/0.0001 using "
            )
            + ";".join(transforms)
            + (
                "; float32, band order B04/B03/B02; all-band invalid pixels stored "
                "as -9999 with internal mask; no model normalization baked in."
            ),
        }
        (output_dir / "sample.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        logging.getLogger(__name__).info(
            "Prepared %s (%d bytes)", image, image.stat().st_size
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    DemoSamplePreparation().run(arguments.output_dir)
