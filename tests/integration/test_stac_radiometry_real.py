"""Opt-in validation of STAC radiometry against pinned public Sentinel-2 COGs.

Run with ``WHIRLD_TEST_STAC_RADIOMETRY=1``. No model checkpoints are downloaded:
both model contracts are tested at the normalized-input boundary. The independent
reference reads stored pixels with rasterio and applies the documented physical
unit conversion explicitly, without using Whirld's radiometry helpers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from whirld.core.contract import translate
from whirld.core.registry import Registry
from whirld.io.stac import read_stac_item

pytestmark = pytest.mark.skipif(
    os.environ.get("WHIRLD_TEST_STAC_RADIOMETRY") != "1",
    reason="set WHIRLD_TEST_STAC_RADIOMETRY=1 to read the pinned public COGs",
)

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "stac_offset_sample.json"


@pytest.mark.parametrize("model", ["clay-v1", "clay-v1.5"])
def test_pinned_stac_radiometry(whirld_home: Path, model: str) -> None:
    """Compare both model input conventions with independent COG arithmetic.

    Args:
        whirld_home: Isolated registry/cache fixture; no user cache is touched.
        model: Reference reflectance or real Clay DN normalization contract.

    Raises:
        AssertionError: Pinned source metadata/pixels drift or conversion differs.
    """
    sample: dict[str, Any] = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    with urlopen(sample["item_url"], timeout=30) as response:
        item = json.load(response)
    assert item["id"] == sample["item_id"]

    entry = Registry().get(model)
    source = read_stac_item(
        sample["item_url"],
        entry=entry,
        sensor=sample["sensor"],
        bbox=tuple(sample["bbox"]),
    )
    actual = translate(source, entry, sample["sensor"]).data
    norm = entry.band_contract.normalization
    sensor = entry.band_contract.sensors[sample["sensor"]]

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True):
        for index, band in enumerate(sensor.bands):
            asset = item["assets"][sample["bands"][band]]
            metadata = asset["raster:bands"][0]
            for key, value in sample["radiometry"].items():
                assert metadata[key] == value

            with rasterio.open(f"/vsicurl/{asset['href']}") as dataset:
                if band == "B02":
                    pixel = sample["source_pixel"]
                    stored = dataset.read(
                        pixel["band"],
                        window=Window(pixel["col"], pixel["row"], 1, 1),
                    ).item()
                    assert stored == pixel["stored_value"]
                    physical = stored * metadata["scale"] + metadata["offset"]
                    assert physical == pytest.approx(pixel["reflectance"])
                    assert physical / 0.0001 == pytest.approx(pixel["clay_dn"])

                with WarpedVRT(
                    dataset,
                    crs=source.crs,
                    transform=source.transform,
                    width=source.width,
                    height=source.height,
                    resampling=Resampling.bilinear,
                    src_nodata=metadata["nodata"],
                    nodata=float("nan"),
                    dtype="float32",
                ) as vrt:
                    raw = vrt.read(1, masked=True).astype(np.float64)

            physical_values = raw * metadata["scale"] + metadata["offset"]
            # The real Clay encoder expects reflectance expressed as canonical
            # DN; the reference encoder's means/std are in reflectance units.
            model_values = (
                physical_values / 0.0001 if model == "clay-v1.5" else physical_values
            )
            expected = (model_values - norm.mean[index]) / norm.std[index]
            expected = expected.filled(entry.band_contract.nodata_fill)
            assert np.isfinite(actual[index]).all()
            np.testing.assert_allclose(actual[index], expected, rtol=2e-5, atol=2e-5)
