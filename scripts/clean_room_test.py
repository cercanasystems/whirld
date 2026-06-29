"""Clean-room smoke test for an installed Whirld (PRD section 17.3).

Run *inside a fresh environment* (e.g. the ``docker/clean-room.Dockerfile`` image)
where only the published package + its base dependencies are installed — no dev
tools, no test harness, no pre-built virtualenv. It exercises the shipped surface a
new user hits in their first five minutes:

1. ``import whirld`` stays lazy (no rasterio/numpy/torch pulled in).
2. ``whirld pull clay-v1`` materializes the offline reference model and verifies it.
3. ``whirld embed <geotiff>`` produces embeddings from a local raster.
4. ``whirld embed <stac item.json>`` produces embeddings from a STAC item via the
   ``/vsicurl/`` reader (here against ``file://`` assets, so it is hermetic).
5. Optionally, if ``WHIRLD_TEST_STAC_URL`` is set, the same against a *real* remote
   STAC item — proving live HTTP range reads on a clean Linux install.

The fixtures are generated inline with rasterio (a base dependency), so this script
depends only on the installed package — never on the repo's ``tests/`` tree.

Exit code 0 means every step passed; any failure prints the reason and exits 1.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

_S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
_COMMON = ["blue", "green", "red", "nir", "swir16", "swir22"]


def _step(message: str) -> None:
    """Print a progress line."""
    print(f"[clean-room] {message}", flush=True)


def _fail(message: str) -> None:
    """Print a failure and exit non-zero."""
    print(f"[clean-room] FAIL: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def _check_lazy_import() -> None:
    """Assert ``import whirld`` does not import heavy modules.

    Run in a *fresh* interpreter so an earlier ``import rasterio`` in this process
    cannot mask an eager import.
    """
    code = (
        "import whirld, sys; "
        "heavy=[m for m in ('rasterio','numpy','torch') if m in sys.modules]; "
        "print(whirld.__version__); "
        "sys.exit('eager: '+repr(heavy) if heavy else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    if result.returncode != 0:
        _fail(f"import whirld is not lazy: {result.stderr.strip()}")
    _step(f"import whirld is lazy (version {result.stdout.strip()})")


def _write_geotiff(path: Path) -> None:
    """Write a tiny six-band Sentinel-2-like GeoTIFF."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(7)
    data = rng.integers(0, 4000, size=(6, 128, 128), dtype=np.uint16)
    profile = {
        "driver": "GTiff",
        "height": 128,
        "width": 128,
        "count": 6,
        "dtype": "uint16",
        "crs": "EPSG:32630",
        "transform": from_origin(320000.0, 5822560.0, 10.0, 10.0),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for idx, band in enumerate(_S2_BANDS, start=1):
            dst.set_band_description(idx, band)
        dst.update_tags(TIFFTAG_IMAGEDESCRIPTION="Sentinel-2 L2A scene")


def _write_stac_item(directory: Path) -> Path:
    """Write a STAC item whose assets are single-band COGs (``file://`` hrefs)."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(7)
    transform = from_origin(320000.0, 5822560.0, 10.0, 10.0)
    assets: dict[str, dict] = {}
    for band, alias in zip(_S2_BANDS, _COMMON, strict=True):
        asset_path = directory / f"{band}.tif"
        profile = {
            "driver": "GTiff",
            "height": 128,
            "width": 128,
            "count": 1,
            "dtype": "uint16",
            "crs": "EPSG:32630",
            "transform": transform,
            "tiled": True,
            "blockxsize": 64,
            "blockysize": 64,
        }
        with rasterio.open(asset_path, "w", **profile) as dst:
            dst.write(rng.integers(0, 4000, size=(1, 128, 128), dtype=np.uint16))
            dst.set_band_description(1, band)
        assets[band] = {
            "href": asset_path.as_uri(),
            "type": "image/tiff",
            "eo:bands": [{"name": band, "common_name": alias}],
        }
    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": "clean-room-item",
        "collection": "sentinel-2-l2a",
        "properties": {
            "datetime": "2024-06-01T10:00:00Z",
            "platform": "sentinel-2a",
            "constellation": "sentinel-2",
        },
        "geometry": None,
        "assets": assets,
    }
    item_path = directory / "item.json"
    item_path.write_text(json.dumps(item, indent=2), encoding="utf-8")
    return item_path


def _run_cli(args: list[str]) -> None:
    """Run the installed ``whirld`` console script and fail on a non-zero exit."""
    result = subprocess.run(
        ["whirld", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _fail(
            f"`whirld {' '.join(args)}` exited {result.returncode}\n"
            f"{result.stdout}\n{result.stderr}"
        )


def _assert_embeddings(npy_path: Path, label: str) -> None:
    """Assert an embeddings file exists with a sane shape."""
    import numpy as np

    if not npy_path.exists():
        _fail(f"{label}: no embeddings written at {npy_path}")
    array = np.load(npy_path)
    if array.ndim != 2 or array.shape[1] != 512 or not np.isfinite(array).all():
        _fail(f"{label}: bad embeddings shape/values {array.shape}")
    _step(f"{label}: embeddings {array.shape} ✓")


def main() -> int:
    """Run the clean-room sequence; return a process exit code."""
    import os

    import rasterio  # base dependency; report the bundled GDAL

    _step(f"python {sys.version.split()[0]}, rasterio {rasterio.__version__}")
    _step(f"GDAL (bundled) {rasterio.__gdal_version__}")
    _check_lazy_import()

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        _step("pull clay-v1 (offline reference)")
        _run_cli(["pull", "clay-v1"])

        tif = work / "scene.tif"
        _write_geotiff(tif)
        emb_tif = work / "scene_embeddings.npy"
        _step("embed local GeoTIFF")
        _run_cli(["embed", "--model", "clay-v1", "--output", str(emb_tif), str(tif)])
        _assert_embeddings(emb_tif, "local GeoTIFF")

        item = _write_stac_item(work)
        emb_stac = work / "stac_embeddings.npy"
        _step("embed STAC item (file:// assets, /vsicurl/ reader)")
        _run_cli(["embed", "--model", "clay-v1", "--output", str(emb_stac), str(item)])
        _assert_embeddings(emb_stac, "STAC item")

        real_url = os.environ.get("WHIRLD_TEST_STAC_URL")
        if real_url:
            emb_real = work / "real_embeddings.npy"
            args = ["embed", "--model", "clay-v1", "--sensor", "sentinel-2-l2a"]
            bbox = os.environ.get("WHIRLD_TEST_STAC_BBOX")
            if bbox:
                args += ["--bbox", bbox]
            args += ["--output", str(emb_real), real_url]
            _step(f"embed REAL STAC item ({real_url})")
            _run_cli(args)
            _assert_embeddings(emb_real, "real STAC item")
        else:
            _step("skipping real STAC item (set WHIRLD_TEST_STAC_URL to enable)")

    _step("ALL CLEAN-ROOM CHECKS PASSED ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
