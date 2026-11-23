# Local GeoTIFF streaming (unreleased)

The current source checkout adds local files to `embed_to_disk()`,
`classify_to_disk()` and `segment_to_disk()`, and their CLI `--stream` modes.
This feature is **not included in the published 0.3.0 artifacts**; no new version
has been assigned. Install the checkout in your `.venv` as described in the
[README](../README.md#install-development).

## Runnable setup and usage

Install the `remoteclip` and `prithvi` extras before running the trained models.
The following small synthetic inputs make the commands reproducible without an
imagery download. They exercise execution, not meaningful prediction accuracy.
For real imagery, substitute files with the same documented sensor/band/units
contract; the [real-world examples](../examples/README.md) explain source data.

```bash
.venv/bin/whirld pull clay-v1
.venv/bin/whirld pull remoteclip
.venv/bin/whirld pull prithvi-burn-scar
.venv/bin/python - <<'PY'
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

root = Path("local-streaming-inputs")
root.mkdir(exist_ok=True)
for filename, names in (
    ("sentinel2-dn.tif", ("B02", "B03", "B04", "B08", "B11", "B12")),
    ("hls-dn.tif", ("B02", "B03", "B04", "B8A", "B11", "B12")),
):
    pixels = np.full((6, 256, 256), 2000, dtype=np.uint16)
    with rasterio.open(
        root / filename, "w", driver="GTiff", width=256, height=256,
        count=6, dtype="uint16", crs="EPSG:32610",
        transform=from_origin(630000, 4260000, 10, 10),
        tiled=True, blockxsize=128, blockysize=128,
    ) as dataset:
        dataset.write(pixels)
        dataset.descriptions = names
PY
```

CLI commands, run from the same directory:

```bash
.venv/bin/whirld embed local-streaming-inputs/sentinel2-dn.tif --model clay-v1 \
  --sensor sentinel-2-l2a --device cpu --stream --batch-size 2 --output local-embeddings.npy
.venv/bin/whirld classify local-streaming-inputs/sentinel2-dn.tif --model remoteclip \
  --sensor sentinel-2-l2a --device cpu --query 'a satellite image of water' \
  --stream --batch-size 2 --cache-mb 8 --all --output local-scores.geojson
.venv/bin/whirld segment local-streaming-inputs/hls-dn.tif --model prithvi-burn-scar \
  --sensor hls --device cpu --stream --batch-size 2 --cache-mb 8 --output local-mask.tif
```

Equivalent Python calls return small typed disk summaries:

```python
import whirld

embeddings = whirld.embed_to_disk(
    "local-streaming-inputs/sentinel2-dn.tif", model="clay-v1",
    sensor="sentinel-2-l2a", batch_size=2, cache_mb=8,
    device="cpu", output="python-local-embeddings.npy",
)
scores = whirld.classify_to_disk(
    "local-streaming-inputs/sentinel2-dn.tif", model="remoteclip",
    sensor="sentinel-2-l2a", query="a satellite image of water", top_k=None,
    batch_size=2, cache_mb=8, device="cpu", output="python-local-scores.geojson",
)
mask = whirld.segment_to_disk(
    "local-streaming-inputs/hls-dn.tif", model="prithvi-burn-scar", sensor="hls",
    batch_size=2, cache_mb=8, device="cpu", output="python-local-mask.tif",
)
```

`Path` inputs and absolute `file://` TIFF URLs work too. Use
`Path("scene.tif").resolve().as_uri()` to encode filenames safely. File URLs permit
an empty or `localhost` authority and no query or fragment. Ordinary filesystem
filenames retain literal `%`, `#` and `?` characters. Each input is bound to an
absolute existing regular file before GDAL opens it; the driver must be GTiff.

## Local input contract

Local streaming preserves eager local-file behavior:

- Values use the model's **canonical digital number (DN)** convention. TIFF
  scales/offsets are ignored, including nonidentity transforms. Already-scaled
  reflectance must be converted to the sensor contract's DN convention first,
  or described through a STAC item with explicit radiometry. For bundled sensors,
  reflectance 0.2 corresponds to canonical DN 2000.
- Sensor precedence is explicit `sensor`, TIFF tags, native band descriptions,
  then resolution. Flood still needs explicit L1C selection or unambiguous
  processing-level metadata; shared Sentinel-2 band names do not identify L1C.
- Named descriptions must uniquely include every required **native band name**,
  case insensitively. Entirely unlabeled files use positional contract order and
  may contain extra trailing bands. When any labels exist, every required native
  band must be named; missing required names and duplicate descriptions fail.
  Extra unlabeled bands are allowed when all required names are present. Local
  files do not borrow the direct COG alias/override rules.
- TIFF nodata, internal/external masks and local PAM metadata remain visible.
  Nonfinite values become missing observations before bilinear resampling.
  `TIFFTAG_DATETIME` and the existing embedding datetime override retain their
  eager meanings.
- Inputs need a north-up, unrotated projected grid in meters. `crs` assigns a
  missing CRS; it does not reproject. Both pixel dimensions are respected, and
  the original affine/dimensions are retained when both resolutions are within
  0.5% of the model target. Otherwise ceil-based resampling covers the source.

Local streaming rejects `bbox` / `--bbox`; crop first or reference the file from
STAC for geographic windowing. `CogOptions` and CLI `--cog`, `--bands`, `--scale`
and `--offset` apply only to direct remote HTTP(S) rasters. Streaming STAC assets
still need a shared CRS; general reprojection remains deferred in
[issue #21](https://github.com/cercanasystems/whirld/issues/21).

## Memory, outputs and verification

The reader owns bounded chip batches, GDAL caches and lazy per-band warp views;
it never creates a scene-sized pixel/chip array. Model/framework buffers and
source decoder blocks also consume memory. Prefer tiled TIFFs: a large untiled
strip can require a large decoder buffer despite a small requested chip. Lower
`batch_size` and `cache_mb` for tighter budgets. Output and temporary disk usage
still grow with scene size.

Existing writers are reused. [Embedding publication](streaming.md#output-publication-and-failures)
stages NPY and JSON with per-file replacement/rollback; the pair is not
crash-atomic. [Classification](classification-streaming.md) uses disk-backed
ranking and atomically replaces GeoJSON after completion.
[Segmentation](segmentation-streaming.md) atomically replaces its completed
GeoTIFF. Failures preserve prior output and remove staging; concurrent runs need
distinct destinations. REST upload endpoints retain their existing eager behavior.

Validation includes **78 local I/O unit checks and 43 integration checks** across
all three workflows. Independent eager comparisons cover native/tolerance grids,
up/downsampling and anisotropic pixels, masks, nodata/NaN/infinities, reordered
bands, source metadata and padded edges. Normalized pixels agree at
`rtol=atol=1e-6`, with unchanged geometry; tests also verify driver restrictions,
window/batch bounds, handle cleanup and GDAL option restoration. These checks
establish pipeline equivalence, not model accuracy.

### Recorded real-model and guide validation

On 2026-11-22, cached curated checkpoints passed fresh SHA-256 checks and local
CPU eager/stream comparisons at streaming batches 1 and 2. Runtime was Python
3.13.7, Torch 2.12.1, NumPy 2.5.0 and Rasterio 1.5.2 on macOS ARM64, with two CPU
threads. Batch 2 used safely encoded `file://` input URLs.

| Model | Local input/output | Agreement with eager |
|---|---|---|
| Prithvi flood | Official projected India sample, 461×513, two chips | Exact classes, invalid mask, CRS and affine; 233029 valid / 109762 positive pixels, classes 0/1 and nodata 255 |
| Clay v1.5 | Ten-band Sentinel-2 subset, four 1024-value embeddings | Bitwise equal float32 embeddings; all sidecar metadata equal except timestamps |
| RemoteCLIP ViT-B/32 | Same subset, six chips and four text prompts | Exact geometry and ranking; maximum score errors 4.18e-6 / 3.22e-6 for batches 1/2, within rtol=1e-4 / atol=1e-5 |

The flood input is the official sample described in the [flood guide](flood-model.md),
used directly with explicit `sentinel-2-l1c`. The other input was materialized from
[the pinned public Sentinel-2 item](https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2A_30UXD_20230630_1_L2A)
at bbox `[-1.32, 52.49, -1.26, 52.53]`, producing a 418×455 projected 10 m TIFF.
Every band declared scale 0.0001 and offset -0.1. Stored samples were converted
once using `(stored * scale + offset) / 0.0001` to canonical DN; the resulting
local TIFF has identity scales/offsets and preserved native names/time/geometry.
This conversion is necessary because the local path ignores TIFF transforms.

Streaming calls took 5.68/6.81 s for flood, 3.71/3.84 s for Clay and 1.78/1.69 s
for RemoteCLIP at batches 1/2. Peak process RSS, including model loading, ranged
from 1.52 to 3.70 GiB; these small-scene measurements do not establish a memory
scaling bound or GPU behavior.

The setup, three CLI commands and three Python calls above also ran unchanged in
a fresh working directory using the cached models. Both interfaces produced
identical embeddings, GeoJSON and mask pixels: one 512-value reference embedding,
four classification features, and an 86×86 mask at 30 m with 7225 class-zero
pixels and 171 nodata pixels. The HLS fixture deliberately starts at 10 m, so this
checks ceil resampling and padded-edge validity as well as command syntax.
