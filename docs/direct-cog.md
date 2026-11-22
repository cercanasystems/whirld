# Direct remote COG input

Whirld accepts an HTTP(S) GeoTIFF URL directly in the Python and CLI embedding,
classification and segmentation workflows. A STAC item is optional. Remote paths
ending in `.tif` or `.tiff` (case insensitive, ignoring query parameters) select
this reader automatically. Use `cog=whirld.CogOptions()` or CLI `--cog` for an
extensionless download endpoint. The server must support HTTP range requests.
Only the GeoTIFF driver is allowed; the filename does not prove COG structure.

## Explicit sensor and band identity

Always supply `sensor` / `--sensor`. Whirld uses complete, unique TIFF band
**descriptions** matching the selected model's native names or aliases. It does
not guess spectral identity from the number of bands. If descriptions are absent,
incomplete or ambiguous, provide names for **every stored band in source order**:

```python
import whirld

result = whirld.embed_to_disk(
    "https://example.org/scene.tif",  # Replace with your range-enabled raster URL.
    model="clay-v1",
    sensor="sentinel-2-l2a",
    cog=whirld.CogOptions(bands=("B02", "B03", "B04", "B08", "B11", "B12")),
    output="scene.npy",
    batch_size=8,
    device="cpu",
)
```

```bash
whirld pull clay-v1
whirld embed 'https://example.org/scene.tif' --model clay-v1 \
  --sensor sentinel-2-l2a --bands B02,B03,B04,B08,B11,B12 \
  --stream --batch-size 8 --output scene.npy
```

These are templates: replace the URL and band labels with your data. Required
bands are selected and reordered to the model contract; additional named bands
are allowed. Explicit names replace TIFF descriptions. Duplicate names or two
source bands matching the same native/alias identity are rejected. The same
`cog` argument and CLI options work with `embed`, `classify`, and `segment`.
COG options apply only to direct remote rasters, not local files or STAC JSON.

## Units, masks and geometry

Transform precedence is per field: explicit `scale` or `offset` overrides its TIFF
counterpart; an unspecified field retains the TIFF value. Scalars apply to all
bands; tuples must describe every source band in source order. Transforms mean
`reflectance = stored_value * scale + offset`, applied before model normalization.

- Nonidentity TIFF transforms declare reflectance conversion.
- Untouched TIFF identity/default values preserve the existing canonical digital
  number convention; TIFF readers cannot reliably distinguish omitted from
  intentionally declared identity metadata.
- For already-scaled reflectance, explicitly pass `CogOptions(scale=1.0)` or
  `--scale 1`. For scaled integers, use the data producer's documented transform,
  for example `scale=0.0001, offset=-0.1` when that is the source convention.
- CLI `--scale` and `--offset` accept a number or comma-separated per-band values.
  All transform values must be finite.

TIFF nodata and internal masks are honored before resampling. Acquisition time
comes from `TIFFTAG_DATETIME`, with the existing `datetime` / `--datetime`
override available for model metadata. `bbox` / `--bbox` uses EPSG:4326 bounds.
Inputs must have north-up projected grids in meters. `crs` assigns a missing CRS;
it does not reproject an unsupported grid. Reprojection remains deferred in
[issue #21](https://github.com/cercanasystems/whirld/issues/21).

## Memory and authentication

`embed_to_disk()` / `embed --stream` reads bounded windows and writes NPY plus
JSON metadata incrementally. Ordinary `embed`, `classify`, and `segment` retain
the selected region in memory. Use a bbox for small eager runs. No remote raster
is staged as a whole local file. Actual transfer size depends on TIFF layout:
use tiled COGs, since a large source strip can require a large decoder buffer.
See the [streaming guide](streaming.md) for cache/batch limits and output guarantees.

Signed URL query parameters are retained for reads and omitted from output
filenames and Whirld diagnostics. Bearer tokens use the existing Python
`stac_token` argument, CLI `--token` (alias `--stac-token`), or
`WHIRLD_STAC_TOKEN`. Credentials are scoped to the source origin and retained
for lazy window reads; redirects strip authorization. Authenticated reads require
GDAL >=3.10. Avoid putting secrets in shell history when constructing commands.

## Reproducible public sample

The official IBM/NASA HLS burn-scar sample is a small **range-readable GeoTIFF**
with row strips, not a tiled COG. This pinned version contains six bands of
already-scaled reflectance. Install the project's Prithvi optional dependencies
and pull `prithvi-burn-scar` first, as described in the
[burn-scar example](../examples/burn-scar.md).

```python
import whirld

url = (
    "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars/"
    "resolve/a3f2c410e45b8ac7417976614528a872f024d831/examples/"
    "subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif"
)
result = whirld.segment(
    url,
    model="prithvi-burn-scar",
    sensor="hls",
    cog=whirld.CogOptions(
        bands=("B02", "B03", "B04", "B8A", "B11", "B12"), scale=1.0
    ),
    device="cpu",
    output="burn-mask.tif",
)
```

On 2026-11-22 this direct URL run produced exactly the same 512 × 512 mask and grid
as the existing STAC-based real-model example: 208989 burned pixels out of 262144
valid pixels. The run took 8.24 seconds on the development CPU with cached weights.
This verifies pipeline equivalence, not prediction accuracy. The upstream sample
and model attribution/licenses are recorded in the linked example guide.

A separate opt-in public pixel check requires no model checkpoint:

```bash
WHIRLD_TEST_COG=1 .venv/bin/python -m pytest tests/integration/io/test_cog_real.py
```

Default integration tests use actual tiled COGs and a controlled HTTP range server.
They check band/radiometry/mask equivalence, bbox byte subsets, scoped credentials,
redirects, API/CLI output, and fresh-process memory as scenes grow from 512 to 4096
pixels per side with fixed chip and batch sizes.
