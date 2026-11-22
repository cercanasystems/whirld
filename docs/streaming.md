# Bounded-memory STAC and COG embeddings

Run commands from the repository root after installing Whirld and pulling the
embedding model (for example, `whirld pull clay-v1`).

## CLI and Python usage

Use `--stream` to read STAC asset or direct remote COG windows, infer bounded chip batches, and write
NPY rows plus JSON chip metadata incrementally:

```bash
whirld embed item.json --model clay-v1 --stream --batch-size 8 --output scene.npy
# Optional: --bbox MIN_LON,MIN_LAT,MAX_LON,MAX_LAT --chip-size 256 --overlap 32
```

The equivalent Python API returns a small disk summary:

```python
import whirld

result = whirld.embed_to_disk(
    "item.json", model="clay-v1", output="scene.npy", batch_size=8, device="cpu"
)
print(result.chip_count, result.output_path, result.meta_path)
# Read slices later without loading the entire output:
import numpy
embeddings = numpy.load(result.output_path, mmap_mode="r")
```

## Supported inputs and numerical behavior

`embed_to_disk()` accepts local STAC JSON paths, file URLs, and HTTP(S) item URLs.
Direct HTTP(S) TIFF URLs are also supported with an explicit sensor and complete
band identity; see [direct COG input](direct-cog.md) for `CogOptions` and CLI flags.
It preserves band order, radiometry, masks/nodata, overlap, padded edge footprints,
row-major chip indices, acquisition time, and per-chip location metadata. It loads
one backend for the run. Ordinary `embed()` continues to return in-memory arrays.
Streaming currently supports embedding models and NPY output with the existing
`*_meta.json` schema; inline JSON, REST streaming,
classification, and segmentation streaming are follow-ups.

Streaming assets must share the reference asset's CRS and use north-up metric
grids; different pixel resolutions are supported. Reproject differently projected
or rotated secondary assets before streaming. The ordinary reader retains its
existing reprojection support for those secondary assets.

For aligned assets, both paths set bilinear sampling scale from the actual pixel
sizes. This avoids GDAL's chunk-dependent scale estimates and can correct previous
results for fractionally aligned bbox windows. See GDAL's
[warp sampling options](https://gdal.org/en/stable/doxygen/structGDALWarpOptions.html).

Optional reprojection is tracked in [issue #21](https://github.com/cercanasystems/whirld/issues/21)
as a low-priority enhancement with no near-term implementation planned.

## Memory budget

The default streaming batch is **8 chips**. `--batch-size` bounds both prepared
chips and model calls; lower it to reduce memory. Python's `cache_mb=32` configures
the GDAL block cache and per-band VSI/warp budgets plus a bounded curl cache.
Multiple source/VRT handles can consume a fixed multiple of these budgets.
Working memory also includes the loaded model, framework buffers, band count,
chip size, and source decoder blocks. Use tiled COGs: a large untiled source strip
can require a large decoder buffer even when the requested window is small.
Output size grows with chip count; completed embeddings and metadata are not kept
in RAM. Reading the full sidecar with `json.load()` afterward will allocate its
complete chip list.

## Output publication and failures

Outputs are staged until every batch succeeds. Inference failures and interrupts
clean temporary files and preserve previous outputs. Publishing the two final
files uses per-file replacement with rollback on ordinary errors; the pair is not
crash-atomic, and concurrent runs must use different output destinations.

## Verification and measurements

Reproduce process-level memory measurements (fixtures are generated before the
measured subprocess; uses the offline reference encoder and bundled GDAL):

```bash
.venv/bin/python scripts/benchmark_streaming.py --sizes 2048 4096 8192
```

Measured on macOS, Python 3.13.7, bundled GDAL 3.12.2, CPU reference encoder,
chip size 128 and batch size 4:

| Scene pixels | Chips | Peak process RSS |
|---|---:|---:|
| 2048 × 2048 | 256 | 146.9 MiB |
| 4096 × 4096 | 1,024 | 147.6 MiB |
| 8192 × 8192 | 4,096 | 151.9 MiB |

These are measurements of the reference path, not a RAM estimate for real model
weights or GPU inference. Tests also measure RSS in fresh subprocesses and check
that small bbox requests transfer only a fraction of a controlled remote COG.
Normalized-pixel equivalence uses `rtol=atol=1e-6` in synthetic tests;
reference embeddings use `rtol=1e-5, atol=1e-6`. The live public COG test retains
`rtol=atol=2e-5` for both Clay normalization contracts. An independent bilinear
calculation covers fractional grid alignment. Large real-checkpoint/GPU inference
remains opt-in.

The [real-model validation workflow](real-model-streaming.md) compares actual
Clay v1.5 embeddings across batch sizes and records CPU resource costs in fresh
processes. It complements the reference-model scaling measurements above.

The focused regression suites mirror the source responsibilities under
`tests/unit/io/`, `tests/unit/core/`, and `tests/integration/`. The opt-in public
COG check runs without model weights:

```bash
WHIRLD_TEST_STAC_RADIOMETRY=1 .venv/bin/python -m pytest tests/integration/test_stac_radiometry_real.py
```

[Return to the README](../README.md)
