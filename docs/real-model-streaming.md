# Real Clay streaming validation

This opt-in acceptance check runs the actual Clay v1.5 encoder against public
Sentinel-2 COGs. It compares the public eager and streaming APIs, including model
loading, radiometry, acquisition/location metadata, edge padding and disk output.
It does not measure embedding quality for a downstream task.

## Reproduce

Use the project's Python 3.13.7 `.venv` with the `hf` extra installed. Model setup
is separate from the measured inference workers:

```bash
export WHIRLD_HOME=/tmp/whirld-real-clay
.venv/bin/whirld pull clay-v1.5
.venv/bin/python scripts/validate_clay_streaming.py \
  --output-dir /tmp/whirld-real-clay-validation \
  --batch-sizes 1 2 3 --chip-size 256 --threads 2
```

Choose a new output directory for each run. The checkpoint is approximately
5.16 GB; the validation runner requires an existing installation and does not
download weights. The initial pull may access Hugging Face. Inference accesses
the public STAC item and asset windows over HTTP. No imagery account is required.

The runner checks the installed artifact against the bundled registry's pinned
checkpoint identity and SHA-256 before starting workers. Clay's curated checkpoint
uses trusted PyTorch deserialization; this workflow is for that pinned checkpoint.

The same acceptance check is available through pytest:

```bash
WHIRLD_TEST_CLAY_STREAMING=1 .venv/bin/python -m pytest \
  tests/integration/test_api_streaming_real.py
```

The ordinary suite skips this test. Keep `WHIRLD_HOME` set to the populated model
home; pytest creates an isolated output directory. Hermetic tests exercise the
report comparison and validation failures without weights or network access.

## Fixed inputs and acceptance criteria

- Model: `clay-v1.5`, version `1.5.0`, Hugging Face repository `made-with-clay/Clay`,
  revision `70200ebcccdf67bf2a0cb9984c77ddee26c10ed2`.
- Artifact: `v1.5/clay-v1.5.ckpt`, SHA-256
  `21432069250b9b3f9a65ffd0071c5ad56b793247285ab0604edf7f531d4798d0`.
- Public item: [S2A_30UXD_20230630_1_L2A](https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2A_30UXD_20230630_1_L2A).
- Comparison bbox (longitude/latitude): `[-1.32, 52.49, -1.26, 52.53]`.
- Larger measurement bbox: `[-1.35, 52.47, -1.23, 52.55]`.
- Chip size: 256 pixels, no overlap. Eager batch size 1 is compared with streaming
  batch sizes 1, 2 and 3. Four base-region chips exercise a partial final batch at 3.
- Require finite float32 arrays with 1,024 values per chip and matching chip counts.
  Compare embeddings with **rtol=1e-4, atol=1e-5**, fixed before the experiment;
  compare all output metadata exactly except the per-run timestamp.

The scene was also used for the earlier radiometry check. Its cloudy pixels are
intentional numerical test inputs. These checks establish implementation
agreement, not usefulness of the embeddings for a particular application.

## Interpreting resource results

Each inference call runs in a fresh process with CPU execution and a fixed thread
count. Total elapsed seconds include model loading, remote item/COG reads,
preprocessing, inference and output publication. Throughput is chip count divided
by that elapsed time; it is not isolated encoder throughput. Setup, checkpoint
checksum verification and checkpoint download are outside this timer.

Peak RSS is the process-lifetime high-water mark, including imports and model
loading. It is not steady-state inference memory. Measurements exclude the parent
process's later output comparisons. HTTP caches, OS file caches and network
conditions can affect times across runs; these are observations, not performance
thresholds. The larger-region run keeps chip and batch sizes fixed. Two small
regions alone do not prove an asymptotic memory bound; the reference-model scaling
experiment and bounded-resource tests remain complementary evidence.

CPU validation does not establish CUDA/MPS equivalence or GPU memory usage. The
streaming restrictions in the [streaming guide](streaming.md) still apply, and
[reprojection issue #21](https://github.com/cercanasystems/whirld/issues/21) remains
deferred.

## Recorded results

Measured on 2026-11-21 with Python 3.13.7, macOS ARM64 26.6.2, Torch 2.12.1,
NumPy 2.5.0, Rasterio 1.5.2 and GDAL 3.12.2; CPU with two intra-op threads and
one inter-op thread. The checkpoint matched the pinned size and SHA-256.

All three streamed outputs matched the eager baseline **exactly** (maximum
absolute error and RMSE both 0.0), within the predeclared tolerance. All metadata
except timestamps matched, including ordered chip geometry. The base region
produced four chips; batch size 3 exercised a partial final batch.

| Mode | Region | Batch | Chips | Peak process RSS (MiB) | Elapsed (s) |
|---|---|---:|---:|---:|---:|
| eager | base | 1 | 4 | 2,618.4 | 12.08 |
| stream | base | 1 | 4 | 2,625.5 | 13.96 |
| stream | base | 2 | 4 | 2,626.3 | 23.20 |
| stream | base | 3 | 4 | 2,625.3 | 23.91 |
| stream | large | 1 | 16 | 2,625.8 | 35.31 |

At fixed batch size 1, four times as many chips increased measured peak RSS by
less than 1 MiB. Larger batches were slower in this run; the shared process loading
peak and network-dependent timings do not establish an optimal batch size.

Each successful run preserves its NPY/sidecar outputs and `.measurement.json`;
`report.json` records runtime/checkpoint identity, exact measurements, comparison
errors and whether a partial final batch occurred. The documented results came
from the standalone runner. The gated pytest wrapper invokes that same runner;
it remained skipped in the ordinary regression suite. GPU inference was not run.
