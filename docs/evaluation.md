# Small labeled segmentation baseline

The 0.4.0 release candidate includes a fixed four-case **burn-scar quality evaluation**
using `prithvi-burn-scar` and openly licensed HLS Burn Scars imagery/labels. This
is pending review and publication, and is absent from the published 0.3.0
artifacts. It complements execution/parity checks with ground-truth metrics for
one existing task. See [release status](releases.md).

The 0.4.0 candidate reran all four cases offline with verified cached data and
weights. Prediction hashes, independently recomputed counts and split metrics
exactly match the pinned baseline below; model/config/data identities are
unchanged. This reproducibility check does not broaden the sample's quality
claims. Current candidate evidence is summarized in the
[release guide](releases.md#candidate-verification).

## Run

Use the project Python 3.13.7 `.venv` and install the Prithvi runtime. Pull the
curated checkpoint once; the runner verifies both its weights and deployment
configuration against the frozen manifest before inference.
Choose a new output directory for each invocation; existing results are preserved.

```bash
.venv/bin/python -m pip install -r requirements-dev.txt -e '.[prithvi]'
.venv/bin/whirld pull prithvi-burn-scar
.venv/bin/python scripts/evaluate_segmentation.py \
  --output-dir evaluation-runs/validation --cache-dir evaluation-runs/cache \
  --split validation
```

After the configuration is fixed, run the held-out test cases with cached data:

```bash
.venv/bin/python scripts/evaluate_segmentation.py \
  --output-dir evaluation-runs/test --cache-dir evaluation-runs/cache \
  --split test --offline
```

To repeat both fixed splits without downloading data:

```bash
.venv/bin/python scripts/evaluate_segmentation.py \
  --output-dir evaluation-runs/repeat --cache-dir evaluation-runs/cache \
  --split all --offline
```

The initial run downloads approximately **52.5 MB** of checksum-pinned data plus
three small official split lists. All four pairs are verified, including headers
for the unselected split, before evaluating the requested cases. The 1.30 GB
model checkpoint is installed separately. No full-dataset archive is required.
Sources and labels are unchanged. A local STAC adapter declares their already
scaled reflectance, native band order and actual nodata -1000 explicitly.

The protocol uses CPU, two threads, fixed threshold **0.5** and the deployment
configuration's seed **2**. There is no fitting, threshold search or selection by
test performance.

`--offline` requires every pinned asset already in the verified cache. Every cache
hit is rehashed. Choose a **new output directory** for each invocation; existing
outputs are preserved. Downloads use bounded buffers and atomic cache replacement.
Failures/interrupts remove only the new run directory and incomplete downloads.
Keep the cache outside that directory, as above, to retain verified assets after
failed runs. `report.json` is published only after all requested
cases finish.

These cases each contain one native 512-pixel chip; changing the batch budget
does not turn this small suite into a multi-chip batching or memory benchmark.
`--cache-mb` and `--threads` are also configurable; all resolved options are
recorded. A compatible custom manifest can be provided with `--manifest`.

## Data, partitions and exposure

The [data guide](evaluation-data.md) documents CC BY 4.0 attribution, immutable
source/model revisions, every image/label digest, source semantics and selection.
The [packaged manifest](../src/whirld/evaluation/data/hls-burn-scars-mini.json)
freezes two validation cases and two test cases before predictions are examined.

The runner checks exact membership against all three hash-pinned official model
split lists. Selected groups must be absent from fine-tuning training; test groups
must also be absent from official validation. Selected validation/test groups and
footprints must separate. Source URLs must identify the declared scene, and real
image/label headers must match each other and their frozen geographic bounds.

The upstream dataset's historical `validation/` download directory is **not**
the model split: the manifest uses the newer model's explicit split lists. The
checkpoint used official validation loss for selection, so validation scores are
descriptive. Pretraining exposure and neighboring training-tile footprints have
not been exhaustively audited. This is a reproducible small baseline, not an
independent dataset or evidence of broad geographic generalization.

## Metric contract

Labels are 0 unburned, 1 burned and -1 ignored. TIFF masks/nodata also exclude
labels; unexpected unmasked classes fail. Predictions are uint8 0/1 with 255
nodata. Both rasters must have one band and exactly the same dimensions, CRS and
affine on a north-up projected meter grid. Labels are never resized or warped to
make a mismatch pass. Scoring reads fixed windows of at most 256×256 pixels.

On valid labels with valid predictions, positive-class counts define:

| Metric | Formula |
|---|---|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2TP / (2TP + FP + FN) |
| IoU | TP / (TP + FP + FN) |
| Accuracy | (TP + TN) / scored pixels |
| Coverage | scored pixels / valid-label pixels |

Zero denominators produce JSON `null`. Missing predictions over valid labels are
retained separately as `prediction_invalid_positive` and
`prediction_invalid_negative`; they are not ordinary TN/FN counts. Quality ratios
are **conditional on scored pixels**, so always compare them together with
coverage and both missing-prediction counts. Empty label/scoring domains fail.
An implementation that drops difficult pixels cannot be judged by IoU alone.

Split metrics sum integer confusion counts before dividing (micro averages).
Validation and test are never pooled, and per-case metrics remain visible.
Accuracy is secondary because a mostly unburned image can have high accuracy
from an all-background prediction. IoU/F1 and recall/precision expose that failure.

## Outputs and verification

Each run writes four metadata adapters, the selected prediction TIFFs and a
strict JSON report. It includes the complete suite, original manifest SHA-256,
actual checkpoint/config hashes, registry fingerprint, runtime versions, CPU
settings, resolved options, prediction hashes, exact grids and case/split counts.
Cached assets must be standalone: unpinned mask, PAM or overview sidecars are
rejected even when the TIFF checksum itself still matches.
The CPU/host runtime owner restores thread counts, CPU/Python/NumPy RNG states
and Lightning seed environment keys after evaluation. The script runs in its own
process; accelerator execution and accelerator-state isolation are outside this
CPU evaluation contract.

Normal tests use offline raster fixtures and independently calculated confusion
counts. They retain real STAC preprocessing, public streaming inference and output
writing while substituting only the optional model/checkpoint boundaries. Real
model/data runs are opt-in and do not add network or large weight downloads to CI.

### Recorded baseline

Executed on 2026-11-23 using the curated model and configuration, Python 3.13.7,
Torch 2.12.1, TerraTorch 1.2.8, NumPy 2.5.0 and Rasterio 1.5.2 on macOS ARM64.
Settings: CPU, two threads, deployment seed 2, threshold 0.5, batch limit 1 and
8 MiB raster cache. Each case is one native 512-pixel chip. These results are from
unchanged labeled source rasters, not synthetic inputs.

| Split | Cases | IoU | F1 | Precision | Recall | Accuracy | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 2 | 0.745975 | 0.854508 | 0.758183 | 0.978871 | 0.969367 | 1.0 |
| Test | 2 | 0.840014 | 0.913052 | 0.848210 | 0.988627 | 0.994663 | 1.0 |

| Case | Split | IoU |
|---|---|---:|
| T10TFM.2018110.v1 | validation | 0.425889 |
| T11SPD.2020184.v1 | validation | 0.927735 |
| T13REQ.2018156.v1 | test | 0.833232 |
| T14SPF.2019345.v1 | test | 0.853179 |

All valid labels had valid predictions. Twelve pixels in the first validation
case were ignored; the other labels were fully valid. The lower-scoring first
case remains in the frozen suite. No threshold, model or case selection changed
after inspecting predictions. These tiny groups do not establish uncertainty
intervals or replicate the model card's full-dataset benchmark.

The [reference report](baselines/hls-burn-scars-mini.json) contains all counts,
metrics and provenance at full precision. The two separate documented split runs
and the offline `--split all` repeat produced identical case counts, metrics,
grids and prediction-file SHA-256 values. An independent calculation from the
saved TIFFs, without the production scorer, reproduced every confusion count and
metric. Quality evaluation uses the existing inference policy unchanged.
