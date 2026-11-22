# Map a burn scar with Prithvi

Run the actual Prithvi EO 2.0 burn-scar head on the small HLS image distributed
with the official model. It produces a georeferenced binary mask, a source-imagery
preview and an orange prediction overlay. The fixed input is one 512 × 512 pixel
HLS S30 scene at 30 m resolution (15.36 km square), tile T10SEH, 9 July 2018.

## Run

From the repository root, use Python 3.13.7 and the project `.venv`. Install the
optional model backend and pull the pinned weights once:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt -e '.[prithvi]'
.venv/bin/whirld pull prithvi-burn-scar
.venv/bin/python -m examples.burn_scar --output-dir /tmp/whirld-burn-scar
```

The default is CPU with two threads. A model download is approximately 1.30 GB;
the source imagery is 6.3 MB. Inference uses the installed weights and downloads
only that one image, with no imagery account or full-dataset download. Choose a
new output directory for each run. See the [examples overview](README.md) for
shared environment and model-cache instructions.

Open `/tmp/whirld-burn-scar/preview.html` in a browser. It and `imagery.html` embed their imagery and require no map service. Open `burn-mask.tif` in QGIS alongside the
downloaded source TIFF for geospatial inspection. Each HTML file is self-contained; the PNGs are also saved separately.

## Inputs and units

The source is pinned to the model repository revision
`a3f2c410e45b8ac7417976614528a872f024d831`:

- Filename: `subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif`.
- SHA-256: `13bc592a5e569d837bd8bb3524bb0d2f28418830bcc7b0750e74033078f8b17e`.
- Size: 6,295,168 bytes; six float32 bands; north-up UTM zone 10 grid.

The [official HLS Burn Scars dataset card](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars/blob/1864285e25010d346a842e4f068b1a1d4248ed6d/README.md)
specifies that these samples are already reflectance, in band order **blue, green,
red, narrow NIR, SWIR1, SWIR2** (`B02/B03/B04/B8A/B11/B12`). The example writes
`hls-item.json` to declare band names, **scale 1**, offset 0 and nodata 0. Whirld
then applies its normal model-unit conversion and normalization exactly once.
Passing the source directly as a bare TIFF would invoke the legacy DN convention
and incorrectly apply an additional 0.0001 factor; use the generated STAC item.
The filename establishes the acquisition day, not an exact time; the item records
that day's interval. Prithvi's single-image head does not need a before/after pair.

Preview RGB uses source bands 3/2/1 and a per-channel 2–98% display stretch.
Only predicted burn pixels receive the orange overlay. This display operation
never changes the model inputs.

## Outputs and interpretation

- `burn-mask.tif`: uint8 prediction, 0 unburned / 1 burned / 255 nodata, with the
  source CRS, affine transform and dimensions verified after inference.
- `imagery.html` and `preview.html`, plus their PNGs: original RGB and prediction
  overlay using exactly the same display stretch.
- `hls-item.json` and the original TIFF: explicit input metadata and unchanged
  checksum-verified source image.
- `report.json`: source identity, pixel counts and fraction, grid information,
  CPU timing and process peak RSS.
- `model.json`: verified model revision/checkpoint SHA-256, platform and runtime
  versions, explicit CPU device and thread count.

The mask is a model prediction, **not an authoritative fire perimeter**. The
reported burned fraction is descriptive, not an accuracy score. This official
model demonstration sample is not an independent held-out benchmark; the example
does not establish generalization or estimate performance against ground truth.
Inspect the original image, overlay and source documentation before downstream use.

## Attribution

Imagery: **HLS Burn Scars**, Phillips, Christopher; Roy, Sujit; Ankur, Kumar;
Ramachandran, Rahul (2023), [DOI 10.57967/hf/0956](https://doi.org/10.57967/hf/0956),
[CC BY 4.0](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars).
The dataset combines HLS v1.4 imagery and MTBS-derived burn-scar labels. This
example uses the imagery provided in the official model repository; it does not
redistribute that imagery in Git. Preparation adds a local metadata adapter and
display previews without altering source pixels.

Model: IBM/NASA **Prithvi EO 2.0 300M BurnScars**,
[model card and citation](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars),
Apache 2.0. The pinned source image is available from the
[official model examples](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars/tree/a3f2c410e45b8ac7417976614528a872f024d831/examples).

## Historical execution (0.1.0)

Executed on **2026-11-22** with actual pinned Prithvi weights: Python 3.13.7,
Torch 2.12.1, macOS ARM64 26.6.2, CPU and two threads. The checkpoint SHA-256
matched the bundled registry before inference.

| Check | Recorded result |
|---|---|
| Output dimensions | 512 × 512, uint8 |
| Valid pixels | 262,144; validity agrees with all source bands |
| Predicted burn pixels | 208,989 (79.72% of valid pixels) |
| Classes / nodata | 0 and 1 / 255 |
| Georeferencing | Source CRS and affine transform preserved |
| Elapsed workflow time | 8.60 seconds |
| Peak process RSS | 3,101.0 MiB |

Timing starts after model-installation verification and thread setup, before the
image download, and includes model loading, inference, output checks and preview
rendering. Peak RSS includes imports and initialization; it is not steady-state
model memory or GPU allocation. Download cache/network conditions change timings.
Allow at least the model's documented 8 GB RAM minimum for a comfortable
run; this observed process peak is not a system-memory guarantee.

The source image and orange overlay were visually inspected: the large dark scar
is visible in the imagery and highlighted in the prediction, while the lake and
much of the farmland remain visible without orange. This is a qualitative display
check, not an accuracy assessment. The full mask was independently checked for
allowed values, dimensions, CRS, transform and valid-pixel agreement. A separate
source-unit check confirmed that normalized pixels equal `(reflectance - mean) /
std` without an extra reflectance factor. Eight hermetic tests cover source units,
mask geometry/classes/nodata, and overlay behavior.

## 0.3.0 candidate revalidation

Re-executed the documented entry point on 2026-11-22 with actual pinned weights,
Python 3.13.7 and CPU/two threads. The source SHA-256, six-band reflectance units,
mask values, source-validity agreement and CRS/affine were independently checked.
The **512 × 512 uint8 mask** is bitwise-identical to the earlier saved result:
**262,144 valid pixels, 208,989 predicted burned pixels**, with valid classes 0/1
and nodata 255. Independent normalization again equals `(reflectance - mean) /
std` within 1e-6; no second reflectance scale is applied. The source imagery and
orange-overlay PNGs were visually inspected, and both self-contained HTML images
match their adjacent PNG bytes and 512-square viewports.

The fresh workflow measured **7.27 seconds / 2,697.1 MiB peak process RSS** under
the timing boundaries above. It completed with no blank GDAL `Cannot read`
diagnostics after the scoped header-file policy correction. These repeat checks
establish compatibility and output fidelity, not prediction accuracy or a lower
system-memory requirement.
