# Labeled baseline data and provenance

The initial suite, `hls-burn-scars-mini`, contains four paired HLS images and
MTBS-derived burn-scar labels. It measures the existing `prithvi-burn-scar`
checkpoint against labels, separately from eager/stream execution comparisons.
The pinned [suite manifest](../src/whirld/evaluation/data/hls-burn-scars-mini.json)
records every URL, byte size, SHA-256, split, geographic group and WGS84 footprint.
Raster bytes are downloaded when requested and are not committed to this project.
The manifest also fixes the model revision, checkpoint SHA-256 and official
deployment-configuration SHA-256, so an identically named but changed installation
cannot silently supply a different baseline.

## Source and license

[HLS Burn Scars](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars/blob/1864285e25010d346a842e4f068b1a1d4248ed6d/README.md)
is explicitly **CC BY 4.0**, including its imagery and labels. Attribute Phillips,
Christopher; Roy, Sujit; Ankur, Kumar; Ramachandran, Rahul (2023), *HLS Foundation
Burnscars Dataset*, [DOI 10.57967/hf/0956](https://doi.org/10.57967/hf/0956).
It combines HLS v1.4 imagery with Monitoring Trends in Burn Severity labels.
Model weights are separately Apache-2.0; that does not replace the dataset license.

The current dataset packages its full collection in a 2.65 GB archive. This suite
uses individually downloadable files from immutable dataset revision
`0376987623e465f458ae6042e7051dcdccf771c8`, before archive-only packaging.
Eight unchanged source files total **52,447,608 bytes** (about 50.0 MiB).
Each image is 12,586,630 bytes; each label is 525,272 bytes. The manifest verifies
these exact artifacts, rather than treating a mutable branch URL as a data pin.

## Fixed cases and split audit

The dataset's older `training/` and `validation/` storage directories are random
splits. They do **not** determine this suite's roles. The
[pinned Prithvi model card](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars/blob/a3f2c410e45b8ac7417976614528a872f024d831/README.md)
describes a later, non-overlapping train/validation/test partition and publishes
its [split lists](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars/tree/a3f2c410e45b8ac7417976614528a872f024d831/splits).
Those lists define suite membership. All eight chosen source files happen to be
stored under the historical dataset directory named `validation/`.

The four cases were fixed before model predictions. Selection used availability
of individual source files and geographic separation, without selecting on scores
or fitting a threshold. Each role has two different MGRS groups:

| Role | Official split identifier | MGRS group | Acquisition day | Approximate location |
|---|---|---|---|---|
| Validation | `T10TFM.2018110.v1` | `T10TFM` | 2018-04-20 | 41.92° N, 121.51° W |
| Validation | `T11SPD.2020184.v1` | `T11SPD` | 2020-07-02 | 38.82° N, 115.00° W |
| Test | `T13REQ.2018156.v1` | `T13REQ` | 2018-06-05 | 30.79° N, 104.15° W |
| Test | `T14SPF.2019345.v1` | `T14SPF` | 2019-12-11 | 36.72° N, 96.80° W |

Each selected identifier is in its declared official split and absent from the
other two lists. In addition, all four selected **MGRS groups** are absent from
the complete fine-tuning training list. Both test groups are absent from the
complete validation list used for checkpoint selection. Suite validation/test
groups and actual WGS84 rectangles do not overlap. A new date from the same tile
would not satisfy this group-separation rule.

These checks establish the declared grouping and footprint separation. MGRS
groups are a conservative spatial grouping convention, not independent fire-event
identifiers, and group names alone cannot prove that neighboring train-tile
footprints never overlap. No claim is made that the foundation model never saw
these locations during pretraining.

The official lists are plain UTF-8, one identifier per line, such as
`T10TFM.2018110.v1`. For that identifier the source basename is
`subsetted_512x512_HLS.S30.T10TFM.2018110.v1.4_merged.tif`; its paired label ends in
`.4.mask.tif`. Validation must compare full identifiers, not substrings or only
the tile prefix. All three lists are fetched and checksum-checked before accepting
membership:

| Official list | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `train.txt` | 524 | 9432 | `f1dca5c488c7a7451f3aa82f20677c18b3b22a2d96b59d68215494e3dba6f393` |
| `val.txt` | 160 | 2880 | `50a98be7974a2fec73af62b5f54063d938bcfe93cd9802b90eb33eb78f4d7f3e` |
| `test.txt` | 120 | 2160 | `c61ddcb5a434da3f5e63edd7c87549af104c6cc69345876a8ed770844bab1788` |

## Raster and label contract

All selected pairs are 512×512 pixels on matching north-up UTM grids at 30 m.
The legacy TIFF CRS is a UTM WKT based on the WGS84 ellipsoid, without an EPSG
authority identifier. Preserve the actual source CRS and affine; do not assign a
different CRS just to simplify its name. Images have six unlabeled **float64**
bands in the documented order B02, B03, B04, B8A, B11, B12. Values are already
reflectance. A local STAC adapter declares native band names, scale **1**, offset
**0**, and source nodata **-1000**. No additional reflectance scaling is applied.
A bare local TIFF would invoke Whirld's legacy DN convention and is unsuitable
for these unchanged reflectance sources.

Labels are one-band int16: **0 unburned, 1 burned, -1 missing**. The TIFF nodata
tag additionally declares -1000; both -1 and masked/nodata observations are
ignored. Numeric zero is a valid image value and a valid negative label. Label
validity must not be inferred from truthiness. Predictions are Whirld's binary
class indices, with 255 reserved for invalid input pixels. The evaluation report
must retain missing-prediction counts and valid-label coverage so abstentions
cannot silently improve a conditional score.

Image and label pixel grids are already identical. Evaluation rejects changed
shape, CRS, affine or label domains; it does not silently resample categorical
ground truth to match a prediction.

## Interpretation and exposure

The model was fine-tuned on HLS Burn Scars. Its published card says the checkpoint
was selected by lowest validation loss across five runs and all epochs. Therefore
the suite's validation cases are descriptive development checks, even though
their MGRS groups are absent from the fine-tuning training list. Test groups avoid
both those training and validation lists, but the dataset and published test
results have already been used in research. This is a reproducible small baseline,
not a new independent dataset or a blind generalization study.

Four positive-centered burn-scar chips cannot estimate performance across global
imagery, unburned landscapes or new sensors. Report per-case and split-level
positive-class metrics and counts; do not compare their values directly with the
model card's full-test benchmark. Keep the operating threshold fixed before the
run and use validation, rather than test scores, for any future tuning.
