# Bounded-memory segmentation

Whirld 0.3.0 adds `segment_to_disk()` and `segment --stream` for
installed segmentation models, including Prithvi burn-scar and flood. Use the
published release or the current source checkout. Install the `prithvi` extra
and pull the selected model first. The 0.4.0 candidate additionally supports
[local GeoTIFF streaming](local-streaming.md), pending review and publication.

```bash
whirld segment item.json --model prithvi-flood --sensor sentinel-2-l1c \
  --stream --batch-size 2 --cache-mb 8 --output flood-mask.tif
```

```python
import whirld

result = whirld.segment_to_disk(
    'item.json', model='prithvi-flood', sensor='sentinel-2-l1c',
    batch_size=2, cache_mb=8, threshold=0.5, device='cpu', output='flood-mask.tif',
)
print(result.width, result.height, result.valid_pixel_count,
      result.positive_pixel_count, result.output_path)
```

`DiskSegmentResult` contains paths, counts, grid and model identity; it does not
retain a mask array or full chip list. Ordinary `segment()` still returns the
in-memory mask. The CLI streaming mode requires `--output`; `--batch-size` and
`--cache-mb` apply only to streaming and are rejected by the eager CLI mode.

## Mask and grid contract

The reader uses the model-native chip size and zero overlap, matching current
eager segmentation. Batches carry model-normalized pixels and a separate validity
mask computed from every required band before missing observations are filled.
Normalized zeros cannot identify missing pixels reliably. Native edge chips use
normalized-zero padding; writes crop padding to the exact target width/height.

The output is a tiled LZW single-band uint8 GeoTIFF. It preserves the target CRS,
affine, bbox crop and complete extent. Valid class 0 remains 0; source-invalid
pixels become nodata 255. Valid class indices must fit the declared model classes,
with 255 reserved for nodata. Shape/class/order errors fail before publication.
Positive-pixel counts include valid classes greater than zero and exclude nodata.

Threshold behavior comes from the existing backend: binary Prithvi uses argmax at
0.5 and positive-class probability thresholding otherwise. Streaming introduces no
probability blending, overlap policy or extra post-processing. Cloud masks must
be supplied by source metadata; predicted water or burned area is not a cloud mask.

## Memory and publication

The default batch is four chips, and the raster cache option defaults to 32 MiB.
One backend remains loaded; prepared pixels, validity, model outputs and geometry
are bounded by batch size. Memory also depends on model/framework buffers, chip
and band dimensions, the source decoder's block size and fixed I/O caches. Use
tiled COGs and lower batch/cache settings when memory is constrained.

The writer owns a separate fixed 8 MiB GDAL cache scope and writes cropped windows
to a staging GeoTIFF on the destination filesystem. It never allocates a full
scene mask. Output disk space grows with scene area; allow space for the staged
file and any previous destination until publication finishes. Single-file atomic
replacement occurs after complete inference and dataset close/fsync. Failures and
interrupts before replacement preserve the old output and clean staging. An
interrupt after replacement can leave the complete new output. Concurrent writers
must choose different destinations; filesystem durability after power loss is
not a promised transaction.

## Inputs and model policy

Streaming accepts STAC JSON paths/file URLs/HTTP(S) URLs and direct HTTP(S)
GeoTIFFs. Direct input requires an explicit sensor and unambiguous band identity;
all [COG options](direct-cog.md), bbox and origin-scoped authentication rules apply.
The 0.4.0 candidate also accepts direct local TIFF paths/file URLs with
the [local streaming contract](local-streaming.md): eager local sensor/band/DN
semantics and no bbox or COG overrides. Candidate publication is pending. REST
streaming remains deferred. Inputs must be north-up projected grids in meters;
STAC assets must share a CRS. Reprojection #21 remains deferred.

Flood requires the explicit Sentinel-2 L1C input contract; burn-scar uses HLS.
See [flood support](flood-model.md) for the checkpoint, normalization and supported
metadata-free policy, which differs from optional choices in the upstream demo.

## Verification

Controlled HTTP tests compare eager/stream masks exactly for both task heads at
multiple thresholds, with non-square edge tiles, nodata/internal masks, bbox crops,
source order and scoped credentials. They exercise CLI aliases, invalid options,
interruptions and previous-output preservation. Fresh subprocesses process actual
512/2048/4096-square COGs at fixed native512/batch2/cache8 settings and enforce
bounded RSS warmup and subsequent growth.

A separate writer-only measurement writes 2048²/4096²/8192² masks with native256
chips and batch4. Output grows about 0.65/2.59/10.34 MB while peak process RSS stays
about 69–70 MiB. This isolates output writing and excludes model and raster-reader
memory; the HTTP process test covers those source buffers separately.

Actual CPU comparisons with installed verified checkpoints passed for flood and
burn-scar at stream batches 1 and 2. Masks, CRS, affine, dimensions and validity
match eager exactly. Flood's projected official example is 461 pixels wide by
513 high (two chips), with 233029 valid and 109762 water/flood predictions.
Burn-scar's official 512-square reflectance sample yields 262144 valid and 208989
burn predictions. The stream runs took 3.32/3.21 s for flood and 3.03/2.11 s for
burn-scar with cached weights and two CPU threads. This proves pipeline
execution/equivalence, not prediction accuracy or GPU performance.
