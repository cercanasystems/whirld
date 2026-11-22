# Bounded-memory classification

Whirld 0.2.0 adds `classify_to_disk()` and `classify --stream` for STAC items and
direct HTTP(S) GeoTIFF/COG inputs. Pull the classification model and install its
extra first; RemoteCLIP needs the `remoteclip` extra and a separate 605 MB checkpoint.
See the [classification example](../examples/classification.md) for setup and
pinned public imagery.

```bash
whirld classify item.json --model remoteclip \
  --query 'a satellite image of farmland' --query 'a satellite image of water' \
  --stream --batch-size 4 --output matches.geojson
```

The default retains the best five matches. Use `--top-k 100` for a different
positive limit, or `--all` to retain every chip passing `--threshold`. `--all` and
`--top-k` are mutually exclusive. Streaming requires `--output`; it never prints
a complete FeatureCollection to stdout. `--batch-size` and `--cache-mb` require
`--stream`; ordinary eager command defaults remain unchanged.

```python
import whirld

result = whirld.classify_to_disk(
    'item.json',
    model='remoteclip',
    query=['a satellite image of farmland', 'a satellite image of water'],
    top_k=None,  # All matches; default is five.
    threshold=0.0,
    batch_size=4,
    cache_mb=8,
    device='cpu',
    output='scores.geojson',
)
print(result.chip_count, result.feature_count, result.output_path)
```

The result is a small typed summary with model, sensor, device and query identity;
it does not retain scores, chip geometry or feature arrays. Ordinary `classify()`
continues returning those arrays in memory. Reading the finished GeoJSON with
`json.load()` later will allocate all its features.

## Numerical and output contract

Both paths use the same normalization, masks, model-native chip size, zero overlap,
padded edge footprints and scoring backend. A single prompt retains RemoteCLIP's
background contrast; multiple prompts retain softmax over the supplied prompts.
The first query drives threshold and ranking. Threshold comparison is inclusive.
Features sort globally by descending primary score; ties retain ascending chip
index, matching eager classification. GeoJSON fields and WGS84 footprints match
the eager output. All-results mode adds no output fields.

Score batches must have exactly one row per chip and one column per prompt, with
finite probabilities in [0, 1]. Invalid model output fails before publication.

## Memory, disk space and failures

One backend stays loaded. The reader prepares at most `batch_size` chips (default
8); `cache_mb` configures the shared raster caches (default 32 MiB). Memory also
depends on model/framework buffers, source decoder blocks, band/query counts and
chip size. Use tiled COGs and lower batch/cache settings for tighter budgets.

A resource-owning SQLite store stages completed features on disk with a 2 MiB
page cache and file-backed sorting. An ordered cursor serializes one feature at
a time to the final GeoJSON staging file, preserving global ranking even with
`top_k=None`. Temporary disk usage grows with qualifying features and includes
the database, sort workspace and output file. Allow several times the expected
GeoJSON size on the output filesystem, plus operating-system temporary sort space.
A small top-k limits output size but does not avoid scanning the scene or staging
qualifying scores. Disk-backed ranking trades I/O for bounded working memory.

The final GeoJSON replaces its destination atomically only after complete
inference, sorting and output flushing. Exceptions and interrupts discard staging
and preserve previous output. Concurrent writers must use distinct destinations;
last replacement wins otherwise. Filesystem durability across power loss is not
a promised transaction.

## Inputs and limits

STAC JSON may be local, a file URL or HTTP(S). Direct rasters must be HTTP(S),
with an explicit sensor and complete band names. All direct COG options, bbox
windows and origin-scoped token rules apply; see [direct COG input](direct-cog.md).
Local raster and REST streaming remain deferred. [Segmentation streaming](segmentation-streaming.md)
is available in the development tree through its separate mask GeoTIFF workflow.
Inputs require north-up projected grids in meters; streamed STAC assets must share
a CRS. Reprojection remains deferred in issue #21.

## Verification

Synthetic integration tests compare eager/stream ranking, scores and geometries
for single/multiple queries, top-k/all/empty results, TIFF masks and edge chips,
STAC/direct COG inputs and authenticated real HTTP ranges. Writer/coordinator
unit tests exercise stable ties, thresholds, invalid batches, incomplete runs,
failures/interrupts and publication cleanup. Fresh subprocesses exercise growing
512/2048/4096-square COG scenes at batch 4 and cache 8 MiB, with bounded warmup and
subsequent RSS growth; fixtures are generated outside measured workers.

A separate writer-only measurement stresses external ranking with fixed batches
and 4,000 / 16,000 / 64,000 features. GeoJSON size grew from 1.55 MB to 24.96 MB,
while peak RSS stayed between 70.2 and 71.2 MiB. Those measurements exclude model
weights and raster processing; the HTTP scene test above covers that reader path.

The pinned real RemoteCLIP comparison uses the public scene in the classification
example, comparing eager output with streaming batches 1 and 3 at predeclared
`rtol=1e-4`, `atol=1e-5`. On the development CPU with cached weights, batches 1 and 3 retained all 16 chips
with identical ranking and geometry. Maximum absolute score errors were
4.95e-6 and 5.37e-6, respectively; those runs took 6.53 and 7.01 seconds including
source/model loading and output publication. Development progress records the
full validation scope.
This establishes pipeline equivalence, not calibrated land-cover accuracy or GPU
performance.
