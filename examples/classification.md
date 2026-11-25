# Compare landscape prompts with RemoteCLIP

This example scores each real image chip against four competing descriptions:
farmland, forest, water and an urban area. It produces per-prompt GeoJSON scores
and an offline imagery preview colored by each chip's highest-scoring prompt.

## Run

Follow the environment setup in [the examples README](README.md), then run from
the repository root:

```bash
.venv/bin/whirld pull remoteclip
.venv/bin/python -m examples.classification --output-dir /tmp/whirld-classification
```

The real RemoteCLIP checkpoint download is approximately **605 MB** and occurs
before the measured workflow. Defaults use CPU and two Torch intra-op threads.
Inference uses the registry's 224-pixel RGB chips. The output directory must be new.

Open `/tmp/whirld-classification/preview.html`. Gold means the farmland prompt won,
green forest, blue water and pink urban area. Labels show the chip index and winning
score. `scores.geojson` retains all **16 expected chips** and all four scores by
requesting `top_k=16` instead of the public API's default five. The example checks
that the pinned scene produces exactly 16 chips and that every chip is returned;
it fails explicitly if the grid changes instead of silently truncating results.

The prompts are exactly:

- `a satellite image of farmland`
- `a satellite image of a forest`
- `a satellite image of water`
- `a satellite image of an urban area`

The example checks finite scores in [0,1], a per-chip sum close to one, unique chip
IDs and WGS84 closed polygon footprints. These checks establish a valid output
contract; they do not measure classification accuracy.

## Data and outputs

The example uses the same July 22, 2024 Sacramento River region as the
[Clay similarity example](similarity.md): pinned item
[`S2A_10SFH_20240722_0_L2A`](https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2A_10SFH_20240722_0_L2A),
bbox `[-121.57, 38.53, -121.48, 38.60]`. This includes farmland, river and urban
landscapes; individual chips can contain several of them.

Contains modified Copernicus Sentinel data (2024), distributed through Element 84
Earth Search and the [Sentinel-2 COG collection](https://registry.opendata.aws/sentinel-2-l2a-cogs/).
No imagery login is required. Whirld selects red/green/blue assets, honors STAC
radiometry and normalizes to the registry's RemoteCLIP contract. The preview's
separate display stretch does not affect inference.

Outputs are `scores.geojson` for QGIS, `preview.html` with its adjacent image,
`scene.json` for exact input selection, `model.json` for verified checkpoint
identity, and `metrics.json` for CPU resource costs.
Padded edge footprints can extend beyond the image crop.

**Interpretation:** softmax scores are relative to these four prompts. They are not
calibrated land-cover confidence, and changing the prompt set changes the scores.
RemoteCLIP was trained on RGB imagery; Sentinel-2 reflectance is an approximate
input mapping. A winning "forest" prompt is a demonstration of model output, not
a verified forest inventory or a pixel-level map.

## Historical run (0.1.0)

Verified on 2026-11-22 with Python 3.13.7, Torch 2.12.1, macOS ARM64 and
CPU/two threads. The pinned real checkpoint produced **16 chip features × four
prompt scores**. Every score was finite and in [0,1]; each chip's scores summed
to one within 1e-5. Chip IDs were unique and all polygon coordinates were WGS84.

The winning prompts were **urban area on 14 chips, farmland on one, and water on
one**; forest won none. The preview shows a largely urban region with river
corridors and fields. These are model outputs, not independently validated
land-cover labels. The full score vector remains available for every chip in
`scores.geojson`.

Measured elapsed time was **10.14 seconds** and peak process RSS
**1,589.1 MiB**. Elapsed time includes model loading, remote reads, scoring and
preview creation, after model verification/thread/output-directory setup. Initial
checkpoint download and verification are excluded from elapsed time; RSS is the
process-lifetime high-water mark. Network, caches and hardware affect these
observations. GPU inference was not tested.

## Large scenes with bounded memory (0.2.0)

The preview script intentionally loads its small 16-chip result to render it. For
large regions use the disk API directly, with the same prompts and source metadata:

```python
import whirld
from examples.classification import PROMPTS
from examples.common import SENTINEL_SCENE

result = whirld.classify_to_disk(
    SENTINEL_SCENE.item_url, model="remoteclip", query=list(PROMPTS),
    sensor=SENTINEL_SCENE.sensor, bbox=SENTINEL_SCENE.bbox,
    top_k=None, batch_size=4, device="cpu", output="streamed-scores.geojson",
)
print(result.chip_count, result.feature_count)
```

Run from the repository checkout after the setup above. This pinned crop produces
16 matches. Enlarge or remove the bbox for a larger scene. Global ranking uses
temporary disk space; [classification streaming](../docs/classification-streaming.md)
explains the memory and publication contract.

## 0.3.0 candidate revalidation

Re-executed the documented entry point on 2026-11-22 using the pinned checkpoint,
public scene and CPU/two-thread setup above. All **16 features × four scores**
remain finite probabilities, with unique chip IDs, inclusive score bounds and
per-chip sums within 1e-5 of one. Independent checks found exact agreement with
the earlier saved feature collection apart from its updated **0.3.0** version
field. Winners remain 14 urban, one farmland and one water. WGS84 footprints,
embedded PNG bytes and all 16 SVG overlay/label positions were checked; the
imagery PNG was visually inspected.

The fresh run measured **11.43 seconds / 1,594.5 MiB peak process RSS** under the
same timing boundaries. It completed with no blank GDAL `Cannot read` diagnostics
after the scoped header-file policy correction. These are repeated execution
checks, not new accuracy or GPU-performance evidence.

The documented `classify_to_disk` snippet was also executed on the candidate with
`top_k=None` and batch size 4. It retained all 16 chips and its complete GeoJSON
matched the fresh eager example exactly, including scores, ranking, geometry and
version metadata. The API call took 17.66 seconds with two CPU threads; unlike the
full example timings, that call excludes preview rendering and the example's
separate checkpoint preflight. It is an execution check, not a performance
comparison. The final-policy log contained no blank `Cannot read` diagnostics.

## 0.4.0 candidate revalidation

The documented default module command ran on **2026-11-25**, using Python
3.13.7, Whirld 0.4.0 and CPU/two threads on macOS ARM64. The same pinned
RemoteCLIP checkpoint was checksum-verified in the existing cache; the run read
the public Sentinel-2 STAC/COG windows into a new result directory.

All **16 features × four prompt scores**, ranked chip IDs and WGS84 geometry
exactly match the saved 0.3.0 collection; only its Whirld version field changes.
Every score is finite and in [0,1], with maximum row-sum error **8.20 × 10⁻⁸**.
Winners remain **14 urban, one farmland and one water**, with no forest winner.
The preview PNG was visually inspected and is byte-identical to the prior image.
Its self-contained HTML embeds those exact PNG bytes; the 797 × 790 viewport,
16 footprint overlays and visible label bounds were independently checked.

Measured workflow time was **10.18 seconds / 1,588.8 MiB peak process RSS** under
the timing boundaries above, excluding model download/checkpoint verification.
These are observations with cached weights and variable network/host activity,
not performance targets. Exact repeatability on this scene does not validate the
prompt labels or turn relative scores into calibrated land-cover confidence.
