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

## Recorded run

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
