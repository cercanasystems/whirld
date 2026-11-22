# Find similar landscapes with Clay

This example asks: **which other image chips look most similar to one reference
chip in Clay's embedding space?** It streams real 1,024-dimensional Clay v1.5
embeddings from a small Sacramento River scene, ranks the other chips by cosine
similarity, and exports ranked footprints for QGIS and an offline HTML preview.

## Run

Follow the environment setup in [the examples README](README.md), then run from
the repository root:

```bash
.venv/bin/whirld pull clay-v1.5
.venv/bin/python -m examples.similarity --output-dir /tmp/whirld-similarity
```

The checkpoint download is approximately **5.16 GB**, separate from the example's
measured workflow. The example checks that the real model is already installed;
it never silently substitutes the offline reference encoder. Defaults use CPU,
two Torch intra-op threads, 256-pixel chips and a streaming batch size of one.
Each run needs a new output directory.

Open `/tmp/whirld-similarity/preview.html` in a browser. The white outline identifies
reference chip **0**; gold outlines identify up to five ranked matches with their
chip IDs and scores. To choose another chip shown in the preview, run:

```bash
.venv/bin/python -m examples.similarity \
  --output-dir /tmp/whirld-similarity-reference5 --reference 5 --top-k 5
```

The query is excluded from its own results. Exact score ties are ordered by chip
index, and zero/nonfinite vectors are rejected rather than assigned misleading
scores. The small example maps the completed embedding file for ranking; the
underlying inference itself streams. This ranking step is not a general-purpose
large-scale vector index.

## Data and outputs

Input is the pinned public Sentinel-2 L2A item
[`S2A_10SFH_20240722_0_L2A`](https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2A_10SFH_20240722_0_L2A),
acquired July 22, 2024. The longitude/latitude bbox
`[-121.57, 38.53, -121.48, 38.60]` includes the Sacramento River, farmland and West
Sacramento. The item reports very low scene-wide cloud cover; that metadata is not
a pixel-level cloud mask.

Contains modified Copernicus Sentinel data (2024), distributed through Element 84
Earth Search and the [Sentinel-2 Cloud-Optimized GeoTIFF collection](https://registry.opendata.aws/sentinel-2-l2a-cogs/).
No imagery login is required. Whirld reads the selected COG windows, applies each
asset's scale/offset/nodata and the model's ten-band contract, and carries acquisition
time and chip location into Clay. Display RGB is stretched separately for viewing;
that display stretch is not applied to model inputs.

Outputs:

- `embeddings.npy` and `embeddings_meta.json`: real embedding rows and ordered,
  georeferenced chip metadata.
- `matches.geojson`: WGS84 polygons, rank, chip index and cosine score. Open in QGIS.
- `preview.html` and its adjacent image: imagery and ranked overlays.
- `scene.json`, `model.json` and `metrics.json`: exact input selection, verified
  checkpoint identity and measured CPU resources.

Chip footprints include padded edge cells and may extend beyond the requested
crop. Cosine similarity is not a probability or a land-cover label. Similar scores
do not establish that sites share a class, condition or downstream property.

## Recorded run

Verified on 2026-11-22 with Python 3.13.7, Torch 2.12.1, macOS ARM64 and
CPU/two threads: **16 chips × 1,024 float32 values**, all finite. With reference
chip 0, the five ranked chips were **5, 2, 4, 9, 8**, with cosine scores
**0.9681, 0.9500, 0.9438, 0.9397, 0.9316**. The query was absent from matches,
and output polygons used WGS84 coordinates.

Measured elapsed time was **43.33 seconds** and peak process RSS **2,596.3 MiB**.
Elapsed time includes model loading, remote imagery reads, ranking and preview
creation, after checkpoint verification/thread/output-directory setup. RSS includes
the process lifetime, including imports and loading. Initial model download and
checkpoint verification are excluded from elapsed time.
These are observations on one CPU host; network/cache/hardware conditions change
runtime and small numerical differences can change close rankings. GPU execution
was not tested.
