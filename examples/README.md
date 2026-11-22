# Three real-world Whirld examples

Run actual model weights on pinned open imagery and inspect the results in a
browser or QGIS. All three workflows use the public Whirld API, default to CPU
with two Torch threads, prepare their own inputs, and preserve previous results
by requiring a new output directory for each run.

| Workflow | Question | Result |
|---|---|---|
| [Clay similarity](similarity.md) | Which chips resemble this reference landscape? | Ranked WGS84 footprints and an imagery map |
| [RemoteCLIP classification](classification.md) | Which of four landscape prompts best matches each chip? | All prompt scores in GeoJSON and a labeled map |
| [Prithvi burn scar](burn-scar.md) | Where does the burn-scar model predict burned pixels? | Georeferenced binary GeoTIFF and an imagery overlay |

## Setup

From a checkout of this repository, use **Python 3.13.7** and a `.venv`. These
commands install the repository's pinned production/development dependencies,
including all three optional model runtimes. No additional plotting or map-server
dependency is required:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt -e '.[hf,remoteclip,prithvi]'
.venv/bin/whirld pull clay-v1.5
.venv/bin/whirld pull remoteclip
.venv/bin/whirld pull prithvi-burn-scar
```

Confirm `python3.13 --version` is 3.13.7 before creating the environment. If you
already have the project's populated `.venv`, use it. Downloads are approximately
5.16 GB, 605 MB and 1.30 GB respectively, plus the Python dependencies. `pull`
verifies checkpoint checksums. Model files live under `~/.whirld` by default;
set `WHIRLD_HOME` consistently for both pull and execution to choose another home.
The pinned checkpoints are curated first-party artifacts, not arbitrary uploaded
pickle files. Plan for at least 8 GB RAM and sufficient cache/disk space.
Each example also checks the active registry against its bundled entry, verifies
the installed checkpoint again, and records its identity and CPU runtime in
`model.json` before starting the measured workflow.

## Run

Execute each command from the repository root, in a separate process:

```bash
.venv/bin/python -m examples.similarity --output-dir example-runs/similarity
.venv/bin/python -m examples.classification --output-dir example-runs/classification
.venv/bin/python -m examples.burn_scar --output-dir example-runs/burn-scar
```

Open each output directory's `preview.html`. The preview embeds its image and
needs no web server or external basemap. `preview.png` is the underlying display
image; the similarity/classification footprint labels are in the HTML/SVG layer.
The burn-scar PNG includes the colored prediction overlay. Open `.geojson` or
`.tif` results directly in QGIS for full spatial inspection. `example-runs/` is
ignored by Git. Choose a new destination to run again; existing directories are
never cleared automatically.

The first two examples fetch small windows from a pinned July 2024 Sentinel-2
scene near Sacramento. Burn-scar mapping downloads a checksum-verified 6.3 MB
sample from the official Prithvi model repository. No imagery login is needed;
internet access is required for setup and imagery retrieval. Source attribution,
licenses, exact regions and radiometric handling are documented in each guide.

## What the checks establish

The scripts fail on invalid output shapes, nonfinite scores, inconsistent spatial
metadata and other contract violations. Hermetic regression tests cover ranking,
probability and geometry checks, display conversion and HLS reflectance handling:

```bash
.venv/bin/python -m pytest tests/unit/examples
```

The measured real runs are recorded in the individual guides. A successfully
executed model is not a validated land-cover product or fire perimeter. Clay
similarity is not a class label; RemoteCLIP scores depend on the competing prompts;
Prithvi masks need independent evaluation before practical use.

Timing starts after checking the installed model and setting CPU threads, then
includes imagery retrieval, model loading, inference, output validation and preview
creation. Model download time is excluded. RSS is the whole process's high-water
mark, including imports/model loading, not steady-state inference or GPU memory.
Network/cache conditions change timings. The default regions are deliberately
small; ranking and preview generation are not designed for full-scene workloads.

## Recorded environment

Real runs use Python 3.13.7 on macOS ARM64, Torch 2.12.1, NumPy 2.5.0, Rasterio
1.5.2/GDAL 3.12.2, open_clip_torch 3.3.0 and TerraTorch 1.2.8. Exact dependency
pins are in `requirements.txt` and `requirements-dev.txt`. CPU execution is
verified; CUDA/MPS is not part of these example measurements.

Recorded on 2026-11-22, with actual pinned weights:

| Example | Output | Elapsed | Peak process RSS |
|---|---|---:|---:|
| Clay | 16 embeddings, five ranked matches | 43.33 s | 2,596.3 MiB |
| RemoteCLIP | 16 features, four scores each | 10.14 s | 1,589.1 MiB |
| Prithvi | 512 x 512 mask | 8.60 s | 3,101.0 MiB |

These are single-host observations under the timing boundaries above. All previews
were visually inspected and spatial/numerical outputs independently checked.
