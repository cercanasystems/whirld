# Whirld

**Local-first CLI and Python library for running geospatial foundation models**

Concept: "Like ollama, but for geospatial foundation models"

Whirld owns the *sensor-to-model translation contract*. Give it a GeoTIFF and a
model name; it detects the sensor, selects the right bands **by spectral alias**,
resamples, normalizes, chips the image, runs inference, and writes a georeferenced
output. You write one command.

```bash
whirld pull clay-v1
whirld embed --model clay-v1 my_sentinel2_scene.tif
# → s2_small_embeddings.npy (512-dim vector per chip) + metadata sidecar
```

---

## Status: walking skeleton (offline)

This repository is an early, end-to-end **walking skeleton** of the Whirld MVP
(see [`whirld-prd-mvp.md`](whirld-prd-mvp.md)). It proves the architecture with a
real, fully tested vertical slice — **`pull` + Clay `embed`** — that runs **entirely
offline**.

### What is real and tested here

- **Band-contract translation** (Whirld's core contribution): alias-based band
  selection, resampling, scale + per-band z-score normalization, chipping with
  edge padding, and georeferenced chip bounding boxes.
- **Sensor detection** with the full precedence chain (TIFF tags → band
  descriptions → resolution → `--sensor` override).
- **Registry**: bundled, schema-validated model YAMLs seeded into `~/.whirld`.
- **`pull` contract**: resolve → acquire → **sha256 verify** → `manifest.json`,
  with the documented exit codes.
- **CLI** (`pull`, `list`, `info`, `embed`, `serve`), **public Python API**,
  structured logging, and a local `usage.jsonl` record.
- **CLI** also includes `classify` (zero-shot text query → GeoJSON scores).
- **REST API** (`whirld serve`): `GET /health`, `GET /models`, `POST /embed`
  (multipart upload → `.npy` + chip-metadata header, or JSON), `POST /classify`
  (multipart upload + `query` → GeoJSON), with models warm-loaded and kept
  resident. `POST /segment` exists and returns `501` until the Prithvi backend lands.
- **Real model weights**: `clay-v1.5` (1024-dim embeddings, `hf` extra) and
  `remoteclip` (zero-shot classification, `remoteclip` extra) both download +
  sha256-verify genuine checkpoints and run the real models.
- Unit + integration tests with synthetic Sentinel-2 fixtures; **88 % coverage**.

### Models

| Model | Paradigm | Backend | Network | Output |
|---|---|---|---|---|
| `clay-v1.5` | embeddings | **real Clay encoder** (torch) | 5.16 GB | 1024-dim `.npy` |
| `remoteclip` | classification | **real RemoteCLIP** (open_clip) | 605 MB | GeoJSON scores |
| `prithvi-burn-scar` | segmentation | **real Prithvi** (TerraTorch) | 1.30 GB | mask GeoTIFF |
| `clay-v1` | embeddings | deterministic reference (numpy) | none | 512-dim `.npy` |

All sit behind the same `ModelBackend` interface, so the translation pipeline,
CLI, REST API, and output code are identical regardless of which one runs. The
offline `clay-v1` reference is kept for no-network / fast-test / no-GPU use.

**Fidelity caveats (documented, not silently shipped):**
- `clay-v1.5`: fully metadata-conditioned — per-chip `lat/lon` (reprojected from the
  raster) and scene `time` (from `--datetime` or `TIFFTAG_DATETIME`) are fed to the
  encoder using Clay's exact formulas. When the acquisition datetime is unknown,
  `time` falls back to zeros (Clay's neutral value); lat/lon is always derived.
- `remoteclip`: scores are `logit_scale`-scaled softmax **probabilities** (0–1) —
  multiple `--query` values give true zero-shot multi-class; a single query is
  softmaxed against a neutral `background` prompt. S2-DN→8-bit RGB tone mapping is
  approximate (the model was trained on 8-bit RGB), so absolute scores on
  Sentinel-2 are approximate.

The real Clay encoder is vendored (Apache-2.0, attributed) under
[`models/_vendor/clay_v15/`](src/whirld/models/_vendor/clay_v15/) — see its
`PROVENANCE.md`. (Clay's own `claymodel` package can't be pip-installed on Python
3.13, so the minimal encoder is vendored. RemoteCLIP needs no vendoring —
`open_clip_torch` installs cleanly; its weights come from the public
`chendelong/RemoteCLIP` mirror since `BAAI/RemoteCLIP` is gated.)

**`prithvi-burn-scar` caveats:** single-image binary segmentation (the PRD's
before/after temporal stacking does not match the published heads); the two task
heads are separate checkpoints, so Whirld exposes `prithvi-burn-scar` /
`prithvi-flood` (the latter is a follow-up), with `prithvi-eo-2 --head <h>` accepted
as an alias. Needs the `prithvi` extra (`pip install 'whirld[prithvi]'`).

### Deferred (extension points left in place)

Prithvi `flood` head, STAC input over the **REST** bodies (the CLI/library STAC path
is implemented) and STAC multi-item *search*, remote registry sync (`update`), and
quantized variants.

---

## Install (development)

Requires Python 3.10+ (developed and tested on 3.13).

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -e . -r requirements-dev.txt
```

## Usage

### CLI

```bash
whirld pull clay-v1            # offline reference model (fast, no network)
whirld pull clay-v1.5          # real Clay v1.5 weights (5.16 GB; needs whirld[hf])
whirld pull remoteclip         # real RemoteCLIP (605 MB; needs whirld[remoteclip])
whirld list                    # installed models
whirld info clay-v1.5          # band contract, sensors, license, citation
whirld embed --model clay-v1.5 --datetime 2024-06-01T10:00:00Z scene.tif   # real 1024-dim embeddings
whirld classify --model remoteclip --query "solar farm" scene.tif          # GeoJSON scores
whirld segment --model prithvi-burn-scar hls_scene.tif                     # mask GeoTIFF
whirld embed --model clay-v1 https://earth-search.aws.element84.com/.../item.json \
  --bbox -1.30,52.50,-1.27,52.52                                           # STAC item (range reads)
whirld rm clay-v1              # remove a cached model (rm --all clears everything)
```

**Input** is a local GeoTIFF/COG **or a STAC item URL**. For a STAC item, Whirld
fetches the item JSON, picks only the band assets the model needs, and reads them with
COG **range requests** (`/vsicurl/`). Pass `--bbox MIN_LON,MIN_LAT,MAX_LON,MAX_LAT`
(EPSG:4326) to window the read, `--sensor` if it can't be inferred from the item, and
`--stac-token` (or `WHIRLD_STAC_TOKEN`) for gated endpoints. Works on
`embed`/`classify`/`segment`; the item path needs no extra dependency.

Useful `embed` options: `--output`, `--format {npy,json}`, `--chip-size`,
`--overlap`, `--device {cuda,mps,cpu}`, `--sensor`, `--crs EPSG:…` (assign a CRS to
a CRS-less input), `--batch-size`, `--datetime` (acquisition time for Clay's
metadata), `--no-warnings`. `classify` options: `--query` (required, **repeatable**
for zero-shot multi-class), `--top-k`, `--threshold`, `--output` (GeoJSON; stdout
default), `--crs`, `--no-warnings`. `segment` options: `--model`, `--head`
(`prithvi-eo-2 --head burn-scar`), `--threshold` (0.5 = argmax), `--output`,
`--crs`, `--no-warnings`. Example multi-class:

```bash
whirld classify --model remoteclip --query "solar farm" --query "forest" scene.tif
#   → per-chip softmax probabilities across the queries (sum to 1)
```

### REST API

```bash
whirld serve --port 8765 --models clay-v1     # preload clay-v1; others load on demand

curl localhost:8765/health
curl localhost:8765/models
curl -F file=@scene.tif -F model=clay-v1 localhost:8765/embed -o embeddings.npy
#   → embeddings.npy (octet-stream); chip bboxes in the X-Whirld-Chips-Meta header
curl -F file=@scene.tif -F model=clay-v1 -F format=json localhost:8765/embed
curl -F file=@scene.tif -F model=remoteclip -F query="solar farm" localhost:8765/classify
#   → GeoJSON FeatureCollection of per-chip scores
curl -F file=@hls.tif -F model=prithvi-burn-scar localhost:8765/segment -o mask.tif
#   → image/tiff single-band mask
```

Requires the `serve` extra: `pip install 'whirld[serve]'`.

### Python

```python
import whirld

whirld.pull("clay-v1")
result = whirld.embed("scene.tif", model="clay-v1")
result.embeddings   # np.ndarray, shape (n_chips, embed_dim)
result.chips        # per-chip metadata with CRS bounding boxes

fc = whirld.classify("scene.tif", model="remoteclip", query="solar farm")
fc.feature_collection   # GeoJSON FeatureCollection of per-chip scores
```

`import whirld` is lazy — it does **not** import numpy, rasterio, torch, or open_clip.

---

## Architecture

```
src/whirld/
  config.py           cache/path layout (WHIRLD_HOME)
  errors.py           exception hierarchy → process exit codes (PRD §13)
  logging_setup.py    app log + usage.jsonl
  api.py              orchestration (pull, embed, classify) — one shared pipeline
  registry_data/      bundled registry: model YAMLs + JSON schema
  core/
    registry.py       typed (Pydantic) registry loading + validation
    fetch.py          pull: acquire → sha256 → manifest
    sensor.py         sensor detection precedence
    contract.py       band-contract translation pipeline
    chips.py          tiling + georeferenced chip metadata
    session.py        ModelSession — warm-loaded backend cache (serve)
  models/
    base.py           ModelBackend ABC + device detection + InferenceContext
    clay.py           Clay offline reference backend (clay-v1)
    clay_torch.py     real Clay v1.5 torch backend (clay-v1.5)
    remoteclip.py     real RemoteCLIP classification backend (remoteclip)
    prithvi.py        real Prithvi segmentation backend (TerraTorch)
    loader.py         registry-driven backend factory (dispatch by `backend:` id)
    _vendor/clay_v15/ vendored Clay encoder (Apache-2.0; see PROVENANCE.md)
  io/
    raster.py         rasterio reader (path + in-memory bytes)
    output.py         npy + meta sidecar; GeoJSON; mask GeoTIFF writers
  server/             FastAPI app + routes (health, models, embed, classify, segment)
  cli/                Typer app + commands
```

The cache lives at `~/.whirld` (override with `WHIRLD_HOME`):

```
~/.whirld/
  registry/models/*.yaml
  models/<name>/{weights, manifest.json}
  logs/{whirld.log, usage.jsonl}
```

### GDAL

Whirld relies on rasterio's bundled GDAL. **System GDAL is not required and is
not used** — a deliberate choice that removes the most common source of
geospatial-Python environment pain.

---

## Development

```bash
.venv/bin/python -m pytest --cov=whirld   # tests + coverage (≥80%)
.venv/bin/black src tests                 # format
.venv/bin/ruff check src tests            # lint
```

**Clean-room install test** (fresh machine, base deps only, no system GDAL — proves
the shipped package + the STAC reader work from scratch). Same smoke test
([scripts/clean_room_test.py](scripts/clean_room_test.py)) on two clean OSes:

```bash
# Linux x86 — Docker
docker build -f docker/clean-room.Dockerfile -t whirld-cleanroom .
docker run --rm whirld-cleanroom                        # hermetic: pull + embed GeoTIFF + STAC item
docker run --rm -e WHIRLD_TEST_STAC_URL=<earth-search item> \
  -e WHIRLD_TEST_STAC_BBOX=<min_lon,min_lat,max_lon,max_lat> whirld-cleanroom   # + live /vsicurl/ reads

# macOS ARM — Tart VM (Apple Silicon; brew install cirruslabs/cli/tart hudochenkov/sshpass/sshpass)
scripts/clean_room_macos.sh                             # throwaway macOS VM, same checks
WHIRLD_TEST_STAC_URL=<earth-search item> \
  WHIRLD_TEST_STAC_BBOX=<min_lon,min_lat,max_lon,max_lat> scripts/clean_room_macos.sh   # + live reads
```

### Adding a model to the registry

Backend selection is **registry-driven**: a model's YAML declares a `backend:` id.

- **Reusing an existing backend** (`clay`, `clay-reference`, `remoteclip`,
  `prithvi`) → add a `models/<name>.yaml` conforming to
  [`registry_data/schema/model.schema.json`](src/whirld/registry_data/schema/model.schema.json).
  **No code change.**
- **A genuinely new architecture / loading mechanism** → add a `ModelBackend`
  subclass and one dispatch branch in
  [`models/loader.py`](src/whirld/models/loader.py), then point YAMLs at it.

### Registry governance (security + license)

Because a model's YAML selects the backend, weights, and license, "schema-valid" is
not a sufficient gate. One policy — [`core/governance.py`](src/whirld/core/governance.py),
documented in
[`registry_data/CONTRIBUTING.md`](src/whirld/registry_data/CONTRIBUTING.md) — governs
what may be pulled, executed, and auto-merged:

- **Weights / RCE.** Pickle formats (`.pt`/`.ckpt`/…) run arbitrary code on load.
  Community entries **must** ship `safetensors`; pickle is allowed only for
  maintainer-curated `first-party` entries. A community + pickle entry is **refused at
  pull** (`SecurityError`, exit 9).
- **Backend dispatch is a closed allowlist** — `load_backend` maps an allowlisted id to
  a hardcoded branch and **never imports a module path from YAML** (test-locked).
- **License.** A model's `license` is required and surfaced before download
  (`whirld pull`) and in `whirld info`. Only an OSS allowlist may auto-merge;
  use-restricted licenses (e.g. the OlmoEarth Artifact License) get a terms warning and
  human review. `scripts/check_registry.py` enforces the machine-checkable parts in CI.

---

## License

Apache-2.0. Bundled model entries carry their own upstream licenses (Clay v1: MIT;
Clay v1.5, RemoteCLIP, Prithvi burn-scar: Apache-2.0) — surfaced by `whirld info` and
`whirld pull`. The registry only auto-accepts OSS-licensed models; use-restricted
licenses require review (see Registry governance above).
