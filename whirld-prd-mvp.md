# Whirld MVP — Product Requirements Document

**Version:** 0.2
**Status:** Decisions Incorporated
**Last Updated:** June 2026

---

## 1. Overview

Whirld is a local-first CLI tool and Python library for running geospatial foundation models. It is the geospatial equivalent of Ollama: it collapses the gap between "a model exists" and "the model runs on my machine against my data."

The name is a portmanteau of *whirl* (the potter's wheel that shapes clay) and *world* (the subject of every model Whirld runs). The pun extends to the tool: Whirld fires Clay.

### 1.1 The Problem

Geospatial foundation models exist. Clay, Prithvi, RemoteCLIP — these are real, open-source, high-quality models. But running any of them requires:

- Resolving fragile conda environments and GDAL version conflicts
- Manually reading source code to understand expected band ordering, normalization parameters, and chip sizes
- Writing bespoke preprocessing pipelines for every model, every sensor combination
- Downloading and managing weights with no cache, no verification, no versioning

The result is that a practitioner who wants to embed a Sentinel-2 scene with Clay spends most of their time fighting infrastructure, not doing science.

### 1.2 The Solution

Whirld owns the *sensor-to-model translation contract*. Given a GeoTIFF or a STAC item URL and a model name, Whirld automatically detects the source sensor, selects the right bands, resamples to the correct resolution, applies the model's normalization, chips the image, runs inference, and writes a georeferenced output. The user writes one command.

### 1.3 The Five-Minute Test

The MVP is not done until this works end-to-end in under five minutes on a fresh machine:

```bash
pip install whirld
whirld pull clay-v1
whirld embed --model clay-v1 my_sentinel2_scene.tif
# → embeddings.npy written, 512-dim vector per chip
```

Everything in this document serves that moment.

---

## 2. Goals and Non-Goals

### 2.1 Goals

- Enable local, offline inference against geospatial foundation models with a single command
- Abstract the sensor/band translation problem completely from the user
- Provide a stable, versioned model registry independent of upstream model repos
- Support both CLI and Python library usage
- Expose a local REST API for integration with notebooks, QGIS, and pipelines
- Run on developer hardware: MacBook (Apple Silicon), Linux workstation, cloud VM

### 2.2 Non-Goals for MVP

The following are explicitly out of scope and should be declined if requested during MVP development:

- Model fine-tuning or training
- Dataset management
- Graphical user interface
- Cloud deployment or managed inference
- Streaming tile outputs
- Multi-model comparison tooling
- Model evaluation / benchmarking
- Authentication or multi-user support
- Windows support (target for v1.1)

---

## 3. Users

### 3.1 Primary: Geospatial ML Practitioner

Works in Python. Runs experiments in Jupyter. Familiar with rasterio, numpy, PyTorch. Wants to generate Clay embeddings for a similarity search index or fine-tune Prithvi on a custom flood dataset. Currently spends 30-60 minutes per project setting up the inference environment.

**Success looks like:** `pip install whirld`, one pull, one command. Embeddings in a numpy array ready for downstream work.

### 3.2 Secondary: Remote Sensing Scientist

Domain expert, moderate Python. Knows satellite data intimately but is not an ML engineer. Wants to run Prithvi's burn scar detection on post-fire Sentinel-2 imagery without writing PyTorch code.

**Success looks like:** A GeoTIFF output they can open in QGIS immediately after running one command.

### 3.3 Tertiary: GIS Analyst / Application Builder

Less Python-fluent. May be calling Whirld from a QGIS plugin or a Node.js web app via the REST API. Cares about the JSON output format, not about model internals.

**Success looks like:** `whirld serve` running in the background, HTTP calls returning clean JSON or GeoTIFF with no setup beyond the initial pull.

---

## 4. Supported Models (MVP)

Three models at launch, chosen to cover distinct capability paradigms:

### 4.1 Clay v1 (`clay-v1`)

| Property | Value |
|---|---|
| Source | `made-with-clay/clay-v1` on HF Hub (pinned commit) |
| Capability | Multi-sensor embeddings |
| Sensors | Sentinel-2, Landsat-8/9, NAIP |
| Output | 512-dimensional embedding vector per chip |
| License | MIT |
| Primary use cases | Similarity search, clustering, fine-tuning downstream classifiers |

### 4.2 Prithvi EO 2.0 (`prithvi-eo-2`)

| Property | Value |
|---|---|
| Source | `ibm-nasa-geospatial/Prithvi-EO-2.0` on HF Hub (pinned commit) |
| Capability | Temporal segmentation with task-specific heads |
| Sensors | Harmonized Landsat Sentinel-2 (HLS) |
| Output | Per-pixel classification mask (GeoTIFF) |
| License | Apache 2.0 |
| Primary use cases | Flood mapping, burn scar detection |

**Task heads in MVP:**

- `flood` — binary flood/no-flood segmentation. Input: two temporally adjacent HLS scenes (before and after event).
- `burn-scar` — binary burned area / unburned segmentation. Input: two temporally adjacent HLS scenes (before and after fire).

Additional heads (crop type, land cover change, etc.) are deferred to v1.1.

### 4.3 RemoteCLIP (`remoteclip`)

| Property | Value |
|---|---|
| Source | `BAAI/RemoteCLIP` on HF Hub (pinned commit) |
| Capability | Zero-shot text-driven classification |
| Sensors | RGB optical (Sentinel-2 RGB, aerial) |
| Output | Class scores + top-k matches (JSON) |
| License | Apache 2.0 |
| Primary use cases | Zero-shot object/scene detection, text-guided search |

---

## 5. CLI Specification

### 5.1 Command Surface

```
whirld pull <model>              Download and cache a model
whirld list                      List installed models
whirld info <model>              Show model metadata and band contract
whirld rm <model>                Remove a cached model
whirld embed <input>             Generate embeddings
whirld segment <input>           Run segmentation/dense prediction
whirld classify <input>          Run classification
whirld serve                     Start local REST API server
whirld update                    Refresh the model registry
```

### 5.2 Command: `whirld pull`

```
whirld pull <model-name>
```

Behavior:
1. Fetch registry YAML for `<model-name>` from `github.com/whirld/registry`
2. Resolve the pinned HF Hub commit hash
3. Download weights to `~/.whirld/models/<model-name>/`
4. Verify sha256 checksum; abort and delete on mismatch
5. Write a local `manifest.json` recording model name, version, download timestamp, and checksum

Options:
- `--quantize int8` — download and convert to int8 quantized variant (CPU-friendly)
- `--force` — re-download even if already cached

Exit codes: 0 success, 1 network error, 2 checksum mismatch, 3 model not found in registry.

Example output:
```
Pulling clay-v1...
  Registry:  github.com/whirld/registry @ a3f8c2d
  Source:    made-with-clay/clay-v1 @ 7b19e04 (HF Hub)
  Size:      847 MB
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 847/847 MB
  Verifying sha256... ✓
  Saved to ~/.whirld/models/clay-v1/
```

### 5.3 Command: `whirld list`

Tabular output of all installed models:

```
NAME            VERSION    SIZE      HARDWARE    MODIFIED
clay-v1         1.0.0      847 MB    MPS         2 days ago
prithvi-eo-2    2.0.0      1.2 GB    MPS         2 days ago
remoteclip      1.0.0      412 MB    CPU         1 hour ago
```

### 5.4 Command: `whirld info`

```
whirld info clay-v1
```

Displays the full band contract, hardware requirements, supported sensors, output spec, license, and citation. Human-readable by default; `--json` flag for machine-readable output.

### 5.5 Command: `whirld embed`

```
whirld embed [OPTIONS] <input>
```

Arguments:
- `<input>` — Local GeoTIFF path or STAC item URL

Options:

| Option | Default | Description |
|---|---|---|
| `--model` | required | Model name (e.g. `clay-v1`) |
| `--output` | `<input_stem>_embeddings.npy` | Output file path |
| `--format` | `npy` | Output format: `npy`, `json` |
| `--chip-size` | model default | Override chip size in pixels |
| `--overlap` | `0` | Chip overlap in pixels |
| `--device` | auto | `cuda`, `mps`, `cpu` |
| `--batch-size` | auto | Inference batch size |

Behavior:
1. Load model (from cache; error if not pulled)
2. Detect input sensor from file metadata
3. Validate sensor is supported by model; raise clear error if not
4. Apply band contract: select bands, resample, normalize
5. Chip image into tiles
6. Run inference in batches
7. Write output with chip coordinates embedded as metadata

Output format for `npy`: shape `(n_chips, embed_dim)` with a sidecar `<stem>_embeddings_meta.json` containing chip bounding boxes in the source CRS.

### 5.6 Command: `whirld segment`

```
whirld segment [OPTIONS] <input> [<input2> ...]
```

Arguments:
- `<input>` — One or more GeoTIFF paths or STAC item URLs. Multiple inputs support temporal stacking (required by Prithvi).

Options:

| Option | Default | Description |
|---|---|---|
| `--model` | required | Model name |
| `--head` | model default | Task head (e.g. `flood`, `burn-scar`) |
| `--output` | `<input_stem>_<head>.tif` | Output GeoTIFF path |
| `--device` | auto | Hardware device |
| `--threshold` | `0.5` | Binary mask threshold |

Output: Single-band GeoTIFF with the same CRS, extent, and resolution as the primary input. Pixel values are class indices (categorical) or probabilities (float32) depending on the head. Compatible with drag-and-drop into QGIS and ArcGIS.

### 5.7 Command: `whirld classify`

```
whirld classify [OPTIONS] <input>
```

Options:

| Option | Default | Description |
|---|---|---|
| `--model` | required | Model name (`remoteclip` for MVP) |
| `--query` | required | Text description (e.g. `"solar farm"`) |
| `--output` | stdout | Output GeoJSON path |
| `--top-k` | `5` | Number of top matches to return |
| `--threshold` | `0.0` | Minimum score threshold |

Output: GeoJSON FeatureCollection. Each Feature is a chip with its bounding box as geometry and classification scores as properties.

### 5.8 Command: `whirld serve`

```
whirld serve [OPTIONS]
```

Options:

| Option | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8765` | Port |
| `--models` | all installed | Comma-separated list of models to load at startup |
| `--device` | auto | Hardware device |

Starts a FastAPI server. Models specified at startup are loaded into memory immediately; all other installed models are loaded on first request and remain warm. Logs each request with model name, input size, inference time, and device used.

### 5.9 Command: `whirld update`

Pulls the latest registry YAMLs from `github.com/whirld/registry`. Does not download any model weights. Prints a summary of new models and updated versions available.

---

## 6. REST API Specification

Base URL: `http://localhost:8765`

All endpoints accept `multipart/form-data` (file upload) or `application/json` (STAC URL). All responses are JSON unless the output is a raster, in which case the response is `image/tiff`.

### 6.1 `GET /health`

Returns server status, loaded models, and hardware info.

```json
{
  "status": "ok",
  "device": "mps",
  "models_loaded": ["clay-v1", "prithvi-eo-2"],
  "version": "0.1.0"
}
```

### 6.2 `GET /models`

Returns list of installed models with metadata.

### 6.3 `POST /embed`

```json
{
  "model": "clay-v1",
  "input": "https://earth-search.aws.element84.com/v1/...",
  "chip_size": 256,
  "format": "npy"
}
```

Response: `application/octet-stream` (npy binary) with `X-Whirld-Chips-Meta` header containing chip bounding boxes as base64-encoded JSON.

File upload variant: `multipart/form-data` with `file` field containing GeoTIFF bytes.

### 6.4 `POST /segment`

```json
{
  "model": "prithvi-eo-2",
  "head": "flood",
  "inputs": [
    "https://stac-url-to-before-scene",
    "https://stac-url-to-after-scene"
  ],
  "threshold": 0.5
}
```

Response: `image/tiff` — single-band GeoTIFF mask.

### 6.5 `POST /classify`

```json
{
  "model": "remoteclip",
  "input": "https://stac-url-or-file",
  "query": "solar farm",
  "top_k": 5
}
```

Response:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Polygon", "coordinates": [...] },
      "properties": {
        "score": 0.847,
        "query": "solar farm",
        "chip_index": 12
      }
    }
  ]
}
```

---

## 7. The Band Contract System

This is Whirld's core technical contribution. Every model in the registry declares a band contract:

```yaml
band_contract:
  sensors:
    sentinel-2-l2a:
      bands: [B02, B03, B04, B08, B11, B12]
      aliases: [blue, green, red, nir, swir16, swir22]
      native_resolution_m: 10
    landsat-8-l2:
      bands: [SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7]
      aliases: [blue, green, red, nir, swir16, swir22]
      native_resolution_m: 30
  target_resolution_m: 10
  chip_size_px: 256
  normalization:
    type: per_band_zscore
    mean: [0.0379, 0.0840, 0.0924, 0.1234, 0.1537, 0.1076]
    std:  [0.0233, 0.0308, 0.0417, 0.0566, 0.0730, 0.0583]
    scale: 0.0001                # DN to reflectance factor
  nodata_fill: 0.0
```

### 7.1 Sensor Detection

Whirld detects sensor from GeoTIFF metadata in order of precedence:

1. `TIFFTAG_IMAGEDESCRIPTION` or `TIFFTAG_SOFTWARE` containing sensor name
2. Band count + band description strings matching known patterns
3. Spatial resolution matching known sensor resolution
4. Explicit `--sensor` flag (user override, always wins)

If sensor cannot be determined, Whirld raises a descriptive error with a suggestion to use `--sensor`.

### 7.2 Translation Pipeline

For every inference request:

1. **Detect** source sensor
2. **Validate** sensor is in model's band contract; fail fast with a clear message if not
3. **Select** the required bands by alias (not by band index — indexes are unreliable across products)
4. **Resample** to target resolution using bilinear interpolation for continuous data, nearest-neighbor for categorical
5. **Apply scale factor** (DN → reflectance where applicable)
6. **Normalize** using model's per-band mean/std
7. **Chip** into non-overlapping (or overlapping) tiles of the required size
8. **Pad** edge chips to full size using nodata fill value
9. **Pass to inference**

Step 3 is the most important: Whirld selects by *spectral alias* (blue, green, red, nir, swir16, swir22), not by band number. A Sentinel-2 scene where band 1 is coastal aerosol and a scene where band 1 is blue will both produce the correct 6-band input for Clay, because Whirld looks for "blue" not "band 1."

---

## 8. Registry Architecture

### 8.1 Repository Structure

```
github.com/whirld/registry/
  models/
    clay-v1.yaml
    prithvi-eo-2.yaml
    remoteclip.yaml
  schema/
    model.schema.json         # JSON Schema for model YAMLs
  CONTRIBUTING.md             # How to add a model
```

### 8.2 Full Model YAML Schema

```yaml
# Metadata
name: clay-v1                  # machine-readable identifier
display_name: Clay v1.0        # human-readable
description: >
  Multi-sensor geospatial foundation model producing 512-dim
  embeddings from optical satellite imagery.
version: 1.0.0
category: embedding            # embedding | segmentation | classification
tags: [multi-sensor, sentinel-2, landsat, embeddings]

# Source
source:
  type: huggingface
  repo: made-with-clay/clay-v1
  revision: a3f8c2d9e8f1b4c7  # pinned commit SHA, never HEAD
  files:
    - clay-v1.ckpt

# Distribution (Phase 2: will add whirld_cdn url)
distribution:
  sha256: 8f3a9c2d1e4b7f0a...
  size_bytes: 888274944

# Band contract (see Section 7)
band_contract:
  ...

# Output specification
output:
  type: embedding
  shape: [n_chips, 512]
  dtype: float32
  format: npy

# Hardware
hardware:
  min_ram_gb: 4
  recommended_vram_gb: 6
  quantized_variants:
    int8:
      size_bytes: 234567890
      sha256: 9a1b2c3d...

# Provenance
license: MIT
license_url: https://github.com/...
citation: >
  Clay Foundation (2024). Clay: A foundation model for Earth.
  https://github.com/Clay-foundation/model
authors:
  - Clay Foundation
source_url: https://clay.foundation

# Compatibility
whirld_min_version: 0.1.0
```

### 8.3 Registry Refresh

`whirld update` clones or pulls the registry repo into `~/.whirld/registry/`. Whirld checks for registry updates at most once per 24 hours automatically; subsequent commands within the 24-hour window use the local cache. The `--no-update` flag suppresses all network calls, enabling fully air-gapped operation once models are pulled.

---

## 9. Input Handling

### 9.1 Local GeoTIFF

Any valid GeoTIFF with a defined CRS. Multi-band TIFFs are the primary input format. Cloud-Optimized GeoTIFF (COG) is supported. GeoTIFFs without a CRS are rejected with a clear error suggesting the `--crs` override flag.

### 9.2 STAC Item URL

Whirld accepts any URL pointing to a STAC Item (JSON). On receiving a STAC URL:

1. Fetch the Item JSON (with auth token if provided)
2. Read the band contract for the requested model
3. Identify only the asset hrefs needed for the required bands
4. Fetch only those assets (range requests for COGs where possible, with auth token if provided)
5. Assemble into an in-memory rasterio dataset
6. Proceed as local GeoTIFF

This means `whirld embed --model clay-v1 https://earth-search.aws.element84.com/v1/...` downloads approximately 200–400 MB (the specific bands needed) rather than the full multi-GB scene.

**Authentication.** The `--stac-token` option passes a bearer token for gated STAC endpoints. Required for Microsoft Planetary Computer collections and any other token-protected API.

```bash
whirld embed --model clay-v1 \
  --stac-token $PC_SDK_SUBSCRIPTION_KEY \
  "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/..."
```

The token is passed as an `Authorization: Bearer <token>` header on all asset fetch requests within that invocation. It is never written to disk, never logged, and never included in `usage.jsonl`.

For environments where passing tokens on the command line is undesirable, the token may also be set via the `WHIRLD_STAC_TOKEN` environment variable.

Supported STAC APIs validated in MVP: Element84 Earth Search (public), Microsoft Planetary Computer (token). Others work if the STAC item follows the specification; no explicit allowlist is enforced.

---

## 10. Output Handling

### 10.1 Embeddings

Primary format: NumPy `.npy` binary. Shape: `(n_chips, embed_dim)`.

Sidecar file: `<stem>_meta.json`:

```json
{
  "model": "clay-v1",
  "model_version": "1.0.0",
  "whirld_version": "0.1.0",
  "timestamp": "2026-06-25T14:32:00Z",
  "crs": "EPSG:32630",
  "embed_dim": 512,
  "chip_size_px": 256,
  "resolution_m": 10.0,
  "chips": [
    {
      "index": 0,
      "bbox": [320000, 5820000, 322560, 5822560],
      "row": 0,
      "col": 0
    }
  ]
}
```

Optional JSON format (`--format json`): the full array serialized to JSON. Useful for small outputs; not recommended for large scenes.

### 10.2 Segmentation Masks

Single-band GeoTIFF. CRS, extent, and resolution match the primary input. Float32 for probability outputs; UInt8 for binary masks. GDAL `COMPRESS=LZW` applied by default to reduce file size.

Nodata value: -9999 for float32, 255 for UInt8. Set in TIFF metadata.

### 10.3 Classification

GeoJSON FeatureCollection. One Feature per chip. Geometry is the chip's footprint polygon in the input CRS. Properties include query, score, model, and chip index. Scores are normalized 0–1.

---

## 11. Cache and Storage

Default cache directory: `~/.whirld/`. Overridable via `WHIRLD_HOME` environment variable.

```
~/.whirld/
  registry/              # pulled from github.com/whirld/registry
    models/
      clay-v1.yaml
      ...
    last_updated         # timestamp of last registry pull
  models/                # downloaded weights
    clay-v1/
      clay-v1.ckpt
      manifest.json      # name, version, sha256, download timestamp
    clay-v1-int8/
      clay-v1-int8.ckpt
      manifest.json
  logs/
    whirld.log           # application log, rotating 10 MB, 3 files
    usage.jsonl          # structured local usage records, rotating 1 MB, 2 files
```

`whirld rm <model>` deletes the model directory and its manifest. `whirld rm --all` clears all cached models but preserves the registry.

There is no automatic cache eviction. Disk usage is the user's responsibility. `whirld list` shows per-model disk usage.

---

## 12. Hardware and Device Handling

### 12.1 Auto-Detection

At startup, Whirld detects available hardware in order:

1. NVIDIA CUDA (via `torch.cuda.is_available()`)
2. Apple MPS (via `torch.backends.mps.is_available()`)
3. CPU fallback

The selected device is logged at INFO level on every inference run.

### 12.2 CPU Inference

CPU inference is fully supported and must not be artificially blocked. When running on CPU with a full-precision model:

- A warning is printed with estimated runtime based on image size
- The `--quantize int8` flag is suggested if not already using a quantized variant
- The warning is suppressible with `--no-warnings`

### 12.3 Quantized Variants

Where available, int8 quantized variants are listed in the registry alongside full-precision weights. They have separate sha256 checksums and are stored separately in cache. The user opts in via `whirld pull --quantize int8` or at inference time via `--quantize int8` (triggers re-pull if the quantized variant is not cached).

---

## 13. Error Handling

Whirld uses exit codes consistently:

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | General error |
| 2 | Model not found in registry |
| 3 | Model not installed (needs `whirld pull`) |
| 4 | Unsupported sensor for model |
| 5 | Network error (download failed) |
| 6 | Checksum mismatch |
| 7 | Invalid input file |
| 8 | Insufficient memory |

All error messages include:
- What went wrong (specific, not generic)
- Why it went wrong (if determinable)
- What to do about it

Example of a good error:
```
Error: Sensor 'SPOT-6' is not supported by clay-v1.
       clay-v1 supports: sentinel-2, landsat-8, landsat-9, naip
       If your file is one of these sensors with non-standard metadata,
       use --sensor to specify it explicitly.
```

Example of a bad error (do not do this):
```
Error: Band contract validation failed
```

---

## 14. Python Library Interface

Whirld is also importable as a Python library. The public API mirrors the CLI surface:

```python
import whirld

# Pull a model programmatically
whirld.pull("clay-v1")

# Embed a local file
result = whirld.embed("scene.tif", model="clay-v1")
# result.embeddings: np.ndarray, shape (n_chips, 512)
# result.chips: list of chip metadata with bounding boxes

# Embed a STAC item
result = whirld.embed(
    "https://earth-search.aws.element84.com/v1/...",
    model="clay-v1"
)

# Segment
mask = whirld.segment(["before.tif", "after.tif"],
                       model="prithvi-eo-2",
                       head="flood")
# mask: rasterio MemoryFile containing the output GeoTIFF

# Classify
results = whirld.classify("scene.tif",
                           model="remoteclip",
                           query="solar farm",
                           top_k=5)
# results: GeoJSON FeatureCollection as dict

# Context manager for model persistence across calls (avoids reload)
with whirld.Session(models=["clay-v1"]) as session:
    r1 = session.embed("scene1.tif", model="clay-v1")
    r2 = session.embed("scene2.tif", model="clay-v1")
```

Lazy imports are used internally; `import whirld` does not import torch.

---

## 15. Installation

### 15.1 Standard Install

```bash
pip install whirld
```

Installs Whirld with CPU-only PyTorch. Works on macOS, Linux.

### 15.2 With CUDA Support

```bash
pip install whirld[cuda]
```

### 15.3 Recommended

```bash
uv add whirld        # uv handles dependency resolution significantly faster
```

### 15.4 Python Version

Requires Python 3.10 or higher. Tested on 3.10, 3.11, 3.12.

### 15.5 GDAL

Whirld uses rasterio binary wheels which bundle a known-good GDAL version. **System GDAL is not required and will not be used.** This is a deliberate design decision that eliminates the most common source of environment pain in geospatial Python.

---

## 16. Logging and Local Usage Records

### 16.1 Application Log

Whirld logs to `~/.whirld/logs/whirld.log` at DEBUG level and to stderr at INFO level by default. Log level is controllable via:

- `--verbose` / `-v` flag: DEBUG to stderr
- `--quiet` / `-q` flag: WARNING and above only
- `WHIRLD_LOG_LEVEL` environment variable

The log file rotates at 10 MB and retains 3 files. Each log entry includes a timestamp, log level, and message.

### 16.2 Local Usage Log

In addition to the application log, Whirld appends a structured record to `~/.whirld/logs/usage.jsonl` after every inference run. This file is local-only and never transmitted anywhere. Its purpose is to give the user (and future debugging tooling) a clean history of what was run, when, and how it performed.

Each line is a JSON object:

```json
{
  "timestamp": "2026-06-25T14:32:17Z",
  "command": "embed",
  "model": "clay-v1",
  "model_version": "1.0.0",
  "input_type": "geotiff",
  "sensor_detected": "sentinel-2-l2a",
  "chip_count": 24,
  "device": "mps",
  "quantized": false,
  "duration_ms": 4820,
  "whirld_version": "0.1.0",
  "python_version": "3.12.3",
  "os": "darwin",
  "error": null
}
```

No file paths, no CRS, no geographic data, no user-identifiable information is written. On error, the `error` field contains the exception class name (e.g. `"UnsupportedSensorError"`) but not the message or traceback — those go to the application log only.

This file serves two purposes:

1. **User self-service debugging.** When something goes wrong, the user can share the relevant lines from `usage.jsonl` in a GitHub issue without revealing anything sensitive about their data or environment.
2. **Foundation for future telemetry.** If opt-in telemetry is added in a future release, this file is the source of truth — the telemetry payload would be drawn from it, not from separate instrumentation.

`usage.jsonl` rotates at 1 MB (approximately 5,000 entries) and retains 2 files.

---

## 17. Testing Requirements

### 17.1 Unit Tests

- Band contract sensor detection for all supported sensor/model combinations
- Band translation pipeline with known inputs and expected outputs
- sha256 verification logic
- Registry YAML parsing and validation against schema
- All CLI commands with mock model weights (no real inference)

### 17.2 Integration Tests

- End-to-end embed with Clay using a small synthetic GeoTIFF
- End-to-end segment with Prithvi using a small synthetic HLS scene
- End-to-end classify with RemoteCLIP using a small synthetic RGB GeoTIFF
- STAC URL input (mock STAC server)
- REST API all endpoints

### 17.3 The Five-Minute Test (Acceptance Criteria)

This test must pass on a clean macOS ARM and clean Ubuntu x86 environment before MVP ships:

```bash
# Fresh virtualenv, no whirld installed
pip install whirld
whirld pull clay-v1
whirld embed --model clay-v1 tests/fixtures/s2_small.tif
# Expected: embeddings.npy written, exit 0, <5 minutes total
```

---

## 18. Success Metrics

### 18.1 MVP Launch Criteria

- All three models pull, embed/segment/classify, and produce verified outputs
- Five-minute test passes on both macOS ARM and Linux x86
- `whirld serve` starts and responds correctly to all three POST endpoints
- STAC URL input works for at least one public STAC endpoint
- 100% of error messages include actionable next steps

### 18.2 Post-Launch Metrics (30 days)

- GitHub stars (qualitative signal of developer interest)
- Number of unique model pulls (tracked via registry YAML fetch count on GitHub)
- Open issues classified as "sensor not supported" (leading indicator for registry expansion)
- `whirld serve` adoption vs CLI-only (inferred from GitHub issue and discussion patterns)

---

## 19. Resolved Decisions

The following questions were open during initial drafting and have been resolved:

**1. Telemetry.** No active telemetry at MVP. Whirld writes structured usage records to `~/.whirld/logs/usage.jsonl` (local only, never transmitted). See Section 16.2. Opt-in telemetry may be revisited post-launch based on community feedback.

**2. Windows support.** Deferred. No target version set. Will be added when there is demonstrated community demand (GitHub issues, discussions). macOS and Linux are the only supported platforms for MVP.

**3. STAC authentication.** `--stac-token` is supported in MVP. See Section 9.2 for updated STAC input handling. Required for Microsoft Planetary Computer collections and any other token-gated STAC endpoint.

**4. Community model submission process.** Merge on passing schema validation. A PR to `github.com/whirld/registry` that adds a valid `models/<name>.yaml` conforming to `schema/model.schema.json` is merged automatically by CI. No manual review required for schema-valid additions. The schema is the gating mechanism. This policy is stated explicitly in `CONTRIBUTING.md`.

**5. Prithvi task heads in MVP.** Flood segmentation and burn scar detection. All other Prithvi heads (crop type, land cover, etc.) are deferred to v1.1 and tracked in a registry issue.

---

## Appendix A: Dependency List

| Package | Purpose | Version Constraint |
|---|---|---|
| `typer` | CLI framework | `>=0.12` |
| `fastapi` | REST API server | `>=0.111` |
| `uvicorn` | ASGI server for FastAPI | `>=0.30` |
| `rasterio` | Raster I/O (bundles GDAL) | `>=1.3` |
| `numpy` | Array operations | `>=1.26` |
| `torch` | Model inference | `>=2.2,<3.0` |
| `huggingface_hub` | Model weight download | `>=0.23` |
| `pystac-client` | STAC API client | `>=0.8` |
| `stackstac` | STAC to xarray/numpy | `>=0.5` |
| `httpx` | Async HTTP for STAC fetching | `>=0.27` |
| `rich` | Terminal formatting | `>=13.0` |
| `pydantic` | Config and schema validation | `>=2.0` |

---

## Appendix B: File Structure

```
whirld/
  pyproject.toml
  README.md
  src/
    whirld/
      __init__.py              # Public Python API
      cli/
        __init__.py            # Typer app entrypoint
        commands/
          pull.py
          list.py
          info.py
          rm.py
          embed.py
          segment.py
          classify.py
          serve.py
          update.py
      core/
        registry.py            # Registry fetch, parse, cache
        fetch.py               # Weight download, sha256 verification
        sensor.py              # Sensor detection logic
        contract.py            # Band contract translation pipeline
        chips.py               # Chipping and reassembly
        session.py             # Model persistence context manager
      models/
        base.py                # Abstract model interface
        clay.py                # Clay wrapper
        prithvi.py             # Prithvi wrapper
        remoteclip.py          # RemoteCLIP wrapper
      server/
        app.py                 # FastAPI application
        routes/
          embed.py
          segment.py
          classify.py
          models.py
          health.py
      io/
        raster.py              # rasterio helpers
        stac.py                # STAC URL resolution and fetch
        output.py              # Write npy, GeoTIFF, GeoJSON
  tests/
    unit/
    integration/
    fixtures/
      s2_small.tif             # Tiny synthetic Sentinel-2 scene
      hls_small.tif            # Tiny synthetic HLS scene
      rgb_small.tif            # Tiny synthetic RGB scene
```
