# Whirld MVP — Walking Skeleton (Clay embed, offline)

Scope chosen with the user: **offline reference backend** + **minimal walking skeleton**.
Full plan: `~/.claude/plans/melodic-popping-map.md`.
Per-pass details and remaining PRD work: `dev_progress.md`.

## Pass 9 — STAC item URL input (done)

- [x] `io/stac.py` `read_stac_item`: stdlib fetch (no new dep; local/file:// offline,
      http(s) + bearer), sensor resolve (override → infer → ask), 3-tier band→asset
      match (native id → eo:bands name → common_name/alias), `/vsicurl/` + WarpedVRT
      range reads onto one grid, datetime + platform stamped into tags
- [x] `api.embed/classify/segment`: shared `_read_input` routes URL/`.json`/`file://`
      to the STAC reader (lazy import); `--bbox` + `--stac-token` threaded;
      `usage.jsonl` records `input_type=stac`
- [x] CLI `--bbox` + `--stac-token` on all three commands (`cli/commands/_stac.py`
      parser + `WHIRLD_STAC_TOKEN` fallback); help text path-or-URL
- [x] `pyproject` `stac` extra slimmed to `pystac-client` (reserved for future search);
      item path needs no extra
- [x] Fixtures `make_stac_item` (file:// per-band COGs, native- and common-name-keyed)
      + tests: `unit/test_stac`, `integration/test_stac_input`, gated `test_stac_real`
      (`WHIRLD_TEST_STAC_URL`); **170 passed / 5 skipped, 89% cov** (stac.py 91%), lint clean
- [x] **Verified:** `embed <item.json>` → (4,512) + `input_type=stac`; `--bbox` windows
      300×300→209×227 (4→1 chips); bad `--bbox` → exit 7; unreachable URL → NetworkError
- [x] **Clean-room container** (`docker/clean-room.Dockerfile` + `scripts/clean_room_test.py`):
      fresh python:3.13-slim, base deps only, no system GDAL → lazy import, pull, embed
      GeoTIFF + STAC item; **real Earth Search item via live /vsicurl/** range reads.
      Found+fixed the `libexpat1` syslib gap. Advances PRD §17.3 acceptance gate.
- [x] **macOS clean-room** (`scripts/clean_room_macos.sh` + `_clean_room_macos_provision.sh`):
      throwaway **Tart** VM, base deps, **no Homebrew GDAL**, runs the same
      `clean_room_test.py` → the macOS ARM half of §17.3. shellcheck-clean; local
      on-demand. (Run requires `brew install cirruslabs/cli/tart hudochenkov/sshpass/sshpass`.)
- Deferred: REST STAC bodies (still 400), multi-item search/mosaic, PC URL signing,
  clean-room CI wiring (Cirrus / self-hosted Apple-Silicon runner for the macOS leg)

## Pass 8 — Registry governance: security + license posture (done)

- [x] `core/governance.py` — OSS-license allowlist + `is_oss_license`;
      `weights_are_pickle` (safetensors/reference safe, pickle + unknown unsafe);
      `TRUSTED_SOURCE_ORGS` + `source_org`; `evaluate_automerge` →
      `AutomergeDecision(auto, reasons)`. Pure stdlib (lazy-import safe).
- [x] `SecurityError` (exit code **9**); `fetch.pull` runtime gate refuses a
      **community + pickle** entry before download (first-party pickle unaffected)
- [x] `trust` field (`first-party|community`): schema-required, `ModelEntry`
      optional-at-parse; 4 bundled YAMLs declare `trust: first-party`; license desc → SPDX
- [x] CLI: `pull` shows license + ⚠ non-OSS terms warning *before* download; `info`
      annotates `(OSS)`/`(non-OSS — review terms)` + trust tier
- [x] Closed backend allowlist locked by test (module-path ids rejected, nothing
      imported); `scripts/check_registry.py` (exit 0/1/2) + `registry_data/CONTRIBUTING.md`
      merge policy (OlmoEarth = declined worked example)
- [x] Tests: governance unit, loader allowlist, pull `SecurityError`, bundled-entry
      invariants, CLI/checker integration; **156 passed / 4 skipped, 88% cov**, lint clean
- [x] **Verified:** `info clay-v1.5` → `Apache-2.0 (OSS)` / `first-party`; checker over
      bundled → review (exit 1), crafted community-OSS-safetensors → auto-merge (exit 0);
      community-pickle pull → exit 9
- Deferred: remote registry sync (the gate now exists for when community PRs arrive)

## Pass 7 — Prithvi segmentation, burn-scar first (done)

- [x] `prithvi` extra (terratorch); `prithvi-burn-scar.yaml` (real coords/sha256,
      HLS contract, mask output); schema `config_file` + `output.classes`
- [x] New output modality: `chips.reassemble_mask` + `io/output` mask GeoTIFF
      (+bytes); `fetch` multi-file download (weight + config)
- [x] `models/prithvi.py` `PrithviBackend.segment` (TerraTorch, lazy); `base.segment`;
      loader; `api.segment`/`segment_raster` + `SegmentResult`; `--head` alias
- [x] `segment` CLI; real `POST /segment` (`image/tiff`); public `whirld.segment`
- [x] Tests (fake-segmodel unit, reassembly/GeoTIFF unit, API/CLI/server integration,
      gated real) + `make_hls` fixture; 135 passed / 4 skipped, 88% cov, lint clean
- [x] **Real proof:** `whirld pull prithvi-burn-scar` (sha256 ✓) + `segment` → real
      uint8 mask GeoTIFF; `--head` alias + `/segment` over HTTP verified
- Deferred: `prithvi-flood` (follow-up); multi-temporal heads

## Pass 6 — Finish the paydown: CPU warning + RemoteCLIP softmax scoring (done)

- [x] CPU full-precision runtime warning (`api._maybe_warn_cpu` in `_run`) +
      `--no-warnings` on embed/classify (reference/quantized/non-CPU stay silent)
- [x] RemoteCLIP scoring → `logit_scale` softmax probabilities; multi-query support
      (`classify --query` repeatable, `/classify` getlist); per-query `scores` in
      GeoJSON; single query softmaxed vs a neutral `background` prompt
- [x] `ModelBackend.classify` → `queries: list[str]`, returns `(n, n_queries)`
- [x] Tests (softmax/probabilities, multi-query, cpu-warning, geojson, server,
      gated real); 117 passed / 3 skipped, 89% cov, lint clean
- [x] **Real proof:** single-query 0–1 prob (0.089 vs background); 3-query softmax
      sums to 1.0; CPU warning fires for clay-v1.5, silent with `--no-warnings`

## Pass 5 — Deferral paydown: Clay metadata fidelity + CLI completeness (done)

- [x] Clay metadata: `models/_clay_metadata.py` (Clay's exact normalize formulas);
      `InferenceContext` + `api._run` derive per-chip lat/lon (reprojected) + scene
      time (`--datetime`/`TIFFTAG_DATETIME`); `clay_torch` datacube uses them
- [x] `rm` command (`rm <model>` / `rm --all`) + `core/fetch` remove helpers
- [x] `--crs` override for CRS-less inputs; `--batch-size` for embed
- [x] Tests (metadata formulas, reprojection, rm, crs, batch-size) + extended gated
      real test (metadata changes the embedding); 109 passed / 3 skipped, 89% cov
- [x] **Real proof:** `--datetime` changes real Clay embedding (Δ 0.018); `--crs`
      embeds a CRS-less S2; `rm`/`rm --all` clear models, registry intact

## Pass 4 — RemoteCLIP `classify` (second real model, done)

- [x] `remoteclip.yaml` (real `chendelong/RemoteCLIP` mirror, RGB CLIP contract,
      `model_name: ViT-B-32`); optional `model_name` schema/registry field
- [x] `models/remoteclip.py` `RemoteCLIPBackend.classify` (open_clip, injectable
      model for tests); GeoJSON writer in `io/output.py`
- [x] `api.py` refactor → one generic `_run`; `classify` + `classify_raster` +
      `ClassifyResult`; `classify` CLI + real `POST /classify` route + public API
- [x] Tests (fake-model unit + integration + server + gated real); 91 passed /
      2 skipped, 88% cov, lint clean
- [x] **Real proof:** `whirld pull remoteclip` (sha256 ✓) + `whirld classify
      --query "solar farm"` → GeoJSON cosine scores; `/classify` over HTTP
- [x] Fixed server double-load bug (warm device reuse)
- Deferred: multi-prompt softmax scoring; S2→8-bit tone mapping fidelity

## Pass 3 — Real Clay v1.5 weights (proof-of-load, done)

- [x] Vendored Clay encoder (Apache-2.0) → `models/_vendor/clay_v15/`
      (+ LICENSE/NOTICE/PROVENANCE; black/ruff/coverage excluded)
- [x] Real `clay-v1.5.yaml` (huggingface source, real sha256, 10-band contract,
      1024-dim); schema + registry gained optional `wavelengths`/`patch_size`
- [x] `models/clay_torch.py` `ClayTorchBackend`; `InferenceContext` threaded
      through `api`; `loader` + `fetch` wiring
- [x] Tiny-encoder unit tests + gated real-weights test + 10-band fixture
- [x] **Real proof:** `whirld pull clay-v1.5` (sha256 ✓) + `whirld embed` →
      real `(n, 1024)`, 265 tensors / 0 missing; 77 passed/1 skipped, 91% cov,
      lint clean
- Deferred: faithful `time`/`latlon` metadata (currently zeros)

## Pass 2 — `serve` REST API (done)

- [x] `core/session.py` (`ModelSession` warm-load cache) + `io/raster.py`
      `read_raster_from_bytes`
- [x] `api.py` refactored to one shared pipeline (`embed` + `embed_raster`)
- [x] `server/` FastAPI app: `GET /health`, `GET /models`, `POST /embed`;
      `segment`/`classify` → 501; `WhirldError`→HTTP mapping
- [x] `whirld serve` CLI command (lazy uvicorn import)
- [x] Tests (TestClient + live curl smoke); 71 passing, 93% coverage, lint clean

## Pass 1 — Walking skeleton plan

- [x] **1. Scaffold** — `pyproject.toml`, `requirements*.txt`, `.gitignore`,
      `.env.example`, `README.md`, `.venv`, `tasks/`
- [x] **2. Foundation** — `config.py` (WHIRLD_HOME paths), `errors.py` (exception
      hierarchy + exit codes), `_version.py`, `logging_setup.py` (app log + usage.jsonl)
- [x] **3. Registry** — `registry_data/schema/model.schema.json`,
      `registry_data/models/clay-v1.yaml`, `core/registry.py` (Pydantic models, load,
      validate, cache seed)
- [x] **4. Pull/fetch** — `core/fetch.py` (offline reference source, sha256 verify,
      manifest.json; HF path wired but not exercised)
- [x] **5. Translation pipeline** — `io/raster.py`, `core/sensor.py`,
      `core/contract.py`, `core/chips.py`
- [x] **6. Model + output** — `models/base.py` (ABC), `models/clay.py` (reference
      backend), `models/loader.py`, `io/output.py` (npy + meta sidecar), `api.py`
- [x] **7. CLI + Python API** — `cli/` (`pull`, `list`, `info`, `embed`),
      `src/whirld/__init__.py` (lazy public API)
- [x] **8. Tests** — fixture generator, unit + integration, `pytest --cov` = 93%
- [x] **9. Polish** — black + ruff clean, end-to-end offline five-minute test, README,
      review section below

## Deferred (extension points left in place)

Prithvi/`segment`, RemoteCLIP/`classify`, `serve`/REST API, `update`/`rm`, STAC input,
quantization, real Hugging Face weight loading.

## Review

**Outcome:** the offline walking skeleton is complete and verified end-to-end.

- **Five-minute-test analog (offline):** `whirld pull clay-v1` then
  `whirld embed --model clay-v1 s2_small.tif` writes `s2_small_embeddings.npy`
  (shape `(4, 512)`, float32) + `s2_small_embeddings_meta.json` (CRS, chip
  bboxes), exit 0 — via the installed `whirld` console script, no network.
- **Tests:** 51 passing, **93% coverage** (`pytest --cov=whirld`). Unit:
  registry, fetch/checksum, sensor detection, contract translation, chipping,
  Clay backend. Integration: full embed via Python API + the CLI surface,
  including the documented error exit codes (2/3/4/7).
- **Quality gates:** `black --check` clean, `ruff check` clean. Full type hints,
  Pydantic models across layer boundaries, custom exception hierarchy, `logging`
  (no prints), docstrings throughout. `import whirld` imports no numpy/rasterio/torch.

**Notable decisions / deviations**

- **NAIP dropped from the Clay contract.** The PRD lists NAIP (4-band) but pairs
  Clay with a 6-value normalization. Shipping that would be an invalid contract,
  so NAIP is omitted with a comment; the three six-band sensors are consistent.
- **Reference `source.type`.** Added a `reference` source type to the registry
  schema for the deterministic offline backend; the real `huggingface` path is
  wired alongside it and ready to switch on.

**Bug found & fixed during the build**

- The app/usage loggers cached handlers globally pinned to the first
  `WHIRLD_HOME`, so a changed home (long-running process, or tests) wrote logs to
  a stale path. Fixed by rebinding handlers when the target path changes
  (`logging_setup.py`).

**Next iteration (deferred, extension points in place):** real HF weight loading,
Prithvi `segment`, RemoteCLIP `classify`, `serve` REST API, `update`/`rm`, STAC
input, quantized variants.
