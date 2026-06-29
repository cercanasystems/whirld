# Whirld — Development Progress Log

A running log of implementation passes against the MVP PRD (`whirld-prd-mvp.md`).
Each entry is timestamped in UTC and records what was implemented and what
remains.

---

## 2026-06-29T10:12:04Z — Clean-room install testing (Linux x86 + macOS ARM, PRD §17.3)

Stood up the **clean-machine "five-minute test"** the PRD's acceptance gate (§17.3)
calls for — a fresh OS with **only the published package + base deps, no system/Homebrew
GDAL** — across **both** target platforms. One OS-agnostic smoke test
([scripts/clean_room_test.py](scripts/clean_room_test.py)) is driven by two thin
environment provisioners (Docker for Linux, a Tart VM for macOS). It immediately earned
its keep by catching a real packaging gap that the dev machine masked. Grew out of the
Pass 9 STAC work (the live `/vsicurl/` path is one of the things it validates).

**Shared smoke test — `scripts/clean_room_test.py`**

OS-agnostic; depends only on the installed package (fixtures generated inline with
rasterio — never the repo's `tests/` tree). Steps: assert `import whirld` is lazy (fresh
interpreter) → `pull clay-v1` → `embed` a local GeoTIFF → `embed` a `file://` STAC item
(hermetic `/vsicurl/`) → optionally, if `WHIRLD_TEST_STAC_URL` is set, a *real* remote
STAC item (live HTTP range reads). Reports python/rasterio/bundled-GDAL versions; any
failure exits non-zero with the reason.

**Linux x86 — `docker/clean-room.Dockerfile`**

Fresh `python:3.13-slim`, base deps only, no system GDAL.
- **Finding (fixed):** `-slim` lacks `libexpat.so.1`, which rasterio's bundled GDAL needs
  to import — `apt-get install libexpat1` (the one syslib the wheel doesn't vendor);
  system GDAL stays uninstalled. Exactly the "works on my Mac" gap a clean room exists to
  catch.
- **Pass:** lazy import → pull → embed GeoTIFF → embed `file://` STAC item → all
  `(1, 512)`, on rasterio 1.5.0 / bundled **GDAL 3.12.1**.
- **Live `/vsicurl/`:** given a current Earth Search Sentinel-2 item (`S2B_43XEA_…_L2A`)
  + a small `--bbox`, fetched the item and range-read its COG assets from S3 (assets
  keyed by **common name** → resolved via the alias tier) → real embeddings. End-to-end
  on a clean Linux box.

**macOS ARM — `scripts/clean_room_macos.sh` + `scripts/_clean_room_macos_provision.sh`**

A throwaway **Tart** VM (Cirrus Labs' Apple-Silicon VM CLI) — the macOS half of the gate,
the leg most likely to expose the classic geospatial pain point (a clean Mac with **no
Homebrew GDAL**).
- Host driver clones/boots a `macos-sequoia-base` VM headless with the repo mounted
  (`--dir`), waits for IP + sshd, pipes the provisioner over SSH (admin/admin), and tears
  the VM down on exit (`KEEP_VM=1` to keep).
- In-guest: `brew install python@3.13` (no GDAL formula), venv, `pip install` (base deps),
  run the shared smoke test. Reuses `clean_room_test.py` verbatim, including the
  `WHIRLD_TEST_STAC_URL` passthrough.
- Scope: **local on-demand**, mirroring the Docker leg; **smoke only** (no full pytest in
  the VM). Both scripts are shellcheck-clean.
- **Honest status:** the scripts are statically validated (syntax + shellcheck) but the
  Tart run was **not executed in-session** (a ~30–40 GB macOS image pull + VM boot isn't a
  safe in-session action). First real run on an Apple-Silicon host is where any
  macOS-specific quirk would surface — and the smoke test prints versions + asserts, so it
  would report, not hide it (the same value the Linux leg delivered).

**Deferrals**

- **CI wiring.** Both legs are local-run today. A CI macOS leg needs **Cirrus CI**
  (Tart-native) or a **self-hosted Apple-Silicon runner** — GitHub-hosted runners can't
  run Tart (no nested virtualization). The Linux leg can run on any runner.
- Published wheel + `whirld[cuda]` validation + multi-version (3.10–3.12) CI remain (PRD
  §15).

---

## 2026-06-27T16:55:00Z — Pass 9: STAC item URL input (PRD §9)

Whirld now accepts a **STAC item URL** as input to `embed` / `classify` / `segment`,
not just a local GeoTIFF. The whole slice hangs off the seam that was already cut: each
command's `resolve()` built one `RasterSource`, and the three `_looks_like_url`
branches simply *raised* "not available in this build." Those raises are replaced with
one new reader; **nothing downstream changed** (sensor detection, alias band selection,
chipping, Clay metadata all run unmodified).

**What was implemented**

- **`io/stac.py` (new) — `read_stac_item()`.** Fetches the item JSON, resolves the
  sensor, maps each contract band to a STAC asset, reads the assets with COG range
  requests onto one grid, and returns a `RasterSource`:
  - **Fetch** uses the **standard library** (`urllib`) — *no new dependency*. Local
    paths / `file://` are read directly (so tests are fully offline); http(s) carries a
    bearer token when given.
  - **Sensor**: explicit `--sensor` → else inferred from item `properties`
    (platform/constellation/collection) → else a clear "pass --sensor" error.
  - **Band → asset matching** is 3-tier and alias-driven: native band id (`B04`) →
    `eo:bands[].name` → `eo:bands[].common_name` / asset key == the contract alias
    (`red`). Assembled band descriptions are the **native band names**, so the existing
    `_select_bands_by_alias` matches by name with zero special-casing.
  - **Assembly** opens each asset via rasterio `/vsicurl/` + `WarpedVRT`, reprojecting
    onto one grid at the contract's target resolution. GDAL is tuned for object stores
    (`GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`, `VSI_CACHE`, bearer via
    `GDAL_HTTP_HEADERS`).
  - Two free reuses: the item's `datetime` → stamped as `TIFFTAG_DATETIME` (Clay gets
    real scene time); the platform → stamped so `_run`'s own sensor detection
    reproduces the answer.
- **`--bbox MIN_LON,MIN_LAT,MAX_LON,MAX_LAT` (EPSG:4326)** windows the read — the COG
  range-request fast path. Verified: a 300×300 fixture → a 209×227 windowed read
  (4 chips → 1). Without `--bbox` the full item is read (with a size warning for large
  scenes).
- **Auth**: `--stac-token` / `WHIRLD_STAC_TOKEN` (already read by `config.py`) →
  `Authorization: Bearer` on the fetch and the asset reads.
- **Wiring**: `api.embed/classify/segment` route URL/`.json`/`file://` inputs through a
  shared `_read_input` helper (lazy-imports `io.stac`); `usage.jsonl` now records
  `input_type="stac"` vs `"geotiff"`. CLI gains `--bbox`/`--stac-token` on all three
  commands (shared `cli/commands/_stac.py` parser).

**Verification**

- `whirld embed <local item.json> --model clay-v1` (file:// assets) → `(4, 512)`
  embeddings, exit 0, `usage` shows `input_type=stac`; sensor inferred from item
  metadata. Common-name-keyed assets resolve via the alias tier.
- `--bbox` reduces the chip count; a malformed `--bbox` → exit 7; an unreachable URL →
  `NetworkError` (proving the path is wired, vs the old deferred message).
- **170 passed, 5 skipped** (gated, incl. the new `WHIRLD_TEST_STAC_URL` real test),
  **89% coverage** (`io/stac.py` 91%). `black` + `ruff` clean. `import whirld` still
  imports no rasterio/numpy/torch (the reader is lazy).

(The clean-room install validation that grew out of this pass is logged as its own
entry, dated 2026-06-29.)

**Scope / deferrals (documented)**

- **Library + CLI only.** REST `/embed|/classify|/segment` STAC bodies still return
  `400` — a small follow-up.
- **Single item, not a search.** Multi-item STAC *search*/mosaic (pystac-client /
  stackstac) is a later slice; the `stac` extra is reserved for it (`pystac-client`).
  The item path itself needs no extra.
- **Bearer auth only.** Planetary-Computer URL *signing* is a follow-up.

---

## 2026-06-27T15:30:00Z — Pass 8: Registry governance (security + license posture)

Settled Whirld's **registry governance** as one policy before remote registry sync
(PRD §8) makes the registry a live attack surface. After the backend refactor a
model's YAML selects its **backend, weights, normalization, and license**, so
"schema-valid" is no longer a sufficient gate for what gets pulled and executed. This
pass decides and *enforces* the security and license posture together.

**Threat model.** The execution risk is **pickle deserialization** —
`torch.load(weights_only=False)` (`.ckpt`/`.pt`) and open_clip flat state_dicts run
arbitrary code on load. Today every entry is first-party/curated; the risk arrives
with community submissions + remote sync.

**What was implemented**

- **`core/governance.py` (new) — one source of truth**, reused by the runtime pull
  gate, the CLI license display, and the registry CI checker (no policy drift). Pure
  stdlib, so the lazy top-level-import promise holds.
  - `OSS_LICENSES` allowlist + `is_oss_license()` (case/space-normalized; unknown ⇒
    non-OSS, fail-safe).
  - `weights_are_pickle(entry)` — primary `source.files[0]` extension; `.safetensors`
    safe, `reference` safe, pickle exts **and anything unrecognized** unsafe.
  - `TRUSTED_SOURCE_ORGS` (`made-with-clay`, `chendelong`, `ibm-nasa-geospatial`) +
    `source_org()`.
  - `evaluate_automerge(entry)` → `AutomergeDecision(auto, reasons)`: **auto ⇔**
    community **∧** OSS license **∧** trusted org **∧** safetensors. First-party
    entries always route to review (maintainer-curated); pickle only ever ships via a
    reviewed first-party entry.
- **Runtime security gate** (`core/fetch.pull`): a **community + pickle** entry is
  refused *before download* with a new **`SecurityError` (exit code 9** — a documented
  Whirld extension beyond the PRD's 1–8). First-party pickle (our shipping models) is
  unaffected; sha256 verification stays.
- **Closed backend allowlist — locked by test.** `models/loader.load_backend` already
  dispatches by an allowlisted id to a hardcoded branch and never imports a module
  path from YAML. A new test asserts dynamic/module-path ids (`os.system`,
  `subprocess`, `whirld.models.clay`, `../etc/passwd`) are rejected and **nothing is
  imported**.
- **Schema + model.** `trust` (`first-party | community`) added — **required** in the
  schema, `ModelEntry.trust = "community"` (optional-at-parse so an older registry
  cache still loads). `license` description tightened to recommend SPDX. The four
  bundled YAMLs declare `trust: first-party`.
- **License surfaced prominently.** `whirld pull` prints the license **before**
  download and a yellow ⚠ terms warning for non-OSS; `whirld info` annotates the
  license `(OSS)` / `(non-OSS — review terms)` and shows the trust tier.
- **Merge-gate enforcement.** `scripts/check_registry.py` loads each entry
  (schema-valid) → `evaluate_automerge` → prints AUTO-MERGE / REVIEW / INVALID; exits
  **0** (all auto-mergeable) / **1** (any needs review) / **2** (schema error) — what
  the external registry repo's CI runs on community PRs.
  `registry_data/CONTRIBUTING.md` (new) replaces "merge on schema validation" with the
  real policy (schema-valid **AND** OSS-allowlist license **AND** trusted source
  **AND** safetensors; pickle only for first-party), the security rationale, the
  license posture with **OlmoEarth as the declined worked example**, and a GitHub
  Actions snippet wiring the checker.

**Verification**

- `whirld info clay-v1.5` → `License: Apache-2.0 (OSS)` + `Trust: first-party`.
- `check_registry.py` over all four bundled entries → REVIEW (maintainer-curated),
  exit 1; a crafted community + OSS + safetensors + trusted-org entry → AUTO-MERGE,
  exit 0; a schema-invalid entry → exit 2.
- A crafted **community + pickle** entry → pull raises `SecurityError` (exit 9) with
  the RCE refusal.
- **156 passed, 4 skipped** (gated real-model), **88% coverage**. `black --check` +
  `ruff` clean. `import whirld` still imports no torch/terratorch/numpy/open_clip/
  fastapi (governance is pure stdlib).

**Design notes / risks**

- **Trust is self-declared in YAML** — safe only because (a) the merge gate (CI +
  human) controls who can set `first-party` for auto-merge, and (b) the runtime
  refuses community pickle regardless. Defense in depth; documented in CONTRIBUTING.
- **License matching is best-effort** (SPDX recommended, not enforced) — conservative:
  unknown/odd strings fall outside the OSS allowlist → warn + no auto-merge.
- Pull **warns** (never blocks) on non-OSS license; the *block* is at merge time. The
  runtime *does* block execution of community pickle. Our bundled models are pickle but
  first-party, so they pass the runtime gate and are correctly "review" (not
  auto-merge) at the CI gate.

---

## 2026-06-27T13:48:34Z — Decision: OlmoEarth NOT integrated (license posture)

Evaluated adding **OlmoEarth** (Ai2's geospatial foundation model) as a 4th model. A
research spike proved it's fully runnable, but the **decision is not to integrate** —
its license does not fit Whirld's all-Apache-2.0 posture. Findings preserved here so a
future revisit needs no re-spike.

**Why declined:** OlmoEarth ships under the custom **"OlmoEarth Artifact License"**
(not OSS). Commercial use is allowed, but with use restrictions — **no
military/defense/surveillance, no extractive-industry use** (oil/gas/mining/
deforestation) — plus required Ai2 attribution and license-propagation on
redistribution. Shipping a use-restricted model in the default registry conflicts with
the project's open posture (the other three are Apache-2.0).

**Spike verdict (GREEN) — for the record:**
- Real, **ungated**, installs + runs on **Python 3.13** via `olmoearth-pretrain`
  (light inference deps: torch/einops/huggingface_hub/numpy; no vendoring, no torch
  downgrade). `[training]` extra is heavy and not needed for embeddings.
- Repos: `allenai/OlmoEarth-v1-{Nano,Tiny,Base,Large}` (+ v1_1/v1_2). Chosen Base rev
  `93589e2dee5b5c95a660d1e9365bc017ea7f35d6`: `weights.pth` 830 MB (sha256
  `551c1cc5…c8b6e11`) + `config.json` (sha256 `bd7759b9…d5512bf`). Plain `torch.save`
  state_dict (pickle, not safetensors).
- Load: `olmoearth_pretrain.model_loader.load_model_from_id(ModelID.OLMOEARTH_V1_BASE)`;
  0 missing/unexpected keys. **Embedding model**, dim **768**.
- Input: 12 S2 L2A bands, resolution-grouped order
  `[B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12,B01,B09]` (omits B10); custom norm
  `(x − (mean−2·std))/(4·std)` — **expressible as our `per_band_zscore`** with derived
  `mean'=(mean−2·std)`, `std'=(4·std)`, `scale=1`. Unusual **BHWTC** tensor layout +
  a `MaskedOlmoEarthSample` (mask + timestamps); tile ≤128, patch 1–8. Output
  `(B,H',W',T,S,D=768)` → mean-pool → per-chip 768-vec.

**If revisited:** it'd be a clean drop-in under the registry-driven design — one new
`olmoearth` backend (BHWTC + masked-sample construction + pooling) + one dispatch line
+ a registry YAML (contract/normalization/output all expressible as data). Start with a
smaller variant (Nano/Tiny), Sentinel-2 only. The license posture is the only blocker.

---

## 2026-06-27T13:41:36Z — Refactor: registry-driven backend selection

**Why:** before adding a fourth model (OlmoEarth), make "everything goes through the
registry" actually true. Audit found the data was fully registry-driven, but backend
*selection* was hard-coded: `models/loader.py` held a `_BACKENDS` dict mapping each
model **name** → a backend, so adding a model — even one reusing an existing backend
— required a Python edit. The pipeline (contract, source, output, normalization) was
already pure data.

**Change:**
- Registry entries now declare a `backend:` id (schema + `core/registry.ModelEntry`).
  All four bundled YAMLs set it (`clay-reference`, `clay`, `remoteclip`, `prithvi`).
- `load_backend` dispatches by `entry.backend` (lazy-importing only that backend);
  the name→backend dict is **gone**. Adding a model that reuses a backend is now
  **pure YAML, zero code**; only a genuinely new architecture needs a new backend
  class + one branch.
- Schema *requires* `backend` (gates new submissions); `ModelEntry.backend` is kept
  optional-at-parse so a registry cache seeded by an older version still loads —
  the loader gives a clear "refresh the registry" error if it's actually missing.

**Verified:** 139 passed, 4 skipped, 88% coverage, black + ruff clean, lazy import
intact. New `unit/test_loader.py` locks in the behavior (every bundled model declares
a backend; unknown/missing backend → clear error).

---

## 2026-06-27T13:21:42Z — Pass 7: Prithvi segmentation (burn-scar first; the 3rd paradigm)

**Why:** the last capability paradigm and the final `501` route. It introduces a
new output modality — a dense **per-pixel mask GeoTIFF** with **tile reassembly**
(every prior model was chip→vector). A spike verified the real mechanics and
corrected the PRD twice. User chose **two models + `--head` alias** and **burn-scar
first**.

**Verified end-to-end (real weights, on this machine):**
- `whirld pull prithvi-burn-scar` → downloads the `.pt` + the TerraTorch config,
  sha256 `0c5f9334…62954d3` ✓.
- `whirld segment --model prithvi-burn-scar <hls>.tif` → TerraTorch loads the real
  weights, forward runs, writes a single-band **uint8 mask GeoTIFF** (320×320,
  EPSG:32613, nodata 0, classes {0,1}).
- `--model prithvi-eo-2 --head burn-scar` resolves to the same model; `POST
  /segment` returns `image/tiff`; gated real test passes.
- Suite: **135 passed, 4 skipped** (gated Clay/RemoteCLIP/Prithvi), **88% coverage**,
  black + ruff clean, `import whirld` imports no terratorch/torch.

### Ground-truth corrections to the PRD
- **TerraTorch installs cleanly on Python 3.13** (`terratorch==1.2.8`) — no vendoring.
- **Single-image (T=1), not before/after (T=2)** — the published flood/burn-scar
  heads are single-scene binary segmentation; the PRD's temporal stacking is wrong.
- **Two separate fine-tuned checkpoints**, not one switchable model — exposed as
  `prithvi-burn-scar` / `prithvi-flood`, with `prithvi-eo-2 --head X` as an alias.
- Real repo `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars` (rev `a3f2c410…`),
  Apache-2.0, ungated; needs **both** the `.pt` and a YAML config.

### What was implemented
- **New output modality.** `core/chips.reassemble_mask` (inverse of chipping:
  place per-tile masks, crop padding); `io/output.write_mask_geotiff` +
  `mask_to_geotiff_bytes` (single-band uint8 LZW GeoTIFF, nodata, carry CRS/transform;
  bytes for the REST `image/tiff`).
- **Backend.** `models/prithvi.py` `PrithviBackend.segment`: lazy `terratorch`, build
  via `LightningInferenceModel.from_config(yaml, ckpt).model`, feed our
  contract-normalized HLS tiles (add the T=1 dim), `argmax`/threshold the
  `(B,2,H,W)` logits → uint8 mask. `model` injectable for tests; `embed`/`classify`
  decline. `models/base.segment` finalized; `loader` maps `prithvi-*` → it.
- **Registry.** `prithvi-burn-scar.yaml` (real coords/sha256, HLS 6-band contract,
  reflectance scale + head means/stds, 512 tiles, `output: mask/geotiff/classes:2`).
  Schema + `core/registry.py` gained `config_file` (model) + `output.classes`.
- **Fetch.** `_pull_huggingface` now downloads **all** `source.files` (weight +
  config), verifying the primary sha256.
- **Surface.** `api.segment`/`segment_raster` + `SegmentResult` (reuse the generic
  `_run`); the `prithvi-eo-2 --head` resolver; `segment` CLI; real `POST /segment`
  (`image/tiff`); public `whirld.segment`.
- **Tests.** Fake-segmodel unit (`PrithviBackend`), reassembly + GeoTIFF writer
  unit, integration (API + CLI + `/segment` via monkeypatched backend), gated real
  (`test_prithvi_real.py`); new HLS fixture (`make_hls`).

### Deferred (documented)
`prithvi-flood` (fast follow-up — needs its config's exact normalization);
multi-temporal/before-after heads; tile overlap/blending (IBM's official script uses
non-overlapping tiles + reflect-pad, which we follow).

---

## 2026-06-26T23:16:46Z — Pass 6: Finish the paydown (CPU warning + RemoteCLIP softmax scoring)

**Why:** clear the last two cheap/uncoupled deferrals so the ledger holds only big
features. (1) The CPU full-precision runtime warning (PRD §12.2); (2) RemoteCLIP's
scoring — the last "shipped-but-not-fully-faithful" item (raw cosine → calibrated
0–1 probabilities).

**Verified:** non-gated **117 passed, 3 skipped, 89% coverage**, black + ruff clean,
`import whirld` still lazy. Real proof on this machine:
- `whirld classify --model remoteclip --query "solar farm"` → single-query 0–1
  **probability** (0.089 vs a neutral background prompt).
- `--query "solar farm" --query "forest" --query "airport"` → per-query softmax
  summing to **1.0** (forest 0.874, solar 0.116, airport 0.010 on the synthetic
  fixture). Gated real test asserts the same.
- `whirld embed --model clay-v1.5 --device cpu` prints the runtime warning
  ("…full-precision on CPU (1 chips)… estimate ~1s… --device mps/cuda…");
  `--no-warnings` silences it; the numpy reference `clay-v1` never warns.

### CPU full-precision warning (PRD §12.2)
- `api._maybe_warn_cpu(model, manifest, device, chip_count, no_warnings)`, called
  from `_run` after chipping. Fires only for real Hugging Face full-precision
  weights on CPU (reference/quantized stay silent); includes a rough
  `chip_count × ~1s` estimate (labeled an estimate) + a `--device` suggestion + a
  note that int8 quantization isn't available yet. Suppressed by `--no-warnings`
  (new option on `embed`/`classify`); the server path passes `no_warnings=True`.

### RemoteCLIP softmax scoring + multi-query
- `ModelBackend.classify` signature `query: str` → `queries: list[str]`, return
  `(n_chips, n_queries)` probabilities. `RemoteCLIPBackend.classify` now uses
  RemoteCLIP's own scoring: `logit_scale.exp() × cosine`, `softmax` over the
  prompts. A single query is softmaxed against a neutral `"background"` prompt
  (calibrated match probability); multiple queries form a true zero-shot
  multi-class. Returns the user-query columns.
- `io/output.build_feature_collection` takes `queries` + the `(n, q)` matrix; each
  Feature carries `score` (primary query) + a per-query `scores` map; FC adds
  `queries`. `api.classify` accepts `query: str | list[str]`; `ClassifyResult`
  gains `queries`. CLI `--query` is repeatable; `POST /classify` reads
  `form.getlist("query")`.

### Notable decision
The single-query score is a softmax against a neutral `"background"` prompt — a
documented, easily-revisited baseline. Multi-query is the unambiguous zero-shot
path. (On random-noise fixtures the absolute values are arbitrary but always valid
probabilities; the gated real test confirms real probabilities on real weights.)

### Tooling
Added `tool.ruff.lint.flake8-bugbear.extend-immutable-calls` for
`typer.Option`/`Argument`/`fastapi.File`/`Form` (their call-in-default is by design).

---

## 2026-06-26T22:45:24Z — Pass 5: Deferral paydown (Clay metadata fidelity + CLI completeness)

**Why:** four passes shipped real models fast, accruing deferrals. The user chose
to pay down two buckets before they accumulate: **Clay metadata fidelity** (the
biggest correctness debt — `clay-v1.5` fed zeros for `time`/`lat-lon`) and **cheap
CLI/IO completeness** (`rm`, `--crs`, `--batch-size`). RemoteCLIP scoring and the
feature-coupled deferrals stay deferred by the user's choice.

**Verified:** non-gated **109 passed, 3 skipped, 89% coverage**, black + ruff
clean, `import whirld` still lazy. Real proof on this machine:
- Gated real Clay test now also asserts metadata **changes** the embedding — and
  it does: `whirld embed --model clay-v1.5 --datetime 2024-06-01T10:00:00Z` differs
  from the no-datetime run (max abs diff 0.018). Proves the metadata is consumed.
- `whirld embed --model clay-v1 --crs EPSG:32630 <no-crs>.tif` succeeds; without
  `--crs` it fails with the clear "no CRS" error (exit 7).
- `whirld rm clay-v1` / `whirld rm --all` clear models; registry stays intact.

### Clay metadata fidelity (the real correctness win)
Verified Clay's **authoritative** formulas from `stacchip.processors.prechip`
(`normalize_timestamp`) and Clay's `inference.ipynb` (`normalize_latlon` /
`prep_datacube`), and reproduced them verbatim:
- `time = [sin(w), cos(w), sin(h), cos(h)]`, `w = isoweek·2π/52`, `h = hour·2π/24`
  (one acquisition datetime per scene).
- `latlon = [sin(la), cos(la), sin(lo), cos(lo)]` per chip (each chip's centroid).

Implementation:
- `models/_clay_metadata.py` (new) — pure-math `normalize_timestamp`/`normalize_latlon`
  + `time_vector`/`latlon_vector` (zeros fallback). Unit-tested against Clay's formula.
- `models/base.InferenceContext` gained `latlons` (per-chip) + `acquisition_datetime`.
- `api._run` computes both after chipping: `_chip_latlons` reprojects each chip's
  bbox centroid (source CRS → EPSG:4326 via `rasterio.warp.transform`);
  `_resolve_datetime` parses `--datetime` (ISO) › `TIFFTAG_DATETIME` › `None`.
- `models/clay_torch._build_datacube` now fills real per-chip `latlon` and scene
  `time`; missing datetime → `time` zeros (Clay's neutral value), lat/lon always
  derived. `usage.jsonl` stays geo/PII-free (metadata used for inference only).

### CLI / IO completeness
- **`rm`** — `core/fetch.remove_model` + `remove_all` (preserves the registry);
  `cli/commands/rm.py` (`whirld rm <model>` / `--all`); registered. Clean errors.
- **`--crs`** — `io/raster.read_raster`/`read_raster_from_bytes` accept a `crs`
  override assigned when the file declares none (still rejects when neither present);
  threaded through `api.embed`/`classify` + both CLI commands.
- **`--batch-size`** — `api.embed` sets `backend._batch_size` when the backend
  batches (Clay torch / RemoteCLIP); the numpy reference ignores it.

### Tests
`unit/test_clay_metadata.py` (formulas vs Clay's exact spec, `_chip_latlons`
reprojection, `_resolve_datetime` precedence, datacube zeros-vs-real),
`unit/test_rm.py`, `integration/test_cli_paydown.py` (rm / --crs / --batch-size),
extended `test_clay_real.py` (metadata changes the real embedding). New CRS-less S2
fixture (`make_sentinel2(crs=None)`).

---

## 2026-06-26T20:40:54Z — Pass 4: RemoteCLIP `classify` (second real model, new paradigm)

**Why:** the user chose RemoteCLIP `classify` as the next slice — the **second
real model** and a new capability paradigm (zero-shot text-driven classification),
which also lights up the dormant `classify` CLI command + REST route (was `501`)
and adds the GeoJSON output format. A research spike proved the real model
classifies correctly (4/4 real aerial images zero-shot).

**Verified end-to-end (real weights, on this machine):**
- `whirld pull remoteclip` → sha256 `60014e39…89af85c4` ✓, manifest written.
- `whirld classify --model remoteclip --query "solar farm" <s2>.tif` → real
  weights load (0 missing / 0 unexpected), GeoJSON FeatureCollection written with
  per-chip cosine scores (~0.17–0.25, the expected RS range) + bbox polygons.
- `POST /classify` over HTTP (TestClient, real weights) → 200 + FeatureCollection.
- Suite: **91 passed, 2 skipped** (gated real tests for Clay + RemoteCLIP),
  **88% coverage**, black + ruff clean, `import whirld` imports no torch/open_clip.

### Ground-truth corrections to the PRD
- **Repo:** the PRD's `BAAI/RemoteCLIP` is now **gated (HTTP 401)**. We use the lead
  author's public mirror `chendelong/RemoteCLIP` (pinned rev + sha256 verified).
- **No vendoring needed** (unlike Clay): `open_clip_torch` installs on Python 3.13.
- RemoteCLIP is **RGB (3-band)** with OpenAI-CLIP preprocessing (0..1 + CLIP mean/std,
  224px) and **512-dim** image/text features; score = image-text cosine similarity.

### What was implemented
- **No new preprocessing.** The band contract already does RemoteCLIP's
  preprocessing — select RGB by alias, scale to 0..1, CLIP-standardize, chip at
  224. `registry_data/models/remoteclip.yaml` expresses it directly
  (`scale: 0.0001`, CLIP mean/std, `chip_size_px: 224`, `model_name: ViT-B-32`,
  `output: {type: scores, format: geojson}`). Added optional `model_name` to the
  schema + `core/registry.py`.
- **Backend** — `models/remoteclip.py` `RemoteCLIPBackend.classify(chips, query)`:
  lazy `open_clip`, build ViT-B-32, load the flat state_dict, tokenize + encode +
  L2-normalize → per-chip cosine. `model`/`tokenizer` injectable for tests; `embed`
  declines (classification model).
- **GeoJSON output** — `io/output.py` `build_feature_collection` + `write_geojson`
  (one Feature/chip: footprint polygon in input CRS + score/query/model/chip_index).
- **Orchestration refactor** — `api.py` now has one generic `_run(..., compute)`;
  `embed` and the new `classify` (+ `classify_raster` for the server) share it,
  including the success/failure `usage.jsonl` recording. Added `ClassifyResult`.
- **Surface** — `classify` CLI command (`--query`/`--top-k`/`--threshold`/`--output`,
  stdout default), real `POST /classify` route (replacing the 501), public
  `whirld.classify`. `models/loader.py` maps `remoteclip` → the backend.
- **Tests** — `unit/test_remoteclip.py` + `unit/test_output_geojson.py` (fake model,
  no download); `integration/test_classify.py` (API + CLI via monkeypatched backend);
  `integration/test_remoteclip_real.py` (gated on `WHIRLD_TEST_REMOTECLIP_CKPT`);
  updated the server `/classify` test (now 422-on-missing-query, not 501).

### Bug found & fixed during the build
The server **double-loaded** models: `serve --models X` preloaded on the server's
device (e.g. cpu), but each request re-ran device auto-detection (→ mps) and loaded
a *second* copy, wasting the warm preload. Fixed in `api.py`: `embed_raster` /
`classify_raster` now default to `session.device`, so warm-preloaded models are
reused (verified: one load, 64 ms warm vs 2526 ms cold). Latent since Pass 2;
masked by the cheap reference backend.

### Deliberate deferral (documented)
Single `--query` → score = **cosine similarity** (multi-prompt softmax probabilities
deferred). S2-DN→RGB tone mapping is approximate (RemoteCLIP trained on 8-bit RGB);
mechanism is correct, absolute scores on Sentinel-2 reflectance are a known caveat.

---

## 2026-06-26T14:31:57Z — Pass 3: Real Clay v1.5 weights (proof-of-load, integrated)

**Why:** the user (correctly) noted the machine is networked, so "no network" was
never the blocker. A research subagent then proved the real Clay model loads and
runs end-to-end. The user chose **proof-of-load first**, so this pass integrates
real weights through the existing `huggingface` source path: `whirld pull
clay-v1.5` downloads + sha256-verifies the real 5.16 GB checkpoint and `whirld
embed --model clay-v1.5` runs the **actual Clay encoder** to produce genuine
1024-dim embeddings. Full metadata fidelity (time/lat-lon) is deliberately deferred.

**Verified end-to-end (real weights, on this machine):**
- `whirld pull clay-v1.5` → sha256 `21432069…d4798d0` ✓, manifest written
  (`weights_file: v1.5/clay-v1.5.ckpt`).
- `whirld embed --model clay-v1.5 <10-band S2>.tif` → encoder loads **265 tensors,
  0 missing / 0 unexpected**, forward runs, writes `(1, 1024)` finite embeddings.
- `whirld info clay-v1.5` renders the real 10-band contract.
- Suite: **77 passed, 1 skipped** (the gated real test, which **passes** when
  pointed at the checkpoint), **91% coverage**, black + ruff clean, and
  `import whirld` still imports no torch (lazy).

### Ground-truth corrections to the PRD (the real model ≠ the spec)
- **Repo:** the PRD's `made-with-clay/clay-v1` does not exist. Real repo is
  `made-with-clay/Clay`; only a **v1.5** checkpoint is published (`v1.5/clay-v1.5.ckpt`).
- **Embedding dim:** **1024** (encoder CLS token), not the PRD's 512.
- **Sentinel-2 bands:** **10** (adds red-edge B05/B06/B07 + narrow-NIR B8A), not 6.
- **Normalization:** per-band standardization of **raw DN** (scale 1.0) with Clay's
  published mean/std — not a reflectance z-score.
- **License:** Apache-2.0 (both code and weights), not the PRD's "MIT".

### What was implemented
- **Vendored encoder** — `models/_vendor/clay_v15/` (`encoder.py` + `backbone.py`,
  `factory.py`, `utils.py`), copied from `claymodel==1.5.0` (Apache-2.0) with
  imports adapted. Includes `LICENSE`, `NOTICE`, `PROVENANCE.md`. Reason for
  vendoring: `claymodel` pins torch 2.4.0 (no Python 3.13 wheel); the encoder alone
  needs only `torch + einops`. black/ruff/coverage exclude this dir (verbatim
  upstream code).
- **Real registry entry** — `registry_data/models/clay-v1.5.yaml`: `huggingface`
  source (real repo/rev/file), real sha256/size, 10-band S2 contract with
  `wavelengths` + `patch_size: 8`, DN mean/std, `embed_dim: 1024`. The offline
  `clay-v1` reference entry is untouched.
- **Schema + models** — `model.schema.json` and `core/registry.py` gained optional
  `wavelengths` (per sensor) and `patch_size` (band contract).
- **Real backend** — `models/clay_torch.py` `ClayTorchBackend`: lazy torch import,
  builds the large encoder, `torch.load(weights_only=False, mmap=True)`, strips the
  `model.encoder.` prefix, runs batched forwards, returns the CLS token as
  `(n, 1024)`. Encoder is injectable for tests.
- **Interface** — added `InferenceContext` (sensor/gsd/wavelengths) to
  `models/base.ModelBackend.embed`; the reference backend ignores it, `api._orchestrate`
  builds it from the detected sensor's contract. `models/loader.py` maps `clay-v1.5`
  → torch backend. `core/fetch.py` now records `weights_file` relative to the model
  dir (HF nests it under `v1.5/`).
- **Tests** — `unit/test_clay_torch.py` (tiny injected encoder: shapes, batching,
  empty, missing-wavelengths, registry values — no 5 GB needed);
  `integration/test_clay_real.py` (gated on `WHIRLD_TEST_CLAY_CKPT`); a 10-band S2
  fixture.

### Deliberate deferral (documented, not silently shipped)
`time` and `latlon` metadata embeddings are passed as **zeros** in
`ClayTorchBackend._build_datacube`. The weights, pixel normalization, wavelengths,
and GSD are faithful; deriving true acquisition time + scene lat/lon is the next
step toward metadata-faithful embeddings.

---

## 2026-06-25T20:37:15Z — Pass 2: `serve` REST API (over the working embed pipeline)

**Scope chosen with the user:** build the **`serve` REST API** next — the
highest-value, lowest-risk slice. It wraps the embed pipeline from Pass 1, runs
fully offline, serves the PRD's Tier-3 user (QGIS/Node via HTTP), and addresses
the "serve adoption" success metric (PRD §6, §5.8). Real HF weights were
explicitly deferred because they can't be downloaded/verified in this environment.

**Verification at time of writing:** 71 tests passing (20 new), **93% coverage**;
`black --check` and `ruff check` clean. Verified two ways: FastAPI `TestClient`
suite, **and** a live `uvicorn` server driven by `curl` (`/health`, `/models`,
multipart `/embed` → `.npy` + decoded `X-Whirld-Chips-Meta` header, `/segment`
501).

### What was implemented this pass

#### Warm model session — `core/session.py` (new)
- `ModelSession` caches loaded backends keyed by `(model, device)`; `get()`
  lazy-loads + caches, `preload()` eager-loads at startup, `loaded` lists resident
  models, `clear()` drops them. `LoadedModel(entry, manifest, backend)` bundle.
  Reuses `Registry`, `load_manifest`, and `load_backend` — one loading path.
  (Also the foundation for a future public `whirld.Session`, still deferred.)

#### Read uploaded bytes — `io/raster.py` (extended)
- `read_raster_from_bytes(data, label)` via `rasterio.MemoryFile` (no temp file).
  Refactored the dataset→`RasterSource` extraction into a shared `_extract()` used
  by both `read_raster` and the new bytes reader — same validation/CRS rules.

#### One shared pipeline — `api.py` (refactored, low-churn)
- Extracted the post-resolve pipeline into a single private `_orchestrate()` that
  records `usage.jsonl` on success and failure. `embed(path, ...)` (unchanged
  public behavior) and the new `embed_raster(raster, *, model, session, ...)`
  (warm-backend, `write=False`, used by the server) both feed it. CLI, library,
  and HTTP now share exactly one code path; HTTP requests are logged to
  `usage.jsonl` too.

#### Server — `server/`
- `app.py` — `create_app(device, preload)`: holds a `ModelSession` in
  `app.state`; a lifespan handler preloads requested models at startup and clears
  on shutdown. A single `WhirldError` exception handler maps exit codes → HTTP
  (2→404, 3→404, 4→422, 7→422, 5→502, 6→500, else 500) with a
  `{"error", "detail"}` envelope.
- `schemas.py` — Pydantic `HealthResponse`, `ModelInfo`, `ModelsResponse`,
  `ErrorResponse` (no raw dicts across the boundary).
- `routes/health.py` — `{status, device, models_loaded, version}` (§6.1).
- `routes/models.py` — every registry model with `installed`/`loaded` flags +
  metadata (§6.2).
- `routes/embed.py` — `POST /embed` multipart upload (`file` + form fields
  `model`, `chip_size?`, `overlap?`, `sensor?`, `format?`). Default response is
  `.npy` bytes (`application/octet-stream`) with chip metadata in
  `X-Whirld-Chips-Meta` (base64 JSON); `format=json` inlines the array. A JSON
  (STAC) body → `400` (deferred); wrong content type / missing fields → `422`.
- `routes/segment.py`, `routes/classify.py` — present but return `501` (their
  backends are deferred), so the documented surface is complete and honest.

#### CLI — `cli/commands/serve.py` (+ registered in `cli/__init__.py`)
- `whirld serve --host --port --models <csv> --device` builds the app and runs
  `uvicorn.run`. Lazy-imports uvicorn; a missing `serve` extra raises a clear
  install hint. Stays off the CLI startup path (no fastapi/uvicorn import unless
  `serve` runs).

#### Dependencies & tests
- Added `python-multipart` to the `serve` extra; pinned `fastapi`, `uvicorn`,
  `python-multipart`, `httpx` in `requirements-dev.txt`.
- New tests: `unit/test_session.py`, `unit/test_raster_bytes.py`,
  `integration/test_server.py` (health, models, embed npy + json, 404/422/400/501
  paths, not-installed). Silenced Starlette's benign TestClient httpx
  deprecation warning.

### Notable decisions
- **`/embed` is multipart-upload-only this pass.** A server can't read a client's
  local path, and STAC URL input is deferred — so a JSON `input` body returns a
  clear `400` rather than pretending. Honest surface.
- **Refactor over duplication.** Rather than copy the pipeline for the server, the
  one-shot and warm paths were unified behind `_orchestrate()`, keeping a single
  source of truth for translation, chipping, inference, and usage logging.
- **`uvicorn.run` blocks**, so the `serve` command itself is covered only by app
  construction; all routes are fully exercised via `TestClient` + a live curl
  smoke test.

---

## 2026-06-25T17:56:17Z — Pass 1: Offline walking skeleton (`pull` + Clay `embed`)

**Scope chosen with the user (via clarifying questions):** an **offline reference
backend** and a **minimal walking skeleton**. The environment cannot download the
real ~847 MB Hugging Face weights and the `github.com/whirld/registry` repo is
fictional, so this pass delivers a real, fully tested, end-to-end vertical slice
that runs entirely offline and proves the architecture. Only the model *math* is
a deterministic stand-in; everything around it is production-shaped.

**Verification at time of writing:** 51 tests passing, 93% coverage
(`pytest --cov=whirld`); `black --check` and `ruff check` clean; the offline
five-minute-test analog (`whirld pull clay-v1` → `whirld embed`) succeeds via the
installed console script with exit 0.

### What was implemented this pass

#### Project scaffold & tooling
- `pyproject.toml` (setuptools, `src/` layout, console-script entrypoint
  `whirld = whirld.cli:app`, optional extras `hf` / `serve` / `stac`, black/ruff/
  pytest/coverage config), pinned `requirements.txt` + `requirements-dev.txt`,
  `.gitignore`, `.env.example`, `README.md`, `tasks/todo.md`, `tasks/lessons.md`.
- `.venv` built on Python 3.13.14 (per CLAUDE.md; the machine default `python3`
  is 3.9 and was not used).

#### Foundation (`src/whirld/`)
- `_version.py` — single source of version truth (`0.1.0`), import-cheap.
- `config.py` — `WhirldPaths` + `WHIRLD_HOME` resolution; the single source of
  truth for the `~/.whirld/{registry,models,logs}` layout (PRD §11). Exposes the
  packaged bundled-registry path and reads `WHIRLD_LOG_LEVEL` / `WHIRLD_STAC_TOKEN`.
- `errors.py` — `WhirldError` hierarchy with `exit_code` attributes mapped exactly
  to PRD §13 (2 not-in-registry, 3 not-installed, 4 unsupported-sensor, 5 network,
  6 checksum, 7 invalid-input, 8 OOM).
- `logging_setup.py` — rotating app log (`whirld.log`, 10 MB × 3) + INFO-to-stderr,
  verbosity from `--verbose`/`--quiet`/`WHIRLD_LOG_LEVEL`; structured local
  `usage.jsonl` writer (1 MB × 2) with the exact PRD §16.2 field set and **no**
  paths/CRS/geo/PII (error field is the exception class name only).

#### Registry (PRD §7, §8)
- `registry_data/schema/model.schema.json` — JSON Schema for model entries
  (the community-submission gate per §19), extended with a `reference` source type.
- `registry_data/models/clay-v1.yaml` — full Clay v1 entry: metadata, source,
  distribution (sha256 + size), **band contract** (sensors, aliases, target
  resolution, chip size, per-band z-score normalization, scale, nodata), output
  spec, hardware, provenance.
- `core/registry.py` — typed **Pydantic** models (`ModelEntry`, `BandContract`,
  `SensorContract`, `Normalization`, `Source`, `Distribution`, `OutputSpec`) with
  validators (aliases↔bands and mean↔std length checks); `Registry` loads,
  validates, lists, and **seeds the cache from the package-bundled registry** on
  first use (guarantees offline availability).

#### Pull / fetch (PRD §5.2)
- `core/fetch.py` — the `pull` contract: resolve entry → acquire artifact →
  **sha256 verify** (abort + delete on mismatch) → write `manifest.json`. Two
  source backends behind one interface:
  - `reference` (offline): deterministically materializes a tiny canonical
    weights blob from the model's seed + embed_dim; its sha256 is what the
    registry declares, so checksum verification is meaningful with no network.
  - `huggingface` (wired, **not exercised**): lazy `huggingface_hub` import +
    download + verify, ready to switch on.
  - Helpers: `sha256_file`, `is_installed`, `load_manifest`, `Manifest` model.

#### Translation pipeline (PRD §7 — Whirld's core contribution)
- `io/raster.py` — rasterio reader → `RasterSource` (data, CRS, transform, band
  descriptions, TIFF tags, resolution); rejects missing/CRS-less inputs.
- `core/sensor.py` — sensor detection with the full precedence chain: explicit
  `--sensor` override → TIFF tags (`IMAGEDESCRIPTION`/`SOFTWARE`) → band
  descriptions → resolution; actionable `UnsupportedSensorError` otherwise.
- `core/contract.py` — `translate()`: validate sensor → **select bands by spectral
  alias** (not index) → resample to target resolution (bilinear) → apply scale
  factor → per-band z-score normalize → `TranslatedRaster`.
- `core/chips.py` — `chip_raster()`: tile into `chip_size` tiles (optional
  overlap), pad edge tiles with nodata fill, compute each chip's CRS bounding box;
  returns the float32 array plus a Pydantic `ChipSet`/`ChipMeta`.

#### Model backend & output
- `models/base.py` — `ModelBackend` ABC (`embed`, plus `segment`/`classify`
  raising `NotImplementedError` as deferred capabilities) + `detect_device`
  (CUDA→MPS→CPU precedence, lazy torch, CPU fallback).
- `models/clay.py` — **Clay offline reference backend**: deterministic 512-dim
  embedding = seeded random projection of per-band pooled stats → `tanh`;
  numpy-only, reproducible, no torch. The real-encoder code path is documented
  inline for drop-in replacement.
- `models/loader.py` — name→backend factory (the only place new models register).
- `io/output.py` — `.npy` writer (shape `(n_chips, embed_dim)`) + `_meta.json`
  sidecar (model, version, CRS, embed_dim, chip size, resolution, per-chip
  bbox/row/col) per PRD §10.1; optional inline `json` format.

#### Orchestration, CLI, public API
- `api.py` — `pull()` and `embed()` wiring the whole pipeline, timing each run and
  appending a `usage.jsonl` record on **both success and failure**; returns
  `EmbedResult` (embeddings, chips, meta, sensor, device, output paths).
- `cli/` (Typer) — `pull`, `list`, `info`, `embed`, with a global
  `--verbose/--quiet` callback and **centralized error→exit-code handling**
  (`WhirldError` → documented process exit code + actionable stderr message).
- `__init__.py` — lazy public API (`whirld.pull`, `whirld.embed`,
  `whirld.EmbedResult`); `import whirld` imports no numpy/rasterio/torch.

#### Tests (PRD §17)
- `tests/fixtures/make_fixtures.py` — generates tiny synthetic Sentinel-2 L2A and
  no-CRS GeoTIFFs (no network, no large files).
- Unit: `test_registry`, `test_fetch` (incl. checksum-mismatch + idempotent pull),
  `test_sensor` (all precedence branches), `test_contract` (shapes, normalization,
  alias-order independence, resampling, unsupported sensor), `test_chips`,
  `test_clay_backend` (determinism, distinctness, empty batch, device).
- Integration: `test_embed_clay` (full Python-API pipeline, default output naming,
  json format, not-installed/STAC/no-CRS errors, usage-record contents),
  `test_cli` (every command + error exit codes via Typer's `CliRunner`).

### Bug found & fixed during the build
The app/usage loggers cached handlers globally pinned to the first `WHIRLD_HOME`,
so a changed home (long-running process, or test isolation) wrote logs to a stale
path. Fixed in `logging_setup.py` by rebinding handlers when the target path
changes.

### Deliberate deviations from the PRD
- **NAIP omitted from the Clay band contract.** The PRD lists NAIP (4-band) for
  Clay but pairs it with a 6-value normalization — an inconsistent/invalid
  contract. Dropped with an explanatory comment; the three six-band sensors
  (Sentinel-2 L2A, Landsat-8/9 L2) are mutually consistent. Re-add once a
  per-sensor normalization block exists.
- **Added a `reference` source type** to the registry schema to support the
  offline deterministic backend, alongside the real `huggingface` type.
- **`requires-python >= 3.10`** in packaging (matches PRD §15.4) even though the
  venv is 3.13.

---

## Remaining PRD work (not yet implemented)

Ordered roughly by the PRD. "Extension point in place" means the interface/seam
already exists and the feature slots in without reworking existing code.

### Models (PRD §4)
- [x] **Real Clay v1.5 weights** *(Pass 3)* — `clay-v1.5` runs the genuine encoder
      (1024-dim) via the `huggingface` source + `ClayTorchBackend`; pull verifies
      the real 5.16 GB checkpoint. The offline `clay-v1` reference remains for the
      no-network/fast path.
- [x] **Clay metadata fidelity** *(Pass 5)* — per-chip `lat/lon` (reprojected) +
      scene `time` (from `--datetime`/`TIFFTAG_DATETIME`) fed to the encoder, using
      Clay's exact formulas; verified the metadata changes the real embedding.
- [x] **RemoteCLIP (`remoteclip`)** *(Pass 4, scoring P6)* — real zero-shot
      text-driven classification (open_clip ViT-B/32); RGB contract; **logit-scale
      softmax probabilities**, single- or multi-query → GeoJSON.
- [x] **Prithvi EO 2.0 burn-scar** *(Pass 7)* — `prithvi-burn-scar` runs the real
      TerraTorch head → single-band mask GeoTIFF; HLS contract; `prithvi-eo-2
      --head burn-scar` alias. Single-image (T=1), not the PRD's before/after.
- [ ] **Prithvi flood** (`prithvi-flood`) — fast follow-up; needs its config's
      normalization. (Multi-temporal/before-after heads remain out of scope.)
- [~] **OlmoEarth** — *evaluated, declined* (see the 2026-06-27 decision entry).
      Fully runnable (768-dim embeddings, works on 3.13), but its custom
      use-restricted license doesn't fit Whirld's all-open posture. A clean drop-in
      under the registry-driven design if the posture ever changes.

### CLI commands (PRD §5)
- [x] `pull`, `list`, `info`, `embed`, `serve`, `classify`, `rm`
      *(serve P2, classify P4, rm P5)*
- [x] `embed` options: `--batch-size`, `--crs`, `--datetime` *(P5)*; `--no-warnings`
      + repeatable `classify --query` *(P6)*.
- [x] **`segment`** (PRD §5.6) *(Pass 7)* — `--model`/`--head`/`--threshold`/`--output`
      → single-band mask GeoTIFF. (Single input; multi-temporal deferred.)
- [ ] **`update`** (PRD §5.9) — refresh registry YAMLs from the remote repo.

### REST API (PRD §6) — `server/` package *(Pass 2)*
- [x] `GET /health`, `GET /models`
- [x] `POST /embed` (multipart upload → npy bytes + `X-Whirld-Chips-Meta`, or
      `?format=json`); models **warm-loaded and kept resident** via `ModelSession`;
      `WhirldError`→HTTP status mapping; per-request usage logging.
- [x] **`POST /classify`** *(Pass 4)* — multipart upload + `query` → GeoJSON
      FeatureCollection (warm `ModelSession`).
- [x] **`POST /segment`** *(Pass 7)* — multipart upload + `model`/`head` →
      `image/tiff` mask (warm `ModelSession`). Raster responses now exist.
- [ ] **STAC URL (JSON body) request handling** — currently returns a clear `400`.
      (The CLI/library STAC path landed in Pass 9; wiring it into the REST bodies is a
      small follow-up.)

### Band contract / sensors (PRD §7)
- [x] Alias-based selection, resample, scale, z-score normalize, chip, pad,
      sensor detection precedence.
- [x] `--crs` override for CRS-less inputs *(Pass 5)*.
- [ ] **NAIP** support (needs per-sensor normalization; see deviation above).
- [ ] Nearest-neighbor resampling for categorical data (only bilinear is wired).

### Registry architecture (PRD §8)
- [x] Local schema-validated loading + bundled seed.
- [x] **Registry-driven backend selection** *(refactor)* — each model declares a
      `backend:` id; `load_backend` dispatches by it (no name→backend dict). Adding a
      model that reuses an existing backend is **pure YAML**; only a new architecture
      needs a backend class + one dispatch branch.
- [x] **Registry governance** *(Pass 8)* — security + license posture as one policy
      (`core/governance.py`): `trust` tier, OSS-license allowlist, pickle/RCE refusal
      at pull (`SecurityError`, exit 9), closed backend allowlist (test-locked), and a
      machine-checkable auto-merge gate (`scripts/check_registry.py` + CONTRIBUTING).
      Settled *before* remote sync so the gate exists when community PRs arrive.
- [ ] **Remote registry sync** from `github.com/whirld/registry` (clone/pull into
      `~/.whirld/registry/`), the 24-hour auto-refresh window, `last_updated`
      tracking, and `--no-update` air-gapped flag.

### Input handling (PRD §9)
- [x] Local GeoTIFF (incl. COG).
- [x] **STAC item URL** input *(Pass 9)* — fetch the item JSON, select only the
      required band assets (alias-matched), read them with COG **range requests**
      (`/vsicurl/` + WarpedVRT) onto one grid, assemble a `RasterSource`. Optional
      `--bbox` windows the read. Library + CLI (`embed`/`classify`/`segment`); REST
      bodies still `400` (deferred).
- [x] **`--stac-token` / `WHIRLD_STAC_TOKEN`** bearer auth *(Pass 9)* — sent on the
      item fetch and the `/vsicurl/` asset reads (via `GDAL_HTTP_HEADERS`).
      Planetary-Computer URL *signing* remains a documented follow-up.

### Output handling (PRD §10)
- [x] Embeddings `.npy` + meta sidecar; optional inline JSON.
- [x] **Classification GeoJSON** FeatureCollection writer *(Pass 4)*.
- [x] **Segmentation GeoTIFF** writer (single-band uint8, LZW, nodata) *(Pass 7)*.

### Hardware / device (PRD §12)
- [x] Device auto-detection precedence (CUDA→MPS→CPU) + override.
- [x] CPU full-precision **runtime warning** + `--no-warnings` *(Pass 6)*. (Mentions
      int8 but the actual `--quantize` suggestion waits for quantization support.)
- [ ] **Quantized (int8) variants** — pull/convert/cache; `--quantize` at pull and
      inference time. *Currently raises a clear deferred-feature error.*

### Installation / packaging (PRD §15)
- [x] `pip install -e .`, console script, extras declared.
- [~] **Clean-machine five-minute test (PRD §17.3)** — both OS legs in place
      *(2026-06-29 entry)*, sharing one OS-agnostic smoke test (`scripts/clean_room_test.py`):
      **Linux x86** via `docker/clean-room.Dockerfile` (fresh `python:3.13-slim`, base
      deps only — surfaced + fixed the `libexpat1` syslib gap), and **macOS ARM** via
      `scripts/clean_room_macos.sh` (throwaway **Tart** VM, base deps only, no Homebrew
      GDAL — proves rasterio's bundled GDAL suffices on a clean Mac). Both run lazy
      import → pull → embed GeoTIFF → embed STAC item, plus an optional live Earth Search
      item. Still TODO: a published wheel, and CI wiring (macOS needs Cirrus CI or a
      self-hosted Apple-Silicon runner — Tart can't run on GitHub-hosted runners).
- [ ] Published distribution; `whirld[cuda]` validated; multi-version CI
      (3.10/3.11/3.12).

### Logging (PRD §16)
- [x] App log + `usage.jsonl` with the specified schema and privacy rules.
- [ ] Nothing outstanding for MVP scope; revisit opt-in telemetry post-launch (§19).

### Explicitly out of scope for MVP (PRD §2.2)
Fine-tuning/training, dataset management, GUI, cloud/managed inference, streaming
tiles, multi-model comparison, eval/benchmarking, auth/multi-user, Windows.
