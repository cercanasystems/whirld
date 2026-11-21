## Pass 11 conformance cleanup

- [x] ~~Replace cross-layer acquisition tag dictionaries with immutable Pydantic
      metadata while preserving datetime precedence, parsing and UTC behavior.~~
- [x] ~~Replace the untyped VRT handle with an explicit optional Rasterio handle and
      an unopened/closed-state guard; retain adapter dictionaries only at GDAL APIs.~~
- [x] ~~Complete parameter/result/error documentation for new nontrivial methods,
      and add a focused streaming guide under `docs/` linked from the README.~~
- [x] ~~Move new feature tests into source-mirroring core/io/CLI/API locations and
      verify collection, regression coverage, formatting and whole-package typing.~~
- [x] ~~Record final conformance evidence and remaining inherited exceptions before
      declaring this pass ready for PR review.~~

Scope: the new Pass 11 implementation and its tests/documentation. Keep the
existing API and numerical behavior; avoid repository-wide schema migrations.
The written plan is reviewed before implementation; execution remains in the
session's developer-selected Default mode.

### Conformance cleanup review

The new reader/coordinator boundary now uses frozen Pydantic `AcquisitionMetadata`.
Its resolve method owns timestamp precedence and UTC normalization; the legacy
raw-tag helper delegates to it for eager API compatibility. A small `_WarpedRaster`
protocol describes the Rasterio read operation, with explicit unopened/closed
handle errors. Nontrivial reader/coordinator/writer/benchmark methods document
arguments, results and failure conditions. The detailed guide lives at
[docs/streaming.md](../docs/streaming.md), linked from the README.

New tests now mirror `core/`, `io/`, and `cli/commands/`; public API tests remain at
the corresponding root. Added regression coverage verifies immutable metadata,
malformed-date fallback, override precedence/UTC, and handle lifetime guards.
Verification: **472 passed / 7 skipped, 93.09% coverage**. Black/Ruff, mypy across
57 source files, annotation/docstring inspection, collection, and diff checks pass.
Independent and dedicated compliance reviews found no new blockers. The latter
also caught and corrected stale README test counts. The new AGENTS.md compliance
reviewer requirement was adopted when it appeared; its user edit is preserved.
Earlier live COG and base-install
verification is recorded in the implementation review below; it was not repeated
for this metadata/docstring/layout cleanup.

Scoped exceptions remain: legacy eager schemas and older flat test files were not
migrated; GDAL keyword dictionaries remain inside external adapters, and Rasterio
still lacks full upstream type stubs. This is a scoped conformance improvement,
not a claim that every legacy repository convention now meets AGENTS.md. No new
dependencies or feature work on deferred issue #21. The implementation and
conformance cleanup are ready for publication together on the feature branch.

## Pass 11 implementation plan — 2026-11-21

- [x] ~~Re-read AGENTS.md and lessons; branch from merged main and reconcile the
      remediation status before starting feature work.~~
- [x] ~~Implement a resource-owning `StacWindowReader` on one common output grid.
      Read clipped windows, reuse band/mask/radiometry rules, normalize before
      padding, yield bounded batches, and explicitly limit GDAL/VSI/warp caches.~~
- [x] ~~Implement an `EmbeddingStreamWriter` owning staged output files. Write the
      known-shape NPY header and batch bytes sequentially; stream chip metadata
      into the existing JSON sidecar layout; clean up incomplete results.~~
- [x] ~~Implement an `EmbeddingStream` coordinator with explicit store/backend,
      reader, and writer ownership. Expose `embed_to_disk()` plus CLI `--stream`;
      return a typed disk-result summary without resident embeddings/chip lists.
      Keep ordinary `embed()` return behavior unchanged. Streaming supports STAC item
      input and NPY output only; invalid combinations fail before processing.~~
- [x] ~~Verify equivalence of pixels/embeddings and metadata for reordered/multiband
      assets, radiometry, masks/nodata, resampling, overlap, bbox and partial edges.
      Verify batch-local location/time metadata and a single backend load.~~
- [x] ~~Investigate live COG resampling differences between full and small-window
      reads, reproduce them hermetically, and fix deterministic sampling without
      whole-scene buffers or relaxed tolerances before completing verification.~~
- [x] ~~Verify real HTTP range traffic, bounded read requests and process peak RSS
      across increasing scene sizes at fixed chip/batch size. Run regression,
      coverage, formatting, lint, typing, packaged install and live COG checks.~~
- [x] ~~Update README and dev_progress.md with usage, measured memory/tolerances,
      validation results and remaining limits; record final review here.~~

Design contract: preserve the full-scene target affine grid, row-major chip order,
padded footprints, and existing NPY/JSON schemas. Memory may depend on model size,
source block size, band count, chip/batch size and bounded caches, never accumulated
scene pixels, embeddings or metadata. New data contracts use typed Pydantic models;
resource management and inference orchestration have explicit object owners.
Checkpoint/GPU integration remains opt-in. No REST streaming, direct remote TIFF,
classification or segmentation streaming in this pass.

### Pass 11 implementation review

Implemented and locally verified on `feat/bounded-memory-cog-inference` after
re-reading AGENTS.md and committing the documentation/plan first (`c2e0288`).
`StacWindowReader` owns source/VRT/cache lifetimes and lazy chip layout;
`EmbeddingStream` owns one model load and bounded execution; `EmbeddingStreamWriter`
owns staged sequential NPY and JSON sidecar output. New contracts are Pydantic
models. Pure UTC/location transforms are shared with the existing eager pipeline.
The public additions are `embed_to_disk()` / `DiskEmbedResult` and CLI `--stream`.

Live validation caught GDAL's per-chunk bilinear-scale estimation changing pixels
for fractional interior windows. Shared `StacWarpPolicy` supplies physical
source/target pixel ratios. Independent four-neighbor interpolation tests with
nonlinear 10/15/20 m sources verify the correction; eager and streaming outputs
match without relaxing tolerances. Aligned eager results can change where the
previous chunk-dependent kernel was incorrect. Streaming explicitly requires
same-CRS north-up assets; eager general reprojection is retained.

Final verification: **463 passed / 7 skipped, 93.07% coverage** on Python 3.13.7;
Black/Ruff, mypy over 57 source files, all bundled registry entries, dependency
consistency, and diff checks passed. Rebuilt base-only installation passed lazy
import, offline pull, GeoTIFF/STAC embedding and streaming CLI equivalence. The
2 live public COG tests passed separately for both Clay input conventions. Real
HTTP tests verify range/subset transfers and credential scope on lazy reads.
Failure/interrupt tests verify output preservation, cleanup, and usage reporting.

Measured reference-encoder peak RSS at chip 128/batch 4 for 2048², 4096², and 8192²
scenes: **146.9 / 147.6 / 151.9 MiB** for 256 / 1,024 / 4,096 chips. The benchmark
builds COGs before fresh measured processes; output grows while memory plateaus.
Source block dimensions, model/framework buffers, and a bounded multiple of GDAL
cache budgets remain memory factors. Output publication rolls back ordinary
failures but is not crash-atomic across two files; concurrent writers require
distinct destinations. Large real-checkpoint/GPU inference was not run. The
implementation and subsequent conformance cleanup are prepared for PR review;
hosted CI results remain separate from this local verification record.

## OOP alignment pass — PR #20

Keep the 18 verified fixes while assigning behavior and mutable state to cohesive
objects, as required by AGENTS.md. Preserve public Python/CLI/REST behavior and
lazy imports; retain simple pure functions where no state or domain owner exists.

- [x] ~~Restore registry validation ownership to `ModelEntry`; make the registry
      checker consume its document constructor instead of a free validation
      function, preserving schema checks and governance verdicts.~~
- [x] ~~Introduce a model store owning paths, registry dependencies, installation
      transactions, recovery, and artifact leases. Remove module-global mutable
      locking state and signature-inspecting decorators; retain thin compatibility
      functions for existing public cache operations.~~
- [x] ~~Introduce an application-owned inference service with explicit session and
      concurrency dependencies; keep HTTP routes as typed transport adapters.~~
- [x] ~~Move per-call batch configuration behind the backend interface, eliminating
      orchestration access to private backend state and preserving isolation.~~
- [x] ~~Close ownership-review gaps: bind relative cache homes once, compare exact
      STAC origins, and prevent RemoteCLIP architecture selectors from bypassing
      the registered artifact source with additional downloads.~~
- [x] ~~Review the remediation diff for remaining misplaced domain behavior, run
      regression/coverage and static checks, and update development documentation.~~

Acceptance: state and policies have explicit owners, dependencies are injectable,
existing regression coverage passes, and no extra inheritance hierarchy or facade
is introduced without a concrete responsibility. Work stays on
`fix/holistic-review-remediation`.

### OOP pass review

`ModelEntry` owns file/schema validation and fingerprints; `ModelStore` owns cache
transactions and binds registry/path identity through session and backend loading.
`InferenceService` owns application lifecycle and admission, with FastAPI dependency
injection. Immutable `InferenceOptions` carry batch policy without touching private
backend fields. `CheckpointReader`, `PrithviDeployment`, and `StacAccessPolicy` own
artifact trust, temporary deployment resources, and credential scope.

`CacheLock` retains only class-owned thread/PID-local lease bookkeeping for
same-thread reentrancy across independent stores; OS locks serialize the same home
across threads/processes. Different homes proceed independently. Pure transforms
and serializers remain functions; existing public cache helpers delegate to stores.
Review also fixed relative-root drift, explicit-port origin comparison, malformed
origin errors, and RemoteCLIP source-selector bypasses. RemoteCLIP now accepts only
the local `RN50`, `ViT-B-32`, and `ViT-L-14` definitions.

Validation on Python 3.13.7: **400 passed / 7 skipped, 92.46% coverage**. Black/Ruff,
mypy (53 source files), all bundled registry entries, dependency consistency, and
diff checks passed. The rebuilt base-only install passed lazy import, pull, local
GeoTIFF embedding, and local STAC embedding smoke tests. Separately, **2 live public
COG radiometry tests passed**. Regression tests cover cache isolation/locking,
service injection/lifecycle, batch isolation after failures and concurrent calls,
credential restoration, and architecture dispatch rejection. Large real-checkpoint
and GPU tests were not run. This pass is locally verified and included in PR #20;
hosted CI results below refer to the previous revision.

## Remediation roadmap — merged in PR #20

### PR #20 — CI follow-up

- [x] ~~Read all four failed checks and identify their shared setup failure.~~
- [x] ~~Pin a runner with a published Python 3.13.7 build and verify workflow syntax.~~
- [x] ~~Push the correction, resolve any subsequent failures, and verify remote checks.~~

Review: both jobs ran for push and pull-request events. All four failed before
dependency installation because `ubuntu-latest` selected Ubuntu 26.04, which has
no Python 3.13.7 build in the setup-python manifest. Both jobs now pin Ubuntu 24.04,
verified against the published x64 builds. All four checks passed for `9f88c3e`
([push run](https://github.com/cercanasystems/whirld/actions/runs/44903259658),
[PR run](https://github.com/cercanasystems/whirld/actions/runs/44903263698)). Linux
results: **347 passed / 7 skipped, 91.78% coverage**, 50 source files type-checked,
format/lint/schema checks and the fresh base-install smoke test all passed.

Address all 18 confirmed findings from the holistic review below before starting
new feature work. Work through R1–R5 in order, with regression tests accompanying
each fix. The remediation gate has passed. All 18 findings are implemented and
verified on
`fix/holistic-review-remediation` and merged into main in PR #20 (`f85153e`).
GitHub issues #2–#19 are closed. Pass 11 is now unblocked.

### R1 — Trust, filesystem, and credential boundaries

- [x] ~~**F01 / P1 — Safe checkpoint loading:** enforce trust and format at the
      deserialization boundary, use a real safetensors reader, and restrict pickle
      loading to explicitly trusted artifacts. Verify a renamed pickle cannot
      execute a harmless test reducer and genuine safetensors still load.
      Tracked in [#2](https://github.com/cercanasystems/whirld/issues/2).~~
- [x] ~~**F02 / P1 — Cache path containment:** validate model identifiers and enforce
      resolved cache containment for reads, writes, and deletion. Test traversal,
      absolute paths, and symlinks against isolated directories outside the cache.
      Tracked in [#3](https://github.com/cercanasystems/whirld/issues/3).~~
- [x] ~~**F03 / P1 — Upload driver restrictions:** accept only supported GeoTIFF
      uploads before reading pixels. Reject VRT and filename/MIME spoofing; prove
      uploaded references cannot read server files or initiate remote reads.
      Tracked in [#4](https://github.com/cercanasystems/whirld/issues/4).~~
- [x] ~~**F04 / P1 — Origin-scoped STAC authentication:** prevent bearer credentials
      from crossing origins through redirects or global GDAL headers. Define
      explicit authorization for asset origins and test with synthetic credentials
      on separate local HTTP origins.
      Tracked in [#5](https://github.com/cercanasystems/whirld/issues/5).~~
- [x] ~~**F14 / P2 — Registry validation:** enforce the shipped schema in both the
      registry checker and runtime ingestion, including safe identifiers and
      compatible backend/format declarations. Invalid fixtures must fail before
      approval, download, or model loading.
      Tracked in [#15](https://github.com/cercanasystems/whirld/issues/15).~~

**Acceptance:** the reproduced trust-boundary failures are covered by passing
regression tests, while supported trusted models, uploads, and authenticated STAC
reads continue to work.

### R2 — Spatial, spectral, missing-data, and temporal correctness

- [x] ~~**F05 / P1 — Required bands:** reject missing or ambiguous named bands;
      permit positional mapping only for wholly unlabeled inputs with a defined
      band order. Verify missing B02 fails before inference rather than reusing B03.
      Tracked in [#6](https://github.com/cercanasystems/whirld/issues/6).~~
- [x] ~~**F06 / P1 — CRS and resolution units:** reproject into a suitable metric
      grid or explicitly reject unsupported grids. Handle both pixel dimensions
      and affine transforms consistently; verify geographic and non-meter CRSs,
      anisotropic pixels, output resolution, and preserved spatial extent.
      Tracked in [#7](https://github.com/cercanasystems/whirld/issues/7).~~
- [x] ~~**F07 / P1 — Interoperable GeoJSON:** transform geometry to WGS84 longitude
      and latitude, retaining source CRS separately if needed. Round-trip through
      an independent GIS reader and check coordinates against known locations.
      Tracked in [#8](https://github.com/cercanasystems/whirld/issues/8).~~
- [x] ~~**F08 / P1 — Input validity:** preserve local TIFF and upload nodata/internal
      masks through resampling and normalization, consistently with STAC inputs.
      Exclude invalid observations from interpolation and apply the model's fill
      convention afterward; verify equivalent inputs across all three paths.
      Tracked in [#9](https://github.com/cercanasystems/whirld/issues/9).~~
- [x] ~~**F10 / P2 — Segmentation validity:** preserve class 0 as valid unburned data;
      use a distinct nodata representation and propagate actual invalid pixels.
      Verify masked raster reads retain both valid classes and exclude only nodata.
      Tracked in [#11](https://github.com/cercanasystems/whirld/issues/11).~~
- [x] ~~**F11 / P2 — Acquisition time:** preserve STAC ISO timestamps, support the
      documented TIFF timestamp format, and normalize timezone-aware instants to
      UTC with an explicit policy for naive timestamps. Verify equivalent instants
      produce identical time vectors and reach the backend correctly.
      Tracked in [#12](https://github.com/cercanasystems/whirld/issues/12).~~

**Acceptance:** independent expected values establish band order, pixel validity,
spatial coordinates, and acquisition time; tests check semantic correctness in
addition to array shapes and internal representations.

### R3 — Offline inference and installation integrity

- [x] ~~**F09 / P1 — Prithvi offline loading:** disable hidden pretrained-backbone
      downloads when loading the installed task checkpoint; explicitly pin and
      verify any artifacts that remain required. Test initialization with an empty
      external model cache and blocked network; document a real-checkpoint check.
      Tracked in [#10](https://github.com/cercanasystems/whirld/issues/10).~~
- [x] ~~**F15 / P2 — Atomic model installation:** stage and verify complete downloads
      before replacing an installation, and validate required artifacts when
      reporting installed status. Test interrupted downloads, checksum failures,
      missing files, and forced pulls; retain the previous usable installation on
      replacement failure and never report a broken installation as successful.
      Tracked in [#16](https://github.com/cercanasystems/whirld/issues/16).~~
- [x] ~~**F16 / P2 — Registry upgrades:** refresh bundled entries across application
      upgrades while preserving user entries under a documented conflict policy.
      Test older caches, new bundled models, schema changes, and actionable errors;
      remove guidance to invoke the nonexistent `whirld update` command.
      Tracked in [#17](https://github.com/cercanasystems/whirld/issues/17).~~

**Acceptance:** a complete installation works offline from a cold external cache,
failed updates leave a truthful and recoverable state, and upgrades expose newly
bundled models without requiring users to delete their registry cache.

### R4 — Public API, server behavior, and output contracts

- [x] ~~**F12 / P2 — Server responsiveness:** move blocking raster/model work off
      the async event loop with bounded concurrency and safe shared model/cache
      access. Verify health requests remain responsive during inference and test
      concurrent requests for state races and resource limits.
      Tracked in [#13](https://github.com/cercanasystems/whirld/issues/13).~~
- [x] ~~**F13 / P2 — Embedding response metadata:** move scene-sized chip metadata
      out of HTTP headers into a documented body/archive or separate resource.
      Define compatibility behavior and verify large responses with a real HTTP
      client/parser, including the reproduced 600-chip case.
      Tracked in [#14](https://github.com/cercanasystems/whirld/issues/14).~~
- [x] ~~**F17 / P2 — Request validation:** constrain chip size, overlap, batch size,
      bbox, thresholds, and supported formats at public boundaries. Reject invalid
      and nonfinite values consistently in Python/CLI and with HTTP 422 in REST;
      ensure OpenAPI reflects the accepted domains.
      Tracked in [#18](https://github.com/cercanasystems/whirld/issues/18).~~
- [x] ~~**F18 / P2 — Default output paths:** derive safe local defaults for URL
      inputs or require explicit destinations under a documented policy. Test
      HTTPS, file URLs, local paths, and query strings across output-producing
      commands; never copy credentials or URL directory structures into filenames.
      Tracked in [#19](https://github.com/cercanasystems/whirld/issues/19).~~

**Acceptance:** invalid requests fail predictably, supported responses work with
ordinary HTTP clients at scene scale, output locations are predictable, and
concurrent inference does not stall unrelated requests.

### R5 — Verification gate and review

- [x] ~~Resolve the whole-package mypy baseline (25 errors in 8 files) with scoped
      fixes and explicit third-party boundaries, rather than blanket suppression.~~
- [x] ~~Add CI for the full hermetic pytest suite with >=80% coverage, Black, Ruff,
      whole-package typing, and registry validation on supported Python 3.13.7.~~
- [x] ~~Verify fresh installation and cold-cache offline behavior; keep live-data
      and real-checkpoint checks explicitly gated and document their execution
      requirements and results separately from model-stand-in tests.~~
- [x] ~~Re-run all 18 finding reproductions as permanent regression tests, review
      the final changes for regressions, and record remaining validation limits.~~
- [x] ~~Update user documentation for changed contracts and compatibility behavior,
      record remediation results here, and only then unblock feature work.~~

**Acceptance:** all F01–F18 tasks and the verification gate are complete, required
checks pass, and any limitations of real-model/platform validation are recorded.
Mark finished tasks checked and strike through their full descriptions only after
verification.

### Review

Implemented F01–F18 on `fix/holistic-review-remediation`, with permanent regression
coverage for every finding. Final Python 3.13.7 suite: **347 passed / 7 skipped,
91.78% coverage**. Black/Ruff, whole-package mypy (**50 source files**), bundled
registry validation, dependency compatibility, and diff checks pass. The two
opt-in live Sentinel-2 radiometry checks also pass without model checkpoints.

A fresh base-only installation in an isolated `.venv` passes lazy import, reference
pull, local GeoTIFF inference, and file-backed STAC inference with empty application
and external-model caches. Independent pyogrio inspection confirms WGS84 GeoJSON
coordinates. Real loopback HTTP/GDAL tests verify credential origin boundaries;
HTTP parser and concurrent-request tests verify bounded headers and responsive
health requests. CI now runs checks and a separate base-install smoke job.

Independent review also caught and fixed interrupted-backup resurrection during
`rm --all`, reserved artifact names overwriting install metadata, status requests
blocking the event loop, and manifest/artifact races during backend loading.

Compatibility changes: REST embeddings default to an archive with complete
metadata; explicit raw `.npy` has no chip header. Unsupported CRS units/orientations
and incomplete named bands fail explicitly. GeoJSON is WGS84; mask nodata is 255;
timestamps use UTC; tokens are origin-scoped; invalid option domains return 422.
README documents these contracts and registry/cache upgrade behavior.

Validation limits: real Clay/RemoteCLIP/Prithvi checkpoint inference and GPU runs
remain gated and were not executed. Prithvi's offline regression verifies the
configuration boundary and actual download trigger; its real-checkpoint test now
blocks network access with an empty external cache when explicitly enabled.
Fresh install validation ran on this macOS host and subsequently on Linux GitHub
Actions after the runner correction recorded above. macOS VM execution remains
unverified. The historical review evidence below is retained unchanged.

### GitHub issue tracking — 2026-11-21

- [x] ~~Check existing repository issues for duplicate findings.~~
- [x] ~~Create one issue per finding F01–F18 with evidence and acceptance criteria.~~
- [x] ~~Link each remediation task to its issue and verify the complete mapping.~~

Tracking review: GitHub issues #2–#19 are closed after PR #20 merged. Read-back
verification confirmed all 18 titles, bodies, and task links match.

## Holistic code review — 2026-11-21

- [x] ~~Assess architecture, API/CLI contracts, packaging, and verification setup.~~
- [x] ~~Review geospatial transformations, model inference, and output semantics.~~
- [x] ~~Review security boundaries, acquisition/cache integrity, and server behavior.~~
- [x] ~~Reproduce suspected defects, run baseline checks, and prioritize findings.~~
- [x] ~~Record actionable findings and validation limits without implementation changes.~~

### Review

Reviewed main `85894cf` holistically. Production files remain unchanged; temporary
reproductions used isolated caches, synthetic credentials, tiny rasters, and model
stand-ins. No large checkpoint downloads, GPU inference, or clean-room VM runs.

Baseline: **218 passed / 7 skipped, 89.04% coverage**; Black/Ruff pass. Targeted
mypy passes for the five Pass 10 source modules, but whole-package mypy (excluding
vendored code, ignoring missing imports) reports **25 errors in 8 files**. Existing
tests often assert shape/internal representation or inject model stand-ins; they
miss trust-boundary, GIS interoperability, cold-cache, and missing-data behavior.

Confirmed findings (review results, not implemented fixes):

- **P1:** community `.safetensors` filename passes governance, but Clay loads it
  through unrestricted pickle (`models/clay_torch.py:146`); harmless reducer ran.
- **P1:** unvalidated model names permit recursive deletion outside the cache
  (`core/fetch.py:161`); reproduced on an unrelated temporary directory.
- **P1:** uploaded bytes accept VRT references to server-side rasters instead of
  enforcing GeoTIFF (`io/raster.py:127`); a separate private TIFF was read.
- **P1:** STAC bearer auth follows cross-origin redirects and is applied globally
  to asset reads (`io/stac.py:151`, `:492`); reproduced with synthetic credentials.
- **P1:** missing named bands silently fall back to positions, duplicating wrong
  spectral channels (`core/contract.py:181`); missing B02 selected B03 twice.
- **P1:** CRS units are treated as meters (`core/contract.py:221`); a 100x100
  EPSG:4326 scene at 0.0001 degrees became 1x1 at 10 degrees.
- **P1:** classification GeoJSON writes projected coordinates with a string CRS
  (`io/output.py:185`); pyogrio read UTM meter bounds as EPSG:4326.
- **P1:** local TIFF nodata/masks are ignored during normalization/resampling
  (`io/raster.py:166`, `core/contract.py:114`); missing pixels became extreme inputs.
- **P1:** Prithvi initialization downloads an unpinned additional backbone before
  loading the pulled task checkpoint (`models/prithvi.py:56`), breaking cold-cache
  offline inference; intercepted the actual download call without fetching weights.
- **P2:** segmentation outputs use valid class 0 as nodata (`io/output.py:249`),
  hiding unburned pixels in masked reads.
- **P2:** STAC ISO dates are discarded by the TIFF-only tag parser; equivalent
  explicit timezone offsets also produce different Clay time vectors
  (`api.py:960`, `:965`; `models/_clay_metadata.py:36`).
- **P2:** synchronous raster/inference work blocks all async server requests
  (`server/routes/embed.py:74`, analogous classify/segment routes).
- **P2:** per-chip metadata in one response header grows beyond client limits
  (`server/routes/embed.py:94`); 600 chips produced a 71,456-byte header rejected
  by Python's standard HTTP client.
- **P2:** registry checker does not enforce shipped JSON Schema
  (`scripts/check_registry.py:43`); invalid path/category/backend/source passed
  with AUTO-MERGE and exit 0.
- **P2:** failed forced pull can delete weights but leave an installed manifest;
  ordinary pull then falsely succeeds (`core/fetch.py:256`, `:381`).
- **P2:** any cached YAML prevents newly bundled registry entries from appearing
  after upgrades (`core/registry.py:269`); suggested `whirld update` does not exist.
- **P2:** request domains are unvalidated: negative chip/overlap returns HTTP 500,
  while zero chip size and unknown output format return 200
  (`server/routes/embed.py:67`, `core/chips.py:83`, `api.py:627`).
- **P2:** default STAC output paths treat URLs as filesystem paths
  (`api.py:835`, `:800`); a run created `https:/example.test/catalog/...` locally.

Recommended order: enforce trust/path/input boundaries; fix spatial, band, nodata,
and temporal correctness; repair offline/cache and server contracts; then Pass 11.
Promote the reproductions into regression tests when implementing fixes, and add
automated CI checks for coverage, whole-package typing, and cold/offline loading.

## Code walkthrough — 2026-11-21

- [x] ~~Trace public entry points and shared inference pipeline.~~
- [x] ~~Verify model behavior and output formats against implementations and tests.~~
- [x] ~~Explain architecture, one concrete execution path, and key caveats with source links.~~

### Review

Verified the walkthrough against public API, shared orchestration, preprocessing,
chipping, model loading, output writers, server session, and existing integration
test assertions. Highlighted the reference/real Clay distinction and positional
band fallback. No application code changed. Tests were not executed: the project
has no `.venv/bin/python`; verification for this explanation was source inspection.

# Whirld MVP — Walking Skeleton (Clay embed, offline)

Scope chosen with the user: **offline reference backend** + **minimal walking skeleton**.
Full plan: `~/.claude/plans/melodic-popping-map.md`.
Per-pass details and remaining PRD work: `dev_progress.md`.

## Pass 10 — STAC asset scale/offset handling (done)

Scope: honor per-band STAC asset radiometric metadata while preserving the units
expected by each model's normalization contract.

Implementation plan (branch `feat/stac-asset-offsets`): retain typed asset-band
metadata and band indices; convert `stored * scale + offset` to canonical DN
using an explicit sensor reflectance scale before existing model normalization.
Keep legacy behavior when no transform is declared, and compatibility defaults
only for existing sensor keys in older registry caches. Exclude nodata during
resampling and fill invalid pixels after normalization. Verify synthetic encoding
equivalence, a pinned live Sentinel-2 sample, the full suite with >=80% coverage,
format/lint checks, and static typing of the changed modules.

- [x] ~~Retain each selected asset band's scale, offset, and nodata metadata in
      the input pipeline; define defaults and validation for missing/invalid values.~~
- [x] ~~Define the conversion from stored pixel values to model input units,
      accounting for asset scale/offset and model normalization exactly once.
      Preserve Clay's expected DN convention and reflectance-based contracts;
      do not blindly apply asset scaling followed by the existing model scaling.~~
- [x] ~~Preserve nodata masks through conversion and resampling so offset values
      do not turn missing pixels into valid observations.~~
- [x] ~~Test nonzero offsets, differing per-band values, absent metadata, already
      corrected assets, and equivalent imagery encoded with different scale/offset
      values; verify expected normalized inputs and unchanged legacy behavior.~~
- [x] ~~Validate a pinned real Sentinel-2 STAC sample against independently computed
      expected values and document the supported metadata/unit conventions.~~

**Acceptance:** equivalent physical observations produce equivalent model inputs
within documented numerical tolerances, with scale/offset applied exactly once
and nodata preserved.

### Review

Implemented typed per-band metadata for Raster extension v1/v2, explicit canonical
DN conventions, legacy-cache compatibility, and nodata-aware conversion/resampling.
A lazy single-band VRT preserves internal masks and prevents other bands from
making a selected band's nodata appear valid during interpolation.

Verification on Python 3.13.7: **218 passed / 7 skipped, 89.04% coverage** (>=80%
gate); 48 new unit cases. The two opt-in live Sentinel-2 tests also pass without
model checkpoints: stored blue pixel 4636 -> reflectance 0.3636 -> Clay DN 3636.
Black/Ruff pass, mypy passes for all five changed source modules, and the installed
dependency set is consistent. Large-checkpoint tests remain opt-in.

Fresh-environment fixes: Typer 0.25.1 satisfies the pinned Hugging Face dependency;
Rasterio 1.5.2 satisfies rioxarray; TorchGeo 0.9.0 retains the trainers API required
by TerraTorch (the Prithvi extra now caps TorchGeo below 0.10). Added pinned mypy
and PyYAML typing stubs. README documents units, precedence, and supported metadata.

## Pass 11 — Bounded-memory COG inference (implemented; locally verified)

**Prerequisite complete:** PR #20 merged; all four PR checks passed and the
remediation issues are closed. Branch: `feat/bounded-memory-cog-inference`.

Scope: embeddings first, using COG assets referenced by STAC items. Extend the
existing remote range reads through preprocessing, inference, and output so a
large scene does not require loading the full region or all chips into RAM.

- [x] ~~Add a windowed reader yielding configurable chip batches on the model's
      common output grid, with optional bbox and existing asset authentication.~~
- [x] ~~Preserve band selection/order, resampling, normalization, overlap, edge
      padding, chip order/bounding boxes, and location/acquisition-time metadata.~~
- [x] ~~Reuse one loaded backend across batches and release each batch's pixel
      arrays after inference; bound GDAL caches as well as application buffers.~~
- [x] ~~Write embeddings and chip metadata incrementally, preserving the existing
      `.npy` and metadata sidecar formats without accumulating all results in RAM.~~
- [x] ~~Expose streaming explicitly for disk output in the CLI/Python API while
      preserving existing in-memory Python return behavior; document usage and
      memory expectations in the README.~~
- [x] ~~Verify streaming against the existing pipeline: equivalent embeddings
      within documented numerical tolerances, matching chip order/coordinates,
      bbox reads, overlap, resampling, and partial edge tiles.~~
- [x] ~~Verify HTTP range reads with a controlled COG fixture and measure peak
      memory across increasing scene sizes at a fixed chip/batch size.~~

**Acceptance:** larger scenes take longer and produce more output, but working
memory stays within a documented bound determined by model size, chip/batch size,
and bounded I/O caches rather than total scene area or chip count.

**Follow-up:** direct remote `.tif` URL input; extend streaming to classification
and segmentation after the embedding path is verified.

- [ ] **Low priority — deferred, no near-term implementation:** optional
      reprojection for cross-CRS or rotated STAC assets. Consider an opt-in,
      disk-backed normalization stage with an explicit target grid; retain
      numerical reproducibility and bounded memory. Outside Pass 11, with no
      scheduled milestone. Tracked in
      [#21](https://github.com/cercanasystems/whirld/issues/21).

### Review

Implemented and verified; see the Pass 11 implementation review at the top of
this file for architecture, numerical correction, measurements and limits.

## Pass 9 — STAC item URL input (done)

- [x] ~~`io/stac.py` `read_stac_item`: stdlib fetch (no new dep; local/file:// offline,
      http(s) + bearer), sensor resolve (override → infer → ask), 3-tier band→asset
      match (native id → eo:bands name → common_name/alias), `/vsicurl/` + WarpedVRT
      range reads onto one grid, datetime + platform stamped into tags~~
- [x] ~~`api.embed/classify/segment`: shared `_read_input` routes URL/`.json`/`file://`
      to the STAC reader (lazy import); `--bbox` + `--stac-token` threaded;
      `usage.jsonl` records `input_type=stac`~~
- [x] ~~CLI `--bbox` + `--stac-token` on all three commands (`cli/commands/_stac.py`
      parser + `WHIRLD_STAC_TOKEN` fallback); help text path-or-URL~~
- [x] ~~`pyproject` `stac` extra slimmed to `pystac-client` (reserved for future search);
      item path needs no extra~~
- [x] ~~Fixtures `make_stac_item` (file:// per-band COGs, native- and common-name-keyed)
      + tests: `unit/test_stac`, `integration/test_stac_input`, gated `test_stac_real`
      (`WHIRLD_TEST_STAC_URL`); **170 passed / 5 skipped, 89% cov** (stac.py 91%), lint clean~~
- [x] ~~**Verified:** `embed <item.json>` → (4,512) + `input_type=stac`; `--bbox` windows
      300×300→209×227 (4→1 chips); bad `--bbox` → exit 7; unreachable URL → NetworkError~~
- [x] ~~**Clean-room container** (`docker/clean-room.Dockerfile` + `scripts/clean_room_test.py`):
      fresh python:3.13-slim, base deps only, no system GDAL → lazy import, pull, embed
      GeoTIFF + STAC item; **real Earth Search item via live /vsicurl/** range reads.
      Found+fixed the `libexpat1` syslib gap. Advances PRD §17.3 acceptance gate.~~
- [x] ~~**macOS clean-room** (`scripts/clean_room_macos.sh` + `_clean_room_macos_provision.sh`):
      throwaway **Tart** VM, base deps, **no Homebrew GDAL**, runs the same
      `clean_room_test.py` → the macOS ARM half of §17.3. shellcheck-clean; local
      on-demand. (Run requires `brew install cirruslabs/cli/tart hudochenkov/sshpass/sshpass`.)~~
- Deferred: REST STAC bodies (still 400), multi-item search/mosaic, PC URL signing,
  clean-room CI wiring (Cirrus / self-hosted Apple-Silicon runner for the macOS leg)

## Pass 8 — Registry governance: security + license posture (done)

- [x] ~~`core/governance.py` — OSS-license allowlist + `is_oss_license`;
      `weights_are_pickle` (safetensors/reference safe, pickle + unknown unsafe);
      `TRUSTED_SOURCE_ORGS` + `source_org`; `evaluate_automerge` →
      `AutomergeDecision(auto, reasons)`. Pure stdlib (lazy-import safe).~~
- [x] ~~`SecurityError` (exit code **9**); `fetch.pull` runtime gate refuses a
      **community + pickle** entry before download (first-party pickle unaffected)~~
- [x] ~~`trust` field (`first-party|community`): schema-required, `ModelEntry`
      optional-at-parse; 4 bundled YAMLs declare `trust: first-party`; license desc → SPDX~~
- [x] ~~CLI: `pull` shows license + ⚠ non-OSS terms warning *before* download; `info`
      annotates `(OSS)`/`(non-OSS — review terms)` + trust tier~~
- [x] ~~Closed backend allowlist locked by test (module-path ids rejected, nothing
      imported); `scripts/check_registry.py` (exit 0/1/2) + `registry_data/CONTRIBUTING.md`
      merge policy (OlmoEarth = declined worked example)~~
- [x] ~~Tests: governance unit, loader allowlist, pull `SecurityError`, bundled-entry
      invariants, CLI/checker integration; **156 passed / 4 skipped, 88% cov**, lint clean~~
- [x] ~~**Verified:** `info clay-v1.5` → `Apache-2.0 (OSS)` / `first-party`; checker over
      bundled → review (exit 1), crafted community-OSS-safetensors → auto-merge (exit 0);
      community-pickle pull → exit 9~~
- Deferred: remote registry sync (the gate now exists for when community PRs arrive)

## Pass 7 — Prithvi segmentation, burn-scar first (done)

- [x] ~~`prithvi` extra (terratorch); `prithvi-burn-scar.yaml` (real coords/sha256,
      HLS contract, mask output); schema `config_file` + `output.classes`~~
- [x] ~~New output modality: `chips.reassemble_mask` + `io/output` mask GeoTIFF
      (+bytes); `fetch` multi-file download (weight + config)~~
- [x] ~~`models/prithvi.py` `PrithviBackend.segment` (TerraTorch, lazy); `base.segment`;
      loader; `api.segment`/`segment_raster` + `SegmentResult`; `--head` alias~~
- [x] ~~`segment` CLI; real `POST /segment` (`image/tiff`); public `whirld.segment`~~
- [x] ~~Tests (fake-segmodel unit, reassembly/GeoTIFF unit, API/CLI/server integration,
      gated real) + `make_hls` fixture; 135 passed / 4 skipped, 88% cov, lint clean~~
- [x] ~~**Real proof:** `whirld pull prithvi-burn-scar` (sha256 ✓) + `segment` → real
      uint8 mask GeoTIFF; `--head` alias + `/segment` over HTTP verified~~
- Deferred: `prithvi-flood` (follow-up); multi-temporal heads

## Pass 6 — Finish the paydown: CPU warning + RemoteCLIP softmax scoring (done)

- [x] ~~CPU full-precision runtime warning (`api._maybe_warn_cpu` in `_run`) +
      `--no-warnings` on embed/classify (reference/quantized/non-CPU stay silent)~~
- [x] ~~RemoteCLIP scoring → `logit_scale` softmax probabilities; multi-query support
      (`classify --query` repeatable, `/classify` getlist); per-query `scores` in
      GeoJSON; single query softmaxed vs a neutral `background` prompt~~
- [x] ~~`ModelBackend.classify` → `queries: list[str]`, returns `(n, n_queries)`~~
- [x] ~~Tests (softmax/probabilities, multi-query, cpu-warning, geojson, server,
      gated real); 117 passed / 3 skipped, 89% cov, lint clean~~
- [x] ~~**Real proof:** single-query 0–1 prob (0.089 vs background); 3-query softmax
      sums to 1.0; CPU warning fires for clay-v1.5, silent with `--no-warnings`~~

## Pass 5 — Deferral paydown: Clay metadata fidelity + CLI completeness (done)

- [x] ~~Clay metadata: `models/_clay_metadata.py` (Clay's exact normalize formulas);
      `InferenceContext` + `api._run` derive per-chip lat/lon (reprojected) + scene
      time (`--datetime`/`TIFFTAG_DATETIME`); `clay_torch` datacube uses them~~
- [x] ~~`rm` command (`rm <model>` / `rm --all`) + `core/fetch` remove helpers~~
- [x] ~~`--crs` override for CRS-less inputs; `--batch-size` for embed~~
- [x] ~~Tests (metadata formulas, reprojection, rm, crs, batch-size) + extended gated
      real test (metadata changes the embedding); 109 passed / 3 skipped, 89% cov~~
- [x] ~~**Real proof:** `--datetime` changes real Clay embedding (Δ 0.018); `--crs`
      embeds a CRS-less S2; `rm`/`rm --all` clear models, registry intact~~

## Pass 4 — RemoteCLIP `classify` (second real model, done)

- [x] ~~`remoteclip.yaml` (real `chendelong/RemoteCLIP` mirror, RGB CLIP contract,
      `model_name: ViT-B-32`); optional `model_name` schema/registry field~~
- [x] ~~`models/remoteclip.py` `RemoteCLIPBackend.classify` (open_clip, injectable
      model for tests); GeoJSON writer in `io/output.py`~~
- [x] ~~`api.py` refactor → one generic `_run`; `classify` + `classify_raster` +
      `ClassifyResult`; `classify` CLI + real `POST /classify` route + public API~~
- [x] ~~Tests (fake-model unit + integration + server + gated real); 91 passed /
      2 skipped, 88% cov, lint clean~~
- [x] ~~**Real proof:** `whirld pull remoteclip` (sha256 ✓) + `whirld classify
      --query "solar farm"` → GeoJSON cosine scores; `/classify` over HTTP~~
- [x] ~~Fixed server double-load bug (warm device reuse)~~
- Deferred: ~~multi-prompt softmax scoring~~ (completed in Pass 6); S2→8-bit tone mapping fidelity

## Pass 3 — Real Clay v1.5 weights (proof-of-load, done)

- [x] ~~Vendored Clay encoder (Apache-2.0) → `models/_vendor/clay_v15/`
      (+ LICENSE/NOTICE/PROVENANCE; black/ruff/coverage excluded)~~
- [x] ~~Real `clay-v1.5.yaml` (huggingface source, real sha256, 10-band contract,
      1024-dim); schema + registry gained optional `wavelengths`/`patch_size`~~
- [x] ~~`models/clay_torch.py` `ClayTorchBackend`; `InferenceContext` threaded
      through `api`; `loader` + `fetch` wiring~~
- [x] ~~Tiny-encoder unit tests + gated real-weights test + 10-band fixture~~
- [x] ~~**Real proof:** `whirld pull clay-v1.5` (sha256 ✓) + `whirld embed` →
      real `(n, 1024)`, 265 tensors / 0 missing; 77 passed/1 skipped, 91% cov,
      lint clean~~
- Deferred: ~~faithful `time`/`latlon` metadata~~ (completed in Pass 5)

## Pass 2 — `serve` REST API (done)

- [x] ~~`core/session.py` (`ModelSession` warm-load cache) + `io/raster.py`
      `read_raster_from_bytes`~~
- [x] ~~`api.py` refactored to one shared pipeline (`embed` + `embed_raster`)~~
- [x] ~~`server/` FastAPI app: `GET /health`, `GET /models`, `POST /embed`;
      `segment`/`classify` → 501; `WhirldError`→HTTP mapping~~
- [x] ~~`whirld serve` CLI command (lazy uvicorn import)~~
- [x] ~~Tests (TestClient + live curl smoke); 71 passing, 93% coverage, lint clean~~

## Pass 1 — Walking skeleton plan

- [x] ~~**1. Scaffold** — `pyproject.toml`, `requirements*.txt`, `.gitignore`,
      `.env.example`, `README.md`, `.venv`, `tasks/`~~
- [x] ~~**2. Foundation** — `config.py` (WHIRLD_HOME paths), `errors.py` (exception
      hierarchy + exit codes), `_version.py`, `logging_setup.py` (app log + usage.jsonl)~~
- [x] ~~**3. Registry** — `registry_data/schema/model.schema.json`,
      `registry_data/models/clay-v1.yaml`, `core/registry.py` (Pydantic models, load,
      validate, cache seed)~~
- [x] ~~**4. Pull/fetch** — `core/fetch.py` (offline reference source, sha256 verify,
      manifest.json; HF path wired but not exercised)~~
- [x] ~~**5. Translation pipeline** — `io/raster.py`, `core/sensor.py`,
      `core/contract.py`, `core/chips.py`~~
- [x] ~~**6. Model + output** — `models/base.py` (ABC), `models/clay.py` (reference
      backend), `models/loader.py`, `io/output.py` (npy + meta sidecar), `api.py`~~
- [x] ~~**7. CLI + Python API** — `cli/` (`pull`, `list`, `info`, `embed`),
      `src/whirld/__init__.py` (lazy public API)~~
- [x] ~~**8. Tests** — fixture generator, unit + integration, `pytest --cov` = 93%~~
- [x] ~~**9. Polish** — black + ruff clean, end-to-end offline five-minute test, README,
      review section below~~

## Deferred (extension points left in place)

~~Prithvi/`segment`~~, ~~RemoteCLIP/`classify`~~, ~~`serve`/REST API~~,
`update`/~~`rm`~~, ~~STAC input~~, quantization,
~~real Hugging Face weight loading~~.

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

**Next iteration (deferred, extension points in place):** ~~real HF weight loading~~,
~~Prithvi `segment`~~, ~~RemoteCLIP `classify`~~, ~~`serve` REST API~~,
`update`/~~`rm`~~, ~~STAC input~~, quantized variants.
