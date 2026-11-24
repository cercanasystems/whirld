## Pass 19 — First real result: setup, doctor and packaged RN50 demo

Branch: `feat/first-run-onboarding`, from main `7ba2f4c`. PR #31 remains open;
this branch is independent of evaluation work. Re-read AGENTS.md/lessons, verified
branch before edits and assigned dedicated compliance plus demo/diagnostics work.
User approved the onboarding recommendation. No version/tag/GitHub/PyPI release
or automatic merge; publication requires separate approval.

- [x] ~~Package a small attributed Sentinel-2 RGB sample and a manifest with exact
      bytes/SHA, source item/ROI/date, canonical-DN conversion, native band order,
      grid and license. Use a reproducible preparation script; preserve masks and
      require a standalone GTiff. Bundle the small derived data, so first-run
      imagery needs no login, STAC request or separate asset publication.~~
- [x] ~~Promote shared preview/score objects from examples into installed package
      code with compatibility re-exports; preserve all three example entrypoints.
      Add an object-owned CPU RN50 DemoRunner returning typed artifacts/provenance,
      using preflight, existing model acquisition/inference and self-contained
      HTML/PNG/GeoJSON. Missing weights may be downloaded; broken runtime or stale
      cache must fail before large acquisition. Reuse verified models. Reserve a
      fresh output directory, scope runtime state and clean owned failures/interrupts.~~
- [x] ~~Add typed doctor diagnostics and installed CLI commands with lazy imports.
      Report interpreter, selected runtime import health, effective model contract,
      actual cached identity, cache permissions and disk estimate. Missing weights
      is a ready-to-download warning, not a runtime failure. Display exact active-
      environment repair commands and upfront model size; install no dependencies
      implicitly and never seed/modify registry as a side effect of diagnosis.~~
- [x] ~~Provide a minimal, exact-version starter dependency profile and one tested uv
      .venv/Python3.13.7 path, with pip alternative. Avoid developer tools, other
      model backends and a repository checkout. Test CPU-friendly PyTorch install
      selection. Prepare distribution metadata/docs without uploading anything.
      Until publication, use an explicitly pinned source archive containing these
      commands; published 0.3.0 lacks them and must not be presented as the installer.~~
- [x] ~~Rewrite README around install -> real demo -> visible result -> own imagery,
      include an actual screenshot, input/result interpretation, download/runtime
      expectations and actionable troubleshooting. Keep contributor setup and the
      deterministic reference smoke path distinct. Maintain TODO/dev_progress.~~
- [x] ~~Verify offline behavior/failure boundaries with meaningful unit/integration
      tests; run actual RN50 weights, independently inspect scores/geometry/source
      identity and render the installed preview. Measure cold and warm paths with
      dependencies/model acquisition clearly separated; promise no arbitrary time.
      Test fresh installed package outside checkout, base-only lazy help/diagnosis,
      supported Mac/Linux starter paths, full >=80% coverage/style/type/registry and
      dedicated final review. Commit/push matching upstream and open a manual PR.~~

Acceptance: a user following the stated preview installation can run doctor and a
real-model demo from a fresh directory, without Git/source checkout or dev packages,
and open a self-contained preview plus GIS-compatible output. All sample/model
identities and setup limits are explicit; unavailable optional runtimes give useful
same-environment fixes before large downloads. Existing inference APIs and examples
remain compatible. No release/version bump or new model support in this pass.

### Review

Completed implementation on the matching upstream and opened
[PR #32](https://github.com/cercanasystems/whirld/pull/32) for manual review/merge.
The exact pinned uv and pip source-archive commands both pass in fresh external
Python3.13.7 `.venv`s, without Git/editable imports or developer/server/Prithvi
packages. A fresh base-only wheel also proves lazy CLI/help, actionable optional-
runtime failures and no network/cache mutation. Sample/notice/profile resources
match their source hashes. Actual installed RN50 runs pass on macOS ARM64 and
Linux x86_64 under emulation; the minimal Linux image needs only `libexpat1`,
which doctor diagnoses explicitly. Network-disabled Linux inference matches
macOS geometry/ranks/identities, with maximum probability error 3.85e-6. Cold and
warm timings and cache conditions are recorded in `docs/quickstart.md`.

Independent verification confirms normalized finite scores, exact WGS84 chip
footprints and source/checkpoint/registry identities; repeated macOS TIFF,
GeoJSON and HTML/PNG outputs are byte-identical. The actual preview screenshot
was inspected. The own-image streaming command executes successfully. Existing
example compatibility tests pass. Final full suite: **1052 passed / 12 skipped,
92.71% coverage**; Black (195 files), Ruff, mypy (82 files), seven registry entries
and diff checks pass. Final compliance review found no blocker. An initial
workflow-context error was corrected and actionlint1.7.12 now validates CI.
Hosted check status remains separate from local verification. No release/version
change, upload or automatic merge.

Native Ubuntu24.04 hosted starter artifacts also match macOS geometry/ranking
and preview pixels, with maximum score error 6.86e-6 within tolerance and exact
input/model identities. This run acquired its weights from an empty cache.
Hosted full-suite feedback identified three CLI assertions sensitive to ANSI
styling/line wrapping; normalize rendered text and retain semantic exit-code,
no-acquisition and no-output assertions before rerunning hosted verification.
The complete 22-test onboarding CLI module passes with forced color, 80 columns
and CI flags, including three additional long-path output assertions. No runtime
code or immutable installer change was needed.

Plan recorded before implementation. Demo imagery is bundled and pinned rather
than relying on an unchanged remote STAC response; preview and inference share
that single input. The installed starter path remains explicitly unreleased until
manual review, merge and separately approved publication.


## Pass 17 — Local GeoTIFF/COG streaming

Branch: `feat/local-raster-streaming`, from main `58b9727` after manually merged
PR #29. Re-read AGENTS.md and lessons and verified branch before edits. Dedicated
compliance review assigned. Focus this PR on local streaming; a labeled evaluation
suite is a subsequent separate milestone. User manually merges PRs; no new release
or version bump from this feature work. Separately finish previously approved
0.3.0 publication from its exact merged commit, excluding feature changes.

- [x] ~~Add a typed metadata-only local source and resource-owning local window
      reader sharing the existing raster grid/window/mask pipeline. Reuse sensor
      detection and local band selection without fake pixels/STAC or full reads.~~
- [x] ~~Preserve eager local named/positional selection, tags/datetime, canonical-DN
      convention (TIFF scale/offset ignored), all-band nodata/masks, 0.5% native-grid
      tolerance and ceil resampling extents. Reject local bbox for the initial
      streaming surface; eager local bbox remains ignored. Retain metric/north-up
      constraints and GTiff-only opens, regular filesystem paths and safe local
      file-URI handling; no virtual/remote GDAL path escape through local input.~~
- [x] ~~Route local paths/file TIFF URLs through all three disk APIs and CLI stream
      modes while preserving STAC/remote behavior and their option validation.
      Reject unsupported local COG metadata options and bbox explicitly; validate
      before model/source I/O. Use existing staged output resource owners.~~
- [x] ~~Prove eager/local-stream equivalence for embeddings, scores/ranks/GeoJSON,
      masks/grid/validity, sensor auto/override, band names/positional ambiguities,
      resampling/native tolerance, nodata/internal masks, missing CRS and edge
      chips. Verify failures/interrupts and that only bounded windows are read.~~
- [x] ~~Measure process RSS on increasing local scenes with fixed buffers across all
      paradigms. Run actual installed-model local workflows using pinned imagery,
      preserving input units; no full-file staging inside the streaming APIs.~~
- [x] ~~Update usage/limits/examples and progress/TODO, run full coverage/style/type/
      registry/base-install checks and dedicated final review.~~
- [x] ~~Commit/push a matching feature upstream and open a PR; stop for manual merge.~~

Acceptance: local raster users need no STAC wrapper for disk streaming. For fixed
source block layout, chip/batch/model/cache settings, working memory is independent
of scene area/output count. Existing output contracts and eager behavior remain
compatible, and metadata decisions match the local pipeline. Untiled strips can
require large decoder buffers; recommend tiled TIFFs for tighter bounds.
No new dependencies, reprojection#21, local crop semantics, evaluation framework,
REST changes, model families or release/version changes in this feature PR.

### Review

Plan recorded before implementation. Local metadata semantics deliberately remain
separate from direct COG/STAC rules; header-only typed metadata feeds common
resource-owned windows and existing output writers. Reprojection stays deferred.

Local I/O tests (78), public/CLI/RSS integration tests (43) and URI routing prove
eager equivalence, safe inputs and bounded windows. Every paradigm passes fresh
512/2048/4096-square fixed-buffer RSS checks. Actual pinned Clay v1.5 embeddings
and flood masks match exactly at batches 1/2; RemoteCLIP ranking/geometry match
with maximum score error 4.18e-6. Canonical-DN conversion of the public Sentinel-2
subset is explicit. All new guide setup/CLI/Python snippets execute unchanged.
Fresh base-only wheel/archive clean-room smoke includes local/STAC eager/stream
parity; formatting, lint, types and registry checks pass. Dedicated design and
compliance review found no blocker. Final verification: **932 passed / 12 skipped,
92.26% package/example coverage**; Black (176 files), Ruff, mypy (72 files), seven
registry entries and diff checks pass. Committed/pushed to the matching upstream
and opened [PR #30](https://github.com/cercanasystems/whirld/pull/30). Hosted CI is
tracked separately from these local checks. Stop for the user’s manual merge; no
feature release/version bump.

### Next separate milestone

- [ ] Build a small labeled evaluation suite after this PR is reviewed/merged:
      pin open imagery and labels, keep spatial train/test separation, define
      task-appropriate metrics and record reproducible model/input provenance.
      Establish a baseline before selecting more model families. Keep execution
      equivalence checks distinct from predictive-quality evaluation.

## Release 0.3.0 preparation and example revalidation

Branch: `release/0.3.0`, from main `afdbb6a` after manually merged PR #28.
Re-read AGENTS.md and lessons; branch verified before edits. User authorized a
release-preparation PR and verification that examples still work. The user
manually merges; no tag, publication, release or PyPI action in this pass.

- [x] ~~Set the single version source to 0.3.0 and prepare accurate release notes,
      README/install guides and model/streaming documentation. Clearly distinguish
      candidate artifacts from a published release and keep the approval boundary.~~
- [x] ~~Re-run all three documented real-world example entry points with actual
      pinned weights and public imagery in fresh output directories. Independently
      check arrays/scores/masks, geometry, model provenance, metrics and previews;
      fix any compatibility problems and update recorded results/commands.~~
- [x] ~~Revalidate new flood/model-variant/streaming example workflows using their
      pinned samples/checkpoints; retain explicit input, metadata and nodata policy.~~
- [x] ~~Build source and wheel candidates, inspect version/metadata/resources/licenses
      and source documentation, and verify independent fresh base-only installs
      plus existing/new streaming smoke workflows outside the checkout.~~
- [x] ~~Run full coverage, Black/Ruff/mypy/registry checks and dedicated compliance
      review; document actual verification and remaining limits in TODO/progress.~~
- [x] ~~Commit/push the preparation branch and open a PR for manual merge, then stop.
      Tagging and release publication require separate explicit approval afterward.~~

Acceptance: runtime/metadata/distribution agree on 0.3.0, all advertised examples
execute with real inputs/weights and valid outputs, candidate archives install
without optional model dependencies, and the PR remains open for user review.
Do not infer model accuracy from execution checks. No automatic merge or release.

### Review

All five documented example paths pass with real pinned weights/public imagery:
three defaults, alternate Clay reference and classification streaming. Independent
arrays/scores/masks/grid/provenance and PNG/SVG checks pass; earlier outputs match
apart from runtime version metadata. Guides retain separate historical/candidate
measurements. The discovered empty GDAL header filename is corrected using an
actual empty source, with inherited-auth isolation/restoration and HTTP regressions.

Final checks: **800 passed / 12 skipped, 91.95% package/example coverage**; three
actual-model checks pass separately. Black (171 files), Ruff, mypy (71 files),
seven-entry registry and diff checks pass. Independent wheel/source base-only
Python 3.13.7 installs pass version/metadata/resources/licenses and local/STAC/new
HTTP streaming workflows. Rebuilt payload includes the policy correction and
preserves inference behavior. Final compliance review is clear. Prepared [PR #29](https://github.com/cercanasystems/whirld/pull/29) on the matching
upstream branch. Hosted CI is tracked separately; no automatic merge, tag or
publication.

Subsequent publication: user manually merged PR #29 and separately authorized
publication. [v0.3.0](https://github.com/cercanasystems/whirld/releases/tag/v0.3.0)
is published from exact merged `58b9727`, excluding local streaming. Main CI,
tag identity and downloaded wheel/source checksums passed. The development log
records final hashes. No automatic merge or PyPI publication.

## Pass 16 — Flood support, segmentation streaming and model variants

Branch: `feat/flood-streaming-models`, from main `d4cb5f8` (v0.2.0).
Re-read AGENTS.md and lessons and verified branch before implementation. Preserve
and include the manual merge/release preference already recorded in lessons.
Dedicated compliance reviewer assigned. Execute the recommended sequence below.
The user merges PRs manually; stop at the reviewable PR. No release/tag action.

### 1. Prithvi flood

- [x] ~~Verify the official EO 2.0 300M TL Sen1Floods11 checkpoint, config, input
      band/unit/normalization/metadata contract and public example. Pin revision,
      artifact hashes and sizes; add a separate prithvi-flood registry entry.~~
- [x] ~~Adapt the existing trusted offline Prithvi deployment only where the actual
      checkpoint requires it; retain burn-scar compatibility. Explicitly support
      the documented Sentinel-2 L1C contract instead of treating L2A/HLS as equal.~~
- [x] ~~Verify registry/pull/API/CLI behavior and real flood model inference against
      the official reference/sample; record class/nodata/grid semantics and limits.~~

### 2. Bounded-memory segmentation

- [x] ~~Extend shared window batches with opt-in validity masks and public source
      dimensions/transform, preserving existing embedding/classification behavior.~~
- [x] ~~Add typed segment_to_disk result/options, resource-owning segmentation
      coordinator and staged windowed GeoTIFF writer. Use native chip size, zero
      overlap and current threshold/class semantics; pad edges and crop on write.
      Preserve all-band source validity and nodata 255, including valid class 0.~~
- [x] ~~Expose segment_to_disk and segment --stream with batch/cache/output controls,
      STAC/direct COG routing and scoped auth; eager/REST remain compatible.
      Publish a single complete mask atomically; failures preserve old outputs.~~
- [x] ~~Verify exact eager/stream masks, dimensions, CRS/affine and validity for
      flood/burn-scar, thresholds, non-square edges, bbox/resampling and masks;
      validate backend batch bounds and output schemas/failures/interrupt cleanup.
      Prove process RSS bounds as actual HTTP scenes/output sizes grow.~~
- [x] ~~Compare real flood and burn-scar eager/stream runs at multiple batch sizes
      and document results; no accuracy claims from execution-only checks.~~

### 3. Selected model variants

- [x] ~~Add RemoteCLIP RN50 and ViT-L/14 as separate pinned registry entries, keeping
      existing ViT-B/32 identity unchanged. Verify upstream architecture, weights,
      preprocessing, licenses and source provenance before claiming support.~~
- [x] ~~Execute real variant inference with public imagery and verify probability,
      geometry and eager/stream contracts. Keep unrelated new sensor/model-family
      expansion outside this pass unless required by the selected models.~~

### Verification and review

- [x] ~~Keep TODO and dev_progress updated, add exact setup/usage/examples and
      limits, run full coverage/Black/Ruff/mypy/registry/fresh package checks,
      and complete dedicated final compliance review.~~
- [x] ~~Commit/push to a matching new upstream and prepare a reviewable PR with
      validation evidence. Pause for the user's manual merge; release separately.~~

Acceptance: new capabilities work with real public artifacts, explicit model
contracts and existing trust/integrity rules. Streaming memory depends on model,
chip/batch/band/cache buffers rather than scene/output pixels. No full-scene
arrays, mask assembly or metadata lists. Source grid/crop semantics and final
mask bytes match eager within predeclared model tolerances. Keep reprojection #21,
segmentation overlap blending, REST streaming, ~~local raster streaming~~
(completed in Pass 17) and new model families deferred. No version/tag/release
changes without explicit later approval.

### Review

All three phases implemented and executed with actual pinned weights and public
imagery. Existing Prithvi deployment loads flood without a runtime adaptation;
the metadata-free policy, padding and output encoding are explicit. Flood and
burn-scar eager/stream masks and grids match exactly at batches 1/2. RN50/L14
loads cover every learned parameter with zero missing/unexpected keys; streamed
scores preserve ranking/geometry within 6.3e-6 maximum absolute error. Controlled
HTTP scene and writer-only growth tests pass. Fresh base-only wheel/source
installs pass existing workflows and both mask tasks. Final coverage: **799 passed / 12 skipped, 91.95%** across package and examples;
Black (170 files), Ruff, mypy (71 files), seven-entry registry and diff checks pass.
Final dedicated review is clear after the source-layout correction. Prepared [PR #28](https://github.com/cercanasystems/whirld/pull/28) on the matching
upstream feature branch. Hosted CI is tracked separately. The user manually
merges; no version/tag/release action.

## Pass 15 — Bounded-memory classification and release 0.2.0

Branch: `feat/bounded-classification`, from released main `b66ee8f` (v0.1.0).
Branch verified before implementation. Re-read AGENTS.md and lessons; dedicated
compliance reviewer assigned. User authorized implementation and the 0.2.0
milestone; release follows verified work and green hosted checks.

- [x] ~~Add typed streaming classification options/result and an owned coordinator
      reusing STAC/direct COG windows, one backend, and bounded inference batches.~~
- [x] ~~Add a resource-owning disk-backed classification writer: preserve existing
      GeoJSON fields, descending primary-score order, stable chip-index ties,
      threshold and positive top-k semantics. Explicit all-results mode uses
      external disk sorting, never a scene-sized Python feature/score list.
      Stage a single output and atomically replace it only after success.~~
- [x] ~~Expose classify_to_disk and classify --stream with batch/cache controls,
      requiring disk output. Preserve existing eager defaults; streaming accepts
      STAC and direct remote COG inputs. Add explicit all-results mode only for
      streaming; reject invalid options before model/source I/O.~~
- [x] ~~Verify eager/stream equivalence for single/multiple queries, thresholds,
      ties/top-k/all/empty results, bbox/masks/edges/radiometry, backend call bounds,
      auth/failures/interrupts and prior-output preservation. Prove RSS bounds
      with increasing scene/output sizes in fresh workers and actual HTTP ranges.~~
- [x] ~~Run actual RemoteCLIP classification against pinned public imagery and
      compare eager/stream at multiple batch sizes within declared tolerances.
      Update example, guides, TODO and development log; run full coverage/style/
      type/base-install/artifact checks and dedicated final compliance review.~~
- [x] ~~Prepare 0.2.0 version/changelog, verified installation assets and release
      notes/checksums for publication after hosted checks.~~

Publication sequence: commit and push the feature branch, open the review PR,
require green hosted checks, then merge/tag/publish verified GitHub assets.
The GitHub release records final tag identity, checksums and publication status.

Acceptance: memory depends on model, band/chip/batch/query/cache sizes, not scene
area or retained completed scores/features. Global ordering uses disk-backed
sorting for all-results output. Existing classify return arrays/REST behavior stay
compatible. Segmentation streaming and reprojection #21 remain deferred. Test
real-model equivalence, not model prediction accuracy. No PyPI publication.

### Review

Implemented on the verified feature branch with explicit coordinator/writer
ownership and strict typed schemas. Final verification: **704 passed / 9 skipped,
91.53% coverage** across package and examples; Black (159 files), Ruff, mypy
(69 files), registry and diff checks pass. Controlled authenticated HTTP tests
preserve eager values, ranking, bbox/masks/edge geometry and scoped credentials.
Fresh workers prove bounded raster memory and disk-sort memory through 64,000
features (about 25 MB output, roughly 70–71 MiB writer RSS). Actual RemoteCLIP
streaming batches 1/3 retain all 16 pinned-scene chips and exact geometry/rank;
maximum score differences stay below 5.4e-6. Fresh wheel/source installs pass
base-only workflow checks and classification equivalence. Dedicated compliance
review is clear, including the SQLite huge-top-k correction. No dependencies added.
Publication follows green hosted checks; final tag/assets are tracked by the
GitHub release. Segmentation streaming and reprojection #21 remain deferred.

## Release 0.1.0

Branch: `release/0.1.0`, from merged PR #25 (`fd47547`). Main CI is green.
Re-read AGENTS.md and lessons; dedicated release compliance reviewer assigned.
User authorized the 0.1.0 GitHub release. No PyPI publication in this pass.

- [x] ~~Reconcile package/runtime version, repository URLs, license inclusion and
      advertised Python support with the verified Python 3.13.7 environment.~~
- [x] ~~Add release notes and exact artifact installation/release instructions;
      update README and development progress with scope and known limitations.~~
- [x] ~~Build wheel and source distribution, inspect metadata/licenses/resources,
      and test installations from both artifacts in fresh `.venv` environments.~~
- [x] ~~Run appropriate regression/style/type checks and final compliance review;
      commit release preparation and verify hosted CI before tagging.~~
- [x] ~~Prepare verified wheel/source assets, checksum manifest and release notes
      for an annotated v0.1.0 GitHub release after hosted CI passes.~~

Publication follows the reviewed preparation commit: tag the merged commit,
upload the verified assets and verify their downloaded checksums. The published
GitHub release records final tag identity, asset hashes and release status.

Acceptance: source/runtime/distribution versions agree at 0.1.0; release artifacts
install and exercise local/STAC/direct COG reference workflows without optional
model dependencies. Notes distinguish tested platform/Python support, model
licenses and deferred streaming/reprojection features. Tag points to the reviewed
CI-green release commit. Record final release URL and artifact hashes externally.

### Review

Release preparation verified: 605 passed / 9 skipped, 90.98% coverage; style,
typing and registry checks pass. Fresh wheel and source installations pass lazy
import, metadata/resource/license inspection, local/STAC/streamed-STAC workflows,
and direct HTTP COG eager/stream equivalence (maximum difference 0.0). Dedicated
compliance review found no remaining blockers. Final rebuilt artifacts preserve
the tested package payload. Publication is authorized and follows green hosted
preparation CI; the GitHub release records final tag and asset verification.
PyPI credentials/setup are not required.

## Pass 14 — Direct remote COG input

Branch: `feat/direct-cog-input`, from merged PR #24 (`d9686ae`). Re-read AGENTS.md
and lessons; dedicated compliance/design review assigned. Written plan checked
before implementation; execution remains in developer-selected Default mode.

- [x] ~~Introduce typed scene/source resolution and reuse one resource-owning grid,
      mask/radiometry and window engine for STAC and direct HTTP(S) GeoTIFF input.
      Preserve existing STAC/local behavior and public reader compatibility.~~
- [x] ~~Add direct COG metadata rules: explicit sensor required; select complete,
      unique TIFF band descriptions or explicit source-position band names; reject
      unnamed/ambiguous mappings. Explicit scalar/per-band scale/offset overrides
      TIFF nonidentity transforms; absent/identity TIFF values keep canonical DN.
      Explicit scale 1 identifies reflectance. Preserve TIFF masks/nodata/time.~~
- [x] ~~Add public CogOptions and automatic .tif/.tiff URL routing (including signed
      queries); allow explicit COG options for extensionless URLs. Expose --cog,
      --bands, --scale and --offset in embed/classify/segment, and --token as an
      alias for existing scoped credentials. Stream embeddings with existing NPY
      output; eager APIs retain resident-result behavior. Require remote HTTP(S)
      for COG options and reject incompatible local/STAC option combinations.~~
- [x] ~~Verify numerical/metadata equivalence for reordered bands, physical units,
      masks/nodata, bbox, resampling, overlap and partial edges; validate URL,
      options and credential boundaries, including failures with signed URLs.~~
- [x] ~~Verify controlled HTTP range/subset reads and fixed-chip/batch RSS as scene
      size grows. Add public API/CLI integration and an opt-in real public remote
      GeoTIFF check; demonstrate a real model run with the official HLS sample.~~
- [x] ~~Document exact usage/metadata precedence, authentication and memory limits;
      update TODO/development progress, run full coverage/style/typing/install
      checks and final compliance review.~~

Acceptance: direct URLs work without a user-created STAC item; semantic band/unit
handling is explicit and numerically equivalent to the established pipeline.
No full-file staging for remote input. Remote GeoTIFF drivers are restricted to
GTiff; authorization is scoped on opens/lazy reads and stripped on redirects.
URLs with query credentials must not leak into new errors, logs or filenames.
Only north-up metric grids are supported; streaming reprojection #21 stays deferred.
COG layout is recommended, not promised by filename; source block size and server
Range support remain requirements/limits. Publication authorized on 2026-11-22.

### Review

Typed source adapters now feed shared window execution without fabricated STAC
files or duplicate pixel algorithms. Controlled HTTP range/memory tests and real
public HLS pixel checks pass. Direct Prithvi inference matches the prior STAC
mask and grid exactly. Final verification: **605 passed / 9 skipped, 90.98%
coverage across whirld + examples**. Black (153 files), Ruff, mypy (67 files),
registry validation and diff checks pass. A fresh base-only Python 3.13.7 install
passes dependency/lazy-import and local/STAC/direct-URL smoke checks; direct eager
and streaming reference embeddings match exactly. Signed-query and bearer
redaction covers actual malformed TIFF DEBUG diagnostics and lazy reads, with
nested cleanup and thread isolation verified. Final dedicated compliance review
found no blocking correctness or AGENTS.md issues. No dependency changes;
reprojection #21 remains deferred. Prepared for PR review; hosted CI is tracked
separately from local validation.

## Pass 13 — Three runnable real-world examples

Branch: `docs/real-world-examples`, from merged PR #23. Re-read AGENTS.md and
lessons; dedicated compliance review assigned. Written plan checked before
implementation; session remains in developer-selected Default mode.

- [x] ~~Select pinned open imagery and verify retrieval, attribution, model input
      contracts and small CPU defaults for all three examples.~~
- [x] ~~Build Clay similarity search: stream real embeddings, rank a reference chip
      by cosine similarity, export georeferenced matches and an inspectable preview.~~
- [x] ~~Build RemoteCLIP classification: score explicit competing prompts on real
      imagery, export GeoJSON and a preview with prompt-dependent score caveats.~~
- [x] ~~Build Prithvi burn-scar segmentation: prepare a genuine HLS sample, infer
      with real weights, export a georeferenced mask and imagery overlay preview.~~
- [x] ~~Execute all examples with real weights; validate spatial/output contracts,
      record runtime/memory/setup costs and visually inspect their previews.~~
- [x] ~~Add focused hermetic tests, exact setup/run instructions, source attribution,
      expected results and README links; run regression/coverage, Black/Ruff/mypy
      and final compliance review; update development progress and this review.~~

Acceptance: every example runs from documented commands against pinned public
inputs with actual model weights; small CPU defaults, automatic preparation,
explicit band/radiometry handling, useful viewable outputs, and recorded evidence.
Do not equate model execution with validated prediction accuracy. Examples live
under `examples/` as requested, with typed, documented object ownership and no
unnecessary new dependencies. Downloads/results stay outside tracked source.
Reprojection #21 remains deferred. Publication authorized on 2026-11-22.

### Review

Implemented and executed all three workflows on CPU with actual checkpoint
weights. Shared `ExampleRuntime` verifies the bundled registry and checkpoint
checksum, reserves a new output directory and writes model/runtime provenance.
Pydantic schemas define options, geospatial results and reports. `Preview` owns
RGB display conversion and self-contained SVG/HTML rendering; source pixels and
model inputs remain separate. Download/results directories are ignored by Git.

- Clay: pinned Sacramento Sentinel-2 item, 16 x 1024 finite float32 embeddings;
  reference 0 ranks chips 5/2/4/9/8; 43.33 s and 2596.3 MiB peak process RSS.
- RemoteCLIP: all 16 chips and four finite normalized prompt scores; 14 urban,
  one farmland and one water winner; 10.14 s and 1589.1 MiB peak process RSS.
- Prithvi: official pinned, checksum-verified HLS sample with explicit reflectance
  scale 1; 512 x 512 binary mask with exact source grid/validity; 208989 predicted
  burned pixels of 262144 valid; 8.60 s and 3101.0 MiB peak process RSS.

Independent review verified saved artifacts and source units. Browser/PNG visual
inspection verified all previews; partial-chip label clipping was fixed and tested.
Live classification exposed an invalid top_k=0 example argument; the final example
requests the pinned region's 16 chips and verifies none are omitted. Production
inference code and dependency pins are unchanged. CPU timing includes imagery,
loading, inference and previews after setup; GPU execution and prediction accuracy
are not established. Attribution and interpretation limits are in each guide.

Verification: **523 passed / 8 skipped; 90.41% combined package/example coverage**.
Black (138 files), Ruff, mypy (62 source/example files), and diff checks pass.
33 new hermetic tests cover example contracts. CI now includes examples in
formatting, lint, typing and the >=80% coverage gate; actual model/data execution
was performed separately from the ordinary suite. README and development progress
are updated. Final compliance review found no blocking issue. Prepared for PR
review on `docs/real-world-examples`; hosted CI is separate from this local
verification record. Deferred issue #21 remains untouched.


## Pass 12 — Real-model streaming validation

Branch: `test/real-model-streaming-validation`, from merged PR #22 (`b863f4b`).
Plan checked against AGENTS.md and lessons; a dedicated compliance reviewer is
assigned. Execution uses the session's developer-selected Default mode.

- [x] ~~Add an opt-in, object-owned real Clay validation runner with typed options
      and reports, using an installed, checksum-verified Clay v1.5 checkpoint.~~
- [x] ~~Compare public API eager/streamed float32 embeddings for one pinned public
      Sentinel-2 scene at batch sizes 1, 2 and 3, including a partial last batch;
      require finite (n, 1024) results, exact metadata except run timestamps, and
      predeclared numerical tolerance rtol=1e-4, atol=1e-5.~~
- [x] ~~Measure real CPU inference in fresh processes with fixed chip/batch sizes
      over increasing public-scene regions. Record process peak RSS and elapsed
      time including model load/I/O, separately from setup/download; report
      throughput and limitations without claiming GPU validation.~~
- [x] ~~Add focused hermetic tests for validation/report failures and a gated live
      real-model integration check; run the real checkpoint/public-data experiment.~~
- [x] ~~Document reproducible commands, pinned input/checkpoint identity, measured
      results and remaining limits; run relevant regression/coverage, formatting,
      lint and typing checks and obtain a final compliance review.~~

Acceptance: identical chip order/geometry and model metadata; eager/streamed
embeddings agree within the predeclared tolerance for all tested batch sizes.
Resource reports describe actual real-model costs, without conflating full-process
loading peaks with steady-state inference. No production inference behavior change
unless validation reveals a reproducible defect. Reprojection issue #21 remains
low priority and outside this pass. Publication authorized on 2026-11-22.

### Review

Completed on `test/real-model-streaming-validation`, branched from the clean
merged main. Added an object-owned, opt-in validation runner with Pydantic settings,
measurements and reports; installed checkpoint identity and SHA-256 are verified
before fresh CPU workers. No production inference or dependency changes.

The actual pinned checkpoint/public Sentinel-2 run passed: four-chip eager output
matched streamed batches 1/2/3 exactly (maximum error and RMSE 0.0), with identical
metadata except timestamps and a partial final batch at size 3. At fixed batch 1,
4 -> 16 chips measured 2625.5 -> 2625.8 MiB peak process RSS and 13.96 -> 35.31 s.
These include loading and remote I/O; GPU and steady-state memory are not measured.
See [the reproducible validation guide](../docs/real-model-streaming.md) for all
measurements, runtime versions, tolerances and interpretation limits.

Verification: **490 passed / 8 skipped, 93.09% coverage**, Black (129 files), Ruff,
mypy (58 source/script files), and diff checks pass. The real acceptance matrix ran
through the standalone runner; its gated pytest wrapper remains skipped in the
ordinary suite. Independent compliance review checked saved arrays/sidecars and
found no blockers. README, streaming guide and development progress are updated.
Merged in PR #23 (`e522f3f`); the preceding record is local verification evidence.
Issue #21 remains deferred.

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
conformance cleanup were merged together in PR #22 (`b863f4b`).

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
implementation and subsequent conformance cleanup are merged in PR #22
(`b863f4b`); this section records local verification evidence.

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

## Pass 11 — Bounded-memory COG inference (merged in PR #22)

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

**Follow-up:** ~~direct remote `.tif` URL input~~ (completed in Pass 14); extend streaming to ~~classification~~ (completed in Pass 15)
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
