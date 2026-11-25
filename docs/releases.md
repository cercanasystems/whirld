# Releases

**0.4.0 is a release candidate, pending manual review and publication.** Its
versioned source is available at immutable revision
`b817fe333be33a6ce96d417915a39a323e2aae37`; no `v0.4.0` tag or published release
assets exist yet. The [quickstart](quickstart.md) installs that candidate without
Git, a checkout or development dependencies. PyPI publication is separate.

The candidate groups local raster streaming (PR #30), labeled evaluation (#31),
installed doctor/demo (#32), and provider contract consistency (#33). See the
[pending changelog](../CHANGELOG.md) for compatibility notes. In particular,
unsupported operations use `UnsupportedOperationError` (CLI 7 / HTTP 422), and
malformed metadata or normalized float32 NCHW/backend output contracts fail
earlier; valid inference math and output formats remain unchanged.

The package version comes from `src/whirld/_version.py`; setuptools resolves it
without importing the inference stack.

## Candidate verification

Candidate checks use Python 3.13.7 and immutable source revision
`b817fe333be33a6ce96d417915a39a323e2aae37` for the remote installer. The local
release tree adds the release documentation and verification record; its only
runtime-source change from main is the version.

The full suite passes **1,472 tests / 12 skipped, 93.53% coverage**. Black (217
files), Ruff, mypy (91 files including scripts), all seven registry documents and
diff checks pass. Recompiling the universal CPU starter profile under its existing
constraints reproduces the exact pinned profile; no dependencies were changed.

A wheel built from the source distribution reports version 0.4.0, Python
>=3.13.7 and the Apache-2.0 license. Package resources and license/notices match
the source bytes; the source archive includes guides, examples, profiles and the
pinned baseline, excluding caches, secrets and weights. Separate fresh 25-package
base environments installed from wheel and source pass dependency checks and the
source archive's clean-room smoke outside the checkout.

Both fresh base installations also pass controlled HTTP-range COG routing checks
for all three public eager/disk workflows. The actual reference encoder produces
bitwise-equal local/HTTP eager/HTTP disk embeddings (9 × 512), with matching
geometry/metadata apart from timestamps. Classification and segmentation use
synthetic inference backends in these base-only checks, retaining real curated
capabilities, source readers, preprocessing and writers: eager/disk GeoJSON and
100 × 100 mask/grid/nodata outputs match exactly. The installed environments
contain no Torch, OpenCLIP, TerraTorch, FastAPI or pytest. Each observes 16 ranged
GETs across six HTTP API calls; these fixtures read the full extent, so this is
routing/output coverage, not a bandwidth or memory benchmark. Actual trained-model
inference is verified separately below.

The exact uv remote archive/profile quickstart also passes in a fresh external
`.venv`, reusing dependency and verified model caches. Installed metadata, runtime
version, `site-packages` location and PEP 610 archive identity are correct; no
development/server/Prithvi packages are present. Doctor reports zero weight
download required. Actual RN50 scores and geometry match the prior baseline
apart from application version; preview PNG bytes are identical. uv reports 51
packages resolved in 2.14 s, three prepared in 0.27 s and 51 installed in 0.40 s;
these are internal phases, not total wall time or cold-install estimates.

The exact pip archive/profile commands in the quickstart passed in a fresh
external `.venv` on macOS ARM64. Installed runtime and distribution versions are
both 0.4.0; imports resolve to that environment's `site-packages`, PEP 610 records
the pinned archive, and `pip check` passes. No development tools, server packages
or Prithvi runtime are installed.

| Candidate pip check | Elapsed |
|---|---:|
| Create `.venv` with an existing Python 3.13.7 | 0.97 s |
| Install remote archive/profile with an existing pip download cache | 23.04 s |
| Doctor with verified cached RN50 weights | 16.52 s |
| Real four-chip demo command with cached weights | 2.53 s |

The demo's internal elapsed time is 2.09 s. These measurements exclude
interpreter provisioning, model acquisition and a failed preliminary sandbox DNS
probe. Independent checks confirm the sample/checkpoint pins, finite normalized
scores, ranking, geometry and preview pixels; outputs match the previous baseline
apart from application version. They establish execution/parity, not prediction
accuracy or GPU performance.

The three documented real-world example entry points also passed on the 0.4.0
source runtime with CPU inference and verified cached model weights. The image
examples made fresh public STAC/COG reads; timings include their data access and
output publication and can vary with network/cache conditions.

| Real workflow | Elapsed | Independent output check |
|---|---:|---|
| Clay v1.5 similarity | 45.18 s | All 16 × 1024 float32 embeddings and matches equal the prior baseline. |
| RemoteCLIP classification | 10.18 s | All 16 four-prompt score vectors, ranking and geometry equal the prior baseline. |
| Prithvi burn-scar mapping | 8.52 s | 512 × 512 mask/grid/nodata equal the prior baseline; 208,989 of 262,144 valid pixels predict burn. |
| Packaged RN50 demo (internal report timer) | 2.41 s | Four features, ranking, geometry and pinned sample/model identities equal the prior baseline. |

Example previews were regenerated, visually inspected and match prior PNGs;
embedded HTML PNGs, geometry counts and label bounds were checked. No new full
browser-rendering claim is made. Application provenance records 0.4.0; prediction
values remain unchanged. Burn-scar source units were independently checked to
apply reflectance normalization once. These are execution/parity checks, not
calibrated land-cover accuracy claims.

The documented four-case labeled evaluation also passed offline with verified
cached imagery/labels and weights. Independently recomputed confusion counts,
metrics and prediction hashes exactly match the pinned baseline: aggregate IoU
**0.7459745 validation / 0.8400137 test**, prediction coverage 1.0, and no missing
predictions. This is a small fixed burn-scar sample; it establishes reproducibility
for those cases, not generalization to other regions, dates or model tasks. See
[evaluation](evaluation.md) for case-level metrics, data licenses and limitations.

Existing dated onboarding measurements remain in the
[quickstart](quickstart.md#recorded-onboarding-verification--2026-11-24). They are
historical observations; none of these timings is a performance promise.

## Install the latest published release (0.3.0)

[Whirld 0.3.0](https://github.com/cercanasystems/whirld/releases/tag/v0.3.0)
was published from merged commit `58b9727`. Its wheel, source distribution and
`SHA256SUMS` were downloaded and verified after publication. Those artifacts
predate doctor/demo, local raster streaming and labeled evaluation.

Download all three assets from that release into one directory, then verify
before installing with Python 3.13.7:

```bash
shasum -a 256 -c SHA256SUMS  # Linux: sha256sum -c SHA256SUMS
python3.13 -m venv .venv
.venv/bin/python -m pip install ./whirld-0.3.0-py3-none-any.whl
.venv/bin/python -m pip check
.venv/bin/whirld --help
```

Model weights download separately. PyPI publication is separate from this GitHub
release; the candidate quickstart does not depend on a PyPI Whirld package.

## Build and verify the 0.4.0 candidate

Use Python 3.13.7 in a `.venv`. From the reviewed source checkout, build the
source distribution, then build the wheel from that archive using an isolated
PEP 517 build. With `uv` available:

```bash
uv build --sdist --out-dir dist
uv build dist/whirld-0.4.0.tar.gz --wheel --out-dir dist
shasum -a 256 dist/whirld-0.4.0.tar.gz dist/whirld-0.4.0-py3-none-any.whl
```

Candidate artifacts remain local verification files until publication is approved.
To install a locally built base wheel in a fresh external directory:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install /absolute/path/to/dist/whirld-0.4.0-py3-none-any.whl
.venv/bin/python -m pip check
.venv/bin/python -c "import whirld; print(whirld.__version__)"
.venv/bin/whirld --help
.venv/bin/whirld pull clay-v1
```

The source archive is also installable:

```bash
.venv/bin/python -m pip install /absolute/path/to/dist/whirld-0.4.0.tar.gz
```

For a starter wheel check, install the same pinned CPU profile used by the
quickstart, selecting the locally built wheel instead of the source archive:

```bash
uv venv --python 3.13.7 .venv
uv pip install --python .venv/bin/python --torch-backend cpu \
  -r https://raw.githubusercontent.com/cercanasystems/whirld/b817fe333be33a6ce96d417915a39a323e2aae37/requirements-quickstart.txt \
  '/absolute/path/to/dist/whirld-0.4.0-py3-none-any.whl[remoteclip]'
.venv/bin/whirld doctor
.venv/bin/whirld demo --output candidate-results
```

Other model/server extras can be installed from a wheel path (`hf`, `prithvi`,
`serve`). Their packages and model weights are separate from the base/reference
installation; see README and the examples for prerequisites and attribution.

## Future 0.4.0 release-asset installation

**These instructions apply only after approved publication.** There is no live
0.4.0 asset download to use yet. After publication, download the wheel, source
archive and `SHA256SUMS` from the announced `v0.4.0` release into one directory:

```bash
shasum -a 256 -c SHA256SUMS  # Linux: sha256sum -c SHA256SUMS
python3.13 -m venv .venv
.venv/bin/python -m pip install ./whirld-0.4.0-py3-none-any.whl
.venv/bin/python -m pip check
.venv/bin/python -c "import whirld; print(whirld.__version__)"
```

The base wheel alone does not install the RemoteCLIP runtime. For doctor/demo,
use that wheel with the pinned starter profile as shown in the candidate wheel
check above, substituting the verified downloaded wheel path.

## Preparation and approval

1. Start from reviewed main on a release preparation branch. Confirm CI on the
   merged implementation, then update the version, changelog, README, TODO and
   development log. Label unpublished release notes as pending.
2. Re-run the documented example entry points with actual pinned imagery and
   weights in new output directories. Check arrays/scores/masks, spatial metadata,
   model/version provenance and saved previews. Record cache conditions and
   execution evidence without claiming model accuracy or GPU support. Keep the
   small labeled evaluation separate from broad model-quality claims.
3. Run formatting, lint, typing, registry and >=80% coverage checks in the pinned
   `.venv`. Build source then wheel as above. Inspect version, Requires-Python,
   URLs, root/vendor licenses, registry/schema, imagery notice, bundled starter
   profile, source docs/examples and runtime resources; exclude secrets, caches
   and weights. Keep tested dependency pins unless verification requires a fix.
4. Install candidates in independent fresh `.venv` environments outside the
   checkout. Verify runtime/distribution versions, dependency integrity, lazy
   imports and source-archive `scripts/clean_room_test.py`. Exercise the exact
   uv/pip candidate quickstarts and real demo, plus direct COG disk workflows.
5. Commit/push preparation and open a PR with verification evidence. **Stop for
   the user's manual merge.** Never merge the PR automatically.
6. After manual merge, **wait for separate explicit publication approval**.
   Verify the merged tree matches candidate build inputs, create final artifacts
   and `SHA256SUMS`, then create/push the annotated `v0.4.0` tag and publish the
   GitHub release with wheel, source archive and checksums. This preparation
   does not authorize any of those publication actions.
7. Download published assets, check checksums and tag identity, and record the
   release URL/hashes. Do not replace published versioned assets; use a patch
   release for corrections. Update pending status and advertised installation
   paths only when the assets are available and verified.

Candidate build timestamps can change archive bytes. Checksums identify the
approved final artifacts; a candidate checksum is not proof of publication.

## PyPI remains a separate publication step

No PyPI upload, account changes or Trusted Publishing configuration is included
in this release preparation. The public `whirld` JSON endpoint returned HTTP 404
during a read-only preflight on 2026-11-24; this dated observation does not reserve
the name or establish current publishing rights.

Before an approved PyPI upload, verify ownership and package-name availability,
render package metadata/long description, make README navigation links absolute,
and repeat the installed starter path from the intended distribution. Configure
accounts or Trusted Publishing only with separate authorization. Model/data
licenses remain distinct from the application license. Never upload a different
payload under an already published version.

Compatible fixes to 0.4.0 use 0.4.x; later feature milestones use a new minor
version. Changes during 0.x must be explained in release notes. Version 1.0 will
require an explicit public API and output-format stability commitment.
