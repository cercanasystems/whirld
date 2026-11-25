# Releases

[Whirld 0.3.0](https://github.com/cercanasystems/whirld/releases/tag/v0.3.0)
is published from merged commit `58b9727`. The wheel, source distribution and
`SHA256SUMS` were downloaded and verified after publication. PyPI publication is
separate. The package version comes from `src/whirld/_version.py`; setuptools
resolves it without importing the inference stack.

The current source checkout adds [local raster streaming](local-streaming.md)
under **Unreleased**. This feature is not part of the 0.3.0 release assets, and its
implementation does not bump the package version. Future release preparation
still requires manual PR merge and separate approval to publish.

## Install the published release

Download all three assets from the linked release into one directory, then verify
before installing with Python 3.13.7:

```bash
shasum -a 256 -c SHA256SUMS  # Linux: sha256sum -c SHA256SUMS
python3.13 -m venv .venv
.venv/bin/python -m pip install ./whirld-0.3.0-py3-none-any.whl
.venv/bin/python -m pip check
.venv/bin/whirld --help
```

Use the source archive instead of the wheel if preferred. Model weights download
separately; model/server extras can be selected on the wheel path as shown below.

## Build and verify release artifacts

### Onboarding preview and future PyPI publication

The first-result preview adds `doctor` and `demo` without changing the package
version. Its quickstart uses an immutable GitHub source archive; the published
0.3.0 wheel predates these commands. Do not upload a different payload under an
existing published version.

PyPI publication is not enabled by this work. The public `whirld` JSON endpoint
returned HTTP 404 during the 2026-11-24 read-only preflight; that does not reserve
the project name or grant publishing rights. After manual merge, prepare a
separate versioned release PR and obtain explicit publication approval.

Before an approved upload, choose the release version, refresh the starter pins,
replace preview install/image URLs with reviewed release refs, and make README
navigation links absolute so they work on PyPI. Build the wheel from the source
archive, validate metadata/long-description rendering, verify the bundled imagery
notice and constraints, and repeat the standalone starter install and real demo.
Configure account ownership/Trusted Publishing only as part of that separately
approved publication step. Model/data licenses remain distinct from package code.

Use Python 3.13.7 in a `.venv`. From the reviewed source checkout, build the
source distribution, then build the wheel from that archive using an isolated
PEP 517 build. With `uv` available:

```bash
uv build --sdist --out-dir dist
uv build dist/whirld-0.3.0.tar.gz --wheel --out-dir dist
```

Future builds remain local verification artifacts until publication is approved.
To install a locally built wheel in a fresh virtual environment:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install ./dist/whirld-0.3.0-py3-none-any.whl
.venv/bin/python -m pip check
.venv/bin/whirld --help
.venv/bin/whirld pull clay-v1
```

The source archive is also installable:

```bash
.venv/bin/python -m pip install ./dist/whirld-0.3.0.tar.gz
```

Install model/server extras using the wheel path, for example:

```bash
.venv/bin/python -m pip install './dist/whirld-0.3.0-py3-none-any.whl[remoteclip]'
```

The base installation includes the offline reference encoder; trained-model
weights are separate downloads. See README and the example guides for setup,
license attribution and runnable workflows. PyPI publication is separate.

## Preparation and approval

1. Start from reviewed main on a release preparation branch. Confirm CI on the
   merged implementation, then update the version, changelog, README, TODO and
   development log. Label unpublished release notes as pending.
2. Re-run the documented example entry points with actual pinned imagery and
   weights in new output directories. Check arrays/scores/masks, spatial metadata,
   model/version provenance and saved previews. Record execution evidence without
   claiming model accuracy or GPU support.
3. Run formatting, lint, typing, registry and coverage checks in the pinned
   `.venv`. Build source then wheel as above. Inspect version, Requires-Python,
   URLs, root/vendor licenses, registry/schema and source docs/examples; exclude
   secrets, caches and weights.
4. Install each candidate in an independent fresh `.venv` outside the checkout.
   Verify runtime/distribution versions, dependency integrity, lazy import and
   `scripts/clean_room_test.py` from the source archive. Exercise direct COG
   embedding/classification/segmentation disk workflows using controlled inputs.
5. Commit/push preparation and open a PR with verification evidence. **Stop for
   the user's manual merge.** Never merge the PR automatically.
6. After the user has merged and separately approved publication, verify the
   merged tree matches candidate build inputs and create final artifacts and a
   SHA-256 checksum file. Only then create/push the new annotated `vX.Y.Z` tag and publish the
   GitHub release with wheel, source archive and `SHA256SUMS`.
7. Download published assets, check their checksums and tag identity, and record
   the release URL/hashes. Do not replace published versioned assets; use a patch
   release for corrections.

After publication, verify downloads with `shasum -a 256 -c SHA256SUMS` on macOS or
`sha256sum -c SHA256SUMS` on Linux. These checksums identify the approved final
artifacts; candidate build timestamps can change archive bytes.

Compatible fixes to 0.3.0 use 0.3.x; later feature milestones use a new minor
version. Changes during 0.x must be explained in release notes. Version 1.0 will
require an explicit public API and output-format stability commitment.
