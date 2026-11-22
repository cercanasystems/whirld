# Releases

Whirld 0.3.0 is a release candidate awaiting manual merge and explicit publication
approval. No 0.3.0 GitHub release or tag has been created. The latest published
release remains [0.2.0](https://github.com/cercanasystems/whirld/releases/tag/v0.2.0).
The package version comes from `src/whirld/_version.py`; setuptools resolves it
without importing the inference stack.

## Build, install and verify the candidate

Use Python 3.13.7 in a `.venv`. From the reviewed source checkout, build the
source distribution, then build the wheel from that archive using an isolated
PEP 517 build. With `uv` available:

```bash
uv build --sdist --out-dir dist
uv build dist/whirld-0.3.0.tar.gz --wheel --out-dir dist
```

These are local verification artifacts until publication is approved. To install
in a fresh virtual environment:

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
   SHA-256 checksum file. Only then create/push annotated `v0.3.0` and publish the
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
