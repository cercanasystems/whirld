# Releases

Whirld 0.1.0 is distributed as a GitHub release with a wheel, source distribution
and `SHA256SUMS`. The package version comes from `src/whirld/_version.py`;
setuptools reads that value without importing the inference stack.

## Install and verify

Use Python 3.13.7 in a `.venv`. Download the artifacts from the
[0.1.0 release](https://github.com/cercanasystems/whirld/releases/tag/v0.1.0).
Verify downloads against the checksum file before installing:

```bash
# macOS; on Linux use sha256sum -c SHA256SUMS
shasum -a 256 -c SHA256SUMS
python3.13 -m venv .venv
.venv/bin/python -m pip install ./whirld-0.1.0-py3-none-any.whl
.venv/bin/python -m pip check
.venv/bin/whirld --help
.venv/bin/whirld pull clay-v1
```

The source archive is also installable:

```bash
.venv/bin/python -m pip install ./whirld-0.1.0.tar.gz
```

Install model/server extras using the wheel path, for example:

```bash
.venv/bin/python -m pip install './whirld-0.1.0-py3-none-any.whl[remoteclip]'
```

The base installation includes the offline reference encoder; real-model weights
are separate downloads. Use the bundled README and example guides for setup,
license attribution and runnable workflows. PyPI publication is a separate step.

## Release procedure

1. Start from reviewed main with green CI, on a release preparation branch.
   Update the version source, changelog, README, TODO and development log.
2. Run formatting, lint, typing, registry checks and the coverage gate using the
   pinned `.venv`. Record actual Python/platform support and known limitations.
3. Build a source distribution and build the wheel **from that archive**, using
   an isolated PEP 517 build. With `uv` available:

   ```bash
   uv build --sdist --out-dir dist
   uv build dist/whirld-0.1.0.tar.gz --wheel --out-dir dist
   ```

4. Inspect both archives: version/Requires-Python/project URLs, Apache license,
   bundled registry/schema and vendored license/notice/provenance. Check source
   archive documentation/examples and exclude secrets, caches and model weights.
   Install each artifact in a fresh `.venv` outside the checkout, verify runtime
   and installed metadata versions, and run `scripts/clean_room_test.py` from the
   source archive. Check direct remote COG eager/stream equivalence as well.
5. Commit and push preparation, open a PR, and require green hosted checks before
   merging. Tag the merged commit with annotated `v0.1.0`; ensure the final source
   tree matches the artifact build inputs. Produce SHA-256 checksums.
6. Publish a GitHub release with the wheel, source archive and checksum file, then
   download the published assets to verify checksums and tag identity. Record the
   release URL and checksums in the GitHub release. Do not replace published
   versioned assets; use a patch release for corrections.

The first 0.x release establishes a reproducible baseline. Compatible fixes use
0.1.x; subsequent feature milestones use a new minor version. Changes during 0.x
must be explained in release notes; 1.0 will require an explicit commitment to
public API and output-format stability. A new feature does not block a release
unless it is required for a promised workflow.
