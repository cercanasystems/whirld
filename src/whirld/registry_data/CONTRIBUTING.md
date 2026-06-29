# Contributing a model to the Whirld registry

The registry is **load-bearing**: a model's YAML selects the backend that runs it,
the weights that get downloaded, the normalization applied to your pixels, and the
license you operate under. Because pulling a model downloads and **executes** code
against your data, "schema-valid" is *not* a sufficient gate. This document is the
governance policy; the machine-checkable parts are enforced by
[`check_registry.py`](../../../scripts/check_registry.py) in CI.

## Merge policy

A pull request that adds a `models/<name>.yaml` is **auto-merged only if all** of
these hold; otherwise it is routed to **human review**:

1. **Schema-valid** — conforms to [`schema/model.schema.json`](schema/model.schema.json).
2. **OSS-allowlist license** — `license` is a recognized open-source license
   (Apache-2.0, MIT, BSD, ISC, MPL-2.0, …; see `whirld.core.governance.OSS_LICENSES`).
   Use-restricted or non-commercial licenses are **never** auto-merged.
3. **Trusted source** — `source.repo`'s org is on the trusted allowlist. A new source
   domain (new Hugging Face org) goes to human review even if everything else passes.
4. **Safetensors weights** — the primary weights file is `.safetensors`. Pickle
   formats (`.pt`, `.pth`, `.ckpt`, `.bin`) are **never** auto-merged.

The maintainer-curated, first-party models (`trust: first-party`) are added by review,
not auto-merge — so they may use pickle weights and any vetted source.

## Why these rules

### Security — pickle is arbitrary code execution
`torch.load` of a `.pt`/`.ckpt` and open_clip flat `state_dict`s use Python pickle,
which **runs arbitrary code on load**. A malicious community checkpoint would execute
on every user who pulls it. So:

- **Community models must ship `safetensors`** (a non-executable tensor container).
- **Pickle is allowed only for `trust: first-party`** entries that a maintainer has
  reviewed. Whirld's runtime **refuses to pull a `community` entry with pickle
  weights** (`SecurityError`, exit 9) — defense in depth even if a bad entry slips
  into a registry cache.
- Backend selection is a **closed allowlist** (`backend:` id → a hardcoded code
  branch in `models/loader.py`). Whirld **never imports a module path from YAML**, so
  a YAML cannot name arbitrary code to run.
- `sha256` of the primary artifact is verified on every pull.

### License — surfaced, allowlisted, propagated
`license` is **required** (SPDX id where possible). Whirld shows it at `whirld pull`
**before** downloading and at `whirld info`. A license outside the OSS allowlist gets
a terms warning at pull time and cannot auto-merge.

**Worked example — declined:** OlmoEarth (Ai2) is fully runnable and ungated, but
ships under the custom **OlmoEarth Artifact License** (no military/surveillance, no
extractive-industry use; attribution + license-propagation required). It is *not* an
OSS-allowlist license, so it would never auto-merge and would warn at pull. Whirld
declined to bundle it on license-posture grounds.

## `trust`

| value | meaning | weights | merge |
|---|---|---|---|
| `first-party` | maintainer-curated | pickle or safetensors | review only |
| `community` | submitted entry | **safetensors required** | auto-merge if rules pass |

A submission **cannot self-declare `first-party`** to bypass the safetensors rule — a
PR setting `trust: first-party` is routed to human review.

## CI wiring (registry repo)

```yaml
# .github/workflows/registry-check.yml
name: registry-gate
on: pull_request
jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.13" }
      - run: pip install whirld pyyaml
      # exit 0 = auto-merge eligible; 1 = needs review; 2 = invalid
      - run: python scripts/check_registry.py $(git diff --name-only origin/main... -- 'models/*.yaml')
```
