# Your first real Whirld result

The onboarding commands are **unreleased**. Use the pinned preview below; the
published 0.3.0 artifacts do not contain `doctor` or `demo`. Publication and a
version bump remain separate review/approval steps.

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then use a
new directory. The tested manager version is uv 0.11.20. It can provision the
supported Python automatically; activation is unnecessary because commands name
the `.venv` interpreter explicitly.

```bash
uv venv --python 3.13.7 .venv
uv pip install --python .venv/bin/python --torch-backend cpu \
  -r https://raw.githubusercontent.com/cercanasystems/whirld/6b70aa931f4dc03bd4abf2e6345c224f4643b277/requirements-quickstart.txt \
  'whirld[remoteclip] @ https://github.com/cercanasystems/whirld/archive/6b70aa931f4dc03bd4abf2e6345c224f4643b277.zip'
.venv/bin/whirld doctor
.venv/bin/whirld demo --output quickstart-results
```

Open `quickstart-results/preview.html` directly in a browser. All imagery is
embedded in the HTML. `scores.geojson` opens in QGIS. You can inspect every prompt
score rather than only the winning label. The four-chip sample is predominantly
urban, and RN50 selects the urban prompt for all four recorded chips.

The profile includes the base package dependencies and RemoteCLIP runtime only.
It omits development tools, the server and the other model backends. CPU PyTorch
is selected explicitly to avoid installing CUDA dependencies on Linux. The
profile is pinned independently of the development environment. Other workflows
can install their named extras later.

### With an existing Python and pip

Use Python **3.13.7** for the verified path. A normal Python-created virtual
environment includes pip. The following uses the same package/profile and the
official CPU wheel index; Whirld itself still comes from the pinned archive.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r https://raw.githubusercontent.com/cercanasystems/whirld/6b70aa931f4dc03bd4abf2e6345c224f4643b277/requirements-quickstart.txt \
  'whirld[remoteclip] @ https://github.com/cercanasystems/whirld/archive/6b70aa931f4dc03bd4abf2e6345c224f4643b277.zip'
.venv/bin/whirld doctor
.venv/bin/whirld demo --output quickstart-results
```

These examples target macOS/Linux shell paths. Python/runtime compatibility on
additional platforms or interpreter versions needs independent verification.
Minimal Linux containers can omit native Python XML libraries: the official
`python:3.13.7-slim-bookworm` image needs Debian's `libexpat1` package. This is
an OS runtime dependency, not a system GDAL installation. Doctor reports this
specific failure before model acquisition.

## What to expect

The demo uses the trained **RemoteCLIP RN50** checkpoint, not the deterministic
reference backend. It compares farmland, forest, water and urban prompts. Scores
are normalized over those four prompts and depend on that choice; Sentinel-2
reflectance is also different from the RGB imagery used in model training.

The source is included in the installed package: 2,060,413 bytes, three 448×448
float32 RGB bands at 10 m resolution, with an internal validity mask. It needs
no imagery account or STAC request at run time. The first model acquisition is
408,397,807 bytes, plus Python/runtime installation downloads. Allow at least
4 GB RAM. Doctor estimates cache permissions and disk capacity; actual writes
can still fail if permissions or space change after inspection.

Doctor does not install packages, download weights, seed the registry or write
cache logs. Missing weights alone are a warning and a successful readiness
result. The demo checks the runtime, output and packaged sample before acquiring
weights. It verifies the cached checkpoint, scopes CPU/host state, then runs the
normal public classification pipeline. Subsequent runs reuse verified weights.

Outputs are a source TIFF, `scores.geojson`, a self-contained HTML/PNG preview and
`report.json`. The report records sample/checkpoint hashes, source attribution,
model revision, active registry fingerprint, runtime versions and elapsed time.
Failures remove only the new directory owned by that run; existing outputs and
installed models remain. Use a different `--output` directory to repeat.

## Troubleshooting

| Observation | Next step |
|---|---|
| `doctor` or `demo` is unknown | Install this preview; 0.3.0 predates these commands. Check `.venv/bin/whirld --help`. |
| Runtime is missing or an import is broken | Run `doctor`; its command targets the active interpreter, package source and pinned constraints. Broken imports use an explicit reinstall. |
| `libexpat.so.1` is missing in a minimal Linux image | Install the distribution's Expat runtime (`libexpat1` on Debian/Ubuntu), then rerun doctor. Reinstalling the Python package alone cannot supply it. |
| Model weights are absent | Run the demo; it displays the size and downloads the curated weights. |
| Model checksum/manifest or registry differs | Follow the explicit re-pull/registry remedy. The demo does not silently replace an override or corrupt installation. |
| Cache cannot be written or disk is insufficient | Choose a writable `WHIRLD_HOME` and rerun doctor. Keep it consistent between commands. |
| Output already exists | Choose a new `--output` path. Previous runs are not erased. |

`doctor --json` provides the same checks as machine-readable output and exits 7
when blockers remain. CPU is the starter default; GPU setup is not needed.

## Try your own image

After the demo, use a suitable north-up, projected-meter Sentinel-2 L2A GeoTIFF
with unambiguous native RGB band names B04/B03/B02 and canonical DN units:

```bash
.venv/bin/whirld classify scene.tif --model remoteclip-rn50 \
  --query 'a satellite image of an urban area' \
  --query 'a satellite image of water' \
  --device cpu --stream --all --output scores.geojson
```

Doctor checks your environment/model, not the spectral correctness of an arbitrary
input. Already-scaled reflectance needs explicit radiometry through STAC or the
[direct COG options](direct-cog.md). See [local streaming](local-streaming.md) for
input restrictions and [larger examples](../examples/README.md) for more workflows.

## Source and attribution

Contains modified Copernicus Sentinel data 2024, via Element 84 Earth Search.
The RGB sample comes from item `S2A_10SFH_20240722_0_L2A`, acquired
2024-07-22T19:03:53.453Z, near Sacramento. The crop and scale/offset conversion to
canonical DN, exact grid and byte identity are in the packaged
[sample manifest](../src/whirld/demo/data/sample.json).

Sample SHA-256: `c058a03a0da3efe368c5517b2eb5c8778cab31bb219ab4b345579665971aca4e`.
The [Copernicus legal notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice)
governs the imagery; the application/model Apache licenses are separate.
`scripts/prepare_demo_sample.py` documents regeneration from the public source.
Remote source or GDAL changes may produce different bytes; distributed sample
identity is always checked against its committed manifest.

## Recorded verification

Verified on 2026-11-24 with Python 3.13.7 and uv 0.11.20. A fresh macOS ARM64
environment used isolated Python, dependency and model download caches. The
measured phases were:

| Phase | Elapsed |
|---|---:|
| Provision Python and create `.venv` | 2.54 s |
| Install pinned starter dependencies | 24.46 s |
| Install the locally built candidate wheel | 0.55 s |
| First doctor, including initial runtime imports | 19.61 s |
| First demo, including the 408.4 MB model acquisition | 35.54 s |
| Repeat demo with verified cached weights | 3.09 s |

These phases exclude installing uv and building a source archive; they are
observations on one machine/network, not a setup-time promise. Source-archive
installation is verified separately below.

The exact uv commands above were then run in a new external `.venv` against the
immutable GitHub archive/profile, reusing the downloaded Python/dependency/model
caches. The package built from the archive, dependency checks passed and the
installed doctor/demo produced the real four-chip result. Imports resolved to
that environment's `site-packages`, without a checkout or development packages.

The pip alternative also passed in a separate fresh Python 3.13.7 `.venv` outside
the checkout. The exact archive/profile install took 42.15 s with a new pip cache
(mostly cold dependency downloads); dependency checks passed. With previously
verified model weights, doctor took 13.13 s and the real demo 2.51 s. The installed
origin records the pinned archive, and no developer/server/Prithvi packages were
present. These times exclude interpreter provisioning and model acquisition.

The installed Linux x86_64 starter path was also tested in the official
`python:3.13.7-slim-bookworm` image under emulation, adding only `libexpat1`.
CPU runtime installation took 40.27 s, excluding OS/Python/uv/model acquisition.
With verified weights preseeded and networking disabled, the real demo took
7.68 s. Its four features, ranking, geometry and sample/checkpoint identities
matched macOS; maximum score difference was 3.85e-6. Neither starter environment
contained development tools, server packages or the Prithvi backend.

Independent checks verified finite normalized scores, exact WGS84 chip geometry
and source/model provenance. Cold/repeat macOS TIFF, GeoJSON and HTML/PNG results
were byte-identical. The screenshot in the README shows that actual preview.
