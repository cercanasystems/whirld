# RemoteCLIP model variants

Whirld's existing `remoteclip` model remains the ViT-B/32 checkpoint. Two new
registry entries expose the other published RemoteCLIP architectures through the
same classification API, native 224-pixel RGB band contract and streaming path:

| Whirld model | OpenCLIP architecture | Checkpoint bytes |
|---|---|---:|
| `remoteclip-rn50` | RN50 | 408,397,807 |
| `remoteclip` | ViT-B-32 | 605,208,421 |
| `remoteclip-vit-l14` | ViT-L-14 | 1,710,613,765 |

These are separate model identities; installing either variant does not replace
the existing ViT-B/32 cache. Architecture choice changes the network and checkpoint,
not the output schema. No accuracy or speed improvement is implied without a
representative evaluation. Larger weights and inference batches can increase RAM
and VRAM use; start with a streaming batch of one on constrained machines.

## Setup and run

From a checkout containing these entries, use Python 3.13.7 and install the
classification extra in `.venv`:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -e '.[remoteclip]'
.venv/bin/whirld pull remoteclip-rn50
.venv/bin/whirld pull remoteclip-vit-l14
```

The checkpoints download separately and are SHA-256 verified before installation.
Use the same text queries and bbox when comparing models. This pinned public crop
produces four native chips, including padded edges:

```bash
.venv/bin/whirld classify \
  'https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2A_10SFH_20240722_0_L2A' \
  --model remoteclip-rn50 --sensor sentinel-2-l2a \
  --bbox=-121.57,38.53,-121.54,38.56 \
  --query 'a satellite image of farmland' --query 'a satellite image of water' \
  --device cpu --stream --batch-size 1 --all --output rn50-scores.geojson
```

Repeat with `--model remoteclip-vit-l14 --output l14-scores.geojson`. The Python
API accepts the same model identifiers:

```python
import whirld

result = whirld.classify_to_disk(
    'https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2A_10SFH_20240722_0_L2A',
    model='remoteclip-vit-l14',
    sensor='sentinel-2-l2a',
    bbox=(-121.57, 38.53, -121.54, 38.56),
    query=['a satellite image of farmland', 'a satellite image of water'],
    top_k=None,
    device='cpu',
    batch_size=1,
    output='l14-scores.geojson',
)
print(result.chip_count, result.feature_count)
```

Ordinary `whirld.classify()` remains available for resident arrays and returns the
same GeoJSON contract. See [classification streaming](classification-streaming.md)
for ordering, threshold/top-k, temporary disk and publication guarantees.

## Preprocessing and interpretation

The selected OpenCLIP architecture is instantiated locally with no pretrained
bootstrap; Whirld's pinned checkpoint is its weights source. RN50 and ViT-L/14
both use 224 × 224 RGB inputs and the published CLIP means/stds. The real tests
compare this normalization against OpenCLIP's image transform on an already-sized
RGB image and require complete state loading, including zero missing/unexpected
checkpoint keys and coverage of learned parameters.

Whirld uses Sentinel-2 B04/B03/B02, scales DN by 0.0001, and applies CLIP
normalization. RemoteCLIP was trained on RGB imagery; this reflectance-to-RGB
mapping remains approximate and does not reproduce display tone mapping or
atmospheric corrections. Scores are relative softmax probabilities over supplied
prompts. A single prompt is compared with the existing neutral background prompt;
these outputs are not calibrated land-cover confidence or validated classifications.

## Provenance and verification

The authors' [official repository](https://github.com/ChenDelong1999/RemoteCLIP)
publishes RN50, ViT-B/32 and ViT-L/14 OpenCLIP-compatible checkpoints. All three
Whirld entries use the [author's checkpoint repository](https://huggingface.co/chendelong/RemoteCLIP/tree/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38)
at revision `bf1d8a3ccf2ddbf7c875705e46373bfe542bce38`, with Apache-2.0 model
licensing and the citation available through `whirld info MODEL`.

The new variants are pinned to:

- RN50: `RemoteCLIP-RN50.pt`, SHA-256
  `11657d33a87fc197e19a7654a76a632415ebea42feb8a61025ef389f8990c0f9`.
- ViT-L/14: `RemoteCLIP-ViT-L-14.pt`, SHA-256
  `fcc2a7e21e171f4ffcb7a9c0206b8b74ac0c9eb83c67b576958b7a4ed6c8cecb`.

The public scene contains modified Copernicus Sentinel data 2024 via Element 84.
It is the same July 22, 2024 Sacramento scene as the
[classification example](../examples/classification.md), using a smaller crop.

To run actual-checkpoint preprocessing/state and eager/stream comparisons, first
pull both variants into a chosen home, then run:

```bash
WHIRLD_TEST_REMOTECLIP_VARIANTS_HOME="$HOME/.whirld" \
  .venv/bin/python -m pytest tests/integration/models/test_remoteclip_variants_real.py
```

The test needs network access to the public STAC/raster source, but does not
download checkpoints. It compares streaming batches one and three against eager
scores at predeclared `rtol=1e-4`, `atol=1e-5`, verifies finite prompt probabilities,
retains all four chips, and checks exact ranking and geometry. Normal tests skip
these large real-weight runs. This establishes model execution and numerical
agreement, not downstream predictive accuracy or GPU performance.

### Recorded CPU validation

On 2026-11-22, the installed pinned checkpoints passed the test above on the
development CPU with two Torch threads. Each run retained all four crop chips,
with identical feature ranking and WGS84 geometry. Every eager/stream checkpoint
load reported zero missing and zero unexpected keys; all learned parameters were
covered. OpenCLIP's native RGB transform matched independent mean/std arithmetic.

| Model | Stream batch | Maximum absolute score error | Stream call seconds |
|---|---:|---:|---:|
| RN50 | 1 | 4.95e-6 | 1.39 |
| RN50 | 3 | 6.26e-6 | 1.29 |
| ViT-L/14 | 1 | 1.91e-6 | 2.69 |
| ViT-L/14 | 3 | 1.91e-6 | 2.43 |

These stream-call timings include model/source access and output publication
after the eager comparison, with installed weights and potentially warm source
caches. They are validation observations, not a controlled model-speed benchmark
or a GPU memory measurement.
