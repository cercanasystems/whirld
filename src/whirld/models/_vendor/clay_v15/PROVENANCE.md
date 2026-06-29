# Provenance — vendored Clay v1.5 encoder

## What this is

The minimal subset of the Clay v1.5 model needed to run the **encoder** and obtain
embeddings. We vendor it (rather than depend on the `claymodel` PyPI package)
because `claymodel==1.5.0` hard-pins `torch==2.4.0`, which has no wheel for
Python 3.13 (this project's interpreter), making it un-installable here. The
encoder itself needs only `torch` + `einops`.

## Upstream source

- Package: `claymodel==1.5.0` (PyPI), source repo
  https://github.com/Clay-foundation/model
- License: **Apache License 2.0** (see `LICENSE` in this directory, retrieved from
  the upstream repository).
- The model weights this encoder loads come from Hugging Face
  `made-with-clay/Clay`, file `v1.5/clay-v1.5.ckpt`, revision
  `70200ebcccdf67bf2a0cb9984c77ddee26c10ed2`
  (sha256 `21432069250b9b3f9a65ffd0071c5ad56b793247285ab0604edf7f531d4798d0`).

## Files and modifications

| File | Origin | Modification |
|---|---|---|
| `encoder.py` | `claymodel/model.py` → the `Encoder` class | Imports adapted to `.backbone` / `.factory` / `.utils`; added the `clay_v15_large_encoder()` factory (the `clay_mae_large` config, `mask_ratio=0.0`, `shuffle=False`). Class body unchanged. |
| `backbone.py` | `claymodel` (vit-pytorch-derived `Transformer`) | Verbatim. |
| `factory.py` | `claymodel` (DOFA-derived `DynamicEmbedding`) | `from src.utils` → `from .utils`. Otherwise verbatim. |
| `utils.py` | `claymodel` (`posemb_sincos_*`) | Verbatim. |

Deliberately **omitted**: `ClayMAE` / `ClayMAEModule` and the DINOv2 teacher,
which require timm + torchvision + lightning and are not needed for inference.

## How weights are loaded

The Lightning checkpoint stores encoder weights under the `model.encoder.` prefix.
Strip that prefix and `load_state_dict(strict=False)` into the encoder built by
`clay_v15_large_encoder()` — this loads 265 tensors with 0 missing / 0 unexpected
keys. `torch.load(..., weights_only=False)` is required (full Lightning ckpt).
