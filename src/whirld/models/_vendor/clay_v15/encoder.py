"""Clay v1.5 ViT-MAE *encoder* — vendored from claymodel 1.5.0.

The :class:`Encoder` body below is copied **verbatim** from ``claymodel/model.py``
(Clay-foundation/model, Apache-2.0). Only the imports were adapted to this
vendored package (``from .backbone`` / ``from .factory`` / ``from .utils``).

Why vendored rather than depended on: ``claymodel==1.5.0`` hard-pins
``torch==2.4.0``, which has no wheel for Python 3.13, so it cannot be pip-installed
on this project's interpreter. The encoder alone needs only ``torch`` + ``einops``
(plus the three helper modules here); the full ``ClayMAE``/``ClayMAEModule`` pull
in timm + torchvision + lightning solely to build the frozen DINOv2 teacher, which
is not needed to produce embeddings.

See ``PROVENANCE.md`` for the exact upstream commit and file list.
"""

import math

import torch
from einops import rearrange, repeat
from torch import nn

from .backbone import Transformer
from .factory import DynamicEmbedding
from .utils import posemb_sincos_2d_with_gsd


class Encoder(nn.Module):
    """Clay v1.5 encoder — copied verbatim from claymodel==1.5.0 model.py."""

    def __init__(
        self,
        mask_ratio,
        patch_size,
        shuffle,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.dim = dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.patch_embedding = DynamicEmbedding(
            wave_dim=128,
            num_latent_tokens=128,
            patch_size=patch_size,
            embed_dim=dim,
            is_decoder=False,
        )
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=int(dim * mlp_ratio),
            fused_attn=True,
        )

    def to_patch_embed(self, cube, waves):
        patches, waves_encoded = self.patch_embedding(cube, waves)
        return patches, waves_encoded

    def add_encodings(self, patches, time, latlon, gsd):
        B, L, D = patches.shape
        grid_size = int(math.sqrt(L))
        self.num_patches = grid_size**2
        pos_encoding = (
            posemb_sincos_2d_with_gsd(
                h=grid_size, w=grid_size, dim=(self.dim - 8), gsd=gsd
            )
            .to(patches.device)
            .detach()
        )
        time_latlon = torch.hstack((time, latlon)).to(patches.device).detach()
        pos_encoding = repeat(pos_encoding, "L D -> B L D", B=B)
        time_latlon = repeat(time_latlon, "B D -> B L D", L=L)
        pos_metadata_encoding = torch.cat((pos_encoding, time_latlon), dim=-1)
        patches = patches + pos_metadata_encoding
        return patches

    def mask_out(self, patches):
        B, L, D = patches.shape
        if self.shuffle:
            noise = torch.randn((B, L), device=patches.device)
        else:
            noise = rearrange(
                torch.arange(B * L, device=patches.device),
                "(B L) -> B L",
                B=B,
                L=L,
            )
        random_indices = torch.argsort(noise, dim=-1)
        reverse_indices = torch.argsort(random_indices, dim=-1)
        num_masked_patches = int(self.mask_ratio * self.num_patches)
        masked_indices, unmasked_indices = (
            random_indices[:, :num_masked_patches],
            random_indices[:, num_masked_patches:],
        )
        masked_matrix = torch.zeros((B, L), device=patches.device)
        masked_matrix[:, :num_masked_patches] = 1
        masked_matrix = torch.gather(masked_matrix, dim=1, index=reverse_indices)
        batch_indices = rearrange(
            torch.arange(B, device=patches.device), "B -> B 1"
        )
        unmasked_patches = patches[batch_indices, unmasked_indices, :]
        _ = patches[batch_indices, masked_indices, :]
        return unmasked_patches, unmasked_indices, masked_indices, masked_matrix

    def forward(self, datacube):
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],
            datacube["time"],
            datacube["latlon"],
            datacube["gsd"],
            datacube["waves"],
        )
        B, C, H, W = cube.shape
        patches, waves_encoded = self.to_patch_embed(cube, waves)
        patches = self.add_encodings(patches, time, latlon, gsd)
        (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        ) = self.mask_out(patches)
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)
        unmasked_patches = torch.cat((cls_tokens, unmasked_patches), dim=1)
        encoded_unmasked_patches = self.transformer(unmasked_patches)
        return (
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )


def clay_v15_large_encoder() -> Encoder:
    """Build the Clay v1.5 "large" encoder (``claymodel`` ``clay_mae_large``).

    Uses ``mask_ratio=0.0`` and ``shuffle=False`` so inference is deterministic
    and every patch (plus the CLS token) passes through the transformer — the
    training-time masking has no effect on the learned weights.

    Returns:
        An untrained :class:`Encoder` with the large architecture; load weights
        from the checkpoint's ``model.encoder.*`` tensors.
    """
    return Encoder(
        mask_ratio=0.0,
        patch_size=8,
        shuffle=False,
        dim=1024,
        depth=24,
        heads=16,
        dim_head=64,
        mlp_ratio=4,
    )
