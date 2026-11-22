"""Curated RemoteCLIP variants preserve the existing RGB/backend contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

from whirld.core.contract import translate
from whirld.core.registry import Registry
from whirld.io.raster import RasterSource
from whirld.models.remoteclip import RemoteCLIPBackend


@pytest.mark.parametrize(
    "name,architecture,filename,size,digest",
    [
        (
            "remoteclip-rn50",
            "RN50",
            "RemoteCLIP-RN50.pt",
            408397807,
            "11657d33a87fc197e19a7654a76a632415ebea42feb8a61025ef389f8990c0f9",
        ),
        (
            "remoteclip-vit-l14",
            "ViT-L-14",
            "RemoteCLIP-ViT-L-14.pt",
            1710613765,
            "fcc2a7e21e171f4ffcb7a9c0206b8b74ac0c9eb83c67b576958b7a4ed6c8cecb",
        ),
    ],
)
def test_variant_identity_and_contract(
    whirld_home: Path,
    name: str,
    architecture: str,
    filename: str,
    size: int,
    digest: str,
) -> None:
    """Select a local allowed architecture and an immutable verified artifact."""
    registry = Registry()
    entry = registry.get(name)
    original = registry.get("remoteclip")
    assert original.model_name == "ViT-B-32"
    assert entry.backend == "remoteclip" and entry.category == "classification"
    assert entry.model_name == architecture
    assert entry.source.repo == "chendelong/RemoteCLIP"
    assert entry.source.revision == "bf1d8a3ccf2ddbf7c875705e46373bfe542bce38"
    assert entry.source.files == [filename]
    assert entry.distribution.size_bytes == size
    assert entry.distribution.sha256 == digest
    assert entry.trust == "first-party" and entry.license == "Apache-2.0"
    assert entry.band_contract == original.band_contract
    assert entry.output == original.output
    backend = RemoteCLIPBackend(name, entry.version, "cpu", arch=architecture)
    assert backend.arch == architecture


@pytest.mark.parametrize("name", ["remoteclip-rn50", "remoteclip-vit-l14"])
def test_variant_translation_is_rgb_clip_standardization(
    whirld_home: Path,
    name: str,
) -> None:
    """Reordered source labels still produce native224 RGB with published constants."""
    entry = Registry().get(name)
    values = np.array([8000.0, 3000.0, 2000.0], dtype=np.float32)
    source = RasterSource(
        data=np.broadcast_to(values[:, None, None], (3, 224, 224)).copy(),
        crs="EPSG:32630",
        transform=from_origin(500000, 4500000, 10, 10),
        band_descriptions=["B02", "B04", "B03"],
        tags={},
    )
    translated = translate(source, entry, "sentinel-2-l2a")
    rgb = np.array([3000.0, 2000.0, 8000.0], dtype=np.float32) * 0.0001
    expected = (rgb - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array(
        [0.26862954, 0.26130258, 0.27577711]
    )
    np.testing.assert_allclose(translated.data[:, 0, 0], expected, atol=1e-6)
    assert translated.data.shape == (3, 224, 224)
    assert translated.valid_mask is not None and translated.valid_mask.all()
