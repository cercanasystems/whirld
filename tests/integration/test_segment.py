"""Integration tests for the segment pipeline (Python API + CLI + REST).

A fake Prithvi backend is injected via monkeypatch so the full pipeline (registry →
raster → sensor → HLS contract → 512 chips → mask → reassemble → GeoTIFF) runs
without the 1.3 GB checkpoint or TerraTorch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

torch = pytest.importorskip("torch")

import whirld  # noqa: E402
from whirld.core.fetch import Manifest  # noqa: E402
from whirld.models.prithvi import PrithviBackend  # noqa: E402


class _FakeSegModel:
    def __call__(self, x):  # type: ignore[no-untyped-def]
        b, _c, _t, h, w = x.shape
        logits = torch.zeros(b, 2, h, w)
        logits[:, 1, :, :] = x.mean(dim=(1, 2, 3, 4)).view(b, 1, 1)
        return logits

    def to(self, _device):  # type: ignore[no-untyped-def]
        return self

    def eval(self):  # type: ignore[no-untyped-def]
        return self


def _fake_manifest(name, paths=None):  # type: ignore[no-untyped-def]
    return Manifest(
        name=name,
        version="2.0.0",
        source_type="huggingface",
        sha256="0" * 64,
        weights_file="Prithvi_EO_V2_300M_BurnScars.pt",
    )


def _fake_backend(entry, manifest, device=None, **kwargs):  # type: ignore[no-untyped-def]
    return PrithviBackend(
        name=entry.name,
        version=manifest.version,
        device="cpu",
        classes=entry.output.classes or 2,
        model=_FakeSegModel(),
    )


@pytest.fixture
def fake_prithvi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the api load path to use the fake Prithvi backend."""
    monkeypatch.setattr(
        "whirld.core.fetch.ModelStore.load_manifest",
        lambda self, name: _fake_manifest(name),
    )
    monkeypatch.setattr("whirld.api.load_backend", _fake_backend)


def test_segment_api(whirld_home: Path, hls_tif: Path, fake_prithvi: None) -> None:
    """segment() writes a single-band uint8 mask GeoTIFF matching the input extent."""
    result = whirld.segment(hls_tif, model="prithvi-burn-scar")
    assert result.mask.dtype == np.uint8
    assert result.mask.shape == (320, 320)  # matches the fixture extent
    assert result.sensor == "hls"
    assert result.output_path is not None and result.output_path.exists()
    with rasterio.open(result.output_path) as ds:
        assert ds.count == 1 and ds.dtypes[0] == "uint8"
        assert ds.crs.to_string() == "EPSG:32613"
        assert ds.width == 320 and ds.height == 320


def test_segment_head_alias(
    whirld_home: Path, hls_tif: Path, fake_prithvi: None
) -> None:
    """`--model prithvi-eo-2 --head burn-scar` resolves to prithvi-burn-scar."""
    result = whirld.segment(hls_tif, model="prithvi-eo-2", head="burn-scar")
    assert result.model == "prithvi-burn-scar"


def test_segment_eo2_requires_head(
    whirld_home: Path, hls_tif: Path, fake_prithvi: None
) -> None:
    """The prithvi-eo-2 alias without --head is a clear error."""
    from whirld.errors import WhirldError

    with pytest.raises(WhirldError):
        whirld.segment(hls_tif, model="prithvi-eo-2")


def test_segment_default_output_name(
    whirld_home: Path, hls_tif: Path, fake_prithvi: None
) -> None:
    """The default mask name is <stem>_<model>.tif."""
    result = whirld.segment(hls_tif, model="prithvi-burn-scar")
    assert result.output_path.name == "hls_small_prithvi-burn-scar.tif"


def test_segment_url_default_output_is_local(
    whirld_home: Path,
    hls_tif: Path,
    fake_prithvi: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote item defaults use a local filename and discard URL query secrets."""
    from whirld.io.raster import read_raster

    raster = read_raster(hls_tif)
    monkeypatch.setattr("whirld.api._read_input", lambda *args, **kwargs: raster)
    monkeypatch.chdir(tmp_path)
    result = whirld.segment(
        "https://example.test/catalog/scene.json?token=secret",
        model="prithvi-burn-scar",
    )
    assert result.output_path == Path("scene_prithvi-burn-scar.tif")
    assert result.output_path.exists()
    assert not (tmp_path / "https:").exists()


def test_segment_usage_record(
    whirld_home: Path, hls_tif: Path, fake_prithvi: None
) -> None:
    """A segment usage record is written (command=segment)."""
    import json

    whirld.segment(hls_tif, model="prithvi-burn-scar")
    usage = whirld_home / "logs" / "usage.jsonl"
    last = json.loads(usage.read_text().strip().splitlines()[-1])
    assert last["command"] == "segment"
    assert last["model"] == "prithvi-burn-scar"
    assert last["error"] is None


def test_segment_cli(
    whirld_home: Path, hls_tif: Path, tmp_path: Path, fake_prithvi: None
) -> None:
    """`whirld segment` writes a mask GeoTIFF and reports the shape."""
    from typer.testing import CliRunner

    from whirld.cli import app

    out = tmp_path / "mask.tif"
    result = CliRunner().invoke(
        app,
        ["segment", str(hls_tif), "--model", "prithvi-burn-scar", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "Segmented" in result.output
    assert out.exists()


def test_segment_rest(whirld_home: Path, hls_tif: Path, monkeypatch) -> None:
    """POST /segment returns an image/tiff mask (warm session, fake backend)."""
    import warnings

    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

    from whirld.server.app import create_app

    # The server session resolves the backend through core.session, not api.
    monkeypatch.setattr(
        "whirld.core.fetch.ModelStore.load_manifest",
        lambda self, name: _fake_manifest(name),
    )
    monkeypatch.setattr("whirld.core.session.load_backend", _fake_backend)

    app = create_app(device="cpu", preload=["prithvi-burn-scar"])
    with TestClient(app) as client, open(hls_tif, "rb") as handle:
        resp = client.post(
            "/segment",
            files={"file": ("hls.tif", handle, "image/tiff")},
            data={"model": "prithvi-burn-scar"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/tiff"
    with rasterio.MemoryFile(resp.content) as mem, mem.open() as ds:
        assert ds.count == 1 and ds.dtypes[0] == "uint8"


@pytest.mark.parametrize("surface", ["api", "rest"])
def test_segment_preserves_input_validity(
    whirld_home: Path,
    hls_tif: Path,
    fake_prithvi: None,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    """Missing input pixels remain nodata while predicted background stays valid."""
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(hls_tif, "r+") as dataset:
            pixels = dataset.read()
            pixels[0, 1, 1] = 65535
            dataset.nodata = 65535
            dataset.write(pixels)
            validity = np.full((dataset.height, dataset.width), 255, dtype=np.uint8)
            validity[3, 3] = 0
            dataset.write_mask(validity)
    if surface == "api":
        result = whirld.segment(hls_tif, model="prithvi-burn-scar", threshold=1.0)
        assert result.output_path is not None
        payload = result.output_path.read_bytes()
    else:
        from fastapi.testclient import TestClient

        from whirld.server.app import create_app

        monkeypatch.setattr(
            "whirld.core.fetch.ModelStore.load_manifest",
            lambda self, name: _fake_manifest(name),
        )
        monkeypatch.setattr("whirld.core.session.load_backend", _fake_backend)
        with TestClient(create_app(device="cpu")) as client:
            response = client.post(
                "/segment",
                files={"file": ("hls.tif", hls_tif.read_bytes(), "image/tiff")},
                data={"model": "prithvi-burn-scar", "threshold": "1"},
            )
        assert response.status_code == 200, response.text
        payload = response.content
    with rasterio.MemoryFile(payload) as memfile, memfile.open() as dataset:
        masked = dataset.read(1, masked=True)
        assert dataset.nodata == 255
        assert masked.mask[1, 1] and masked.mask[3, 3]
        assert masked.count() == masked.size - 2
        assert (masked.compressed() == 0).all()
