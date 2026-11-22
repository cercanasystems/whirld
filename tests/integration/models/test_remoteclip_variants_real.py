"""Opt-in actual-checkpoint preprocessing and public eager/stream comparisons."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

_HOME_ENV = "WHIRLD_TEST_REMOTECLIP_VARIANTS_HOME"


class VariantMeasurement(BaseModel):
    """Persisted actual CPU comparison, including checkpoint coverage observations."""

    model_config = ConfigDict(frozen=True)
    model: str
    chip_count: int
    batch_size: int
    max_abs_error: float
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    elapsed_seconds: float


@pytest.mark.skipif(
    not os.environ.get(_HOME_ENV),
    reason=f"set {_HOME_ENV} to an installed variant home",
)
@pytest.mark.parametrize("name", ["remoteclip-rn50", "remoteclip-vit-l14"])
def test_real_variant_preprocessing_and_public_streaming(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pinned variants load all state and preserve prompt probabilities and geometry.

    Requires pre-pulled verified weights and network access to the pinned public
    Sentinel-2 item. No checkpoint is downloaded by this test. A small crop retains
    four native224 chips and exercises a partial final batch of three.
    """
    torch = pytest.importorskip("torch")
    open_clip = pytest.importorskip("open_clip")
    image_module = pytest.importorskip("PIL.Image")
    import whirld
    from examples.common import SENTINEL_SCENE
    from whirld.core.registry import Registry

    home = os.environ[_HOME_ENV]
    monkeypatch.setenv("WHIRLD_HOME", home)
    torch.set_num_threads(2)
    entry = Registry().get(name)
    assert (
        open_clip.get_model_config(entry.model_name)["vision_cfg"]["image_size"] == 224
    )
    normalization = entry.band_contract.normalization
    rgb = np.random.default_rng(762).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    official = open_clip.image_transform(224, is_train=False)(
        image_module.fromarray(rgb)
    )
    independent = (
        rgb.transpose(2, 0, 1).astype(np.float32) / 255
        - np.array(normalization.mean, dtype=np.float32)[:, None, None]
    ) / np.array(normalization.std, dtype=np.float32)[:, None, None]
    np.testing.assert_allclose(official.numpy(), independent, rtol=1e-6, atol=1e-6)

    coverage: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    real_load = torch.nn.Module.load_state_dict

    def observe_load(
        module: Any,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Observe the real checkpoint load without weakening its validation."""
        result = real_load(module, state_dict, strict=strict, assign=assign)
        learned = set(dict(module.named_parameters()))
        assert not learned.difference(state_dict)
        coverage.append((tuple(result.missing_keys), tuple(result.unexpected_keys)))
        assert not result.missing_keys and not result.unexpected_keys
        return result

    monkeypatch.setattr(torch.nn.Module, "load_state_dict", observe_load)
    prompts = ["a satellite image of farmland", "a satellite image of water"]
    bbox = (-121.57, 38.53, -121.54, 38.56)
    eager = whirld.classify(
        SENTINEL_SCENE.item_url,
        model=name,
        query=prompts,
        top_k=4,
        bbox=bbox,
        sensor=SENTINEL_SCENE.sensor,
        device="cpu",
        no_warnings=True,
    )
    expected = json.loads(json.dumps(eager.feature_collection))
    assert eager.scores.shape == (4, 2)
    assert np.isfinite(eager.scores).all()
    np.testing.assert_allclose(eager.scores.sum(axis=1), 1, atol=1e-6)
    assert ((eager.scores >= 0) & (eager.scores <= 1)).all()
    measurements = []
    for size in (1, 3):
        started = time.perf_counter()
        output = tmp_path / f"{name}-{size}.geojson"
        result = whirld.classify_to_disk(
            SENTINEL_SCENE.item_url,
            model=name,
            query=prompts,
            top_k=None,
            bbox=bbox,
            sensor=SENTINEL_SCENE.sensor,
            device="cpu",
            batch_size=size,
            no_warnings=True,
            output=output,
        )
        actual = json.loads(result.output_path.read_text())
        assert result.chip_count == result.feature_count == 4
        assert {key: value for key, value in actual.items() if key != "features"} == {
            key: value for key, value in expected.items() if key != "features"
        }
        errors = []
        for found, wanted in zip(actual["features"], expected["features"], strict=True):
            assert found["geometry"] == wanted["geometry"]
            assert (
                found["properties"]["chip_index"] == wanted["properties"]["chip_index"]
            )
            values = [found["properties"]["scores"][prompt] for prompt in prompts]
            reference = [wanted["properties"]["scores"][prompt] for prompt in prompts]
            np.testing.assert_allclose(values, reference, rtol=1e-4, atol=1e-5)
            errors.extend(np.abs(np.asarray(values) - reference).tolist())
        measurements.append(
            VariantMeasurement(
                model=name,
                chip_count=4,
                batch_size=size,
                max_abs_error=max(errors),
                missing_keys=coverage[-1][0],
                unexpected_keys=coverage[-1][1],
                elapsed_seconds=time.perf_counter() - started,
            )
        )
    assert len(coverage) == 3
    (tmp_path / f"{name}-report.json").write_text(
        json.dumps([record.model_dump() for record in measurements], indent=2)
    )
