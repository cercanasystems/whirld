"""Behavioral tests for per-call execution options on resident model backends."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from whirld.core.options import InferenceOptions  # noqa: E402
from whirld.errors import InvalidInputError, WhirldError  # noqa: E402
from whirld.models.base import InferenceContext, ModelBackend  # noqa: E402
from whirld.models.clay_torch import ClayTorchBackend  # noqa: E402
from whirld.models.prithvi import PrithviBackend  # noqa: E402
from whirld.models.remoteclip import RemoteCLIPBackend  # noqa: E402


class RecordingModel(torch.nn.Module):
    """Minimal tensor interfaces recording actual model invocation batch sizes."""

    def __init__(self, kind: str) -> None:
        """Select the backend's output shape and begin with no recorded calls."""
        super().__init__()
        self.kind = kind
        self.sizes: list[int] = []
        self.fail_next = False
        self.logit_scale = torch.tensor(1.0)

    def forward(self, value: Any) -> Any:
        """Record real execution and return deterministic encoder/logit tensors."""
        pixels = value["pixels"] if isinstance(value, dict) else value
        self.sizes.append(int(pixels.shape[0]))
        if self.fail_next:
            self.fail_next = False
            raise WhirldError("simulated model failure")
        if self.kind == "clay":
            return (torch.ones(pixels.shape[0], 1, 2),)
        return torch.ones(pixels.shape[0], 2, pixels.shape[-2], pixels.shape[-1])

    def encode_image(self, pixels: Any) -> Any:
        """Record image batch size using the same failure controls as forward."""
        self.forward(pixels)
        return torch.ones(pixels.shape[0], 2)

    def encode_text(self, tokens: Any) -> Any:
        """Return constant two-component text features."""
        return torch.ones(tokens.shape[0], 2)


def _context(batch_size: int | None = None) -> InferenceContext:
    """Create immutable execution policy with enough metadata for all backends."""
    return InferenceContext(
        sensor="test",
        gsd_m=10,
        wavelengths=[0.4, 0.5, 0.6],
        options=InferenceOptions(batch_size=batch_size),
    )


@pytest.mark.parametrize("kind", ["clay", "remoteclip", "prithvi"])
def test_model_batch_policy_is_local_to_each_call(kind: str) -> None:
    """Overrides change actual forward calls without changing later default calls."""
    model = RecordingModel(kind)
    backend: ModelBackend
    if kind == "clay":
        backend = ClayTorchBackend(
            "test", "1", "cpu", embed_dim=2, patch_size=1, encoder=model, batch_size=3
        )
    elif kind == "remoteclip":
        backend = RemoteCLIPBackend(
            "test",
            "1",
            "cpu",
            arch="ViT-B-32",
            model=model,
            tokenizer=lambda prompts: torch.ones(len(prompts), 1),
            batch_size=3,
        )
    else:
        backend = PrithviBackend(
            "test", "1", "cpu", classes=2, model=model, batch_size=3
        )
    chips = np.ones((5, 6 if kind == "prithvi" else 3, 2, 2), dtype=np.float32)

    def run(context: InferenceContext) -> np.ndarray:
        """Exercise each public backend method, with its unchanged call signature."""
        if kind == "clay":
            return backend.embed(chips, context)
        if kind == "remoteclip":
            return backend.classify(chips, ["forest"], context)
        return backend.segment(chips, context=context)

    override = run(_context(2))
    assert model.sizes == [2, 2, 1]
    model.sizes.clear()
    default = run(_context())
    assert model.sizes == [3, 2]
    np.testing.assert_array_equal(override, default)
    model.fail_next = True
    with pytest.raises(WhirldError, match="simulated model failure"):
        run(_context(1))
    model.sizes.clear()
    np.testing.assert_array_equal(default, run(_context()))
    assert model.sizes == [3, 2]


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_batch_options_raise_domain_errors(size: object) -> None:
    """Public option constructors reject nonpositive and noninteger batch sizes."""
    with pytest.raises(InvalidInputError):
        InferenceOptions.from_arguments(batch_size=size)


def test_inference_options_are_immutable() -> None:
    """A call cannot mutate the execution policy shared with its context."""
    from pydantic import ValidationError

    options = InferenceOptions(batch_size=2)
    with pytest.raises(ValidationError, match="frozen"):
        options.batch_size = 3


def test_concurrent_calls_keep_independent_batch_policies() -> None:
    """Interleaved requests on one resident model retain their own batch sizes."""
    barrier = Barrier(2)
    observed: dict[int, list[int]] = {}

    class InterleavedModel(RecordingModel):
        """Synchronize each request's first forward call to force overlap."""

        def forward(self, value: Any) -> Any:
            """Record batch sizes separately using each request's pixel marker."""
            marker = int(value[0, 0, 0, 0, 0])
            sizes = observed.setdefault(marker, [])
            sizes.append(int(value.shape[0]))
            if len(sizes) == 1:
                barrier.wait(timeout=5)
            return super().forward(value)

    backend = PrithviBackend(
        "test", "1", "cpu", classes=2, model=InterleavedModel("prithvi"), batch_size=3
    )

    def run(marker: int, batch_size: int) -> np.ndarray:
        """Invoke one request with distinct pixels and a per-call batch policy."""
        chips = np.full((5, 6, 2, 2), marker, dtype=np.float32)
        return backend.segment(chips, context=_context(batch_size))

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run, 1, 1)
        second = executor.submit(run, 2, 4)
        np.testing.assert_array_equal(
            first.result(timeout=10), second.result(timeout=10)
        )
    assert observed == {1: [1, 1, 1, 1, 1], 2: [4, 1]}
