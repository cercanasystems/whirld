"""Evaluation CPU scopes restore caller state changed by Lightning-style seeding."""

from __future__ import annotations

import os
import random
import subprocess
import sys
from contextlib import nullcontext
from typing import Literal, cast

import numpy as np
import pytest
import torch

from whirld.evaluation.errors import EvaluationError
from whirld.evaluation.runtime import EvaluationCpuRuntime


class HostSnapshot:
    """Capture only CPU/host state for independent restoration assertions."""

    def __init__(self) -> None:
        """Copy caller generators, thread count and both Lightning environment keys."""
        self.python = random.getstate()
        self.numpy = np.random.get_state(legacy=True)
        self.torch = torch.random.get_rng_state().clone()
        self.threads = torch.get_num_threads()
        self.environment = {
            key: os.environ.get(key) for key in ("PL_GLOBAL_SEED", "PL_SEED_WORKERS")
        }

    def assert_restored(self) -> None:
        """Require exact generator states, threads and environment presence/values."""
        assert random.getstate() == self.python
        np.testing.assert_equal(np.random.get_state(legacy=True), self.numpy)
        assert torch.equal(torch.random.get_rng_state(), self.torch)
        assert torch.get_num_threads() == self.threads
        assert {
            key: os.environ.get(key) for key in self.environment
        } == self.environment


@pytest.mark.parametrize("failure", ["none", "error", "interrupt"])
@pytest.mark.parametrize("environment", ["absent", "present", "mixed"])
def test_lightning_seed_side_effects_are_restored(
    monkeypatch: pytest.MonkeyPatch,
    failure: Literal["none", "error", "interrupt"],
    environment: str,
) -> None:
    """Normal returns and BaseExceptions restore state and absent/present seed keys."""
    for key in ("PL_GLOBAL_SEED", "PL_SEED_WORKERS"):
        monkeypatch.delenv(key, raising=False)
    if environment == "present":
        monkeypatch.setenv("PL_GLOBAL_SEED", "123")
        monkeypatch.setenv("PL_SEED_WORKERS", "0")
    elif environment == "mixed":
        monkeypatch.setenv("PL_GLOBAL_SEED", "0")
    before = HostSnapshot()
    expected = (
        nullcontext()
        if failure == "none"
        else pytest.raises(RuntimeError if failure == "error" else KeyboardInterrupt)
    )
    with expected:
        with EvaluationCpuRuntime(1).activate():
            assert torch.get_num_threads() == 1
            assert torch.initial_seed() == 2
            random.seed(2)
            np.random.seed(2)
            torch.manual_seed(2)
            os.environ["PL_GLOBAL_SEED"] = "2"
            os.environ["PL_SEED_WORKERS"] = "1"
            random.random()
            np.random.random()
            torch.rand(1)
            torch.set_num_threads(2)
            if failure == "error":
                raise RuntimeError("simulated backend failure")
            if failure == "interrupt":
                raise KeyboardInterrupt()
    before.assert_restored()


def test_nested_cpu_scopes_restore_the_enclosing_state() -> None:
    """Each owned scope restores its immediate caller rather than a global snapshot."""
    before = HostSnapshot()
    with EvaluationCpuRuntime(1).activate():
        torch.rand(7)
        random.random()
        np.random.random()
        outer = HostSnapshot()
        with EvaluationCpuRuntime(2).activate():
            random.seed(99)
            np.random.seed(99)
            torch.default_generator.manual_seed(99)
            os.environ["PL_GLOBAL_SEED"] = "99"
            os.environ.pop("PL_SEED_WORKERS", None)
        outer.assert_restored()
    before.assert_restored()


def test_cpu_snapshot_does_not_probe_or_initialize_accelerators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly empty device set keeps runtime ownership on the CPU."""

    def unexpected(*args: object, **kwargs: object) -> None:
        """Fail if the CPU context queries or initializes CUDA for its snapshots."""
        raise AssertionError("unexpected accelerator operation")

    for name in (
        "is_available",
        "is_initialized",
        "device_count",
        "get_rng_state",
        "get_rng_state_all",
        "manual_seed_all",
        "_lazy_init",
    ):
        monkeypatch.setattr(torch.cuda, name, unexpected)
    before = HostSnapshot()
    with EvaluationCpuRuntime(1).activate():
        assert torch.initial_seed() == 2
        torch.rand(1)
    before.assert_restored()


@pytest.mark.parametrize("threads", [0, -1, True, False, 1.5, "2"])
def test_thread_count_must_be_a_positive_strict_integer(threads: object) -> None:
    """Direct runtime construction rejects ambiguous values before touching state."""
    with pytest.raises(EvaluationError, match="threads"):
        EvaluationCpuRuntime(cast(int, threads))


def test_runtime_import_stays_lazy() -> None:
    """Importing the owner does not load NumPy or the optional Torch dependency."""
    code = (
        "import sys; from whirld.evaluation.runtime import EvaluationCpuRuntime; "
        "assert 'numpy' not in sys.modules and 'torch' not in sys.modules; "
        "EvaluationCpuRuntime(1)"
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_activation_uses_the_pinned_configuration_seed() -> None:
    """Python, NumPy and CPU Torch all start from the deployment's fixed seed 2."""
    import torch

    before = HostSnapshot()
    generator = torch.Generator().manual_seed(2)
    expected = torch.rand(4, generator=generator)
    with EvaluationCpuRuntime(1).activate():
        assert random.random() == random.Random(2).random()
        assert np.random.random() == np.random.RandomState(2).random_sample()
        assert torch.equal(torch.rand(4), expected)
    before.assert_restored()
