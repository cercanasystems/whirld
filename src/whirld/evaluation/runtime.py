"""Owned CPU and host state for reproducible evaluation with Lightning backends."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager

from .errors import EvaluationError


class EvaluationCpuRuntime:
    """Scope CPU inference and restore host state changed by Lightning seeding.

    The resource owner restores Python's random generator, NumPy's legacy global
    generator, Torch's CPU generator, intra-op threads and Lightning seed settings.
    It does not snapshot accelerator generators or claim to restore GPU state.
    Use serially: these host settings are process-wide, not thread-local.
    """

    _SEED_ENVIRONMENT = ("PL_GLOBAL_SEED", "PL_SEED_WORKERS")

    def __init__(self, threads: int) -> None:
        """Bind a positive strict CPU thread count without importing Torch.

        Args:
            threads: Torch intra-op threads for this evaluation scope.

        Raises:
            EvaluationError: The requested count is not a positive strict integer.
        """
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise EvaluationError("Evaluation threads must be a positive integer.")
        self._threads = threads

    @staticmethod
    def _restore_environment(previous: Mapping[str, str | None]) -> None:
        """Restore captured seed keys, preserving absent versus present values."""
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Own runtime changes through completion, errors and interrupts.

        Yields:
            A CPU scope seeded to 2, matching the pinned deployment configuration.
            Lightning may additionally seed Python/NumPy and change its environment
            keys while loading the backend; their caller states are restored too.

        Raises:
            ImportError: NumPy or the optional Torch runtime is unavailable.
            BaseException: Errors and interrupts propagate after restoration.

        Torch is imported only at activation. Explicit empty accelerator devices
        keep RNG snapshots on the CPU without probing accelerator availability.
        """
        import numpy as np
        import torch

        previous_environment = {
            key: os.environ.get(key) for key in self._SEED_ENVIRONMENT
        }
        with ExitStack() as resources:
            resources.callback(torch.set_num_threads, torch.get_num_threads())
            resources.callback(self._restore_environment, previous_environment)
            resources.callback(np.random.set_state, np.random.get_state(legacy=True))
            resources.callback(random.setstate, random.getstate())
            resources.enter_context(torch.random.fork_rng(devices=[]))
            torch.set_num_threads(self._threads)
            random.seed(2)
            np.random.seed(2)
            torch.default_generator.manual_seed(2)
            yield
