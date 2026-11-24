"""Shared CPU setup and packaged preview exports for runnable examples."""

from __future__ import annotations

import importlib.metadata
import logging
import os
import platform
import resource
import sys
import time
from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel

from whirld.core.fetch import ModelStore, sha256_file
from whirld.core.registry import ModelEntry
from whirld.demo.preview import (
    SENTINEL_SCENE as SENTINEL_SCENE,
)
from whirld.demo.preview import (
    OverlayPolygon as OverlayPolygon,
)
from whirld.demo.preview import (
    Preview as Preview,
)
from whirld.demo.preview import (
    RasterImage as RasterImage,
)
from whirld.demo.preview import (
    SceneSpec as SceneSpec,
)
from whirld.errors import InvalidInputError


class RunMetrics(BaseModel):
    """Wall time from the caller's start and process-lifetime peak CPU RSS."""

    elapsed_seconds: float
    peak_rss_bytes: int

    @classmethod
    def capture(cls, started: float) -> RunMetrics:
        """Capture time/RSS after a workflow; RSS includes imports and model loading.

        Args:
            started: Monotonic timestamp after setup and before data preparation.

        Returns:
            Total workflow seconds and process peak bytes, not GPU allocations.
        """
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return cls(
            elapsed_seconds=time.monotonic() - started,
            peak_rss_bytes=int(rss if sys.platform == "darwin" else rss * 1024),
        )


class ExampleRuntime:
    """Own an exclusive output directory and explicit CPU setup for one example."""

    def __init__(self, output_dir: Path, threads: int = 2) -> None:
        """Bind output and thread settings; reject nonpositive thread counts."""
        if threads < 1:
            raise InvalidInputError("threads must be positive")
        self.output_dir = output_dir
        self.threads = threads

    def prepare(self, model: str) -> None:
        """Check installed weights, configure CPU threads and reserve a fresh output.

        Args:
            model: Registry identifier already installed using whirld pull.

        Raises:
            ModelNotInstalledError: The model is absent or its manifest is invalid.
            InvalidInputError: Registry or checkpoint differs from the curated pin.
            FileExistsError: Output exists; previous example results are preserved.
            ImportError: Required torch extra is missing.
        """
        import torch

        logging.basicConfig(level=os.environ.get("WHIRLD_LOG_LEVEL", "INFO").upper())
        store = ModelStore()
        entry = store.registry.get(model)
        manifest = store.load_manifest(model)
        bundled = ModelEntry.from_registry_file(
            Path(str(files("whirld").joinpath(f"registry_data/models/{model}.yaml")))
        )
        if entry != bundled or not manifest.matches_entry(entry):
            raise InvalidInputError(f"Re-pull {model}: expected bundled model registry")
        weights = store.paths.model_file(model, manifest.weights_file)
        if sha256_file(weights) != entry.distribution.sha256:
            raise InvalidInputError(f"Re-pull {model}: checkpoint checksum differs")
        torch.set_num_threads(self.threads)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        provenance = ModelProvenance(
            model=model,
            revision=entry.source.revision or "",
            checkpoint_sha256=manifest.sha256,
            python_version=platform.python_version(),
            platform=platform.platform(),
            torch_version=importlib.metadata.version("torch"),
            threads=self.threads,
        )
        (self.output_dir / "model.json").write_text(
            provenance.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )


class ModelProvenance(BaseModel):
    """Verified curated weights and CPU runtime identity for one recorded run."""

    model: str
    revision: str
    checkpoint_sha256: str
    python_version: str
    platform: str
    torch_version: str
    threads: int
    device: str = "cpu"
