"""A real RN50 first result from verified installed imagery and model weights."""

from __future__ import annotations

import importlib.metadata
import logging
import platform
import random
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import whirld

from ..core.contract import translate
from ..core.fetch import ModelStore, sha256_file
from ..core.registry import ModelEntry
from ..errors import InvalidInputError
from ..io.raster import read_raster
from ..logging_setup import configure_logging
from ..onboarding.doctor import DiagnosticsService
from .preview import OverlayPolygon, Preview, RasterImage
from .sample import BundledDemoSample, DemoSampleSpec
from .scores import COLORS, PROMPTS, ScoreCollection

_LOG = logging.getLogger(__name__)
MODEL: Literal["remoteclip-rn50"] = "remoteclip-rn50"


class DemoOptions(BaseModel):
    """CPU demo settings; outputs must be a fresh directory at execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    output_dir: Path
    threads: int = Field(default=2, strict=True, gt=0)
    verbose: bool = False
    quiet: bool = False


class DemoRuntimeIdentity(BaseModel):
    """Actual runtime versions for the measured small CPU workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    python: str
    platform: str
    whirld: str
    torch: str
    open_clip: str
    numpy: str
    rasterio: str


class DemoReport(BaseModel):
    """Verified source/model identities and paths to the user's first result."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    model: Literal["remoteclip-rn50"] = MODEL
    model_version: str
    model_revision: str
    checkpoint_sha256: str
    registry_sha256: str
    sample: DemoSampleSpec
    output_dir: Path
    scores_path: Path
    preview_path: Path
    feature_count: int = Field(gt=0)
    threads: int = Field(gt=0)
    device: Literal["cpu"] = "cpu"
    elapsed_seconds: float = Field(ge=0)
    runtime: DemoRuntimeIdentity


class _ExpectedRn50Initialization(logging.Filter):
    """Suppress only OpenCLIP's known temporary construction notice in this thread."""

    def __init__(self, thread: int) -> None:
        """Bind the inference thread whose verified checkpoint follows construction."""
        super().__init__()
        self._thread = thread

    def filter(self, record: logging.LogRecord) -> bool:
        """Retain every warning/error except the exact pre-checkpoint RN50 notice."""
        return not (
            record.levelno == logging.WARNING
            and record.thread == self._thread
            and record.pathname.replace("\\", "/").endswith("/open_clip/factory.py")
            and record.getMessage()
            == (
                "No pretrained weights loaded for model 'RN50'. "
                "Model initialized randomly."
            )
        )


class DemoCpuRuntime:
    """Own CPU threads and host RNG state consumed while constructing RN50."""

    def __init__(self, threads: int) -> None:
        """Bind the positive thread count already validated by DemoOptions."""
        self._threads = threads

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Restore host random state and CPU threads on success, errors and interrupts.

        Yields:
            A CPU inference scope; accelerator generators are not initialized.

        Raises:
            ImportError: The optional Torch runtime is absent.
            BaseException: Inference exceptions propagate after host restoration.
        """
        import torch

        threads = torch.get_num_threads()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        logger = logging.getLogger()
        notice_filter = _ExpectedRn50Initialization(threading.get_ident())
        logger.addFilter(notice_filter)
        try:
            with torch.random.fork_rng(devices=[]):
                torch.set_num_threads(self._threads)
                yield
        finally:
            logger.removeFilter(notice_filter)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_num_threads(threads)


class DemoRunner:
    """Own preflight, automatic weight acquisition, real inference and a preview."""

    def __init__(self, options: DemoOptions) -> None:
        """Bind settings and the environment's model store without loading weights."""
        self.options = options
        self._store = ModelStore()

    def _verify_model(self) -> tuple[ModelEntry, str]:
        """Require the bundled RN50 contract and hash its installed checkpoint.

        Returns:
            Active curated model contract and actual checkpoint SHA-256.

        Raises:
            InvalidInputError: Active registry or model bytes differ from the pin.
            WhirldError: Model installation is missing or corrupt.
        """
        entry = self._store.registry.get(MODEL)
        resource = files("whirld").joinpath(f"registry_data/models/{MODEL}.yaml")
        with as_file(resource) as path:
            bundled = ModelEntry.from_registry_file(path)
        manifest = self._store.load_manifest(MODEL)
        if entry != bundled or not manifest.matches_entry(entry):
            raise InvalidInputError(
                "Demo requires the bundled RemoteCLIP RN50 contract."
            )
        checkpoint = self._store.paths.model_file(MODEL, manifest.weights_file)
        actual = sha256_file(checkpoint)
        if actual != entry.distribution.sha256:
            raise InvalidInputError(
                "Demo checkpoint checksum differs from curated pin."
            )
        return entry, actual

    @staticmethod
    def _preview(
        sample: Path,
        output: Path,
        collection: ScoreCollection,
        entry: ModelEntry,
        spec: DemoSampleSpec,
    ) -> None:
        """Render the same checked local pixels and classified WGS84 footprints.

        Args:
            sample: Verified canonical-DN TIFF copied into this run.
            output: HTML path; a companion RGB PNG is also produced.
            collection: Validated real four-prompt model scores and footprints.
            entry: Curated normalization used for inference and display reversal.
            spec: Source attribution and input semantics retained in the report.

        Raises:
            WhirldError: Source metadata or model-unit translation is invalid.
            OSError: Preview files cannot be written.
        """
        translated = translate(read_raster(sample), entry, spec.sensor)
        normalization = entry.band_contract.normalization
        reflectance = translated.data * np.asarray(normalization.std)[:, None, None]
        reflectance += np.asarray(normalization.mean)[:, None, None]
        polygons: list[OverlayPolygon] = []
        for feature in collection.features:
            scores = feature.properties.scores
            winner = max(range(len(PROMPTS)), key=lambda index: scores[PROMPTS[index]])
            prompt = PROMPTS[winner]
            polygons.append(
                OverlayPolygon(
                    coordinates=feature.geometry.coordinates[0],
                    label=(
                        f"chip {feature.properties.chip_index}: "
                        f"{prompt.removeprefix('a satellite image of ')} "
                        f"{scores[prompt]:.2f}"
                    ),
                    color=COLORS[winner],
                )
            )
        Preview.write_raster(
            output,
            "Your first Whirld result: landscape prompts",
            RasterImage(
                rgb=Preview.stretch(reflectance),
                transform=translated.transform,
                crs=translated.crs,
            ),
            polygons,
            "RemoteCLIP RN50 compares farmland (gold), forest (green), water (blue) "
            "and urban area (pink). Each footprint shows its highest-scoring prompt. "
            "Scores are relative to these prompts, not calibrated confidence. "
            "Sentinel-2 reflectance differs from the model's training RGB imagery. "
            + spec.attribution,
        )

    def run(self) -> DemoReport:
        """Produce a real scored preview after readiness checks and verified downloads.

        Returns:
            Small typed report pointing to GeoJSON, preview and source provenance.

        Raises:
            WhirldError: Readiness, acquisition, source verification or inference fails.
            OSError: A fresh output cannot be reserved or artifacts cannot be written.
            KeyboardInterrupt: Propagates after output and host-state cleanup.

        A missing model is pulled only after runtime/storage readiness passes.
        Existing output directories are never overwritten. Only the new directory
        reserved by this run is removed on failure; installed model caches remain.
        """
        started = time.monotonic()
        output = self.options.output_dir.resolve()
        if output.exists():
            raise InvalidInputError(
                "Demo output already exists; choose a new directory."
            )
        readiness = DiagnosticsService().inspect(model=MODEL)
        if not readiness.ready_for_demo:
            blockers = "; ".join(
                check.message + (f" {check.remedy}" if check.remedy else "")
                for check in readiness.checks
                if check.status == "fail"
            )
            raise InvalidInputError("Demo setup needs attention: " + blockers)
        output.mkdir(parents=True, exist_ok=False)
        try:
            spec, sample = BundledDemoSample().prepare(output)
            configure_logging(
                verbose=self.options.verbose, quiet=self.options.quiet, force=True
            )
            if not readiness.model_installed:
                _LOG.info(
                    "Downloading RN50 weights: %s bytes", readiness.model_download_bytes
                )
                whirld.pull(MODEL)
            with (
                self._store.transaction(MODEL),
                DemoCpuRuntime(self.options.threads).activate(),
                rasterio.Env(
                    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                    GDAL_PAM_ENABLED=False,
                    GDAL_GEOREF_SOURCES="INTERNAL",
                ),
            ):
                entry, checkpoint = self._verify_model()
                scores_path = output / "scores.geojson"
                result = whirld.classify(
                    sample,
                    model=MODEL,
                    query=list(PROMPTS),
                    top_k=4,
                    threshold=0.0,
                    sensor=spec.sensor,
                    device="cpu",
                    no_warnings=True,
                    output=scores_path,
                )
                try:
                    collection = ScoreCollection.model_validate(
                        result.feature_collection
                    )
                except ValidationError as exc:
                    raise InvalidInputError(
                        "Demo classification metadata is invalid."
                    ) from exc
                collection.validate_probabilities()
                if (
                    collection.model != MODEL
                    or collection.queries != list(PROMPTS)
                    or collection.query != PROMPTS[0]
                    or any(
                        feature.properties.model != MODEL
                        for feature in collection.features
                    )
                    or result.sensor != spec.sensor
                    or result.device != "cpu"
                ):
                    raise InvalidInputError(
                        "Demo output identity differs from its request."
                    )
                if len(collection.features) != 4 or len(result.chips) != 4:
                    raise InvalidInputError(
                        "Demo sample must produce exactly four chips."
                    )
                preview_path = output / "preview.html"
                self._preview(sample, preview_path, collection, entry, spec)
            report = DemoReport(
                model_version=entry.version,
                model_revision=entry.source.revision or "",
                checkpoint_sha256=checkpoint,
                registry_sha256=entry.fingerprint(),
                sample=spec,
                output_dir=output,
                scores_path=scores_path,
                preview_path=preview_path,
                feature_count=len(collection.features),
                threads=self.options.threads,
                elapsed_seconds=time.monotonic() - started,
                runtime=DemoRuntimeIdentity(
                    python=platform.python_version(),
                    platform=platform.platform(),
                    whirld=whirld.__version__,
                    torch=importlib.metadata.version("torch"),
                    open_clip=importlib.metadata.version("open_clip_torch"),
                    numpy=importlib.metadata.version("numpy"),
                    rasterio=importlib.metadata.version("rasterio"),
                ),
            )
            (output / "report.json").write_text(report.model_dump_json(indent=2) + "\n")
            _LOG.info("Open %s", report.preview_path)
            return report
        except BaseException:
            shutil.rmtree(output)
            raise
