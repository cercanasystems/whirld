"""Opt-in real Clay/public Sentinel-2 streaming validation in fresh CPU processes.

Install the pinned ``clay-v1.5`` checkpoint first, then run this script with a new
``--output-dir``. No worker downloads model weights. Each process measures one
public API call, including model loading, remote I/O and output publication; RSS
is process-lifetime peak, not a steady-state inference or GPU allocation metric.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
import platform
import subprocess
import sys
import time
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from whirld.core.chips import ChipMeta
from whirld.core.fetch import ModelStore, sha256_file
from whirld.core.registry import ModelEntry
from whirld.errors import WhirldError
from whirld.io.embedding_stream import EmbeddingMetadata

ITEM_URL = (
    "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/"
    "S2A_30UXD_20230630_1_L2A"
)
BASE_BBOX = (-1.32, 52.49, -1.26, 52.53)
LARGE_BBOX = (-1.35, 52.47, -1.23, 52.55)
RTOL = 1e-4
ATOL = 1e-5
PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class ValidationFailure(WhirldError):
    """Real-model validation found an incompatible installation or output."""


class ValidationOptions(BaseModel):
    """Immutable runner settings; output must be a new directory at execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    output_dir: Path
    chip_size: PositiveInt = 256
    batch_sizes: tuple[PositiveInt, ...] = (1, 2, 3)
    threads: PositiveInt = 2
    timeout: PositiveInt = 1800

    @field_validator("chip_size")
    @classmethod
    def validate_chip_size(cls, value: int) -> int:
        """Require positive Clay patch multiples; raise ValueError otherwise."""
        if value % 8:
            raise ValueError("Clay chip size must be divisible by patch size 8")
        return value

    @field_validator("batch_sizes")
    @classmethod
    def validate_batches(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Require at least two distinct batches including one; reject duplicates."""
        if len(value) < 2 or len(set(value)) != len(value) or 1 not in value:
            raise ValueError(
                "Provide distinct batch sizes including 1 and another size"
            )
        return value


class RunSpec(BaseModel):
    """One fresh process's fully specified inference request."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    mode: Literal["eager", "stream"]
    region: Literal["base", "large"]
    batch_size: PositiveInt
    chip_size: PositiveInt
    threads: PositiveInt
    output_path: Path

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return the pinned region's geographic west/south/east/north bounds."""
        return BASE_BBOX if self.region == "base" else LARGE_BBOX


class RunMeasurement(BaseModel):
    """Process-lifetime RSS and API-call time; output checks occur afterward."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    mode: Literal["eager", "stream"]
    region: Literal["base", "large"]
    bbox: tuple[float, float, float, float]
    batch_size: PositiveInt
    chip_size: PositiveInt
    chip_count: PositiveInt
    embed_dim: PositiveInt
    dtype: str
    finite: bool
    output_path: Path
    meta_path: Path
    peak_rss_bytes: PositiveInt
    elapsed_seconds: float = Field(gt=0, allow_inf_nan=False)
    chips_per_second: float = Field(gt=0, allow_inf_nan=False)


class Comparison(BaseModel):
    """Measured numerical deviation after metadata and tolerance checks pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    batch_size: PositiveInt
    chip_count: PositiveInt
    max_abs_error: float = Field(ge=0, allow_inf_nan=False)
    rmse: float = Field(ge=0, allow_inf_nan=False)
    rtol: float = RTOL
    atol: float = ATOL


class OutputChipMetadata(ChipMeta):
    """Strict chip schema so comparison cannot silently discard unknown fields."""

    model_config = ConfigDict(extra="forbid", strict=True)


class OutputMetadata(EmbeddingMetadata):
    """Typed existing sidecar schema, including ordered chip geometry."""

    chips: list[OutputChipMetadata]


class CheckpointIdentity(BaseModel):
    """Installed registry and artifact identity verified before measured workers."""

    model: str
    revision: str
    sha256: str
    size_bytes: int
    registry_fingerprint: str


class RuntimeIdentity(BaseModel):
    """Runtime versions and explicit CPU/thread configuration for reproducibility."""

    python: str
    platform: str
    numpy: str
    torch: str
    rasterio: str
    gdal: str
    device: Literal["cpu"] = "cpu"
    threads: int


class ValidationReport(BaseModel):
    """Complete acceptance evidence; does not imply accelerator validation."""

    options: ValidationOptions
    item_url: str = ITEM_URL
    checkpoint: CheckpointIdentity
    runtime: RuntimeIdentity
    measurements: list[RunMeasurement]
    comparisons: list[Comparison]
    partial_batch_exercised: bool
    timing_scope: str = (
        "API call including model loading, remote I/O and output writing"
    )
    rss_scope: str = "Fresh-process lifetime peak through API completion; CPU only"


class EmbeddingComparison:
    """Validate NPY artifacts and compare typed sidecars plus embedding values."""

    @staticmethod
    def validate_array(path: Path) -> np.ndarray:
        """Map a finite float32 (n,1024) output without copying its data.

        Args:
            path: Completed NPY artifact to inspect.

        Returns:
            A read-only memory mapped array with at least one chip.

        Raises:
            ValidationFailure: The artifact has wrong dtype/shape or nonfinite data.
            OSError: The output cannot be read.
        """
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            array.ndim != 2
            or array.shape[0] < 1
            or array.shape[1] != 1024
            or array.dtype != np.float32
        ):
            raise ValidationFailure(f"Invalid real Clay shape/dtype: {path}")
        for start in range(0, len(array), 64):
            if not np.isfinite(array[start : start + 64]).all():
                raise ValidationFailure(f"Nonfinite real Clay embedding: {path}")
        return array

    @classmethod
    def compare(cls, eager: RunMeasurement, streamed: RunMeasurement) -> Comparison:
        """Require identical chip geometry and the predeclared numeric tolerance.

        Args:
            eager: Baseline eager output and measurement record.
            streamed: Streamed output for the same item region and chip size.

        Returns:
            Maximum absolute and root mean square errors across all components.

        Raises:
            ValidationFailure: Shapes, metadata, record counts or values differ.
            OSError: An output or metadata file cannot be read.
            pydantic.ValidationError: A sidecar violates the existing schema.
        """
        left = cls.validate_array(eager.output_path)
        right = cls.validate_array(streamed.output_path)
        if left.shape != right.shape or left.shape[0] != eager.chip_count:
            raise ValidationFailure("Eager/streamed shape or baseline count differs")
        if right.shape[0] != streamed.chip_count:
            raise ValidationFailure("Streamed output count differs from measurement")
        headers = [
            OutputMetadata.model_validate_json(path.read_text(encoding="utf-8"))
            for path in (eager.meta_path, streamed.meta_path)
        ]
        if headers[0].model_dump(exclude={"timestamp"}) != headers[1].model_dump(
            exclude={"timestamp"}
        ):
            raise ValidationFailure("Eager/streamed metadata differs")
        if len(headers[0].chips) != len(left):
            raise ValidationFailure("Sidecar chip count differs from embeddings")
        maximum = 0.0
        squared = 0.0
        for start in range(0, len(left), 64):
            expected = left[start : start + 64]
            actual = right[start : start + 64]
            if not np.allclose(actual, expected, rtol=RTOL, atol=ATOL):
                raise ValidationFailure(
                    f"Embedding values differ at batch {streamed.batch_size}; "
                    f"required rtol={RTOL}, atol={ATOL}"
                )
            difference = actual.astype(np.float64) - expected.astype(np.float64)
            maximum = max(maximum, float(np.abs(difference).max()))
            squared += float(np.square(difference).sum())
        return Comparison(
            batch_size=streamed.batch_size,
            chip_count=len(left),
            max_abs_error=maximum,
            rmse=float(np.sqrt(squared / left.size)),
        )


class ClayStreamingValidation:
    """Own checkpoint verification, fresh workers, and their persisted report."""

    def __init__(self, options: ValidationOptions) -> None:
        """Bind validated immutable options without creating files or loading models."""
        self.options = options

    def verify_checkpoint(self) -> CheckpointIdentity:
        """Verify the installed model matches the bundled entry and artifact digest.

        Returns:
            Pinned checkpoint provenance for the report.

        Raises:
            ValidationFailure: Cached configuration or bytes differ from the bundle.
            WhirldError: No valid installed model is available.
            OSError: The checkpoint cannot be read.
        """
        store = ModelStore()
        entry = store.registry.get("clay-v1.5")
        bundled_path = files("whirld").joinpath("registry_data/models/clay-v1.5.yaml")
        bundled = ModelEntry.from_registry_file(Path(str(bundled_path)))
        manifest = store.load_manifest("clay-v1.5")
        if entry != bundled or not manifest.matches_entry(entry):
            raise ValidationFailure("Installed Clay configuration differs from bundle")
        path = store.paths.model_file(entry.name, manifest.weights_file)
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != entry.distribution.size_bytes or digest != entry.distribution.sha256:
            raise ValidationFailure("Installed Clay checkpoint checksum/size mismatch")
        return CheckpointIdentity(
            model=entry.name,
            revision=entry.source.revision or "",
            sha256=digest,
            size_bytes=size,
            registry_fingerprint=entry.fingerprint(),
        )

    def measure(self, spec: RunSpec) -> RunMeasurement:
        """Run one isolated worker and persist its validated measurement.

        Args:
            spec: Inference mode, pinned region, batch size and output destination.

        Returns:
            Process peak RSS, elapsed API-call time and verified output dimensions.

        Raises:
            ValidationFailure: The worker exits unsuccessfully or times out.
            pydantic.ValidationError: Worker output has an invalid report schema.
            OSError: The worker or measurement artifact cannot be created.
        """
        environment = {
            **os.environ,
            "OMP_NUM_THREADS": str(self.options.threads),
            "OPENBLAS_NUM_THREADS": str(self.options.threads),
            "MKL_NUM_THREADS": str(self.options.threads),
        }
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    spec.model_dump_json(),
                ],
                env=environment,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.options.timeout,
            )
        except subprocess.CalledProcessError as exc:
            raise ValidationFailure(f"Worker failed: {exc.stderr}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ValidationFailure(f"Worker exceeded {self.options.timeout}s") from exc
        measurement = RunMeasurement.model_validate_json(process.stdout)
        spec.output_path.with_suffix(".measurement.json").write_text(
            measurement.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return measurement

    def run(self) -> ValidationReport:
        """Verify installation, measure all cases, compare outputs, and save report.

        Returns:
            Complete real CPU model validation and resource evidence.

        Raises:
            ValidationFailure: Any checkpoint, worker or acceptance check fails.
            FileExistsError: Output directory exists; previous evidence is preserved.
            OSError: Output evidence cannot be written.
        """
        import rasterio

        checkpoint = self.verify_checkpoint()
        self.options.output_dir.mkdir(parents=True, exist_ok=False)
        runtime = RuntimeIdentity(
            python=platform.python_version(),
            platform=platform.platform(),
            numpy=np.__version__,
            torch=importlib.metadata.version("torch"),
            rasterio=rasterio.__version__,
            gdal=rasterio.__gdal_version__,
            threads=self.options.threads,
        )
        specs = [self._spec("eager", "base", 1)]
        specs.extend(self._spec("stream", "base", b) for b in self.options.batch_sizes)
        specs.append(self._spec("stream", "large", 1))
        measurements = []
        comparisons = []
        for spec in specs:
            logging.getLogger(__name__).info(
                "Measuring %s %s batch=%d", spec.mode, spec.region, spec.batch_size
            )
            measurement = self.measure(spec)
            measurements.append(measurement)
            if spec.mode == "stream" and spec.region == "base":
                comparisons.append(
                    EmbeddingComparison.compare(measurements[0], measurement)
                )
        count = measurements[0].chip_count
        partial = any(
            count % batch for batch in self.options.batch_sizes if batch < count
        )
        if not partial:
            raise ValidationFailure(
                "Selected region/batches did not exercise a partial final batch"
            )
        if measurements[-1].chip_count <= count:
            raise ValidationFailure(
                "Larger resource region did not increase chip count"
            )
        report = ValidationReport(
            options=self.options,
            checkpoint=checkpoint,
            runtime=runtime,
            measurements=measurements,
            comparisons=comparisons,
            partial_batch_exercised=partial,
        )
        (self.options.output_dir / "report.json").write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return report

    def _spec(
        self,
        mode: Literal["eager", "stream"],
        region: Literal["base", "large"],
        batch: int,
    ) -> RunSpec:
        """Build a typed run request with a unique output name inside the report dir."""
        return RunSpec(
            mode=mode,
            region=region,
            batch_size=batch,
            chip_size=self.options.chip_size,
            threads=self.options.threads,
            output_path=self.options.output_dir / f"{mode}-{region}-batch{batch}.npy",
        )


class InferenceWorker:
    """Own one real public API invocation inside a fresh measured interpreter."""

    def __init__(self, spec: RunSpec) -> None:
        """Bind the validated request; loading and inference occur only in run."""
        self.spec = spec

    def run(self) -> RunMeasurement:
        """Run CPU inference, capture RSS/time, then validate completed artifacts.

        Returns:
            Output shape and process resources, excluding subsequent artifact scans.

        Raises:
            WhirldError: Model loading, remote reads or publication fail.
            ValidationFailure: The output is not finite float32 real Clay embeddings.
            OSError: Output artifacts cannot be read.
        """
        import resource

        import torch

        import whirld

        torch.set_num_threads(self.spec.threads)
        torch.set_num_interop_threads(1)
        started = time.monotonic()
        if self.spec.mode == "eager":
            eager = whirld.embed(
                ITEM_URL,
                model="clay-v1.5",
                output=self.spec.output_path,
                chip_size=self.spec.chip_size,
                batch_size=self.spec.batch_size,
                bbox=self.spec.bbox,
                device="cpu",
                sensor="sentinel-2-l2a",
                no_warnings=True,
            )
            chip_count = len(eager.chips)
        else:
            streamed = whirld.embed_to_disk(
                ITEM_URL,
                model="clay-v1.5",
                output=self.spec.output_path,
                chip_size=self.spec.chip_size,
                batch_size=self.spec.batch_size,
                bbox=self.spec.bbox,
                device="cpu",
                sensor="sentinel-2-l2a",
                no_warnings=True,
            )
            chip_count = streamed.chip_count
        elapsed = time.monotonic() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(rss if sys.platform == "darwin" else rss * 1024)
        array = EmbeddingComparison.validate_array(self.spec.output_path)
        if len(array) != chip_count:
            raise ValidationFailure("Output array count differs from API result")
        return RunMeasurement(
            mode=self.spec.mode,
            region=self.spec.region,
            bbox=self.spec.bbox,
            batch_size=self.spec.batch_size,
            chip_size=self.spec.chip_size,
            chip_count=chip_count,
            embed_dim=array.shape[1],
            dtype=str(array.dtype),
            finite=True,
            output_path=self.spec.output_path,
            meta_path=self.spec.output_path.with_name(
                self.spec.output_path.stem + "_meta.json"
            ),
            peak_rss_bytes=peak,
            elapsed_seconds=elapsed,
            chips_per_second=chip_count / elapsed,
        )


def main() -> int:
    """Parse CLI settings and serialize validation evidence or a private worker result.

    Returns:
        Zero on successful validation, one for a domain validation failure.

    Raises:
        SystemExit: Arguments are invalid or help was requested.
        OSError: Evidence cannot be read or written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chip-size", type=int, default=256)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("WHIRLD_LOG_LEVEL", "INFO").upper())
    try:
        if args.worker:
            result = InferenceWorker(RunSpec.model_validate_json(args.worker)).run()
            sys.stdout.write(result.model_dump_json() + "\n")
        else:
            if args.output_dir is None:
                parser.error("--output-dir is required and must not exist")
            options = ValidationOptions(
                output_dir=args.output_dir.resolve(),
                chip_size=args.chip_size,
                batch_sizes=tuple(args.batch_sizes),
                threads=args.threads,
                timeout=args.timeout,
            )
            report = ClayStreamingValidation(options).run()
            sys.stdout.write(report.model_dump_json(indent=2) + "\n")
    except WhirldError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
