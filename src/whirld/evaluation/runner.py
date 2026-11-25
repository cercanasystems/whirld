"""Reproducible, split-isolated HLS burn-scar evaluation via the public disk API."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import re
import shutil
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from urllib.parse import unquote, urlsplit

import rasterio
from rasterio.errors import RasterioError
from rasterio.warp import transform_bounds

import whirld

from ..core.fetch import Manifest, ModelStore, sha256_file
from ..core.registry import ModelEntry
from ..errors import InvalidInputError
from ..io.local import LocalRasterAccessPolicy
from ..io.raster import validate_metric_grid
from .assets import EvaluationAssetCache
from .errors import EvaluationError
from .metrics import BinaryRasterScorer
from .runtime import EvaluationCpuRuntime
from .schemas import (
    CaseResult,
    ConfusionCounts,
    EvaluationCase,
    EvaluationOptions,
    EvaluationReport,
    EvaluationSuite,
    MetricSummary,
    ModelIdentity,
    PreparedCase,
    RuntimeIdentity,
    Split,
    SplitResult,
    SuiteDocument,
)

_LOG = logging.getLogger(__name__)


class SuiteLoader:
    """Load a packaged or explicit frozen manifest without model/network access."""

    @staticmethod
    def load(path: Path | None = None) -> SuiteDocument:
        """Validate manifest JSON and retain its original SHA-256.

        Args:
            path: Custom compatible manifest, or None for the packaged mini suite.

        Returns:
            Validated data/model pins and serialized manifest identity.

        Raises:
            EvaluationError: Manifest is malformed, too large or unavailable.
        """
        try:
            resource = path or files("whirld.evaluation").joinpath(
                "data/hls-burn-scars-mini.json"
            )
            with resource.open("rb") as source:
                raw = source.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise EvaluationError("Evaluation manifest exceeds 1 MiB.")
            return SuiteDocument(
                suite=EvaluationSuite.model_validate_json(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        except (OSError, ValueError) as exc:
            raise EvaluationError(
                "Evaluation manifest is unavailable or invalid."
            ) from exc


class HlsCasePreparer:
    """Validate paired headers and write scale-one STAC without altering pixels."""

    @staticmethod
    def prepare(
        case: EvaluationCase, image: Path, label: Path, directory: Path
    ) -> PreparedCase:
        """Require the pinned HLS geometry and bind its native reflectance units.

        Args:
            case: Frozen source identity, split, geographic group and bounds.
            image: Checksum-verified six-band local reflectance TIFF.
            label: Checksum-verified one-band int16 ground-truth TIFF.
            directory: Owned run directory for a local metadata adapter.

        Returns:
            A typed pair and STAC path, retaining source CRS/affine/nodata exactly.

        Raises:
            EvaluationError: Header/grid/bounds differ from the documented contract.
            OSError: Adapter output cannot be written.
        """
        policy = LocalRasterAccessPolicy(str(image))
        try:
            with (
                policy.asset_environment(str(image)),
                rasterio.open(image, driver="GTiff") as source,
                rasterio.open(label, driver="GTiff") as target,
            ):
                if (
                    source.count != 6
                    or source.shape != (512, 512)
                    or any(
                        dtype not in {"float32", "float64"} for dtype in source.dtypes
                    )
                    or source.nodata != -1000
                    or target.count != 1
                    or target.dtypes != ("int16",)
                    or target.nodata != -1000
                    or source.crs is None
                    or source.crs != target.crs
                    or source.shape != target.shape
                    or source.transform != target.transform
                ):
                    raise EvaluationError(
                        "HLS image/label headers differ from contract."
                    )
                if validate_metric_grid(source.crs.to_string(), source.transform) != (
                    30.0,
                    30.0,
                ):
                    raise EvaluationError("HLS evaluation requires native 30 m pixels.")
                bounds = transform_bounds(source.crs, "EPSG:4326", *source.bounds)
                expected = (
                    case.bounds.west,
                    case.bounds.south,
                    case.bounds.east,
                    case.bounds.north,
                )
                if not all(
                    math.isclose(actual, frozen, rel_tol=0, abs_tol=1e-8)
                    for actual, frozen in zip(bounds, expected, strict=True)
                ):
                    raise EvaluationError("Actual HLS footprint differs from manifest.")
        except (RasterioError, ValueError, InvalidInputError) as exc:
            raise EvaluationError("Could not validate HLS raster pair.") from exc
        item = directory / f"{case.id}.stac.json"
        # Raw mappings stay within this STAC serialization adapter boundary.
        item.write_text(
            json.dumps(
                {
                    "type": "Feature",
                    "stac_version": "1.0.0",
                    "id": case.id,
                    "properties": {"datetime": None},
                    "geometry": None,
                    "bbox": list(bounds),
                    "links": [],
                    "assets": {
                        "reflectance": {
                            "href": image.resolve().as_uri(),
                            "type": "image/tiff; application=geotiff",
                            "eo:bands": [
                                {"name": name}
                                for name in ("B02", "B03", "B04", "B8A", "B11", "B12")
                            ],
                            "raster:bands": [
                                {"scale": 1.0, "offset": 0.0, "nodata": -1000.0}
                                for _ in range(6)
                            ],
                        }
                    },
                },
                allow_nan=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return PreparedCase(case=case, image=image, label=label, item=item)


class EvaluationRunner:
    """Own verified assets, fixed inference and atomic final-report publication."""

    def __init__(
        self,
        document: SuiteDocument,
        options: EvaluationOptions,
    ) -> None:
        """Bind typed inputs and storage owners without fetching or loading weights."""
        self.document = document
        self.options = options
        self._store = ModelStore()
        self._cache = EvaluationAssetCache(options.cache_dir, options.offline)

    def _model_identity(self) -> tuple[ModelEntry, Manifest, ModelIdentity]:
        """Hash installed model/config and require exact curated suite pins.

        Returns:
            The registry contract, verified installation and report provenance.

        Raises:
            EvaluationError: Registry/checkpoint/config differ from frozen pins.
            WhirldError: The model is absent or its installation is invalid.
        """
        suite = self.document.suite
        entry = self._store.registry.get(suite.model)
        manifest = self._store.load_manifest(suite.model)
        if (
            entry.source.revision != suite.model_revision
            or entry.distribution.sha256 != suite.model_sha256
            or not manifest.matches_entry(entry)
            or entry.config_file is None
        ):
            raise EvaluationError("Installed model contract differs from suite pins.")
        checkpoint = sha256_file(
            self._store.paths.model_file(entry.name, manifest.weights_file)
        )
        configuration = sha256_file(
            self._store.paths.model_file(entry.name, entry.config_file)
        )
        if (
            checkpoint != suite.model_sha256
            or configuration != suite.model_config_sha256
        ):
            raise EvaluationError(
                "Installed checkpoint/config SHA-256 differs from suite."
            )
        return (
            entry,
            manifest,
            ModelIdentity(
                name=entry.name,
                version=manifest.version,
                revision=suite.model_revision,
                checkpoint_sha256=checkpoint,
                config_sha256=configuration,
                registry_sha256=entry.fingerprint(),
            ),
        )

    def _verify_splits(self) -> None:
        """Check full official identifiers and exclude exposed geographic groups.

        Raises:
            EvaluationError: Lists overlap, contain malformed/duplicate identities,
                or selected cases/groups violate the frozen split protocol.
        """
        partitions: dict[str, set[str]] = {}
        for pin in self.document.suite.split_pins:
            path = self._cache.fetch(pin.asset)
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError as exc:
                raise EvaluationError(
                    "Official split list must contain UTF-8 text."
                ) from exc
            if (
                not lines
                or len(set(lines)) != len(lines)
                or any(
                    not re.fullmatch(r"T\d{2}[A-Z]{3}\.\d{7}\.v1", line)
                    for line in lines
                )
            ):
                raise EvaluationError(
                    "Official split contains invalid/duplicate identities."
                )
            partitions[pin.split] = set(lines)
        for split, members in partitions.items():
            if any(
                members & other for name, other in partitions.items() if name != split
            ):
                raise EvaluationError("Official split memberships overlap.")
        train_groups = {name.split(".")[0] for name in partitions["train"]}
        validation_groups = {name.split(".")[0] for name in partitions["validation"]}
        for case in self.document.suite.cases:
            basename = f"subsetted_512x512_HLS.S30.{case.source_name}.4"
            if (
                case.source_name not in partitions[case.split]
                or case.spatial_group != case.source_name.split(".")[0]
                or case.spatial_group in train_groups
                or (case.split == "test" and case.spatial_group in validation_groups)
                or unquote(urlsplit(case.image.url).path).rsplit("/", 1)[-1]
                != basename + "_merged.tif"
                or unquote(urlsplit(case.label.url).path).rsplit("/", 1)[-1]
                != basename + ".mask.tif"
            ):
                raise EvaluationError("Case/group violates official split isolation.")

    def _prepare_cases(self, directory: Path) -> tuple[PreparedCase, ...]:
        """Verify all frozen pairs before inference, including the unselected split."""
        self._verify_splits()
        return tuple(
            HlsCasePreparer.prepare(
                case,
                self._cache.fetch(case.image),
                self._cache.fetch(case.label),
                directory,
            )
            for case in self.document.suite.cases
        )

    @staticmethod
    def _runtime() -> RuntimeIdentity:
        """Record actual dependency versions; missing optional runtimes fail clearly."""
        try:
            return RuntimeIdentity(
                python=platform.python_version(),
                platform=platform.platform(),
                whirld=whirld.__version__,
                numpy=importlib.metadata.version("numpy"),
                rasterio=importlib.metadata.version("rasterio"),
                torch=importlib.metadata.version("torch"),
                terratorch=importlib.metadata.version("terratorch"),
            )
        except importlib.metadata.PackageNotFoundError as exc:
            raise EvaluationError(
                "Install Whirld's prithvi extra before evaluation."
            ) from exc

    def _evaluate(self, prepared: PreparedCase, directory: Path) -> CaseResult:
        """Run public streaming inference and score one exact-grid raster in windows."""
        output = directory / f"{prepared.case.id}.prediction.tif"
        whirld.segment_to_disk(
            prepared.item,
            model=self.document.suite.model,
            sensor=self.document.suite.sensor,
            output=output,
            device="cpu",
            threshold=self.options.threshold,
            batch_size=self.options.batch_size,
            cache_mb=self.options.cache_mb,
            no_warnings=True,
        )
        counts = BinaryRasterScorer().score(output, prepared.label)
        with rasterio.open(output, driver="GTiff") as source:
            result = CaseResult(
                id=prepared.case.id,
                split=prepared.case.split,
                counts=counts,
                metrics=MetricSummary.from_counts(counts),
                prediction_sha256=sha256_file(output),
                prediction_file=output.name,
                width=source.width,
                height=source.height,
                crs=source.crs.to_string(),
                transform=(
                    source.transform.a,
                    source.transform.b,
                    source.transform.c,
                    source.transform.d,
                    source.transform.e,
                    source.transform.f,
                ),
            )
        _LOG.info(
            "Evaluated %s (%s): IoU=%s", result.id, result.split, result.metrics.iou
        )
        return result

    def run(self) -> EvaluationReport:
        """Evaluate selected cases and publish report.json only after complete success.

        Returns:
            Strict JSON-compatible provenance and separate case/split results.

        Raises:
            EvaluationError: Verification, preparation or scoring fails.
            WhirldError: Public inference fails or installed weights are unavailable.
            OSError: A fresh output directory cannot be reserved or written.
            KeyboardInterrupt: Propagates after cleanup and runtime restoration.

        Existing outputs are never overwritten. Failed runs remove only the new
        directory owned by this invocation; verified cached data remain reusable.
        """
        directory = self.options.output_dir.resolve()
        directory.mkdir(parents=True, exist_ok=False)
        try:
            runtime = self._runtime()
            prepared = self._prepare_cases(directory)
            with (
                EvaluationCpuRuntime(self.options.threads).activate(),
                self._store.transaction(self.document.suite.model),
            ):
                entry, manifest, identity = self._model_identity()
                with self._store.installation_lease(entry, manifest):
                    results = tuple(
                        self._evaluate(case, directory)
                        for case in prepared
                        if self.options.split == "all"
                        or case.case.split == self.options.split
                    )
            splits: list[SplitResult] = []
            split_names: tuple[Split, ...] = ("validation", "test")
            for split in split_names:
                members = tuple(case for case in results if case.split == split)
                if members:
                    counts = ConfusionCounts.combine(
                        tuple(case.counts for case in members)
                    )
                    splits.append(
                        SplitResult(
                            split=split,
                            case_count=len(members),
                            counts=counts,
                            metrics=MetricSummary.from_counts(counts),
                        )
                    )
            report = EvaluationReport(
                suite=self.document.suite,
                manifest_sha256=self.document.sha256,
                model=identity,
                runtime=runtime,
                options=self.options,
                created_at=datetime.now(UTC),
                cases=results,
                splits=tuple(splits),
            )
            stage = directory / ".report.json.part"
            stage.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
            stage.replace(directory / "report.json")
            return report
        except BaseException:
            shutil.rmtree(directory)
            raise
