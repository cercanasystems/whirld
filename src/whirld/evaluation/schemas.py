"""Immutable evaluation manifests, explicit metric denominators and reports."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Count = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Identity = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")]
Split = Literal["validation", "test"]


class EvaluationSchema(BaseModel):
    """Reject undeclared fields and nonfinite numbers at evaluation boundaries."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class AssetPin(EvaluationSchema):
    """Public immutable source location with an exact byte identity."""

    url: str
    sha256: Digest
    size_bytes: PositiveInt

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """Require HTTP(S) without credentials/fragments; reject malformed URLs."""
        try:
            parsed = urlsplit(value)
            valid = (
                parsed.scheme in {"https", "http"}
                and parsed.hostname
                and parsed.username is None
                and parsed.password is None
                and not parsed.fragment
            )
            _ = parsed.port
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("asset URL must be public HTTP(S) without credentials")
        return value


class SpatialBounds(EvaluationSchema):
    """WGS84 bounding rectangle used to verify the frozen geographic partition."""

    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)

    @model_validator(mode="after")
    def validate_extent(self) -> Self:
        """Reject empty, reversed or antimeridian-spanning rectangles."""
        if self.west >= self.east or self.south >= self.north:
            raise ValueError("bounds must have positive width and height")
        return self

    def overlaps(self, other: SpatialBounds) -> bool:
        """Return whether rectangles share positive area; touching edges do not."""
        return max(self.west, other.west) < min(self.east, other.east) and max(
            self.south, other.south
        ) < min(self.north, other.north)


class SplitPin(EvaluationSchema):
    """Official model split list, verified before accepting case membership."""

    split: Literal["train", "validation", "test"]
    asset: AssetPin


class EvaluationCase(EvaluationSchema):
    """One paired raster with its preselected official split and footprint."""

    id: Identity
    split: Split
    spatial_group: Identity
    source_name: Identity
    image: AssetPin
    label: AssetPin
    bounds: SpatialBounds


class EvaluationSuite(EvaluationSchema):
    """Frozen HLS reflectance/binary burn-scar baseline, not a training dataset.

    The first suite deliberately supports one documented input/label contract.
    Model-card split provenance does not prove absence from pretraining corpora.
    """

    schema_version: Literal[1] = 1
    id: Identity
    dataset_url: str
    dataset_revision: str
    license: Literal["CC-BY-4.0"]
    attribution: str = Field(min_length=1)
    selection_policy: str = Field(min_length=1)
    exposure_note: str = Field(min_length=1)
    model: Literal["prithvi-burn-scar"] = "prithvi-burn-scar"
    model_revision: Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")]
    model_sha256: Digest
    model_config_sha256: Digest
    sensor: Literal["hls"] = "hls"
    band_names: tuple[str, ...] = ("B02", "B03", "B04", "B8A", "B11", "B12")
    input_units: Literal["reflectance"] = "reflectance"
    split_pins: tuple[SplitPin, ...] = Field(min_length=3, max_length=3)
    cases: tuple[EvaluationCase, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_partition(self) -> Self:
        """Require unique cases and geographically disjoint validation/test sets.

        Raises:
            ValueError: Split lists, case identities, bands or partitions conflict.
        """
        if {pin.split for pin in self.split_pins} != {"train", "validation", "test"}:
            raise ValueError(
                "exactly one train, validation and test split pin required"
            )
        if self.band_names != ("B02", "B03", "B04", "B8A", "B11", "B12"):
            raise ValueError(
                "HLS band order must match the documented six-band contract"
            )
        if {case.split for case in self.cases} != {"validation", "test"}:
            raise ValueError("both validation and test cases are required")
        for values in (
            [case.id for case in self.cases],
            [case.source_name for case in self.cases],
            [case.image.sha256 for case in self.cases],
            [case.label.sha256 for case in self.cases],
        ):
            if len(set(values)) != len(values):
                raise ValueError("case identities and source artifacts must be unique")
        for index, case in enumerate(self.cases):
            for other in self.cases[index + 1 :]:
                if case.split != other.split and (
                    case.spatial_group == other.spatial_group
                    or case.bounds.overlaps(other.bounds)
                ):
                    raise ValueError(
                        "validation/test groups and footprints must separate"
                    )
        return self


class ConfusionCounts(EvaluationSchema):
    """Integer binary confusion counts with ignored labels and abstentions explicit."""

    tp: Count = 0
    fp: Count = 0
    tn: Count = 0
    fn: Count = 0
    label_ignored: Count = 0
    prediction_invalid_positive: Count = 0
    prediction_invalid_negative: Count = 0

    @property
    def scored_pixels(self) -> int:
        """Return valid-label pixels with a valid binary prediction."""
        return self.tp + self.fp + self.tn + self.fn

    @property
    def label_valid_pixels(self) -> int:
        """Return all valid labels, including those without a valid prediction."""
        return (
            self.scored_pixels
            + self.prediction_invalid_positive
            + self.prediction_invalid_negative
        )

    @property
    def total_pixels(self) -> int:
        """Return all raster pixels, including ignored labels."""
        return self.label_ignored + self.label_valid_pixels

    @classmethod
    def combine(cls, counts: tuple[ConfusionCounts, ...]) -> ConfusionCounts:
        """Sum integer counts before computing a split's micro-averaged ratios."""
        return cls(
            **{
                field: sum(getattr(value, field) for value in counts)
                for field in cls.model_fields
            }
        )


class MetricSummary(EvaluationSchema):
    """Metrics conditional on scored pixels, accompanied by valid-label coverage."""

    precision: float | None
    recall: float | None
    f1: float | None
    iou: float | None
    accuracy: float | None
    coverage: float | None

    @classmethod
    def from_counts(cls, counts: ConfusionCounts) -> MetricSummary:
        """Compute positive-class ratios; zero denominators produce JSON null."""

        def ratio(numerator: int, denominator: int) -> float | None:
            """Return a finite ratio or undefined when its denominator is zero."""
            return numerator / denominator if denominator else None

        return cls(
            precision=ratio(counts.tp, counts.tp + counts.fp),
            recall=ratio(counts.tp, counts.tp + counts.fn),
            f1=ratio(2 * counts.tp, 2 * counts.tp + counts.fp + counts.fn),
            iou=ratio(counts.tp, counts.tp + counts.fp + counts.fn),
            accuracy=ratio(counts.tp + counts.tn, counts.scored_pixels),
            coverage=ratio(counts.scored_pixels, counts.label_valid_pixels),
        )


class EvaluationOptions(EvaluationSchema):
    """Fixed-threshold CPU baseline settings; evaluation never tunes on test labels."""

    output_dir: Path
    cache_dir: Path
    split: Literal["validation", "test", "all"] = "validation"
    threshold: float = Field(default=0.5, strict=True, ge=0.5, le=0.5)
    batch_size: PositiveInt = 1
    cache_mb: PositiveInt = 8
    threads: PositiveInt = 2
    offline: bool = False


class ModelIdentity(EvaluationSchema):
    """Verified checkpoint/config bytes and registry contract used for inference."""

    name: str
    version: str
    revision: str
    checkpoint_sha256: Digest
    config_sha256: Digest
    registry_sha256: Digest


class RuntimeIdentity(EvaluationSchema):
    """Runtime versions and explicit CPU execution settings for reproducibility."""

    python: str
    platform: str
    whirld: str
    numpy: str
    rasterio: str
    torch: str
    terratorch: str
    device: Literal["cpu"] = "cpu"
    seed: Literal[2] = 2


class CaseResult(EvaluationSchema):
    """One source-pinned case with its exact-grid counts and conditional metrics."""

    id: str
    split: Split
    counts: ConfusionCounts
    metrics: MetricSummary
    prediction_sha256: Digest
    prediction_file: str
    width: PositiveInt
    height: PositiveInt
    crs: str
    transform: tuple[float, float, float, float, float, float]


class SplitResult(EvaluationSchema):
    """Separate split micro metrics, computed from integer count sums."""

    split: Split
    case_count: PositiveInt
    counts: ConfusionCounts
    metrics: MetricSummary


class EvaluationReport(EvaluationSchema):
    """Complete suite/options/provenance and split-isolated quality measurements."""

    schema_version: Literal[1] = 1
    suite: EvaluationSuite
    manifest_sha256: Digest
    model: ModelIdentity
    runtime: RuntimeIdentity
    options: EvaluationOptions
    created_at: datetime
    cases: tuple[CaseResult, ...]
    splits: tuple[SplitResult, ...]
    metric_domain: str = (
        "Quality ratios are conditional on valid labels with valid predictions; "
        "coverage and missing-prediction positive/negative counts must accompany them."
    )


class SuiteDocument(EvaluationSchema):
    """Validated manifest with the identity of the original serialized bytes."""

    suite: EvaluationSuite
    sha256: Digest


class PreparedCase(EvaluationSchema):
    """Verified local raster pair and explicit reflectance metadata adapter."""

    case: EvaluationCase
    image: Path
    label: Path
    item: Path
