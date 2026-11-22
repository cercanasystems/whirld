"""Disk-backed classification ranking and atomic incremental GeoJSON output."""

from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from types import TracebackType

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .._version import __version__
from ..core.chips import ChipMeta, ChipSet
from ..errors import InvalidInputError, WhirldError
from ..logging_setup import get_logger
from .output import build_feature_collection

_log = get_logger("io.classification_stream")


class ClassificationMetadata(BaseModel):
    """Immutable source/model/query identity for a classification FeatureCollection.

    Queries retain their original spelling and order, including repeated prompts,
    matching the eager API. Invalid empty query collections or blank prompts raise
    Pydantic ValidationError before opening any output resources.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    model: str
    model_version: str
    crs: str
    queries: tuple[str, ...] = Field(min_length=1)
    whirld_version: str = __version__

    @field_validator("queries")
    @classmethod
    def valid_queries(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject blank prompts while preserving eager query spelling/order."""
        if any(not query.strip() for query in value):
            raise ValueError("queries must contain nonempty prompts")
        return value

    def header_json(self) -> str:
        """Serialize the small GeoJSON header without allocating feature records.

        Returns:
            JSON object containing eager-compatible provenance and query fields.
        """
        return json.dumps(
            {
                "type": "FeatureCollection",
                "source_crs": self.crs,
                "model": self.model,
                "model_version": self.model_version,
                "whirld_version": self.whirld_version,
                "query": self.queries[0],
                "queries": self.queries,
            },
            allow_nan=False,
        )


class ClassificationStreamWriter:
    """Own disk sorting, staged output and atomic single-file publication.

    Args:
        output_path: GeoJSON destination; its exact filename is preserved.
        chip_count: Nonnegative total number of expected ordered chip records.
        metadata: Typed model/source/query provenance for the collection.
        top_k: Positive maximum number of matches, or None to emit all matches.
        threshold: Inclusive primary-query probability cutoff in [0, 1].

    SQLite retains completed features on disk, with a 2 MiB page cache and file
    backed temporary sorting. Memory depends on the supplied batch/query sizes,
    SQLite's fixed cache and one serialized feature, never total scene size or
    top-k. Disk storage grows with qualifying features. The writer emits globally
    descending primary scores, breaking ties by ascending global chip index.
    Explicit finish publishes only after every expected chip has been received.
    Failures and interruptions preserve previous destination contents.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        chip_count: int,
        metadata: ClassificationMetadata,
        top_k: int | None = 5,
        threshold: float = 0.0,
    ) -> None:
        """Validate result limits and bind resources without creating files.

        Raises:
            InvalidInputError: Counts, threshold or top-k violate the contract.
        """
        if isinstance(chip_count, bool) or not isinstance(chip_count, int):
            raise InvalidInputError("Classification chip_count must be an integer.")
        if chip_count < 0:
            raise InvalidInputError("Classification chip_count must be nonnegative.")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
        ):
            raise InvalidInputError("Classification top_k must be positive or None.")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise InvalidInputError("Classification threshold must be in [0, 1].")
        self._output_path = Path(output_path).absolute()
        self._chip_count = chip_count
        self._metadata = metadata
        self._top_k = top_k
        self._threshold = threshold
        self._written = 0
        self._feature_count: int | None = None
        self._stage: Path | None = None
        self._connection: sqlite3.Connection | None = None
        self._entered = False
        self._finished = False

    @property
    def output_path(self) -> Path:
        """Return the final GeoJSON destination without changing its suffix."""
        return self._output_path

    @property
    def feature_count(self) -> int:
        """Return published match count, rejecting access before successful finish.

        Raises:
            WhirldError: No complete output has been published yet.
        """
        if not self._finished or self._feature_count is None:
            raise WhirldError("Classification output has not been finished.")
        return self._feature_count

    def __enter__(self) -> ClassificationStreamWriter:
        """Create an owned database beside the destination and return this writer.

        Raises:
            WhirldError: Writer is reused or staging/database setup fails.
        """
        if self._entered:
            raise WhirldError("A classification stream writer cannot be reused.")
        self._entered = True
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._stage = Path(
                tempfile.mkdtemp(
                    prefix=".whirld-classify-", dir=self.output_path.parent
                )
            )
            connection = sqlite3.connect(self._stage / "scores.sqlite")
            self._connection = connection
            connection.execute("PRAGMA cache_size=-2048")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA mmap_size=0")
            connection.execute(
                "CREATE TABLE features (chip_index INTEGER PRIMARY KEY, "
                "score REAL NOT NULL, payload TEXT NOT NULL)"
            )
        except BaseException as exc:
            self._cleanup()
            if isinstance(exc, (OSError, sqlite3.Error)):
                raise WhirldError(
                    "Could not open streaming classification output."
                ) from exc
            raise
        return self

    def _active(self) -> sqlite3.Connection:
        """Require the active unfinished database connection.

        Raises:
            WhirldError: Writer is unopened, closed or already finished.
        """
        if self._connection is None or self._finished:
            raise WhirldError("Classification stream writer is not open for writing.")
        return self._connection

    def _validate_batch(self, scores: np.ndarray, chips: list[ChipMeta]) -> None:
        """Validate a bounded batch completely before modifying staged results.

        Args:
            scores: Numeric matrix with one probability for each chip/query pair.
            chips: Consecutive global chip metadata following the previous batch.

        Raises:
            InvalidInputError: Shape, probabilities, count or geometry are invalid.
        """
        if scores.ndim != 2 or scores.shape != (
            len(chips),
            len(self._metadata.queries),
        ):
            raise InvalidInputError(
                "Classification scores must match chips and queries."
            )
        if scores.dtype.kind not in "fiu" or not np.isfinite(scores).all():
            raise InvalidInputError(
                "Classification scores must be finite probabilities."
            )
        if np.any(scores < 0) or np.any(scores > 1):
            raise InvalidInputError("Classification scores must lie in [0, 1].")
        if self._written + len(chips) > self._chip_count:
            raise InvalidInputError("Classification exceeds the declared chip count.")
        for offset, chip in enumerate(chips):
            if chip.index != self._written + offset:
                raise InvalidInputError(
                    "Classification chip indices must be contiguous."
                )
            if (
                chip.row < 0
                or chip.col < 0
                or len(chip.bbox) != 4
                or not all(math.isfinite(value) for value in chip.bbox)
                or chip.bbox[0] >= chip.bbox[2]
                or chip.bbox[1] >= chip.bbox[3]
            ):
                raise InvalidInputError("Classification chip geometry must be valid.")

    def _feature_json(self, scores: np.ndarray, chip: ChipMeta) -> tuple[float, str]:
        """Serialize one eager-compatible feature at the GeoJSON adapter boundary.

        Args:
            scores: One-row query probability matrix for this chip.
            chip: Source-grid chip geometry and global identity.

        Returns:
            Effective primary probability and one WGS84 GeoJSON feature string.

        Raises:
            InvalidInputError: Source CRS/geometry cannot produce finite GeoJSON.
        """
        try:
            collection = build_feature_collection(
                scores,
                ChipSet(chips=[chip], chip_size_px=1, n_rows=1, n_cols=1),
                queries=list(self._metadata.queries),
                model=self._metadata.model,
                model_version=self._metadata.model_version,
                crs=self._metadata.crs,
            )
            feature = collection["features"][0]
            return feature["properties"]["score"], json.dumps(feature, allow_nan=False)
        except (ValueError, OverflowError) as exc:
            raise InvalidInputError(
                "Could not serialize classification geometry."
            ) from exc

    def write(self, scores: np.ndarray, chips: list[ChipMeta]) -> None:
        """Stage one ordered batch without retaining completed features in memory.

        Args:
            scores: Matrix shaped (len(chips), len(queries)), probabilities [0, 1].
            chips: Global row-major chip metadata matching the score rows.

        Raises:
            InvalidInputError: Batch violates shape/order/probability/geometry rules.
            WhirldError: Writer is inactive or staging the batch fails.
        """
        connection = self._active()
        self._validate_batch(scores, chips)
        try:
            with connection:
                for index, chip in enumerate(chips):
                    score, payload = self._feature_json(scores[index : index + 1], chip)
                    if score >= self._threshold:
                        connection.execute(
                            "INSERT INTO features VALUES (?, ?, ?)",
                            (chip.index, score, payload),
                        )
        except (OSError, sqlite3.Error) as exc:
            raise WhirldError("Could not stage classification scores.") from exc
        self._written += len(chips)

    def finish(self) -> Path:
        """Emit globally ranked GeoJSON and atomically publish its complete file.

        Returns:
            Published output path. Repeated successful calls return the same path.

        Raises:
            InvalidInputError: Fewer than the declared total chips were received.
            WhirldError: Serialization, output flushing or atomic replacement fails.
        """
        if self._finished:
            return self.output_path
        connection = self._active()
        if self._written != self._chip_count:
            raise InvalidInputError(
                f"Classification output incomplete: {self._written} of "
                f"{self._chip_count} chips received."
            )
        assert self._stage is not None
        staged = self._stage / "classification.geojson"
        count = 0
        try:
            query = "SELECT payload FROM features ORDER BY score DESC, chip_index ASC"
            parameters: tuple[int, ...] = ()
            if self._top_k is not None and self._top_k < self._chip_count:
                query += " LIMIT ?"
                parameters = (self._top_k,)
            with staged.open("w", encoding="utf-8") as output:
                output.write(self._metadata.header_json()[:-1] + ',"features":[')
                for (payload,) in connection.execute(query, parameters):
                    if count:
                        output.write(",")
                    output.write(payload)
                    count += 1
                output.write("]}\n")
                output.flush()
                os.fsync(output.fileno())
            self._close_connection()
            staged.replace(self.output_path)
        except (OSError, sqlite3.Error) as exc:
            raise WhirldError(
                "Could not finalize streaming classification output."
            ) from exc
        self._feature_count = count
        self._finished = True
        self._cleanup()
        _log.info("Wrote %s", self.output_path.name)
        return self.output_path

    def _close_connection(self) -> None:
        """Release the owned SQLite handle; safe to call repeatedly.

        Raises:
            sqlite3.Error: Closing the database fails.
        """
        connection = self._connection
        self._connection = None
        if connection is not None:
            connection.close()

    def _cleanup(self) -> None:
        """Release database and staging without masking an active failure."""
        try:
            self._close_connection()
        except sqlite3.Error:
            _log.debug(
                "Could not close classification staging database.", exc_info=True
            )
        if self._stage is not None:
            shutil.rmtree(self._stage, ignore_errors=True)
            self._stage = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Clean staging; reject normal context exit without successful finish.

        Args:
            exc_type: Active exception type, or None on a normal exit.
            exc: Active exception instance, never suppressed.
            traceback: Original traceback, never suppressed.

        Raises:
            WhirldError: A normal exit omitted successful publication.
        """
        self._cleanup()
        if exc_type is None and not self._finished:
            raise WhirldError(
                "Classification stream was not finished; output discarded."
            )
