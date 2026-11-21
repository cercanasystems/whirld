"""Bounded-memory, staged NPY and JSON-sidecar output for embedding batches."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, TextIO

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .._version import __version__
from ..core.chips import ChipMeta
from ..errors import InvalidInputError, WhirldError
from ..logging_setup import get_logger
from .output import _meta_path_for, _utc_now_iso

_log = get_logger("io.embedding_stream")


class EmbeddingMetadata(BaseModel):
    """Small, immutable sidecar header; chip records are streamed separately.

    Fields match the existing embedding metadata schema. The timestamp is UTC,
    and dimensions/resolution must be positive. No scene-length data is retained.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model: str
    model_version: str
    whirld_version: str = __version__
    timestamp: str = Field(default_factory=_utc_now_iso)
    crs: str
    embed_dim: int = Field(gt=0)
    chip_size_px: int = Field(gt=0)
    resolution_m: float = Field(gt=0, allow_inf_nan=False)


class EmbeddingStreamWriter:
    """Own staged files and write ordered batches without mapping the full output.

    Args:
        output_path: NPY destination; ``.npy`` is appended when absent.
        chip_count: Total number of rows expected in the final NPY array.
        metadata: Typed header defining the embedding width and sidecar fields.

    Use as a context manager, call ``write`` for each batch, then explicitly call
    ``finish`` to publish. Aborted or unfinished writes leave prior output files
    untouched. Publication replaces each file individually and rolls back ordinary
    failures; two-file replacement is not crash-atomic or concurrent-writer safe.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        chip_count: int,
        metadata: EmbeddingMetadata,
    ) -> None:
        """Bind destination identity and validate total count before opening files.

        Args:
            output_path: Destination filename; append ``.npy`` when absent.
            chip_count: Nonnegative total number of embedding rows to publish.
            metadata: Validated dimensions and provenance for the JSON sidecar.

        Raises:
            InvalidInputError: chip_count is negative or not an integer.
        """
        if isinstance(chip_count, bool) or not isinstance(chip_count, int):
            raise InvalidInputError("Streaming chip_count must be an integer.")
        if chip_count < 0:
            raise InvalidInputError("Streaming chip_count must be nonnegative.")
        path = Path(output_path).absolute()
        if path.suffix != ".npy":
            path = path.with_suffix(path.suffix + ".npy")
        self._output_path = path
        self._meta_path = _meta_path_for(path)
        self._chip_count = chip_count
        self._metadata = metadata
        self._written = 0
        self._data: BinaryIO | None = None
        self._meta: TextIO | None = None
        self._stage: Path | None = None
        self._entered = False
        self._finished = False
        self._preserve_staging = False

    @property
    def output_path(self) -> Path:
        """Return the bound NPY destination, including its normalized suffix."""
        return self._output_path

    @property
    def meta_path(self) -> Path:
        """Return the companion JSON metadata destination."""
        return self._meta_path

    def __enter__(self) -> EmbeddingStreamWriter:
        """Open staging files and emit small fixed headers; return this writer.

        Returns:
            This writer with both temporary files ready to accept batches.

        Raises:
            WhirldError: The writer was reused or a staging file cannot be opened.
        """
        if self._entered:
            raise WhirldError("An embedding stream writer cannot be reused.")
        self._entered = True
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._stage = Path(
                tempfile.mkdtemp(prefix=".whirld-embed-", dir=self.output_path.parent)
            )
            self._data = (self._stage / "embeddings.npy").open("wb")
            self._meta = (self._stage / "metadata.json").open("w", encoding="utf-8")
            np.lib.format.write_array_header_2_0(
                self._data,
                {
                    "descr": np.dtype(np.float32).str,
                    "fortran_order": False,
                    "shape": (self._chip_count, self._metadata.embed_dim),
                },
            )
            header = self._metadata.model_dump_json()
            self._meta.write(header[:-1] + ',"chips":[')
        except OSError as exc:
            self._cleanup()
            raise WhirldError("Could not open streaming embedding outputs.") from exc
        return self

    def write(self, embeddings: np.ndarray, chips: list[ChipMeta]) -> None:
        """Append one batch, checking shape, finite values, and contiguous indices.

        Input rows must match metadata records and continue the preceding batch.
        Invalid batches raise InvalidInputError before modifying either file.
        Only the supplied batch is converted to contiguous float32 bytes.

        Args:
            embeddings: Numeric array shaped ``(len(chips), embed_dim)``. Values
                must remain finite when converted to float32.
            chips: Ordered geometry records with consecutive global indices.

        Returns:
            None. Appends rows and records to staging; does not publish files.

        Raises:
            InvalidInputError: Shape, values, geometry, indices, or count violate
                the declared output contract.
            WhirldError: The writer is inactive or writing either file fails.
        """
        data, metadata = self._handles()
        if embeddings.ndim != 2 or embeddings.shape != (
            len(chips),
            self._metadata.embed_dim,
        ):
            raise InvalidInputError(
                "Streaming embeddings must match chip count and width."
            )
        if self._written + len(chips) > self._chip_count:
            raise InvalidInputError("Streaming output exceeds the declared chip count.")
        for offset, chip in enumerate(chips):
            if chip.index != self._written + offset:
                raise InvalidInputError(
                    "Streaming chip indices must be contiguous and ordered."
                )
            if (
                chip.row < 0
                or chip.col < 0
                or len(chip.bbox) != 4
                or not all(math.isfinite(value) for value in chip.bbox)
            ):
                raise InvalidInputError(
                    "Streaming chip geometry must be finite and valid."
                )
        if embeddings.dtype.kind not in "fiu":
            raise InvalidInputError("Streaming embeddings must contain numeric values.")
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.ascontiguousarray(embeddings, dtype=np.float32)
        if not np.isfinite(array).all():
            raise InvalidInputError(
                "Streaming embeddings must contain finite float32 values."
            )
        if not chips:
            return
        try:
            data.write(memoryview(array).cast("B"))
            for chip in chips:
                if self._written:
                    metadata.write(",")
                metadata.write(chip.model_dump_json())
                self._written += 1
        except OSError as exc:
            self._cleanup()
            raise WhirldError("Could not write streaming embedding output.") from exc

    def finish(self) -> tuple[Path, Path]:
        """Publish a complete pair and return its paths; repeated calls are harmless.

        Returns:
            The final NPY path and JSON sidecar path, in that order.

        Raises:
            InvalidInputError: Fewer rows were written than the declared total.
            WhirldError: Files could not be flushed or published. Ordinary publish
                failures restore previous destinations; failed rollback preserves
                staging backups and reports their location for recovery.
        """
        if self._finished:
            return self.output_path, self.meta_path
        data, metadata = self._handles()
        if self._written != self._chip_count:
            raise InvalidInputError(
                f"Streaming output is incomplete: wrote {self._written} of "
                f"{self._chip_count} chips."
            )
        try:
            metadata.write("]}\n")
            data.flush()
            metadata.flush()
            os.fsync(data.fileno())
            os.fsync(metadata.fileno())
            self._close_handles()
            self._publish()
        except OSError as exc:
            self._cleanup()
            raise WhirldError("Could not finalize streaming embedding output.") from exc
        self._finished = True
        self._cleanup()
        _log.info("Wrote %s and %s", self.output_path.name, self.meta_path.name)
        return self.output_path, self.meta_path

    def _handles(self) -> tuple[BinaryIO, TextIO]:
        """Require an active unfinished writer before modifying staged files.

        Returns:
            The open binary embedding stream and text metadata stream.

        Raises:
            WhirldError: The writer is unopened, closed, or already finished.
        """
        if self._data is None or self._meta is None or self._finished:
            raise WhirldError("Embedding stream writer is not open for writing.")
        return self._data, self._meta

    def _publish(self) -> None:
        """Replace staged files, backing up previous outputs for rollback.

        Returns:
            None. Successful publication installs both destination files.

        Raises:
            OSError: A destination cannot be replaced; previous files are restored
                before propagation when rollback succeeds.
            WhirldError: Rollback fails; staging backups remain for recovery.

        Interruptions also attempt rollback before propagating. The pair is not
        crash-atomic and must not be published concurrently to the same paths.
        """
        assert self._stage is not None
        pairs = (
            (self._stage / "embeddings.npy", self.output_path),
            (self._stage / "metadata.json", self.meta_path),
        )
        backups: list[tuple[Path, Path]] = []
        published: list[Path] = []
        try:
            for index, (_, target) in enumerate(pairs):
                if target.is_dir():
                    raise OSError(f"Output destination is a directory: {target}")
                if target.exists() or target.is_symlink():
                    backup = self._stage / f"previous-{index}"
                    target.replace(backup)
                    backups.append((backup, target))
            for source, target in pairs:
                source.replace(target)
                published.append(target)
        except BaseException:
            try:
                for target in reversed(published):
                    target.unlink(missing_ok=True)
                for backup, target in reversed(backups):
                    backup.replace(target)
            except OSError as rollback_error:
                self._preserve_staging = True
                raise WhirldError(
                    "Streaming output replacement and rollback failed; "
                    f"recovery files remain in {self._stage}."
                ) from rollback_error
            raise

    def _close_handles(self) -> None:
        """Clear owned handles and attempt to close both streams.

        Returns:
            None. Safe to call again after handles have been cleared.

        Raises:
            OSError: A close fails; both streams are attempted before the first
                close error is propagated.
        """
        handles = (self._data, self._meta)
        self._data = None
        self._meta = None
        failure: OSError | None = None
        for handle in handles:
            if handle is not None:
                try:
                    handle.close()
                except OSError as exc:
                    failure = failure or exc
        if failure is not None:
            raise failure

    def _cleanup(self) -> None:
        """Release handles and remove staging unless rollback needs manual recovery.

        Returns:
            None. Close errors are logged; staging-removal errors are ignored so
            cleanup does not replace the original inference or publication error.
        """
        try:
            self._close_handles()
        except OSError:
            _log.debug("Error closing aborted streaming output.", exc_info=True)
        if self._stage is not None and not self._preserve_staging:
            shutil.rmtree(self._stage, ignore_errors=True)
            self._stage = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Discard unfinished output; require explicit finish on a successful exit.

        Args:
            exc_type: Exception class leaving the context, or None on normal exit.
            exc: Active exception instance, if any; it is not suppressed.
            traceback: Original exception traceback, if any.

        Returns:
            None, preserving propagation of an existing exception.

        Raises:
            WhirldError: The context exits normally without a successful finish.
        """
        self._cleanup()
        if exc_type is None and not self._finished:
            raise WhirldError(
                "Embedding stream was not finished; staged output discarded."
            )
