"""Incremental embedding artifacts remain compatible, bounded, and recoverable."""

from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path

import numpy as np
import pytest

from whirld.core.chips import ChipMeta, ChipSet
from whirld.errors import InvalidInputError, WhirldError
from whirld.io.embedding_stream import EmbeddingMetadata, EmbeddingStreamWriter
from whirld.io.output import build_meta


def _metadata() -> EmbeddingMetadata:
    """Return a small header matching the existing sidecar contract."""
    return EmbeddingMetadata(
        model="clay-v1",
        model_version="1",
        crs="EPSG:32630",
        embed_dim=3,
        chip_size_px=8,
        resolution_m=10.0,
    )


def _chips(start: int, count: int) -> list[ChipMeta]:
    """Build contiguous chip identities without retaining previous batches."""
    return [
        ChipMeta(index=i, row=0, col=i, bbox=[i * 80, 0, (i + 1) * 80, 80])
        for i in range(start, start + count)
    ]


def test_streamed_npy_and_sidecar_match_existing_formats(tmp_path: Path) -> None:
    """NPY handles noncontiguous float64 batches and sidecar retains exact schema."""
    embeddings = np.arange(30, dtype=np.float64).reshape(5, 6)[:, ::2]
    metadata = _metadata()
    output = tmp_path / "nested" / "result.data"
    with EmbeddingStreamWriter(output, chip_count=5, metadata=metadata) as writer:
        writer.write(embeddings[:2], _chips(0, 2))
        writer.write(embeddings[2:], _chips(2, 3))
        assert not writer.output_path.exists()
        assert not writer.meta_path.exists()
        paths = writer.finish()
        assert paths == writer.finish()
        with pytest.raises(WhirldError, match="not open"):
            writer.write(embeddings[:1], _chips(0, 1))
    data_path, meta_path = paths
    assert data_path.name == "result.data.npy"
    assert meta_path.name == "result.data_meta.json"
    actual = np.load(data_path, allow_pickle=False)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, embeddings)
    expected = build_meta(
        model=metadata.model,
        model_version=metadata.model_version,
        crs=metadata.crs,
        embed_dim=metadata.embed_dim,
        chip_size_px=metadata.chip_size_px,
        resolution_m=metadata.resolution_m,
        chipset=ChipSet(chips=_chips(0, 5), chip_size_px=8, n_rows=1, n_cols=5),
    )
    expected["timestamp"] = metadata.timestamp
    assert json.loads(meta_path.read_text()) == expected
    assert not list(data_path.parent.glob(".whirld-embed-*"))


def test_writer_does_not_retain_batch_arrays_or_chip_records(tmp_path: Path) -> None:
    """Completed batches become collectable even while the output remains open."""
    with EmbeddingStreamWriter(
        tmp_path / "emb.npy", chip_count=40, metadata=_metadata()
    ) as writer:
        for index in range(40):
            array = np.full((1, 3), index, dtype=np.float32)
            chips = _chips(index, 1)
            array_ref = weakref.ref(array)
            chip_ref = weakref.ref(chips[0])
            writer.write(array, chips)
            del array, chips
            gc.collect()
            assert array_ref() is None
            assert chip_ref() is None
        writer.finish()


def test_failed_inference_preserves_existing_pair(tmp_path: Path) -> None:
    """An exception before finish removes staging and preserves old output bytes."""
    output = tmp_path / "result.npy"
    sidecar = tmp_path / "result_meta.json"
    output.write_bytes(b"old-array")
    sidecar.write_text("old-metadata")
    with pytest.raises(RuntimeError, match="inference failed"):
        with EmbeddingStreamWriter(
            output, chip_count=2, metadata=_metadata()
        ) as writer:
            writer.write(np.ones((1, 3)), _chips(0, 1))
            raise RuntimeError("inference failed")
    assert output.read_bytes() == b"old-array"
    assert sidecar.read_text() == "old-metadata"
    assert not list(tmp_path.glob(".whirld-embed-*"))


def test_incomplete_or_unfinished_stream_is_not_published(tmp_path: Path) -> None:
    """Normal exit requires explicit completion, and missing rows cannot finish."""
    with pytest.raises(InvalidInputError, match="incomplete"):
        with EmbeddingStreamWriter(
            tmp_path / "missing.npy", chip_count=2, metadata=_metadata()
        ) as writer:
            writer.write(np.ones((1, 3)), _chips(0, 1))
            writer.finish()
    with pytest.raises(WhirldError, match="not finished"):
        with EmbeddingStreamWriter(
            tmp_path / "unfinished.npy", chip_count=1, metadata=_metadata()
        ) as writer:
            writer.write(np.ones((1, 3)), _chips(0, 1))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing", [False, True])
def test_second_replace_failure_restores_both_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    """Publishing the sidecar cannot leave a new array paired with old metadata."""
    output = tmp_path / "result.npy"
    sidecar = tmp_path / "result_meta.json"
    if existing:
        output.write_bytes(b"old-array")
        sidecar.write_text("old-metadata")
    original_replace = Path.replace

    def replace(source: Path, destination: str | Path) -> Path:
        """Fail exactly the second staged-file installation, permitting rollback."""
        if source.name == "metadata.json" and Path(destination) == sidecar:
            raise OSError("simulated sidecar publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(WhirldError, match="finalize"):
        with EmbeddingStreamWriter(
            output, chip_count=1, metadata=_metadata()
        ) as writer:
            writer.write(np.ones((1, 3)), _chips(0, 1))
            writer.finish()
    if existing:
        assert output.read_bytes() == b"old-array"
        assert sidecar.read_text() == "old-metadata"
    else:
        assert not output.exists()
        assert not sidecar.exists()
    assert not list(tmp_path.glob(".whirld-embed-*"))


@pytest.mark.parametrize(
    "array,chips",
    [
        (np.ones((1, 2)), _chips(0, 1)),
        (np.ones((2, 3)), _chips(0, 1)),
        (np.ones((1, 3)), _chips(1, 1)),
        (np.ones((2, 3)), _chips(0, 2)),
        (np.full((1, 3), np.nan), _chips(0, 1)),
        (np.full((1, 3), 1e100), _chips(0, 1)),
        (np.full((1, 3), "value"), _chips(0, 1)),
        (np.ones((1, 3)), [ChipMeta(index=0, row=0, col=0, bbox=[0, 0, 1, np.nan])]),
    ],
)
def test_invalid_batch_does_not_advance_stream(
    tmp_path: Path, array: np.ndarray, chips: list[ChipMeta]
) -> None:
    """Reject malformed data before either artifact is modified, allowing retry."""
    with EmbeddingStreamWriter(
        tmp_path / "result.npy", chip_count=1, metadata=_metadata()
    ) as writer:
        with pytest.raises(InvalidInputError):
            writer.write(array, chips)
        writer.write(np.ones((1, 3)), _chips(0, 1))
        output, sidecar = writer.finish()
    np.testing.assert_array_equal(np.load(output), np.ones((1, 3)))
    assert len(json.loads(sidecar.read_text())["chips"]) == 1


def test_empty_output_is_valid_and_context_cannot_be_reused(tmp_path: Path) -> None:
    """Zero-row output has valid headers, while resource lifetimes remain single-use."""
    writer = EmbeddingStreamWriter(
        tmp_path / "empty", chip_count=0, metadata=_metadata()
    )
    with pytest.raises(WhirldError, match="not open"):
        writer.finish()
    with writer:
        writer.write(np.empty((0, 3)), [])
        output, sidecar = writer.finish()
    assert np.load(output).shape == (0, 3)
    assert json.loads(sidecar.read_text())["chips"] == []
    with pytest.raises(WhirldError, match="cannot be reused"):
        with writer:
            pass


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_invalid_total_count_fails_before_creating_files(
    tmp_path: Path, count: int
) -> None:
    """Invalid counts cannot create directories or outputs."""
    with pytest.raises(InvalidInputError):
        EmbeddingStreamWriter(
            tmp_path / "child" / "output", chip_count=count, metadata=_metadata()
        )
    assert list(tmp_path.iterdir()) == []


def test_publish_interruption_restores_existing_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A keyboard interrupt during publication still restores previous outputs."""
    output = tmp_path / "result.npy"
    sidecar = tmp_path / "result_meta.json"
    output.write_bytes(b"old-array")
    sidecar.write_text("old-metadata")
    original_replace = Path.replace

    def replace(source: Path, destination: str | Path) -> Path:
        """Interrupt only new metadata publication, leaving rollback possible."""
        if source.name == "metadata.json":
            raise KeyboardInterrupt()
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(KeyboardInterrupt):
        with EmbeddingStreamWriter(
            output, chip_count=1, metadata=_metadata()
        ) as writer:
            writer.write(np.ones((1, 3)), _chips(0, 1))
            writer.finish()
    assert output.read_bytes() == b"old-array"
    assert sidecar.read_text() == "old-metadata"
    assert not list(tmp_path.glob(".whirld-embed-*"))


def test_failed_rollback_preserves_recovery_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed restoration preserves backups and reports their recovery location."""
    output = tmp_path / "result.npy"
    sidecar = tmp_path / "result_meta.json"
    output.write_bytes(b"old-array")
    sidecar.write_text("old-metadata")
    original_replace = Path.replace

    def replace(source: Path, destination: str | Path) -> Path:
        """Fail publication and backup restoration while permitting original moves."""
        if source.name == "metadata.json" or source.name.startswith("previous-"):
            raise OSError("simulated unavailable filesystem")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(WhirldError, match="recovery files remain") as error:
        with EmbeddingStreamWriter(
            output, chip_count=1, metadata=_metadata()
        ) as writer:
            writer.write(np.ones((1, 3)), _chips(0, 1))
            writer.finish()
    staging = list(tmp_path.glob(".whirld-embed-*"))
    assert len(staging) == 1
    assert str(staging[0]) in str(error.value)
    assert (staging[0] / "previous-0").read_bytes() == b"old-array"
    assert (staging[0] / "previous-1").read_text() == "old-metadata"


def test_flush_failure_leaves_existing_outputs_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disk synchronization failure aborts before publishing either artifact."""
    import whirld.io.embedding_stream as stream

    output = tmp_path / "result.npy"
    sidecar = tmp_path / "result_meta.json"
    output.write_bytes(b"old-array")
    sidecar.write_text("old-metadata")

    def fail_sync(descriptor: int) -> None:
        """Simulate a failed output filesystem without changing real disk state."""
        raise OSError("simulated synchronization failure")

    monkeypatch.setattr(stream.os, "fsync", fail_sync)
    with pytest.raises(WhirldError, match="finalize"):
        with EmbeddingStreamWriter(
            output, chip_count=1, metadata=_metadata()
        ) as writer:
            writer.write(np.ones((1, 3)), _chips(0, 1))
            writer.finish()
    assert output.read_bytes() == b"old-array"
    assert sidecar.read_text() == "old-metadata"
    assert not list(tmp_path.glob(".whirld-embed-*"))
