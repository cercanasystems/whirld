"""Disk-ranked classification output matches eager semantics and survives failures."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from whirld.core.chips import ChipMeta, ChipSet
from whirld.errors import InvalidInputError, WhirldError
from whirld.io.classification_stream import (
    ClassificationMetadata,
    ClassificationStreamWriter,
)
from whirld.io.output import build_feature_collection


def metadata(
    queries: tuple[str, ...] = ("farmland", "forest")
) -> ClassificationMetadata:
    """Provide source/model identity matching the eager comparison fixture."""
    return ClassificationMetadata(
        model="remoteclip", model_version="1.0", crs="EPSG:32630", queries=queries
    )


def chips(count: int) -> list[ChipMeta]:
    """Create identifiable projected chips, including a non-square final footprint."""
    return [
        ChipMeta(
            index=i,
            row=i // 4,
            col=i % 4,
            bbox=[500000 + i * 120, 4500000, 500120 + i * 120, 4500240],
        )
        for i in range(count)
    ]


def eager(
    scores: np.ndarray,
    records: list[ChipMeta],
    meta: ClassificationMetadata,
    *,
    top_k: int | None,
    threshold: float,
) -> dict[str, Any]:
    """Use the existing serializer/filter/ranking contract as an independent oracle."""
    collection = build_feature_collection(
        scores,
        ChipSet(chips=records, chip_size_px=224, n_rows=3, n_cols=4),
        queries=list(meta.queries),
        model=meta.model,
        model_version=meta.model_version,
        crs=meta.crs,
    )
    features = [
        feature
        for feature in collection["features"]
        if feature["properties"]["score"] >= threshold
    ]
    features.sort(key=lambda feature: feature["properties"]["score"], reverse=True)
    collection["features"] = features if top_k is None else features[:top_k]
    return json.loads(json.dumps(collection))


@pytest.mark.parametrize("top_k", [1, 5, 100, 2**80, None])
@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_batched_disk_ranking_matches_eager(
    tmp_path: Path, top_k: int | None, threshold: float
) -> None:
    """Partial batches, ties, thresholds and all/top-k preserve exact GeoJSON values."""
    scores = np.random.default_rng(713).random((13, 2), dtype=np.float32)
    scores[[0, 2, 9], 0] = 0.75
    records = chips(len(scores))
    path = tmp_path / "classified.json"
    with ClassificationStreamWriter(
        path,
        chip_count=len(scores),
        metadata=metadata(),
        top_k=top_k,
        threshold=threshold,
    ) as writer:
        for start in range(0, len(scores), 4):
            writer.write(scores[start : start + 4], records[start : start + 4])
        assert not path.exists()
        assert writer.finish() == path
        assert writer.finish() == path
        actual = json.loads(path.read_text())
        assert writer.feature_count == len(actual["features"])
    assert actual == eager(
        scores, records, metadata(), top_k=top_k, threshold=threshold
    )
    assert list(tmp_path.glob(".whirld-classify-*")) == []


def test_duplicate_queries_retain_legacy_primary_score(tmp_path: Path) -> None:
    """Repeated prompts use the final duplicate column, as eager's scores map does."""
    meta = metadata(("farm", "forest", "farm"))
    scores = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]], dtype=np.float32)
    records = chips(2)
    path = tmp_path / "duplicates.geojson"
    with ClassificationStreamWriter(path, chip_count=2, metadata=meta) as writer:
        writer.write(scores[:1], records[:1])
        writer.write(scores[1:], records[1:])
        writer.finish()
    actual = json.loads(path.read_text())
    assert actual == eager(scores, records, meta, top_k=5, threshold=0.0)
    assert actual["features"][0]["properties"]["chip_index"] == 1


def test_single_query_and_empty_scene(tmp_path: Path) -> None:
    """A single prompt and a zero-chip scene each produce valid GeoJSON."""
    for count in (0, 3):
        path = tmp_path / f"{count}.geojson"
        scores = np.full((count, 1), 0.5, dtype=np.float32)
        records = chips(count)
        meta = metadata(("farmland",))
        with ClassificationStreamWriter(
            path, chip_count=count, metadata=meta, top_k=None
        ) as writer:
            writer.write(scores, records)
            writer.finish()
        assert json.loads(path.read_text()) == eager(
            scores, records, meta, top_k=None, threshold=0.0
        )


@pytest.mark.parametrize(
    "values",
    [
        {"chip_count": -1},
        {"chip_count": True},
        {"chip_count": 1.2},
        {"top_k": 0},
        {"top_k": -1},
        {"top_k": True},
        {"top_k": 1.5},
        {"threshold": -0.1},
        {"threshold": 1.1},
        {"threshold": float("nan")},
        {"threshold": float("inf")},
        {"threshold": True},
    ],
)
def test_invalid_configuration_does_not_create_output(
    tmp_path: Path, values: dict[str, Any]
) -> None:
    """Result limits are checked before staging paths or opening resources."""
    with pytest.raises(InvalidInputError):
        ClassificationStreamWriter(
            tmp_path / "subdir/out.json",
            metadata=metadata(),
            **({"chip_count": 1} | values),
        )
    assert not (tmp_path / "subdir").exists()


@pytest.mark.parametrize("queries", [(), ("",), ("  ",), ("valid", "")])
def test_invalid_query_metadata(queries: tuple[str, ...]) -> None:
    """Empty or blank prompts cannot define the output columns."""
    with pytest.raises(ValidationError):
        metadata(queries)


@pytest.mark.parametrize(
    "scores",
    [
        np.ones((1,), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.array([[0.1, float("nan")]]),
        np.array([[0.1, float("inf")]]),
        np.array([[-0.1, 0.1]]),
        np.array([[1.1, 0.1]]),
        np.array([[True, False]]),
        np.array([["0.1", "0.2"]]),
        np.array([[complex(0.1, 0), complex(0.2, 0)]]),
    ],
)
def test_invalid_scores_leave_batch_unwritten(
    tmp_path: Path, scores: np.ndarray
) -> None:
    """Invalid matrices cannot consume chip positions or corrupt later valid writes."""
    path = tmp_path / "scores.geojson"
    with ClassificationStreamWriter(path, chip_count=1, metadata=metadata()) as writer:
        with pytest.raises(InvalidInputError):
            writer.write(scores, chips(1))
        writer.write(np.array([[0.3, 0.7]]), chips(1))
        writer.finish()
    assert len(json.loads(path.read_text())["features"]) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"index": 3},
        {"row": -1},
        {"col": -1},
        {"bbox": [0, 0, 1]},
        {"bbox": [0, 0, float("nan"), 1]},
        {"bbox": [1, 0, 0, 1]},
        {"bbox": [0, 1, 1, 0]},
    ],
)
def test_invalid_geometry_preserves_existing_output(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    """Bad chip geometry aborts a run before replacing an earlier result."""
    path = tmp_path / "previous.geojson"
    path.write_text("previous result")
    record = chips(1)[0].model_copy(update=changes)
    with pytest.raises(InvalidInputError):
        with ClassificationStreamWriter(
            path, chip_count=1, metadata=metadata()
        ) as writer:
            writer.write(np.array([[0.3, 0.7]]), [record])
    assert path.read_text() == "previous result"
    assert not list(tmp_path.glob(".whirld-classify-*"))


def test_incomplete_and_excess_counts_fail_without_publication(tmp_path: Path) -> None:
    """Declared scene count is enforced before any result is published."""
    path = tmp_path / "result.json"
    with pytest.raises(InvalidInputError, match="incomplete"):
        with ClassificationStreamWriter(
            path, chip_count=2, metadata=metadata()
        ) as writer:
            writer.write(np.array([[0.3, 0.7]]), chips(1))
            writer.finish()
    assert not path.exists()
    with pytest.raises(InvalidInputError, match="declared"):
        with ClassificationStreamWriter(
            path, chip_count=0, metadata=metadata()
        ) as writer:
            writer.write(np.array([[0.3, 0.7]]), chips(1))
    assert not path.exists()


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_consumer_failures_preserve_previous_output(
    tmp_path: Path, failure: type[BaseException]
) -> None:
    """Inference exceptions and interrupts discard staging and retain old output."""
    path = tmp_path / "old.json"
    path.write_text("old")
    with pytest.raises(failure):
        with ClassificationStreamWriter(
            path, chip_count=2, metadata=metadata()
        ) as writer:
            writer.write(np.array([[0.3, 0.7]]), chips(1))
            raise failure("inference failed")
    assert path.read_text() == "old"
    assert not list(tmp_path.glob(".whirld-classify-*"))


def test_unfinished_context_and_writer_reuse_are_rejected(tmp_path: Path) -> None:
    """Explicit finish and a single resource lifetime prevent accidental output loss."""
    writer = ClassificationStreamWriter(
        tmp_path / "out.json", chip_count=0, metadata=metadata()
    )
    with pytest.raises(WhirldError, match="not been finished"):
        _ = writer.feature_count
    with pytest.raises(WhirldError, match="not open"):
        writer.write(np.empty((0, 2)), [])
    with pytest.raises(WhirldError, match="not finished"):
        with writer:
            pass
    with pytest.raises(WhirldError, match="reused"):
        with writer:
            pass


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_atomic_replacement_failures_preserve_previous_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    """Publication failures and interrupts do not remove or truncate the destination."""
    path = tmp_path / "out.json"
    path.write_text("old")

    def fail_replace(source: Path, target: Path) -> Path:
        """Reject only the publication operation, leaving cleanup available."""
        raise failure("simulated replacement failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    expected = WhirldError if failure is OSError else KeyboardInterrupt
    with pytest.raises(expected):
        with ClassificationStreamWriter(
            path, chip_count=0, metadata=metadata()
        ) as writer:
            writer.finish()
    assert path.read_text() == "old"
    assert not list(tmp_path.glob(".whirld-classify-*"))


def test_database_setup_failure_closes_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed staging database open is a domain failure and leaves no temp files."""

    def fail_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        """Simulate disk/database failure before the connection can be acquired."""
        raise sqlite3.OperationalError("disk unavailable")

    monkeypatch.setattr(sqlite3, "connect", fail_connect)
    with pytest.raises(WhirldError, match="open"):
        with ClassificationStreamWriter(
            tmp_path / "out.json", chip_count=0, metadata=metadata()
        ):
            pass
    assert not list(tmp_path.glob(".whirld-classify-*"))


def test_invalid_source_crs_is_a_domain_failure(tmp_path: Path) -> None:
    """Projection errors preserve old output and surface as actionable input errors."""
    path = tmp_path / "out.json"
    path.write_text("old")
    meta = metadata().model_copy(update={"crs": "invalid source CRS"})
    with pytest.raises(InvalidInputError, match="geometry"):
        with ClassificationStreamWriter(path, chip_count=1, metadata=meta) as writer:
            writer.write(np.array([[0.3, 0.7]]), chips(1))
    assert path.read_text() == "old"
    assert not list(tmp_path.glob(".whirld-classify-*"))


def test_database_insert_failure_preserves_old_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed database write cannot publish partial scores or leak its staging."""
    path = tmp_path / "out.json"
    path.write_text("old")
    connect = sqlite3.connect

    def deny_inserts(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        """Permit database setup but simulate an unwritable score staging table."""
        connection = connect(*args, **kwargs)

        def authorize(
            action: int,
            table: str | None,
            column: str | None,
            database: str | None,
            trigger: str | None,
        ) -> int:
            """Deny inserting feature rows; leave SQLite setup operations intact."""
            if action == sqlite3.SQLITE_INSERT and table == "features":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        return connection

    monkeypatch.setattr(sqlite3, "connect", deny_inserts)
    with pytest.raises(WhirldError, match="stage"):
        with ClassificationStreamWriter(
            path, chip_count=1, metadata=metadata()
        ) as writer:
            writer.write(np.array([[0.3, 0.7]]), chips(1))
    assert path.read_text() == "old"
    assert not list(tmp_path.glob(".whirld-classify-*"))
