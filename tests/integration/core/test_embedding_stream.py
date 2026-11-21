"""Streaming integration tests for the corresponding source interface."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import whirld
from whirld.config import get_paths
from whirld.core import embedding_stream
from whirld.errors import WhirldError
from whirld.models.base import InferenceContext
from whirld.models.clay import ClayBackend


@pytest.mark.parametrize("override", [None, "2024-06-01T13:00:00+02:00"])
def test_streaming_reuses_backend_and_preserves_context(
    whirld_home: Path,
    stac_item: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: str | None,
) -> None:
    """One model load sees batch-local locations and shared UTC acquisition time."""
    whirld.pull("clay-v1")
    contexts: list[InferenceContext] = []
    original = ClayBackend.embed
    loads = []
    load = embedding_stream.load_backend

    def recording_load(*args: object, **kwargs: object) -> ClayBackend:
        """Track actual backend construction without substituting its inference."""
        backend = load(*args, **kwargs)
        loads.append(backend)
        return backend

    def recording_embed(
        self: ClayBackend, chips: np.ndarray, context: InferenceContext
    ) -> np.ndarray:
        """Record the context paired with each real inference batch."""
        assert len(context.latlons) == chips.shape[0]
        contexts.append(context)
        return original(self, chips, context)

    monkeypatch.setattr(embedding_stream, "load_backend", recording_load)
    monkeypatch.setattr(ClayBackend, "embed", recording_embed)
    result = whirld.embed_to_disk(
        stac_item,
        model="clay-v1",
        output=tmp_path / "context.npy",
        chip_size=128,
        batch_size=2,
        device="cpu",
        datetime=override,
    )
    assert len(loads) == 1
    assert len(contexts) > 1
    assert all(c.options.batch_size == 2 for c in contexts)
    assert all(
        c.acquisition_datetime
        == datetime(2024, 6, 1, 11 if override else 10, tzinfo=timezone.utc)
        for c in contexts
    )
    assert sum(len(c.latlons) for c in contexts) == result.chip_count
    assert contexts[0].latlons != contexts[1].latlons


@pytest.mark.parametrize("error_type", [WhirldError, KeyboardInterrupt])
def test_failed_stream_preserves_previous_outputs(
    whirld_home: Path,
    stac_item: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    """An inference failure after one batch cannot publish a partial result pair."""
    whirld.pull("clay-v1")
    output, sidecar = tmp_path / "result.npy", tmp_path / "result_meta.json"
    output.write_bytes(b"previous data")
    sidecar.write_text("previous metadata")
    calls = 0
    original = ClayBackend.embed

    def failing_embed(
        self: ClayBackend, chips: np.ndarray, context: InferenceContext
    ) -> np.ndarray:
        """Fail on a later batch after output staging has started."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error_type("inference stopped")
        return original(self, chips, context)

    monkeypatch.setattr(ClayBackend, "embed", failing_embed)
    with pytest.raises(error_type, match="inference stopped"):
        whirld.embed_to_disk(
            stac_item,
            model="clay-v1",
            output=output,
            chip_size=128,
            batch_size=1,
            device="cpu",
        )
    assert calls == 2
    assert output.read_bytes() == b"previous data"
    assert sidecar.read_text() == "previous metadata"
    assert not list(tmp_path.glob(".whirld-embed-*"))
    usage = json.loads(get_paths().usage_log.read_text().splitlines()[-1])
    assert usage["error"] == error_type.__name__
