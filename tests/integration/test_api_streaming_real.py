"""Opt-in public API equivalence using installed real Clay weights and live COGs.

Set WHIRLD_TEST_CLAY_STREAMING=1 and WHIRLD_HOME to a model home populated by
``whirld pull clay-v1.5``. This test performs no checkpoint download itself.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("WHIRLD_TEST_CLAY_STREAMING") != "1",
    reason="set WHIRLD_TEST_CLAY_STREAMING=1 with installed real Clay weights",
)


def test_real_clay_streaming_public_api(tmp_path: Path) -> None:
    """Verify real embeddings across batch boundaries and record resource costs.

    Args:
        tmp_path: Isolated output root; installed model files remain unchanged.

    Raises:
        ValidationFailure: Checkpoint, embeddings, metadata or workers fail checks.
    """
    from scripts.validate_clay_streaming import (
        ClayStreamingValidation,
        ValidationOptions,
    )

    report = ClayStreamingValidation(
        ValidationOptions(output_dir=tmp_path / "validation")
    ).run()
    assert len(report.comparisons) == 3
    assert {comparison.batch_size for comparison in report.comparisons} == {1, 2, 3}
