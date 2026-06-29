"""Security-posture invariants: the runtime pull gate + bundled-registry compliance."""

from __future__ import annotations

from pathlib import Path

import pytest

from whirld.core import fetch, governance
from whirld.core.registry import Registry, Source
from whirld.errors import SecurityError


def test_pull_refuses_community_pickle(whirld_home: Path) -> None:
    """A community entry shipping pickle weights is refused at pull (SecurityError)."""
    registry = Registry()
    evil = registry.get("clay-v1.5").model_copy(
        update={
            "name": "evil-community",
            "trust": "community",
            "source": Source(
                type="huggingface", repo="someone/Evil", files=["payload.pt"]
            ),
        }
    )

    class _FakeRegistry:
        def get(self, name):  # type: ignore[no-untyped-def]
            return evil

    with pytest.raises(SecurityError) as excinfo:
        fetch.pull("evil-community", registry=_FakeRegistry())
    assert excinfo.value.exit_code == 9
    assert "pickle" in excinfo.value.message


def test_pull_allows_first_party_pickle(whirld_home: Path) -> None:
    """First-party pickle entries pass the gate (the helper does not raise)."""
    entry = Registry().get("clay-v1.5")  # first-party, .ckpt pickle
    # Directly exercise the gate (full pull would hit the network).
    fetch._enforce_weights_security(entry)  # must not raise


def test_pull_allows_reference(whirld_home: Path) -> None:
    """Reference (offline) models have no executable weights; the gate passes."""
    fetch._enforce_weights_security(Registry().get("clay-v1"))


def test_bundled_entries_are_policy_compliant(whirld_home: Path) -> None:
    """Every bundled entry declares trust + license, first-party-or-safetensors."""
    registry = Registry()
    for name in registry.available():
        entry = registry.get(name)
        assert entry.trust in (governance.FIRST_PARTY, governance.COMMUNITY)
        assert entry.license, f"{name} has no license"
        # Our shipped models may be pickle only because they are first-party.
        if governance.weights_are_pickle(entry):
            assert (
                entry.trust == governance.FIRST_PARTY
            ), f"{name} ships pickle weights but is not first-party"
