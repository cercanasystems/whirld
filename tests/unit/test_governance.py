"""Unit tests for the registry governance policy."""

from __future__ import annotations

from pathlib import Path

from whirld.core import governance as g
from whirld.core.registry import Registry, Source


def _entry(whirld_home: Path):
    """A real bundled entry to copy + override for cases."""
    return Registry().get(
        "clay-v1.5"
    )  # first-party, pickle .ckpt, Apache, made-with-clay


def test_is_oss_license() -> None:
    """OSS licenses are recognized (case-insensitive); restricted ones are not."""
    assert g.is_oss_license("Apache-2.0")
    assert g.is_oss_license("mit")
    assert g.is_oss_license("BSD-3-Clause")
    assert not g.is_oss_license("OlmoEarth Artifact License")
    assert not g.is_oss_license("CC-BY-NC-4.0")
    assert not g.is_oss_license("")
    assert not g.is_oss_license(None)


def test_weights_are_pickle(whirld_home: Path) -> None:
    """Pickle extensions are flagged; safetensors is safe; reference is safe."""
    base = _entry(whirld_home)
    assert g.weights_are_pickle(base)  # clay-v1.5 ships a .ckpt

    safe = base.model_copy(
        update={
            "source": Source(type="huggingface", repo="x/y", files=["m.safetensors"])
        }
    )
    assert not g.weights_are_pickle(safe)

    pt = base.model_copy(
        update={"source": Source(type="huggingface", repo="x/y", files=["m.pt"])}
    )
    assert g.weights_are_pickle(pt)

    reference = Registry().get("clay-v1")  # reference source → no real weights
    assert not g.weights_are_pickle(reference)


def test_source_org(whirld_home: Path) -> None:
    """The org is the part before '/' in source.repo."""
    assert g.source_org(_entry(whirld_home)) == "made-with-clay"


def test_automerge_first_party_requires_review(whirld_home: Path) -> None:
    """First-party entries never auto-merge (maintainer-curated)."""
    decision = g.evaluate_automerge(_entry(whirld_home))
    assert decision.auto is False
    assert any("maintainer review" in r for r in decision.reasons)
    # First-party may use pickle, so no safetensors complaint is raised.
    assert not any("safetensors" in r for r in decision.reasons)


def test_automerge_olmoearth_worked_example(whirld_home: Path) -> None:
    """OlmoEarth: community + non-OSS + untrusted + pickle → review, all reasons."""
    olmo = _entry(whirld_home).model_copy(
        update={
            "trust": "community",
            "license": "OlmoEarth Artifact License",
            "source": Source(
                type="huggingface",
                repo="allenai/OlmoEarth-v1-Base",
                files=["weights.pth"],
            ),
        }
    )
    decision = g.evaluate_automerge(olmo)
    assert decision.auto is False
    joined = " ".join(decision.reasons)
    assert "OSS allowlist" in joined
    assert "not yet trusted" in joined
    assert "safetensors" in joined  # community + pickle


def test_automerge_community_safetensors_oss_trusted(whirld_home: Path) -> None:
    """A clean community entry (OSS + trusted org + safetensors) auto-merges."""
    good = _entry(whirld_home).model_copy(
        update={
            "trust": "community",
            "license": "MIT",
            "source": Source(
                type="huggingface",
                repo="made-with-clay/Community-Model",
                files=["model.safetensors"],
            ),
        }
    )
    decision = g.evaluate_automerge(good)
    assert decision.auto is True
    assert decision.reasons == []
