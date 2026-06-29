"""Registry governance policy — security + license posture in one place.

The registry is load-bearing: a model's YAML selects its backend, weights,
normalization, and license. "Schema-valid" is therefore not a sufficient gate for
what gets pulled and executed. This module is the single source of truth for the
governance rules, reused by:

* the runtime pull gate (:func:`whirld.core.fetch.pull`),
* the CLI license display (`whirld pull` / `whirld info`), and
* the registry CI auto-merge checker (``scripts/check_registry.py``).

Two concerns:

**Security (weights provenance / RCE).** Pickle deserialization —
``torch.load(weights_only=False)`` for ``.ckpt``/``.pt`` and open_clip flat
state_dicts — is arbitrary code execution on load. Community (non-first-party)
models must therefore ship ``safetensors``; pickle is allowed only for first-party,
maintainer-curated entries.

**License.** Only a curated allowlist of OSS licenses may auto-merge; use-restricted
or non-commercial licenses (e.g. the OlmoEarth Artifact License) are surfaced with a
terms warning and routed to human review.

Pure-stdlib so it imports cheaply and keeps the lazy top-level import promise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .registry import ModelEntry

# --- License posture --------------------------------------------------------

# Recognized open-source licenses (normalized SPDX-ish ids). A model whose license
# is outside this set is treated as non-OSS: it is surfaced with a terms warning and
# is not eligible for auto-merge.
OSS_LICENSES: frozenset[str] = frozenset(
    {
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
        "isc",
        "mpl-2.0",
        "cc0-1.0",
        "cc-by-4.0",
        "unlicense",
        "gpl-3.0",
        "lgpl-3.0",
        "agpl-3.0",
    }
)


def normalize_license(value: str | None) -> str:
    """Normalize a license string for comparison (lowercase, trimmed).

    Args:
        value: The raw license string (or ``None``).

    Returns:
        The normalized id, or ``""`` if unset.
    """
    return (value or "").strip().lower()


def is_oss_license(value: str | None) -> bool:
    """Return whether a license is in the recognized OSS allowlist.

    Conservative: unknown or use-restricted strings are treated as non-OSS so the
    policy fails safe (warn + no auto-merge).

    Args:
        value: The license string.
    """
    return normalize_license(value) in OSS_LICENSES


# --- Weights provenance (RCE surface) --------------------------------------

# Pickle-bearing weight formats execute arbitrary code on load.
PICKLE_EXTS: frozenset[str] = frozenset({".pt", ".pth", ".ckpt", ".bin", ".pkl"})
# Safe, non-executable tensor container.
SAFETENSORS_EXTS: frozenset[str] = frozenset({".safetensors"})


def weights_are_pickle(entry: ModelEntry) -> bool:
    """Return whether a model's primary weights file is a pickle format.

    Inferred from the first declared source file's extension. Unknown extensions are
    treated as pickle (unsafe) so the policy fails safe. Reference (offline) models
    have no real weights and are never pickle.

    Args:
        entry: The registry entry.
    """
    if entry.source.type == "reference":
        return False
    files = entry.source.files
    if not files:
        return True  # no declared weights → cannot prove safe
    ext = PurePosixPath(files[0]).suffix.lower()
    if ext in SAFETENSORS_EXTS:
        return False
    return True  # pickle exts and anything unrecognized → unsafe


# --- Source provenance ------------------------------------------------------

# Hugging Face orgs whose models have been vetted. A submission pointing at a new
# org is routed to human review rather than auto-merged.
TRUSTED_SOURCE_ORGS: frozenset[str] = frozenset(
    {
        "made-with-clay",
        "chendelong",
        "ibm-nasa-geospatial",
    }
)


def source_org(entry: ModelEntry) -> str | None:
    """Return the source org (the part before ``/`` in ``source.repo``), if any."""
    repo = entry.source.repo
    if not repo or "/" not in repo:
        return None
    return repo.split("/", 1)[0]


def is_trusted_source(entry: ModelEntry) -> bool:
    """Return whether the entry's source org is on the trusted allowlist."""
    return source_org(entry) in TRUSTED_SOURCE_ORGS


# --- Trust + the auto-merge decision ---------------------------------------

FIRST_PARTY = "first-party"
COMMUNITY = "community"


@dataclass(frozen=True)
class AutomergeDecision:
    """Outcome of evaluating a registry entry against the auto-merge policy.

    Attributes:
        auto: True if the entry may be auto-merged; False routes to human review.
        reasons: Why it cannot auto-merge (empty when ``auto`` is True).
    """

    auto: bool
    reasons: list[str] = field(default_factory=list)


def evaluate_automerge(entry: ModelEntry) -> AutomergeDecision:
    """Evaluate a registry entry against the auto-merge governance policy.

    Auto-merge requires **all** of: a community submission (first-party / curated
    entries are added by maintainers, not auto-merged), an OSS-allowlist license, a
    trusted source org, and safetensors weights (pickle only ships via reviewed
    first-party entries).

    Args:
        entry: The validated registry entry.

    Returns:
        An :class:`AutomergeDecision` with the verdict and any blocking reasons.
    """
    reasons: list[str] = []
    trust = entry.trust or COMMUNITY
    if trust != COMMUNITY:
        # First-party/curated entries are added by maintainer review, never
        # auto-merge — so the safetensors rule doesn't apply (pickle is allowed).
        reasons.append(
            f"trust='{entry.trust}' — first-party/curated entries require "
            f"maintainer review (not auto-merge)"
        )
    elif weights_are_pickle(entry):
        reasons.append(
            "weights are pickle — community models must ship safetensors "
            "(pickle is an arbitrary-code-execution risk on load)"
        )
    if not is_oss_license(entry.license):
        reasons.append(
            f"license '{entry.license}' is not in the OSS allowlist "
            f"(use-restricted/non-commercial licenses require review)"
        )
    if not is_trusted_source(entry):
        reasons.append(
            f"source org '{source_org(entry)}' is not yet trusted "
            f"(new source domains require human review)"
        )
    return AutomergeDecision(auto=not reasons, reasons=reasons)
