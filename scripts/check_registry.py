#!/usr/bin/env python3
"""Registry auto-merge gate — the machine-checkable part of the governance policy.

The Whirld registry's CI runs this on community pull requests. For each model YAML
it (1) validates against the schema/Pydantic model and (2) evaluates the auto-merge
policy (OSS-allowlist license + trusted source org + safetensors weights + a
community submission). It prints a per-entry verdict and exits:

* ``0`` — every entry is auto-merge eligible,
* ``1`` — at least one entry needs human review (with reasons),
* ``2`` — at least one entry is invalid (schema/parse error).

Usage::

    python scripts/check_registry.py models/<name>.yaml [more.yaml ...]

Single source of truth for the rules is ``whirld.core.governance``; this script is
a thin CLI over it so the policy can't drift between runtime and CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from whirld.core import governance
from whirld.core.registry import ModelEntry


def check_entry(path: Path) -> int:
    """Validate + evaluate one registry YAML, printing the verdict.

    Args:
        path: Path to a model YAML.

    Returns:
        ``0`` auto-merge, ``1`` needs review, ``2`` invalid.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        entry = ModelEntry.model_validate(raw)
    except Exception as exc:  # parse or schema failure
        print(f"INVALID  {path.name}: {exc}")
        return 2

    decision = governance.evaluate_automerge(entry)
    if decision.auto:
        print(f"AUTO-MERGE  {path.name}")
        return 0
    print(f"REVIEW   {path.name}:")
    for reason in decision.reasons:
        print(f"           - {reason}")
    return 1


def main(argv: list[str]) -> int:
    """Run the gate over the given YAML paths.

    Args:
        argv: Command-line arguments (YAML paths).

    Returns:
        The worst exit code across all entries (2 > 1 > 0).
    """
    if not argv:
        print(__doc__)
        return 2
    worst = 0
    for arg in argv:
        worst = max(worst, check_entry(Path(arg)))
    return worst


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
