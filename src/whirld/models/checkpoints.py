"""Checkpoint deserialization with format and trust checks at the read boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import SecurityError, WhirldError


def require_trusted_pickle(path: Path, trusted: bool) -> None:
    """Reject executable checkpoint formats without explicit first-party trust."""
    if not trusted:
        raise SecurityError(
            f"Refusing executable checkpoint '{path.name}': only first-party "
            "curated models may load pickle checkpoints."
        )


def load_checkpoint(path: Path | None, *, trusted: bool = False) -> dict[str, Any]:
    """Read safetensors safely or an explicitly trusted PyTorch checkpoint.

    A safetensors filename selects the safetensors parser exclusively; malformed
    files never fall back to pickle. A flat tensor dictionary is returned for
    safetensors; trusted Lightning checkpoints may include their state_dict.

    Raises:
        SecurityError: Untrusted pickle or invalid safetensors content.
        WhirldError: No artifact path or a required optional dependency is missing.
    """
    if path is None:
        raise WhirldError("A checkpoint path is required to load model weights.")
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise WhirldError("Install safetensors to load this checkpoint.") from exc
        try:
            return dict(load_file(str(path), device="cpu"))
        except Exception as exc:
            raise SecurityError(
                f"Invalid safetensors checkpoint '{path.name}'."
            ) from exc
    require_trusted_pickle(path, trusted)
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(checkpoint, dict):
        raise WhirldError("The checkpoint must contain a tensor state dictionary.")
    return checkpoint
