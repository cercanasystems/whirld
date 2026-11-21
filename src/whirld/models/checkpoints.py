"""Checkpoint deserialization with format and trust checks at the read boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import SecurityError, WhirldError


class CheckpointReader:
    """Own artifact format dispatch and the trust boundary for deserialization.

    Binding the artifact and its trust once keeps those decisions together for
    direct state loading and framework-managed loading such as TerraTorch.
    Malformed safetensors never fall back to an executable pickle parser.
    """

    def __init__(self, path: Path, *, trusted: bool = False) -> None:
        """Bind a local artifact to an explicit, default-deny trust decision."""
        self._path = path
        self._trusted = trusted

    @property
    def path(self) -> Path:
        """The bound local artifact, for third-party framework loading."""
        return self._path

    def require_trusted_execution(self) -> None:
        """Authorize framework/pickle loading only for explicitly trusted artifacts."""
        if not self._trusted:
            raise SecurityError(
                f"Refusing executable checkpoint '{self.path.name}': only first-party "
                "curated models may load pickle checkpoints."
            )

    def read_state_dict(self) -> dict[str, Any]:
        """Read safetensors safely or an explicitly trusted PyTorch checkpoint.

        Returns a flat state dictionary for safetensors; trusted Lightning files
        may contain a nested state_dict. Invalid content raises SecurityError or
        WhirldError; missing optional dependencies produce an actionable error.
        """
        if self.path.suffix.lower() == ".safetensors":
            return self._read_safetensors()
        self.require_trusted_execution()
        import torch

        checkpoint = torch.load(
            self.path, map_location="cpu", weights_only=False, mmap=True
        )
        if not isinstance(checkpoint, dict):
            raise WhirldError("The checkpoint must contain a tensor state dictionary.")
        return checkpoint

    def _read_safetensors(self) -> dict[str, Any]:
        """Read nonexecutable tensor data exclusively through the safetensors parser."""
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise WhirldError("Install safetensors to load this checkpoint.") from exc
        try:
            return dict(load_file(str(self.path), device="cpu"))
        except Exception as exc:
            raise SecurityError(
                f"Invalid safetensors checkpoint '{self.path.name}'."
            ) from exc


def require_trusted_pickle(path: Path, trusted: bool) -> None:
    """Compatibility facade for the checkpoint reader's execution policy."""
    CheckpointReader(path, trusted=trusted).require_trusted_execution()


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
    return CheckpointReader(path, trusted=trusted).read_state_dict()
