"""Domain failures for pinned evaluation inputs and invalid scoring contracts."""

from ..errors import WhirldError


class EvaluationError(WhirldError):
    """An evaluation input, artifact, grid or reproducibility check failed."""

    exit_code = 7
