"""``whirld rm`` — remove a cached model (PRD sections 5, 11)."""

from __future__ import annotations

import typer
from rich.console import Console

from ...core.fetch import remove_all, remove_model
from ...errors import WhirldError

console = Console()


def rm_command(
    model: str | None = typer.Argument(None, help="Model name to remove."),
    all_models: bool = typer.Option(
        False, "--all", help="Remove all cached models (keeps the registry)."
    ),
) -> None:
    """Remove a cached model's weights and manifest, or all of them with ``--all``.

    Args:
        model: Model identifier to remove (omit when using ``--all``).
        all_models: Remove every cached model.

    Raises:
        WhirldError: Neither a model nor ``--all`` was given, or both were.
    """
    if all_models:
        if model is not None:
            raise WhirldError("Pass either a model name or --all, not both.")
        removed = remove_all()
        if removed:
            console.print(f"Removed {len(removed)} model(s): {', '.join(removed)}")
        else:
            console.print("No models installed; nothing to remove.")
        return

    if model is None:
        raise WhirldError("Specify a model to remove, or use --all.")

    remove_model(model)
    console.print(f"Removed [bold]{model}[/bold].")
