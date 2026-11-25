"""A packaged real-model first result, with no source checkout or imagery login."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console

from ...errors import InvalidInputError


def demo_command(
    ctx: typer.Context,
    output: Path = typer.Option(
        Path("quickstart-results"),
        "--output",
        help="New directory for scores and preview.",
    ),
    threads: int = typer.Option(
        2, "--threads", min=1, help="CPU threads (default: 2)."
    ),
) -> None:
    """Run real RemoteCLIP RN50 on the small bundled Sentinel-2 sample.

    Args:
        ctx: Root verbosity flags, applied only after setup preflight succeeds.
        output: Fresh result directory; previous runs are preserved.
        threads: Positive number of CPU inference threads.

    Raises:
        WhirldError: Setup, verified model download or inference fails.
        InvalidInputError: Output settings or local writes are invalid.
        KeyboardInterrupt: Resources and incomplete owned output are cleaned.
    """
    try:
        from ...demo.runner import DemoOptions, DemoRunner
    except (ImportError, OSError) as exc:
        raise InvalidInputError(
            "The demo runtime could not load. Run whirld doctor for setup repairs."
        ) from exc

    console = Console()
    console.print("[bold]Your first Whirld result[/bold]")
    console.print("RemoteCLIP RN50 · CPU · bundled Sentinel-2 imagery")
    console.print("Model weights: 408.4 MB on the first run; verified cache reused.")
    try:
        flags = ctx.find_root().params
        report = DemoRunner(
            DemoOptions(
                output_dir=output,
                threads=threads,
                verbose=bool(flags.get("verbose", False)),
                quiet=bool(flags.get("quiet", False)),
            )
        ).run()
    except (ValidationError, OSError) as exc:
        raise InvalidInputError(f"Demo could not write its results: {exc}") from exc
    console.print(f"Scored {report.feature_count} image chips.")
    console.print("Open this file in your browser:")
    console.print(str(report.preview_path), markup=False, highlight=False)
    console.print(
        "GIS scores: " + str(report.scores_path), markup=False, highlight=False
    )
    console.print(
        "Scores compare the four prompts; they are not land-cover confidence."
    )
