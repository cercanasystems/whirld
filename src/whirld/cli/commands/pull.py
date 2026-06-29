"""``whirld pull`` — download/materialize, verify, and cache a model."""

from __future__ import annotations

import typer
from rich.console import Console

from ... import api
from ...core import governance
from ...core.registry import Registry

console = Console()


def pull_command(
    model: str = typer.Argument(..., help="Model name, e.g. clay-v1."),
    force: bool = typer.Option(
        False, "--force", help="Re-download even if already cached."
    ),
    quantize: str | None = typer.Option(
        None, "--quantize", help="Quantized variant, e.g. int8 (deferred)."
    ),
) -> None:
    """Download and cache a model, verifying its sha256 checksum.

    Args:
        model: Model identifier to pull.
        force: Re-acquire even if cached.
        quantize: Quantized variant (deferred in this build).
    """
    entry = Registry().get(model)
    console.print(f"Pulling [bold]{model}[/bold]...")
    console.print(f"  Source:    {entry.source.type}", highlight=False)
    console.print(
        f"  Size:      {entry.distribution.size_bytes} bytes", highlight=False
    )
    oss = governance.is_oss_license(entry.license)
    console.print(
        f"  License:   {entry.license} {'(OSS)' if oss else '(non-OSS)'}",
        highlight=False,
    )
    # Surface the license terms *before* downloading (PRD §8 governance).
    if not oss:
        console.print(
            f"  [yellow]⚠ '{entry.license}' is not a recognized open-source "
            f"license — it may carry use restrictions.[/yellow]\n"
            f"  [yellow]  Review the terms before use: "
            f"{entry.license_url or entry.source_url or '(see the model card)'}"
            f"[/yellow]"
        )

    manifest = api.pull(model, force=force, quantize=quantize)

    console.print("  Verifying sha256... [green]✓[/green]")
    console.print(
        f"  Saved to ~/.whirld/models/{manifest.name}/ " f"(version {manifest.version})"
    )
