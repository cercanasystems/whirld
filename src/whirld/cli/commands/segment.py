"""``whirld segment`` — per-pixel segmentation / dense prediction (PRD section 5.6)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ... import api
from ._stac import parse_bbox, resolve_stac_token

console = Console()


def segment_command(
    input: str = typer.Argument(
        ..., help="Local GeoTIFF path or a STAC item URL (…/item.json)."
    ),
    model: str = typer.Option(
        ..., "--model", help="Model name (e.g. prithvi-burn-scar)."
    ),
    head: str | None = typer.Option(
        None, "--head", help="Task head (e.g. burn-scar, flood) for prithvi-eo-2."
    ),
    output: str | None = typer.Option(
        None, "--output", help="Output GeoTIFF path (default: <stem>_<model>.tif)."
    ),
    threshold: float = typer.Option(
        0.5, "--threshold", help="Binary mask threshold (0.5 = argmax)."
    ),
    device: str | None = typer.Option(
        None, "--device", help="Device: cuda, mps, cpu (default: auto)."
    ),
    sensor: str | None = typer.Option(
        None, "--sensor", help="Override sensor detection."
    ),
    crs: str | None = typer.Option(
        None, "--crs", help="Assign a CRS when the input has none (e.g. EPSG:32630)."
    ),
    no_warnings: bool = typer.Option(
        False, "--no-warnings", help="Suppress the CPU runtime warning."
    ),
    bbox: str | None = typer.Option(
        None,
        "--bbox",
        help="STAC window 'min_lon,min_lat,max_lon,max_lat' (EPSG:4326).",
    ),
    stac_token: str | None = typer.Option(
        None, "--stac-token", help="Bearer token for gated STAC endpoints."
    ),
) -> None:
    """Run segmentation and write a single-band mask GeoTIFF.

    Args:
        input: Local GeoTIFF path or STAC item URL.
        model: Model identifier.
        head: Task head (required for the ``prithvi-eo-2`` alias).
        output: Output path override.
        threshold: Binary mask threshold (0.5 = argmax).
        device: Inference device override.
        sensor: Sensor override.
        crs: CRS to assign when the input declares none.
        no_warnings: Suppress the CPU full-precision runtime warning.
        bbox: STAC read window (EPSG:4326); ignored for local files.
        stac_token: Bearer token for gated STAC endpoints.
    """
    result = api.segment(
        input,
        model=model,
        head=head,
        output=output,
        threshold=threshold,
        device=device,
        sensor=sensor,
        crs=crs,
        no_warnings=no_warnings,
        bbox=parse_bbox(bbox),
        stac_token=resolve_stac_token(stac_token),
    )
    h, w = result.mask.shape
    positive = int((result.mask > 0).sum())
    console.print(
        f"[green]✓[/green] Segmented {h}x{w} ({result.classes} classes; "
        f"{positive} positive px) — model [bold]{result.model}[/bold], "
        f"device {result.device}"
    )
    if result.output_path:
        console.print(f"  Mask: {Path(result.output_path).name}")
