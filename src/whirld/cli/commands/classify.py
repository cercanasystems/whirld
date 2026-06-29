"""``whirld classify`` — zero-shot text-driven classification (PRD section 5.7)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from ... import api
from ._stac import parse_bbox, resolve_stac_token

console = Console()


def classify_command(
    input: str = typer.Argument(
        ..., help="Local GeoTIFF path or a STAC item URL (…/item.json)."
    ),
    model: str = typer.Option(..., "--model", help="Model name (e.g. remoteclip)."),
    query: list[str] = typer.Option(
        ...,
        "--query",
        help='Text query (repeatable for zero-shot multi-class), e.g. "solar farm".',
    ),
    top_k: int = typer.Option(5, "--top-k", help="Number of top matches to return."),
    threshold: float = typer.Option(
        0.0, "--threshold", help="Minimum score threshold (primary query)."
    ),
    output: str | None = typer.Option(
        None, "--output", help="Output GeoJSON path (default: stdout)."
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
    """Score chips against one or more text queries and emit GeoJSON.

    Args:
        input: Local GeoTIFF path or STAC item URL.
        model: Model identifier.
        query: One or more free-text queries (the first is primary).
        top_k: Number of top matches to keep (by the primary query).
        threshold: Minimum primary-query score threshold.
        output: GeoJSON output path; prints to stdout when omitted.
        device: Inference device override.
        sensor: Sensor override.
        crs: CRS to assign when the input declares none.
        no_warnings: Suppress the CPU full-precision runtime warning.
        bbox: STAC read window (EPSG:4326); ignored for local files.
        stac_token: Bearer token for gated STAC endpoints.
    """
    result = api.classify(
        input,
        model=model,
        query=query,
        top_k=top_k,
        threshold=threshold,
        output=output,
        device=device,
        sensor=sensor,
        crs=crs,
        no_warnings=no_warnings,
        bbox=parse_bbox(bbox),
        stac_token=resolve_stac_token(stac_token),
    )
    if output:
        n = len(result.feature_collection["features"])
        console.print(
            f"[green]✓[/green] Classified — {n} matches for "
            f'"{result.query}" (model {result.model}, device {result.device})'
        )
        console.print(f"  GeoJSON: {Path(output).name}")
    else:
        # Print the FeatureCollection to stdout (machine-readable default).
        print(json.dumps(result.feature_collection, indent=2))
