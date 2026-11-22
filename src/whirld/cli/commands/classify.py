"""``whirld classify`` — zero-shot text-driven classification (PRD section 5.7)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from ... import api
from ...errors import InvalidInputError
from ._cog import CogArguments
from ._stac import parse_bbox, resolve_stac_token

console = Console()


def classify_command(
    input: str = typer.Argument(
        ..., help="Local GeoTIFF, STAC item, or direct HTTP(S) COG URL."
    ),
    model: str = typer.Option(..., "--model", help="Model name (e.g. remoteclip)."),
    query: list[str] = typer.Option(
        ...,
        "--query",
        help='Text query (repeatable for zero-shot multi-class), e.g. "solar farm".',
    ),
    top_k: int | None = typer.Option(
        None, "--top-k", help="Number of top matches (default: 5)."
    ),
    threshold: float = typer.Option(
        0.0, "--threshold", help="Minimum score threshold (primary query)."
    ),
    output: str | None = typer.Option(
        None, "--output", help="Output GeoJSON path (default: stdout)."
    ),
    stream: bool = typer.Option(
        False,
        "--stream",
        help="Classify STAC/COG windows with bounded memory; requires --output.",
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size", help="Streaming chips per batch (default: 8)."
    ),
    cache_mb: int | None = typer.Option(
        None, "--cache-mb", help="Streaming raster cache in MiB (default: 32)."
    ),
    all_results: bool = typer.Option(
        False,
        "--all",
        help="Stream all threshold-passing matches; incompatible with --top-k.",
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
        help="STAC/COG window 'min_lon,min_lat,max_lon,max_lat' (EPSG:4326).",
    ),
    stac_token: str | None = typer.Option(
        None,
        "--token",
        "--stac-token",
        help="Bearer token scoped to the STAC/COG origin.",
    ),
    cog: bool = typer.Option(
        False, "--cog", help="Treat an extensionless HTTP(S) URL as a direct GeoTIFF."
    ),
    bands: str | None = typer.Option(
        None, "--bands", help="Direct COG band names in source order, comma-separated."
    ),
    scale: str | None = typer.Option(
        None,
        "--scale",
        help="COG reflectance scale: scalar or comma-separated per band.",
    ),
    offset: str | None = typer.Option(
        None,
        "--offset",
        help="COG reflectance offset: scalar or comma-separated per band.",
    ),
) -> None:
    """Score chips against one or more text queries and emit GeoJSON.

    Args:
        input: Local GeoTIFF, STAC item or direct HTTP(S) COG URL.
        model: Model identifier.
        query: One or more free-text queries (the first is primary).
        top_k: Number of top matches to keep (default five).
        stream: Use bounded STAC/COG window processing and staged disk output.
        batch_size: Streaming batch bound; omitted defaults to eight.
        cache_mb: Streaming raster cache budget; omitted defaults to 32 MiB.
        all_results: Stream every threshold-passing chip, sorted on disk.
        threshold: Minimum primary-query score threshold.
        output: GeoJSON output path; prints to stdout when omitted.
        device: Inference device override.
        sensor: Sensor override.
        crs: CRS to assign when the input declares none.
        no_warnings: Suppress the CPU full-precision runtime warning.
        bbox: STAC/COG read window (EPSG:4326); ignored for local files.
        stac_token: Bearer token scoped to the STAC/COG origin.
        cog: Force direct COG routing for extensionless HTTP(S) URLs.
        bands: Source-position COG band names, overriding TIFF descriptions.
        scale: COG reflectance scale, one value or a value for every source band.
        offset: COG reflectance offset, one value or a value for every source band.
    """
    if stream:
        if output is None:
            raise InvalidInputError("--stream requires --output for GeoJSON.")
        if all_results and top_k is not None:
            raise InvalidInputError("--all and --top-k are mutually exclusive.")
        disk = api.classify_to_disk(
            input,
            model=model,
            query=query,
            output=output,
            top_k=None if all_results else (5 if top_k is None else top_k),
            threshold=threshold,
            batch_size=8 if batch_size is None else batch_size,
            cache_mb=32 if cache_mb is None else cache_mb,
            device=device,
            sensor=sensor,
            crs=crs,
            no_warnings=no_warnings,
            bbox=parse_bbox(bbox),
            stac_token=resolve_stac_token(stac_token),
            cog=CogArguments(
                enabled=cog, bands=bands, scale=scale, offset=offset
            ).options(),
        )
        console.print(
            f"[green]✓[/green] Streamed {disk.chip_count} chips — "
            f"{disk.feature_count} matches"
        )
        console.print(f"  GeoJSON: {disk.output_path.name}")
        return
    if all_results or batch_size is not None or cache_mb is not None:
        raise InvalidInputError("--all, --batch-size and --cache-mb require --stream.")
    result = api.classify(
        input,
        model=model,
        query=query,
        top_k=5 if top_k is None else top_k,
        threshold=threshold,
        output=output,
        device=device,
        sensor=sensor,
        crs=crs,
        no_warnings=no_warnings,
        bbox=parse_bbox(bbox),
        stac_token=resolve_stac_token(stac_token),
        cog=CogArguments(
            enabled=cog, bands=bands, scale=scale, offset=offset
        ).options(),
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
