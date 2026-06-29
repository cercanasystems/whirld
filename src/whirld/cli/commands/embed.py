"""``whirld embed`` — generate embeddings for a raster input."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ... import api
from ._stac import parse_bbox, resolve_stac_token

console = Console()


def embed_command(
    input: str = typer.Argument(
        ..., help="Local GeoTIFF path or a STAC item URL (…/item.json)."
    ),
    model: str = typer.Option(..., "--model", help="Model name, e.g. clay-v1."),
    output: str | None = typer.Option(
        None, "--output", help="Output path (default: <stem>_embeddings.npy)."
    ),
    fmt: str = typer.Option("npy", "--format", help="Output format: npy or json."),
    chip_size: int | None = typer.Option(
        None, "--chip-size", help="Override chip size in pixels."
    ),
    overlap: int = typer.Option(0, "--overlap", help="Chip overlap in pixels."),
    device: str | None = typer.Option(
        None, "--device", help="Device: cuda, mps, cpu (default: auto)."
    ),
    sensor: str | None = typer.Option(
        None, "--sensor", help="Override sensor detection."
    ),
    crs: str | None = typer.Option(
        None, "--crs", help="Assign a CRS when the input has none (e.g. EPSG:32630)."
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size", help="Inference batch size (default: model's)."
    ),
    datetime: str | None = typer.Option(
        None,
        "--datetime",
        help="Acquisition datetime (ISO-8601) for metadata-aware models (Clay).",
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
    """Embed a GeoTIFF or STAC item and write a ``.npy`` array plus a sidecar.

    Args:
        input: Local GeoTIFF path or STAC item URL.
        model: Model identifier.
        output: Output path override.
        fmt: Output format (``npy`` or ``json``).
        chip_size: Chip size override in pixels.
        overlap: Chip overlap in pixels.
        device: Inference device override.
        sensor: Sensor override.
        crs: CRS to assign when the input declares none.
        batch_size: Inference batch size override.
        datetime: Acquisition datetime (ISO-8601) for metadata-aware models.
        no_warnings: Suppress the CPU full-precision runtime warning.
        bbox: STAC read window (EPSG:4326); ignored for local files.
        stac_token: Bearer token for gated STAC endpoints.
    """
    result = api.embed(
        input,
        model=model,
        output=output,
        fmt=fmt,
        chip_size=chip_size,
        overlap=overlap,
        device=device,
        sensor=sensor,
        crs=crs,
        batch_size=batch_size,
        datetime=datetime,
        no_warnings=no_warnings,
        bbox=parse_bbox(bbox),
        stac_token=resolve_stac_token(stac_token),
    )
    n_chips, embed_dim = result.embeddings.shape
    console.print(
        f"[green]✓[/green] Embedded {n_chips} chips "
        f"({embed_dim}-dim) — sensor [bold]{result.sensor}[/bold], "
        f"device {result.device}"
    )
    if result.output_path:
        console.print(f"  Embeddings: {Path(result.output_path).name}")
    if result.meta_path and result.meta_path != result.output_path:
        console.print(f"  Metadata:   {Path(result.meta_path).name}")
