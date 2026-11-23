"""``whirld embed`` — generate embeddings for a raster input."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ... import api
from ._cog import CogArguments
from ._stac import parse_bbox, resolve_stac_token

console = Console()


def embed_command(
    input: str = typer.Argument(
        ..., help="Local GeoTIFF, STAC item, or direct HTTP(S) COG URL."
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
        help="STAC/COG window 'min_lon,min_lat,max_lon,max_lat' (EPSG:4326).",
    ),
    stac_token: str | None = typer.Option(
        None,
        "--token",
        "--stac-token",
        help="Bearer token scoped to the STAC/COG origin.",
    ),
    stream: bool = typer.Option(
        False,
        "--stream",
        help="Stream local/STAC/COG embeddings to NPY (default batch: 8).",
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
    """Embed a GeoTIFF or STAC item and write a ``.npy`` array plus a sidecar.

    Args:
        input: Local GeoTIFF, STAC item or direct HTTP(S) COG URL.
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
        bbox: STAC/COG EPSG:4326 window; rejected by local streaming and
            ignored for eager local reads.
        stac_token: Bearer token scoped to the STAC/COG origin.
        cog: Force direct COG routing for extensionless HTTP(S) URLs.
        bands: Source-position COG band names, overriding TIFF descriptions.
        scale: COG reflectance scale, one value or a value for every source band.
        offset: COG reflectance offset, one value or a value for every source band.
        stream: Bound local/STAC/COG batches and write results incrementally.
    """
    if stream:
        disk_result = api.embed_to_disk(
            input,
            model=model,
            output=output,
            fmt=fmt,
            chip_size=chip_size,
            overlap=overlap,
            device=device,
            sensor=sensor,
            crs=crs,
            batch_size=batch_size if batch_size is not None else 8,
            datetime=datetime,
            no_warnings=no_warnings,
            bbox=parse_bbox(bbox),
            stac_token=resolve_stac_token(stac_token),
            cog=CogArguments(
                enabled=cog, bands=bands, scale=scale, offset=offset
            ).options(),
        )
        console.print(
            f"[green]✓[/green] Streamed {disk_result.chip_count} chips "
            f"({disk_result.embed_dim}-dim) — "
            f"sensor [bold]{disk_result.sensor}[/bold], "
            f"device {disk_result.device}"
        )
        console.print(f"  Embeddings: {disk_result.output_path.name}")
        console.print(f"  Metadata:   {disk_result.meta_path.name}")
        return
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
        cog=CogArguments(
            enabled=cog, bands=bands, scale=scale, offset=offset
        ).options(),
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
