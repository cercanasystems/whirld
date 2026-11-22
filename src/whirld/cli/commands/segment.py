"""``whirld segment`` — per-pixel segmentation / dense prediction (PRD section 5.6)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ... import api
from ...errors import InvalidInputError
from ._cog import CogArguments
from ._stac import parse_bbox, resolve_stac_token

console = Console()


def segment_command(
    input: str = typer.Argument(
        ..., help="Local GeoTIFF, STAC item, or direct HTTP(S) COG URL."
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
    stream: bool = typer.Option(
        False,
        "--stream",
        help="Segment STAC/COG windows with bounded memory; requires --output.",
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size", help="Streaming chips per batch (default: 4)."
    ),
    cache_mb: int | None = typer.Option(
        None, "--cache-mb", help="Streaming raster cache in MiB (default: 32)."
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
    """Run segmentation and write a single-band mask GeoTIFF.

    Args:
        input: Local GeoTIFF, STAC item or direct HTTP(S) COG URL.
        model: Model identifier.
        head: Task head (required for the ``prithvi-eo-2`` alias).
        output: Output path override.
        threshold: Binary mask threshold (0.5 = argmax).
        stream: Use native window batches and staged GeoTIFF output.
        batch_size: Streaming batch bound; omitted defaults to four.
        cache_mb: Streaming raster cache budget; omitted defaults to 32 MiB.
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
            raise InvalidInputError("--stream requires --output for GeoTIFF.")
        disk = api.segment_to_disk(
            input,
            model=model,
            head=head,
            output=output,
            threshold=threshold,
            batch_size=4 if batch_size is None else batch_size,
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
            f"{disk.height}x{disk.width} mask ({disk.positive_pixel_count} positive px)"
        )
        console.print(f"  Mask: {disk.output_path.name}")
        return
    if batch_size is not None or cache_mb is not None:
        raise InvalidInputError("--batch-size and --cache-mb require --stream.")
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
        cog=CogArguments(
            enabled=cog, bands=bands, scale=scale, offset=offset
        ).options(),
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
