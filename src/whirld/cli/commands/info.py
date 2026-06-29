"""``whirld info`` — show a model's metadata and band contract."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from ...core import governance
from ...core.registry import Registry

console = Console()


def info_command(
    model: str = typer.Argument(..., help="Model name, e.g. clay-v1."),
    as_json: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of a table."
    ),
) -> None:
    """Display a model's band contract, hardware, sensors, license, and citation.

    Args:
        model: Model identifier.
        as_json: If true, print the full registry entry as JSON.
    """
    entry = Registry().get(model)

    if as_json:
        console.print_json(json.dumps(entry.model_dump(mode="json")))
        return

    console.print(f"[bold]{entry.display_name}[/bold] ({entry.name})  v{entry.version}")
    if entry.description:
        console.print(entry.description.strip())
    console.print(f"Category:  {entry.category}")
    if governance.is_oss_license(entry.license):
        console.print(f"License:   {entry.license} (OSS)")
    else:
        console.print(
            f"License:   {entry.license} "
            f"[yellow](non-OSS — review terms before use)[/yellow]"
        )
    console.print(f"Trust:     {entry.trust}")
    console.print(f"Sensors:   {', '.join(entry.supported_sensors())}")

    contract = entry.band_contract
    console.print(
        f"Output:    {entry.output.type} "
        f"(dim {entry.output.embed_dim}, {entry.output.format})"
    )
    console.print(
        f"Chips:     {contract.chip_size_px}px @ {contract.target_resolution_m} m"
    )

    table = Table(title="Band contract")
    for col in ("SENSOR", "BANDS", "ALIASES", "NATIVE RES (m)"):
        table.add_column(col)
    for sensor_key, sensor in contract.sensors.items():
        table.add_row(
            sensor_key,
            ", ".join(sensor.bands),
            ", ".join(sensor.aliases),
            str(sensor.native_resolution_m),
        )
    console.print(table)

    if entry.citation:
        console.print(f"\nCitation:  {entry.citation.strip()}")
