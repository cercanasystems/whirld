"""Installed setup diagnostics with human-readable and strict JSON output."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape


def doctor_command(
    model: str = typer.Option(
        "remoteclip-rn50", "--model", help="Model runtime to check."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit a structured setup report."
    ),
) -> None:
    """Check the active environment and cache before running a real model.

    Args:
        model: Model name; the default matches the packaged starter demo.
        json_output: Write machine-readable JSON instead of the human summary.

    Raises:
        WhirldError: Model identity is invalid or unavailable.
        typer.Exit: Exit seven if setup blockers require attention. Missing
            weights alone are a warning and keep the exit status successful.
    """
    from ...onboarding.doctor import DiagnosticsService

    report = DiagnosticsService().inspect(model=model)
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        console = Console()
        console.print(f"[bold]Whirld setup: {escape(report.model)}[/bold]")
        console.print(f"Python {escape(report.python_version)} · CPU")
        console.print(f"Interpreter: {escape(report.executable)}")
        console.print(f"Model cache: {escape(str(report.cache_path))}")
        for check in report.checks:
            color = {"pass": "green", "warn": "yellow", "fail": "red"}[check.status]
            console.print(
                f"[{color}]{check.status.upper()}[/{color}] {escape(check.message)}"
            )
            if check.remedy:
                console.print("  " + escape(check.remedy))
        if report.ready_for_demo:
            console.print("[green]Ready for CPU inference.[/green]")
            if not report.model_installed:
                amount = report.model_download_bytes / 1_000_000
                if report.model == "remoteclip-rn50":
                    console.print(
                        f"The first demo will download {amount:.1f} MB of weights."
                    )
                else:
                    console.print(
                        f"Pull this model before inference ({amount:.1f} MB)."
                    )
    if not report.ready_for_demo:
        raise typer.Exit(code=7)
