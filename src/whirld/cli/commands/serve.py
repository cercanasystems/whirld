"""``whirld serve`` — start the local REST API server (PRD section 5.8)."""

from __future__ import annotations

import typer
from rich.console import Console

from ...errors import WhirldError

console = Console()


def serve_command(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8765, "--port", help="Port to listen on."),
    models: str | None = typer.Option(
        None,
        "--models",
        help="Comma-separated models to preload at startup (default: none).",
    ),
    device: str | None = typer.Option(
        None, "--device", help="Device: cuda, mps, cpu (default: auto)."
    ),
) -> None:
    """Start a FastAPI server exposing /health, /models, and /embed.

    Models named in ``--models`` are loaded immediately; any other installed model
    loads on first request and stays warm.

    Args:
        host: Bind address.
        port: Port to listen on.
        models: Comma-separated model names to preload.
        device: Inference device override.

    Raises:
        WhirldError: The optional ``serve`` extra is not installed.
    """
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError as exc:
        raise WhirldError(
            "The REST server requires the optional 'serve' extra.\n"
            "       Install it:  pip install 'whirld[serve]'"
        ) from exc

    from ...server.app import create_app  # noqa: PLC0415

    preload = [m.strip() for m in models.split(",") if m.strip()] if models else None
    app = create_app(device=device, preload=preload)

    console.print(
        f"Starting Whirld server on [bold]http://{host}:{port}[/bold]"
        + (f" (preloading: {', '.join(preload)})" if preload else "")
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
