"""``whirld list`` — show installed models in a table."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from ... import config
from ...core.fetch import load_manifest
from ...core.registry import Registry

console = Console()


def _human_size(num_bytes: int) -> str:
    """Format a byte count as a short human-readable string.

    Args:
        num_bytes: Size in bytes.

    Returns:
        A string like ``847 MB`` or ``412 KB``.
    """
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit != "B" else f"{size:.0f} B"
        size /= 1024
    return f"{size:.0f} TB"


def _dir_size(path: Path) -> int:
    """Return the total size in bytes of all files under a directory."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def list_command() -> None:
    """List installed models with version, size, device, and modified time."""
    paths = config.get_paths()
    registry = Registry(paths)
    installed = (
        sorted(d.name for d in paths.models_dir.glob("*") if d.is_dir())
        if paths.models_dir.exists()
        else []
    )

    table = Table(title="Installed models")
    for col in ("NAME", "VERSION", "SIZE", "HARDWARE", "SOURCE"):
        table.add_column(col)

    if not installed:
        console.print("No models installed. Pull one with: whirld pull clay-v1")
        return

    for name in installed:
        try:
            manifest = load_manifest(name, paths)
            version = manifest.version
            source = manifest.source_type
        except Exception:
            version = "?"
            source = "?"
        try:
            hardware = registry.get(name).hardware.get("recommended_vram_gb", "-")
        except Exception:
            hardware = "-"
        size = _human_size(_dir_size(paths.model_dir(name)))
        table.add_row(name, version, size, str(hardware), source)

    console.print(table)
