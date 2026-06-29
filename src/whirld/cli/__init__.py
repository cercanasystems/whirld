"""Whirld command-line interface (Typer).

Exposes the walking-skeleton command surface: ``pull``, ``list``, ``info``, and
``embed``. Deferred commands (``segment``, ``classify``, ``serve``, ``update``,
``rm``) are intentionally absent and tracked in ``tasks/todo.md``.

Domain errors (:class:`~whirld.errors.WhirldError`) are caught centrally and
converted to the documented process exit codes (PRD section 13), printing the
error's actionable message to stderr.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import typer
from rich.console import Console

from ..errors import WhirldError
from ..logging_setup import configure_logging
from .commands.classify import classify_command
from .commands.embed import embed_command
from .commands.info import info_command
from .commands.list import list_command
from .commands.pull import pull_command
from .commands.rm import rm_command
from .commands.segment import segment_command
from .commands.serve import serve_command

_err_console = Console(stderr=True)

_F = TypeVar("_F", bound=Callable[..., Any])

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Whirld — local-first geospatial foundation models.",
)


@app.callback()
def _main(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="DEBUG logging to stderr."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="WARNING and above only."),
) -> None:
    """Configure logging before any command runs.

    Args:
        verbose: Enable DEBUG stderr logging.
        quiet: Restrict stderr logging to WARNING and above.
    """
    configure_logging(verbose=verbose, quiet=quiet, force=True)


def _handle_errors(func: _F) -> _F:
    """Wrap a command so domain errors map to documented exit codes.

    Args:
        func: The command function to wrap.

    Returns:
        The wrapped function (signature preserved for Typer via ``functools.wraps``).
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except WhirldError as exc:
            _err_console.print(f"[red]Error:[/red] {exc.message}")
            raise typer.Exit(code=exc.exit_code) from None

    return wrapper  # type: ignore[return-value]


app.command("pull")(_handle_errors(pull_command))
app.command("list")(_handle_errors(list_command))
app.command("info")(_handle_errors(info_command))
app.command("rm")(_handle_errors(rm_command))
app.command("embed")(_handle_errors(embed_command))
app.command("classify")(_handle_errors(classify_command))
app.command("segment")(_handle_errors(segment_command))
app.command("serve")(_handle_errors(serve_command))


if __name__ == "__main__":  # pragma: no cover
    app()
