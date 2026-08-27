"""Command-line surface. Mirrors the HTTP API (section 11.4).

Commands are added as the pipeline stages land; this file stays a thin shell
over the library so that anything reachable from the CLI is also reachable
programmatically.
"""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

import papersynth
from papersynth.schemas import SCHEMA_DIR, load_schema, validator_for

app = typer.Typer(
    name="papersynth",
    help="Synthesize a verified implementation spec from a set of papers.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"papersynth {papersynth.__version__} (spec {papersynth.SPEC_VERSION})")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """PaperSynth: N papers in, one verified implementation spec out."""


@app.command("validate-schemas")
def validate_schemas() -> None:
    """Check that every bundled schema is well-formed and its $refs resolve.

    Run in CI. A schema whose $ref silently fails to resolve would validate
    literally anything, which would quietly disable the provenance gate.
    """
    names = sorted(p.name for p in SCHEMA_DIR.glob("*.json"))
    if not names:
        console.print("[red]No schemas found[/red]")
        raise typer.Exit(1)

    table = Table(title="Bundled schemas", show_lines=False)
    table.add_column("schema", style="cyan")
    table.add_column("$id")
    table.add_column("status")

    failed = 0
    for name in names:
        try:
            schema = load_schema(name)
            validator = validator_for(name)
            validator.check_schema(schema)
            # Force $ref resolution by validating a deliberately empty instance.
            list(validator.iter_errors({}))
            table.add_row(name, schema.get("$id", "-"), "[green]ok[/green]")
        except Exception as exc:
            failed += 1
            table.add_row(name, "-", f"[red]{type(exc).__name__}: {exc}[/red]")

    console.print(table)
    if failed:
        console.print(f"[red]{failed} schema(s) failed[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{len(names)} schemas valid[/green]")


@app.command("schema")
def show_schema(
    name: Annotated[str, typer.Argument(help="Schema filename, e.g. spec.schema.json")],
) -> None:
    """Print a bundled schema."""
    try:
        console.print_json(json.dumps(load_schema(name)))
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
