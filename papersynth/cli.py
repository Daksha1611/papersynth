"""Command-line surface, mirroring the HTTP API (section 11.4).

A thin shell over the library: anything reachable here is reachable
programmatically. Exit codes are meaningful because scripts/run.sh branches on
them - notably `conflicts --quiet`, which exits non-zero when blocking
conflicts remain so a pipeline can stop before emitting.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Annotated, Any, cast, get_args

import httpx
import typer
import yaml
from rich.console import Console
from rich.table import Table

import papersynth
from papersynth.core.config import ProviderId, get_settings
from papersynth.core.errors import AllProvidersExhausted, PaperSynthError
from papersynth.core.ledger import Ledger
from papersynth.core.models import Resolution, utcnow
from papersynth.core.run import Pipeline, Workspace
from papersynth.llm.stub import StubProvider
from papersynth.schemas import SCHEMA_DIR, load_schema, validator_for
from papersynth.store import RunStore
from papersynth.synth import SpecBuilder, SpecValidator, render_review

app = typer.Typer(
    name="papersynth",
    help="Synthesize a verified implementation spec from a set of papers.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err = Console(stderr=True)

_SEVERITY_STYLE = {"BLOCKING": "bold red", "MATERIAL": "yellow", "COSMETIC": "dim"}


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


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    arxiv: Annotated[str | None, typer.Option("--arxiv", help="arXiv ID or URL.")] = None,
    pdf: Annotated[Path | None, typer.Option("--pdf", help="Path to a PDF.")] = None,
    latex: Annotated[
        Path | None, typer.Option("--latex", help="Path to .tex, a source dir, or a tarball.")
    ] = None,
    prefer_latex: Annotated[
        bool, typer.Option("--prefer-latex/--prefer-pdf", help="Prefer e-print source.")
    ] = True,
    no_grobid: Annotated[
        bool,
        typer.Option("--no-grobid", help="Allow pdftotext only. Degrades math fidelity."),
    ] = False,
    out: Annotated[Path | None, typer.Option("--out", help="Write the document JSON here.")] = None,
) -> None:
    """Ingest one paper into a canonical StructuredDocument."""
    from papersynth.ingest import ingest as ingest_ref

    ref = arxiv or (str(pdf) if pdf else None) or (str(latex) if latex else None)
    if not ref:
        err.print("[red]Pass one of --arxiv, --pdf, or --latex.[/red]")
        raise typer.Exit(2)

    try:
        doc = ingest_ref(ref, prefer_latex=prefer_latex, allow_no_grobid=no_grobid)
    except PaperSynthError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    fidelity_style = "green" if doc.math_fidelity == "latex_native" else "yellow"
    console.print(
        f"[green]OK[/green] {doc.paper_id}  {doc.title}\n"
        f"   {doc.ingest_method} | [{fidelity_style}]{doc.math_fidelity}[/{fidelity_style}] | "
        f"{len(doc.sections)} sections | {len(doc.equations)} eq | "
        f"{len(doc.algorithms_raw)} alg"
    )
    for warning in doc.warnings:
        console.print(f"   [yellow]warning:[/yellow] {warning}")

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc.model_dump(), indent=2, default=str), encoding="utf-8")
        console.print(f"   wrote {out}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    papers: Annotated[
        str, typer.Option("--papers", help="Comma-separated arXiv IDs or file paths.")
    ],
    objective: Annotated[str, typer.Option("--objective", help="What is being implemented.")],
    out: Annotated[Path, typer.Option("--out", help="Run workspace directory.")],
    extractors: Annotated[
        str, typer.Option("--extractors", help="Comma-separated extractor names.")
    ] = "hyperparameter",
    provider_chain: Annotated[
        str | None, typer.Option("--provider-chain", help="Override the provider chain.")
    ] = None,
    no_entailment: Annotated[
        bool,
        typer.Option("--no-entailment", help="Skip the LLM entailment check. Cheaper."),
    ] = False,
    split_gate: Annotated[
        bool,
        typer.Option(
            "--split-gate",
            help="Review multi-paper clusters with the SplitterAgent. Recommended "
            "with --embedding-merges; on its own it can over-split.",
        ),
    ] = False,
    embedding_merges: Annotated[
        bool,
        typer.Option(
            "--embedding-merges",
            help="Also align differently-named claims by similarity. Needs --split-gate.",
        ),
    ] = False,
    adversarial_gaps: Annotated[
        bool,
        typer.Option(
            "--adversarial-gaps",
            help="Ask an implementer what they would have to guess. One extra call.",
        ),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Reuse per-paper artifacts already in --out. Use after quotas reset.",
        ),
    ] = False,
    no_grobid: Annotated[bool, typer.Option("--no-grobid", help="Allow pdftotext only.")] = False,
) -> None:
    """Run the full pipeline over a set of papers."""
    from papersynth.ingest import ingest as ingest_ref
    from papersynth.llm import build_router

    settings = get_settings()
    if provider_chain:
        settings.provider_chain = _parse_chain(provider_chain)

    refs = [p.strip() for p in papers.split(",") if p.strip()]
    if not refs:
        err.print("[red]--papers is empty.[/red]")
        raise typer.Exit(2)

    documents = []
    for ref in refs:
        try:
            doc = ingest_ref(ref, settings=settings, allow_no_grobid=no_grobid)
        except PaperSynthError as exc:
            # One paper failing must not abort the run (NFR-09).
            err.print(f"[yellow]skipping {ref}: {exc}[/yellow]")
            continue
        documents.append(doc)
        console.print(f"[green]ingested[/green] {doc.paper_id}  {doc.title}")

    if not documents:
        err.print("[red]No papers could be ingested.[/red]")
        raise typer.Exit(1)

    workspace = Workspace(out.parent or Path("."), out.name)
    ledger = Ledger(workspace.ledger_path)

    try:
        router = build_router(settings, ledger=ledger, workspace=workspace.root)
    except PaperSynthError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    pipeline = Pipeline(
        router,
        settings=settings,
        workspace=workspace,
        extractors=[e.strip() for e in extractors.split(",") if e.strip()],
        entailment=not no_entailment,
        split_gate=split_gate or embedding_merges,
        embedding_merges=embedding_merges,
        adversarial_gaps=adversarial_gaps,
        resume=resume,
        ledger=ledger,
    )

    try:
        result = pipeline.run(
            documents, objective=objective, run_id=out.name, papers_requested=len(refs)
        )
    except PaperSynthError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    _print_run_summary(result, workspace.root)
    if result.blocking:
        raise typer.Exit(2)


def _gap_breakdown(gaps: list[Any]) -> str:
    """Total plus a split by criticality.

    A bare count says nothing about whether a reviewer must act now. Nine gaps
    that are all cosmetic and nine that all block are the same number and
    completely different situations.
    """
    if not gaps:
        return "0"
    counts: dict[str, int] = {}
    for gap in gaps:
        counts[gap.criticality] = counts.get(gap.criticality, 0) + 1
    parts = [
        f"{counts[k]} {k.lower()}" for k in ("BLOCKING", "MATERIAL", "COSMETIC") if k in counts
    ]
    return f"{len(gaps)}  ({', '.join(parts)})"


def _print_run_summary(result: Any, root: Path) -> None:
    console.print()
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("claims", str(len(result.claims)))
    table.add_row("contradictions", str(len(result.contradictions)))
    table.add_row("open conflicts", str(len(result.open_conflicts)))
    table.add_row("gaps", _gap_breakdown(result.gaps))
    table.add_row("provenance", f"{result.provenance_completeness:.0%}")
    console.print(table)

    for warning in result.warnings[:5]:
        console.print(f"[yellow]warning:[/yellow] {warning}")

    summary = (result.spec or {}).get("verification_summary", {})
    requested = summary.get("papers_requested", 0) or summary.get("papers_ingested", 0)
    contributing = summary.get("papers_contributing", 0)
    if requested and contributing < requested:
        # "0 contradictions" reads as "the papers agree" when it can equally
        # mean "only one paper was read". The distinction has to be loud.
        console.print(
            f"\n[bold yellow]PARTIAL[/bold yellow] - only {contributing} of {requested} "
            "papers contributed claims. Cross-paper reconciliation did not happen "
            "for the rest, so the conflict count above is not a finding about the "
            "corpus you asked for."
        )

    if result.blocking:
        console.print(
            f"\n[bold red]BLOCKED[/bold red] - {len(result.blocking)} blocking conflict(s). "
            "The spec was not emitted."
        )
        console.print(f"  -> papersynth conflicts {root} --status open --severity BLOCKING")
    else:
        console.print(f"\n[green]spec emitted[/green] {root / 'implementation_spec.yaml'}")
        console.print(f"  review: {root / 'SPEC_REVIEW.md'}")


# ---------------------------------------------------------------------------
# review flow
# ---------------------------------------------------------------------------


@app.command()
def conflicts(
    run_dir: Annotated[Path, typer.Argument(help="Run workspace directory.")],
    status: Annotated[str, typer.Option("--status", help="open | resolved | all")] = "open",
    severity: Annotated[str | None, typer.Option("--severity")] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Print nothing; exit non-zero if any match."),
    ] = False,
) -> None:
    """List contradictions. Exits 2 when any open conflict matches."""
    loaded = _load(run_dir)

    if status == "resolved":
        resolutions = (
            [r for r in loaded.reconciliation.resolutions if not r.is_open]
            if loaded.reconciliation
            else []
        )
        if not quiet:
            _print_resolutions(loaded, resolutions)
        raise typer.Exit(0)

    matching = loaded.open_contradictions(severity)
    if status == "all":
        matching = [c for c in loaded.contradictions if not severity or c.severity == severity]

    if quiet:
        raise typer.Exit(2 if matching else 0)

    if not matching:
        console.print("[green]No matching conflicts.[/green]")
        raise typer.Exit(0)

    for contradiction in matching:
        _print_conflict(loaded, contradiction)
    raise typer.Exit(2)


def _print_conflict(loaded: Any, contradiction: Any) -> None:
    style = _SEVERITY_STYLE.get(contradiction.severity, "")
    console.print(
        f"[bold]{contradiction.contradiction_id}[/bold]  {contradiction.type}  "
        f"[{style}]{contradiction.severity}[/{style}]  {contradiction.cluster_id}"
    )
    console.print(f"  {contradiction.description}")
    console.print()

    for label, position in zip("ABCDEFGH", contradiction.positions, strict=False):
        claim = loaded.claims.get(position.claim_id)
        where = f"§{claim.provenance.section} p{claim.provenance.page or '?'}" if claim else "§?"
        support = position.support
        details = [f"specificity {support.specificity}"]
        if support.venue:
            details.append(support.venue)
        if support.year:
            details.append(str(support.year))
        if support.is_primary:
            details.append("primary")
        if not support.stated_explicitly:
            details.append("[yellow]inferred, not stated[/yellow]")

        console.print(
            f"  [{label}] {position.claim_id}  {position.paper_id}  {where}   "
            f"value = [bold]{position.position}[/bold]"
        )
        console.print(f"      support: {' · '.join(details)}")

    resolution = (
        loaded.reconciliation.for_contradiction(contradiction.contradiction_id)
        if loaded.reconciliation
        else None
    )
    if resolution is None:
        verdict = "no rule fired; fallback ESCALATED"
    elif resolution.rule_fired:
        verdict = f"{resolution.rule_fired} - {resolution.rationale}"
    else:
        # rule_fired is None precisely when the fallback applied, and the
        # rationale already says so; prefixing "no rule fired" repeats it.
        verdict = resolution.rationale
    console.print(f"\n  policy: {verdict}\n")


def _print_resolutions(loaded: Any, resolutions: list[Resolution]) -> None:
    if not resolutions:
        console.print("[dim]Nothing has been resolved.[/dim]")
        return
    table = Table(title="Resolved conflicts")
    table.add_column("contradiction", style="cyan")
    table.add_column("outcome")
    table.add_column("rule")
    table.add_column("by")
    for resolution in resolutions:
        table.add_row(
            resolution.contradiction_id,
            resolution.outcome,
            resolution.rule_fired or "-",
            resolution.resolved_by,
        )
    console.print(table)


@app.command()
def resolve(
    run_dir: Annotated[Path, typer.Argument()],
    contradiction_id: Annotated[str, typer.Argument()],
    select: Annotated[str, typer.Option("--select", help="Claim ID to adopt.")],
    note: Annotated[str, typer.Option("--note", help="Why. Recorded in the audit trail.")] = "",
    reviewer: Annotated[str, typer.Option("--reviewer")] = "",
) -> None:
    """Record a human resolution, then re-emit the spec."""
    store = RunStore(run_dir)
    loaded = _load(run_dir)

    contradiction = loaded.contradiction(contradiction_id)
    if contradiction is None:
        err.print(f"[red]{contradiction_id} is not a contradiction in this run.[/red]")
        raise typer.Exit(1)

    valid = {p.claim_id for p in contradiction.positions}
    if select not in valid:
        err.print(
            f"[red]{select} is not a position on {contradiction_id}.[/red] "
            f"Choose one of: {', '.join(sorted(valid))}"
        )
        raise typer.Exit(2)

    from papersynth.core import ids

    resolution = Resolution(
        resolution_id=ids.resolution_id(contradiction_id),
        contradiction_id=contradiction_id,
        outcome="SELECTED",
        selected_claim_id=select,
        rule_fired=None,
        rationale=note or "Resolved by human review.",
        resolved_by="human",
        resolved_at=utcnow(),
        human_note=note or None,
    )

    reconciliation = loaded.reconciliation
    if reconciliation is None:
        from papersynth.core.models import ReconciliationResult

        reconciliation = ReconciliationResult(policy_version="1.0.0", resolutions=[])

    # Idempotent by contradiction_id: resolving twice replaces, never appends.
    reconciliation.resolutions = [
        r for r in reconciliation.resolutions if r.contradiction_id != contradiction_id
    ] + [resolution]
    store.save_reconciliation(reconciliation)
    loaded.reconciliation = reconciliation

    console.print(f"[green]OK[/green] {resolution.resolution_id} recorded -> {select}")
    remaining = len(loaded.blocking)
    console.print(f"   remaining BLOCKING: {remaining}")

    _rebuild(store, loaded, reviewer=reviewer or None)


@app.command()
def gaps(run_dir: Annotated[Path, typer.Argument()]) -> None:
    """List what the corpus does not supply."""
    loaded = _load(run_dir)
    if not loaded.gaps:
        console.print("[green]No gaps recorded.[/green]")
        return

    table = Table(title=f"missing_but_critical ({len(loaded.gaps)})")
    table.add_column("gap", style="cyan")
    table.add_column("field")
    table.add_column("criticality")
    table.add_column("question", overflow="fold")
    for gap in loaded.gaps:
        style = _SEVERITY_STYLE.get(gap.criticality, "")
        table.add_row(gap.gap_id, gap.field, f"[{style}]{gap.criticality}[/{style}]", gap.question)
    console.print(table)
    console.print(
        "\n[dim]A gap means no verified claim supplies the field, which is not "
        "the same as the papers being silent - check the source before treating "
        "one as genuinely absent.[/dim]"
    )


@app.command()
def spec(
    run_dir: Annotated[Path, typer.Argument()],
    fmt: Annotated[str, typer.Option("--format", help="yaml | json")] = "yaml",
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Re-assemble from stage artifacts.")
    ] = False,
) -> None:
    """Print the emitted spec, or rebuild it from artifacts."""
    store = RunStore(run_dir)
    loaded = _load(run_dir)

    if rebuild or loaded.spec is None:
        _rebuild(store, loaded)
        loaded = store.load()

    if loaded.spec is None:
        err.print("[red]No spec could be produced for this run.[/red]")
        raise typer.Exit(1)

    if fmt == "json":
        console.print_json(json.dumps(loaded.spec, default=str))
    else:
        console.print(yaml.safe_dump(loaded.spec, sort_keys=False, allow_unicode=True, width=100))


@app.command()
def approve(
    run_dir: Annotated[Path, typer.Argument()],
    reviewer: Annotated[str, typer.Option("--reviewer", help="Who approved it.")],
    note: Annotated[str, typer.Option("--note")] = "",
) -> None:
    """Mark the spec approved. Required before handing it to a coding agent."""
    store = RunStore(run_dir)
    loaded = _load(run_dir)

    if loaded.blocking:
        err.print(
            f"[red]{len(loaded.blocking)} blocking conflict(s) remain; "
            "the spec is not approvable.[/red]"
        )
        raise typer.Exit(2)

    if loaded.spec is None:
        err.print("[red]No spec to approve. Run `papersynth spec --rebuild` first.[/red]")
        raise typer.Exit(1)

    loaded.spec["review"] = {
        "status": "approved",
        "reviewer": reviewer,
        "approved_at": utcnow(),
        "notes": note or None,
    }
    path = store.save_spec(loaded.spec)
    console.print(
        f"[green]approved[/green] by {reviewer} at {loaded.spec['review']['approved_at']}"
    )
    console.print(f"   {path}")


@app.command()
def cost(
    run_dir: Annotated[Path, typer.Argument()],
    by_provider: Annotated[bool, typer.Option("--by-provider")] = False,
) -> None:
    """Show what the run spent. Always $0.00 on the default free-tier chain."""
    ledger = Ledger.load(Path(run_dir) / "ledger.jsonl")
    summary = ledger.summary()

    table = Table(show_header=False, box=None)
    table.add_row("calls", str(summary.calls))
    table.add_row("cached", str(summary.cached_calls))
    table.add_row("input tokens", f"{summary.input_tokens:,}")
    table.add_row("output tokens", f"{summary.output_tokens:,}")
    table.add_row("fallbacks", str(summary.fallbacks))
    table.add_row("cost", f"[green]${summary.cost_usd:.2f}[/green]")
    console.print(table)

    if by_provider and summary.by_provider:
        console.print()
        breakdown = Table(title="by provider")
        breakdown.add_column("provider", style="cyan")
        breakdown.add_column("calls", justify="right")
        for provider_id, calls in sorted(summary.by_provider.items()):
            breakdown.add_row(provider_id, str(calls))
        console.print(breakdown)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


@app.command("validate-schemas")
def validate_schemas() -> None:
    """Check that every bundled schema is well-formed and its $refs resolve.

    Run in CI. A schema whose $ref silently fails to resolve would validate
    literally anything, which would quietly disable the provenance gate.
    """
    names = sorted(p.name for p in SCHEMA_DIR.glob("*.json"))
    if not names:
        err.print("[red]No schemas found[/red]")
        raise typer.Exit(1)

    table = Table(title="Bundled schemas")
    table.add_column("schema", style="cyan")
    table.add_column("$id")
    table.add_column("status")

    failed = 0
    for name in names:
        try:
            schema = load_schema(name)
            validator = validator_for(name)
            validator.check_schema(schema)
            list(validator.iter_errors({}))
            table.add_row(name, schema.get("$id", "-"), "[green]ok[/green]")
        except Exception as exc:
            failed += 1
            table.add_row(name, "-", f"[red]{type(exc).__name__}: {exc}[/red]")

    console.print(table)
    if failed:
        err.print(f"[red]{failed} schema(s) failed[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{len(names)} schemas valid[/green]")


@app.command("schema")
def show_schema(name: Annotated[str, typer.Argument(help="e.g. spec.schema.json")]) -> None:
    """Print a bundled schema."""
    try:
        console.print_json(json.dumps(load_schema(name)))
    except FileNotFoundError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@app.command()
def models(
    provider: Annotated[
        str, typer.Option("--provider", help="groq | gemini | openrouter | vllm")
    ] = "groq",
) -> None:
    """List the models a provider currently serves.

    Free-tier lineups rotate without notice - the documented default for Groq
    was delisted between this project being designed and first run (R-13). This
    turns "check the provider's catalogue" from advice into one command.
    """
    settings = get_settings()
    key = settings.api_key(provider)

    if provider == "gemini":
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        headers = {"x-goog-api-key": key} if key else {}
    else:
        base = {
            "groq": "https://api.groq.com/openai/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "vllm": settings.vllm_url,
        }.get(provider)
        if base is None:
            err.print(f"[red]Unknown provider {provider!r}.[/red]")
            raise typer.Exit(2)
        url = f"{base}/models"
        headers = {"Authorization": f"Bearer {key}"} if key else {}

    try:
        response = httpx.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        err.print(f"[red]Cannot reach {provider}: {exc}[/red]")
        raise typer.Exit(1) from exc

    if provider == "gemini":
        ids = sorted(
            entry["name"].removeprefix("models/")
            for entry in payload.get("models") or []
            if "generateContent" in entry.get("supportedGenerationMethods", [])
        )
    else:
        ids = sorted(entry.get("id", "?") for entry in payload.get("data") or [])
    configured = settings.model_for(provider)

    table = Table(title=f"{provider} models ({len(ids)})")
    table.add_column("model", style="cyan")
    table.add_column("")
    for model_id in ids:
        table.add_row(model_id, "[green]<- configured[/green]" if model_id == configured else "")
    console.print(table)

    if configured not in ids:
        err.print(
            f"[yellow]The configured model {configured!r} is not in this list. "
            "A run would fail with model-not-found.[/yellow]"
        )


@app.command()
def diff(
    before: Annotated[Path, typer.Argument(help="The earlier run directory.")],
    after: Annotated[Path, typer.Argument(help="The later run directory.")],
    fmt: Annotated[str, typer.Option("--format", help="table | json")] = "table",
) -> None:
    """Show what changed between two emitted specs (FR-17).

    Exits 3 when a value an implementer may already have built against moved,
    so a pipeline can gate on it. A changed number is the dangerous case: code
    written against the old value still compiles against the new one.
    """
    from papersynth.synth import diff_specs

    first, second = _load(before), _load(after)
    if first.spec is None or second.spec is None:
        err.print("[red]Both runs must have an emitted spec to diff.[/red]")
        raise typer.Exit(1)

    result = diff_specs(first.spec, second.spec)

    if fmt == "json":
        console.print_json(json.dumps(result.to_dict(), default=str))
        raise typer.Exit(3 if result.breaking else 0)

    if result.identical:
        console.print("[green]No change to anything an implementer depends on.[/green]")
        raise typer.Exit(0)

    if result.values_changed:
        table = Table(title="Values changed")
        table.add_column("parameter", style="cyan")
        table.add_column("condition")
        table.add_column("from")
        table.add_column("to")
        table.add_column("resolved by")
        for entry in result.values_changed:
            table.add_row(
                entry["canonical_name"],
                entry["condition"] or "-",
                repr(entry["from"]),
                repr(entry["to"]),
                entry.get("resolved_from") or "-",
            )
        console.print(table)

    for label, items, style in (
        ("values added", [v["canonical_name"] for v in result.values_added], "green"),
        ("values removed", [v["canonical_name"] for v in result.values_removed], "red"),
        ("conflicts opened", result.conflicts_opened, "yellow"),
        ("conflicts closed", result.conflicts_closed, "green"),
        ("gaps opened", result.gaps_opened, "yellow"),
        ("gaps closed", result.gaps_closed, "green"),
        ("papers added", result.papers_added, "green"),
        ("papers removed", result.papers_removed, "red"),
    ):
        if items:
            console.print(f"[{style}]{label}:[/{style}] {', '.join(items)}")

    if result.review_from != result.review_to:
        console.print(f"review: {result.review_from or '-'} -> {result.review_to or '-'}")

    if result.breaking:
        console.print(
            "\n[bold red]Breaking for anyone who built against the earlier spec:[/bold red]"
        )
        for line in result.breaking:
            console.print(f"  {line}")
        raise typer.Exit(3)


@app.command()
def doctor() -> None:
    """Check that everything a run needs is actually present.

    The runbook opens with a list of things to verify by hand - GROBID
    reachable, a key set, the configured model still served. Every one of them
    has failed at least once during development, and each failed in a way that
    surfaced much later as a confusing error somewhere else. This asks all of
    them at once, before a run spends anything.
    """
    settings = get_settings()
    table = Table(title="Preflight")
    table.add_column("check", style="cyan")
    table.add_column("result")
    table.add_column("detail", overflow="fold")

    problems = 0

    def row(name: str, ok: bool | None, detail: str) -> None:
        nonlocal problems
        if ok is None:
            table.add_row(name, "[dim]skip[/dim]", detail)
            return
        if not ok:
            problems += 1
        table.add_row(name, "[green]ok[/green]" if ok else "[red]FAIL[/red]", detail)

    row("python", True, sys.version.split()[0])

    configured = list(settings.provider_chain)
    with_keys = [p for p in configured if settings.api_key(p) or p == "vllm"]
    row(
        "provider keys",
        bool(with_keys),
        f"chain {configured}; usable: {with_keys or 'none - set GROQ_API_KEY in .env'}",
    )

    for provider_id in with_keys:
        if provider_id == "vllm":
            continue
        try:
            served = _served_models(provider_id, settings)
        except Exception as exc:
            row(f"{provider_id} models", False, str(exc)[:90])
            continue
        wanted = settings.model_for(provider_id)
        row(
            f"{provider_id} model",
            wanted in served,
            wanted
            if wanted in served
            else f"{wanted!r} is not served. Free lineups rotate; see `papersynth models`.",
        )

    try:
        alive = httpx.get(f"{settings.grobid_url}/api/isalive", timeout=5.0).status_code == 200
    except httpx.HTTPError:
        alive = False
    row(
        "grobid",
        alive or None,
        f"{settings.grobid_url}"
        + ("" if alive else " unreachable - only the LaTeX path will work"),
    )

    row(
        "pdftotext",
        shutil.which("pdftotext") is not None,
        shutil.which("pdftotext") or "install poppler-utils for the PDF fallback",
    )

    from papersynth.ingest.math_layer import build_recoverer

    recoverer = build_recoverer(enabled=True)
    row(
        "math recovery",
        recoverer.available or None,
        "pix2tex ready"
        if recoverer.available
        else "not installed - damaged math is detected and flagged, not recovered",
    )

    for label, path in (
        ("policy", settings.policy),
        ("range rules", settings.range_rules),
        ("checklist", settings.checklist),
    ):
        row(f"config: {label}", Path(path).exists(), str(path))

    console.print(table)
    if problems:
        err.print(f"[red]{problems} check(s) failed.[/red]")
        raise typer.Exit(1)
    console.print("[green]Ready to run.[/green]")


def _served_models(provider_id: str, settings: Any) -> set[str]:
    key = settings.api_key(provider_id)
    if provider_id == "gemini":
        response = httpx.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": key or ""},
            timeout=20.0,
        )
        response.raise_for_status()
        return {
            m["name"].removeprefix("models/")
            for m in response.json().get("models") or []
            if "generateContent" in m.get("supportedGenerationMethods", [])
        }

    base = {
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }[provider_id]
    response = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=20.0)
    response.raise_for_status()
    return {m.get("id", "") for m in response.json().get("data") or []}


@app.command("eval-gaps")
def eval_gaps(
    adversarial: Annotated[
        bool, typer.Option("--adversarial/--checklist-only", help="Include Pass B.")
    ] = True,
    fp_only: Annotated[
        bool,
        typer.Option(
            "--fp-only",
            help="Skip the ablation sweep; one call, asks only whether the "
            "reference spec has converged.",
        ),
    ] = False,
) -> None:
    """Measure gap recall by ablation, and false positives on a complete spec.

    Recall is self-labelling: delete a field, check the gap appears. False
    positives are counted on the untouched spec, where zero is correct - a
    noisy gap list gets skimmed and then ignored, at which point the real
    entries are invisible too.
    """
    from papersynth.eval import evaluate_gaps
    from papersynth.llm import build_router

    settings = get_settings()
    try:
        provider = build_router(settings) if adversarial else StubProvider([])
    except PaperSynthError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    try:
        report = evaluate_gaps(provider, adversarial=adversarial, ablate_sweep=not fp_only)
    except AllProvidersExhausted as exc:
        # Expected on a day of heavy use, and not a failure of the harness.
        # The traceback this used to print buried that.
        err.print(f"[yellow]{exc}[/yellow]")
        err.print(
            "[dim]Run with --checklist-only to measure Pass A alone, which needs no provider.[/dim]"
        )
        raise typer.Exit(3) from exc

    console.print(report.render())

    for note in report.notes:
        console.print(f"[dim]  {note}[/dim]")

    # Only warn when recall was actually measured. With --fp-only there is no
    # sweep, so recall is 0/0, and "0.00 below target" would invent a failure
    # out of a measurement that was never taken.
    if not fp_only and report.recall < 0.80:
        err.print(
            f"[yellow]recall {report.recall:.2f} is below the 0.80 target (section 13.1)[/yellow]"
        )


@app.command()
def extractors() -> None:
    """List registered extractors, including third-party plugins."""
    from papersynth.extract import registry

    table = Table(title="Extractors")
    table.add_column("claim type", style="cyan")
    table.add_column("version")
    table.add_column("sections read", overflow="fold")
    for name, info in registry.describe().items():
        table.add_row(name, info["version"], info["sections"])
    console.print(table)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_chain(raw: str) -> list[ProviderId]:
    """Validate a --provider-chain override.

    A typo silently producing an empty or unknown chain would surface much
    later as "no providers configured", pointing at .env rather than at the
    flag that actually caused it.
    """
    valid = set(get_args(ProviderId))
    requested = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [part for part in requested if part not in valid]
    if unknown:
        err.print(
            f"[red]Unknown provider(s): {', '.join(unknown)}.[/red] "
            f"Available: {', '.join(sorted(valid))}"
        )
        raise typer.Exit(2)
    return cast("list[ProviderId]", requested)


def _load(run_dir: Path) -> Any:
    try:
        return RunStore(run_dir).load()
    except PaperSynthError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


def _rebuild(store: RunStore, loaded: Any, *, reviewer: str | None = None) -> None:
    """Re-assemble and re-validate the spec from stage artifacts.

    Cheap by design: a human decision on one conflict re-emits the spec without
    re-extracting a single paper.
    """
    builder = SpecBuilder(
        run_id=loaded.run_id,
        objective=loaded.objective,
        documents=loaded.documents,
        claims=loaded.claims,
    )
    rebuilt = builder.build(
        contradictions=loaded.contradictions,
        reconciliation=loaded.reconciliation,
        gaps=loaded.gaps,
        reports=loaded.reports,
        reviewer=reviewer,
    )

    report = SpecValidator(loaded.claims).validate(
        rebuilt,
        contradictions=loaded.contradictions,
        reconciliation=loaded.reconciliation,
    )
    rebuilt["verification_summary"]["provenance_completeness"] = report.provenance_completeness

    store.save_review(
        render_review(
            rebuilt,
            contradictions=loaded.contradictions,
            reconciliation=loaded.reconciliation,
            gaps=loaded.gaps,
            blocking=report.blocking_conflicts,
        )
    )

    if report.blocking_conflicts:
        store.save_spec(rebuilt, draft=True)
        err.print(
            f"[yellow]{len(report.blocking_conflicts)} blocking conflict(s) remain; "
            "wrote implementation_spec.draft.yaml instead.[/yellow]"
        )
        return

    if not report.ok:
        # Provenance or schema failure is a builder bug. The runbook is
        # explicit that this is fixed, never bypassed.
        err.print("[red]Spec failed validation:[/red]")
        for problem in (
            report.unclosed_provenance + report.dependency_cycle + report.schema_errors
        )[:5]:
            err.print(f"  {problem}")
        raise typer.Exit(1)

    path = store.save_spec(rebuilt)
    console.print(f"[green]spec re-emitted[/green] {path}")


if __name__ == "__main__":
    sys.exit(app())
