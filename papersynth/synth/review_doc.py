"""SPEC_REVIEW.md - the human-facing side of the spec (section 4.3).

The spec is for the coding agent; this is for the person who has to approve it.
It leads with what blocks emission, because that is the only thing the reviewer
must act on before anything can proceed.
"""

from __future__ import annotations

from typing import Any

from papersynth.core.models import Contradiction, Gap, ReconciliationResult


def render(
    spec: dict[str, Any],
    *,
    contradictions: list[Contradiction],
    reconciliation: ReconciliationResult | None,
    gaps: list[Gap],
    blocking: list[str],
) -> str:
    lines: list[str] = [
        f"# Spec review - {spec['run_id']}",
        "",
        f"**Objective:** {spec['objective']}",
        "",
        f"Generated {spec['generated_at']} from {len(spec['source_papers'])} paper(s).",
        "",
    ]

    summary_counts = spec.get("verification_summary", {})
    ingested = summary_counts.get("papers_ingested", len(spec["source_papers"]))
    contributing = summary_counts.get("papers_contributing", ingested)
    if contributing < ingested:
        silent = [p["paper_id"] for p in spec["source_papers"] if not p.get("claims_contributed")]
        lines += [
            "## Partial corpus - read this first",
            "",
            f"Only {contributing} of {ingested} papers contributed anything to this spec.",
            f"Nothing was extracted from: {', '.join(silent)}.",
            "",
            "Cross-paper reconciliation is the point of this tool, and it cannot",
            "have happened for papers that produced no claims. Any absence of",
            "conflicts below reflects the papers that were read, not the corpus",
            "you asked for. Check the run warnings, then re-run.",
            "",
        ]

    if blocking:
        lines += [
            "## Blocking - resolve before the spec can be emitted",
            "",
            f"{len(blocking)} contradiction(s) block emission. Nothing else in this",
            "document matters until these are decided.",
            "",
        ]
        for contradiction in contradictions:
            if contradiction.contradiction_id not in blocking:
                continue
            lines += _render_conflict(contradiction, reconciliation)
        lines += [
            "Resolve with:",
            "",
            "```",
            f'papersynth resolve <run> {blocking[0]} --select <claim_id> --note "..."',
            "```",
            "",
        ]
    else:
        lines += ["## No blocking conflicts", "", "The spec can be emitted.", ""]

    open_conflicts = spec.get("open_conflicts", [])
    lines += [
        f"## Open conflicts ({len(open_conflicts)})",
        "",
    ]
    if open_conflicts:
        lines += [
            "These do not block emission. They ride along in the spec as",
            "annotations, and the implementer chooses.",
            "",
        ]
        for entry in open_conflicts:
            lines += [
                f"### {entry['contradiction_id']} - {entry['type']}",
                "",
                entry["summary"],
                "",
            ]
            for position in entry["positions"]:
                provenance = position.get("provenance") or {}
                where = provenance.get("section", "?")
                lines.append(
                    f"- **{position['position']}** - {position['paper_id']} "
                    f"(§{where}, `{position['claim_id']}`)"
                )
            if entry.get("guidance"):
                lines += ["", f"> {entry['guidance']}", ""]
    else:
        lines += ["None.", ""]

    missing = spec.get("missing_but_critical", [])
    lines += [f"## Missing but critical ({len(missing)})", ""]
    if missing:
        lines += [
            "No verified claim supplies these. That is not quite the same as",
            "the papers being silent - extraction can miss a stated value - so",
            "check the source before treating any of them as genuinely absent.",
            "Either way, do not let a coding agent invent them.",
            "",
        ]
        for gap in missing:
            lines.append(f"- **{gap['field']}** ({gap['criticality']}) - {gap['question']}")
            if gap.get("suggested_sources"):
                lines.append(f"  - try: {', '.join(gap['suggested_sources'])}")
        lines.append("")
    else:
        lines += ["None.", ""]

    resolved = spec.get("resolved_conflicts", [])
    if resolved:
        lines += [
            f"## Auto-resolved ({len(resolved)})",
            "",
            "Recorded for audit. Each names the rule that closed it - if a rule",
            "looks wrong, the policy is config and can be changed.",
            "",
        ]
        for entry in resolved:
            lines.append(
                f"- `{entry['contradiction_id']}` -> {entry['outcome']} "
                f"via **{entry['rule_fired']}** ({entry['resolved_by']})"
            )
        lines.append("")

    assumptions = spec.get("assumptions", [])
    if assumptions:
        lines += [f"## Assumptions ({len(assumptions)})", ""]
        for entry in assumptions:
            lines.append(f"- {entry['statement']}")
        lines.append("")

    summary = spec.get("verification_summary", {})
    lines += [
        "## Verification",
        "",
        f"- claims extracted: {summary.get('claims_total', 0)}",
        f"- verified: {summary.get('verified', 0)}",
        f"- rejected: {summary.get('rejected', 0)}",
        f"- provenance completeness: {summary.get('provenance_completeness', 0):.0%}",
        "",
    ]
    reasons = summary.get("rejection_reasons") or {}
    if reasons:
        lines += ["Rejections by check:", ""]
        lines += [f"- {check}: {count}" for check, count in sorted(reasons.items())]
        lines.append("")

    lines += [
        "## Handing off",
        "",
        "Once approved, give the spec to a coding agent with this instruction:",
        "",
        "> Implement `implementation_spec.yaml`. Do not fill in anything listed",
        "> under `missing_but_critical` - stop and ask me instead.",
        "",
    ]

    return "\n".join(lines)


def _render_conflict(
    contradiction: Contradiction, reconciliation: ReconciliationResult | None
) -> list[str]:
    lines = [
        f"### {contradiction.contradiction_id} - {contradiction.type} ({contradiction.severity})",
        "",
        contradiction.description,
        "",
    ]
    for position in contradiction.positions:
        support = position.support
        details = [f"specificity {support.specificity}"]
        if support.venue:
            details.append(support.venue)
        if support.year:
            details.append(str(support.year))
        if support.is_primary:
            details.append("primary source")
        if not support.stated_explicitly:
            details.append("inferred, not stated")
        lines.append(
            f"- **{position.position}** - {position.paper_id} "
            f"(`{position.claim_id}`) - {', '.join(details)}"
        )

    resolution = (
        reconciliation.for_contradiction(contradiction.contradiction_id) if reconciliation else None
    )
    if resolution is not None:
        rule = resolution.rule_fired or "no rule fired"
        lines += ["", f"> policy: {rule} - {resolution.rationale}", ""]
    else:
        lines += ["", "> policy: no rule fired; fallback ESCALATED", ""]
    return lines
