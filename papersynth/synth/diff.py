"""Machine-readable diff between two emitted specs (FR-17).

The question this answers is not "what bytes changed" but "did anything I
depend on change". A spec is handed to a coding agent; when it is re-emitted
after a resolution, a re-run, or a policy change, the implementer needs to know
whether a value they already built against has moved.

So the diff is structured by what a reader acts on - values that changed,
conflicts that opened or closed, gaps that appeared - rather than by document
position. Two specs differing only in `generated_at` diff as identical, because
a timestamp is not a change to anything anyone implements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Fields that differ between any two runs and mean nothing to an implementer.
_INCIDENTAL = frozenset({"generated_at", "run_id"})


@dataclass
class SpecDiff:
    """What changed, grouped by what a reader would do about it."""

    from_run: str = ""
    to_run: str = ""
    papers_added: list[str] = field(default_factory=list)
    papers_removed: list[str] = field(default_factory=list)
    values_added: list[dict[str, Any]] = field(default_factory=list)
    values_removed: list[dict[str, Any]] = field(default_factory=list)
    values_changed: list[dict[str, Any]] = field(default_factory=list)
    conflicts_opened: list[str] = field(default_factory=list)
    conflicts_closed: list[str] = field(default_factory=list)
    gaps_opened: list[str] = field(default_factory=list)
    gaps_closed: list[str] = field(default_factory=list)
    review_from: str = ""
    review_to: str = ""

    @property
    def identical(self) -> bool:
        """True when nothing an implementer depends on moved."""
        return not any(
            (
                self.papers_added,
                self.papers_removed,
                self.values_added,
                self.values_removed,
                self.values_changed,
                self.conflicts_opened,
                self.conflicts_closed,
                self.gaps_opened,
                self.gaps_closed,
                self.review_from != self.review_to,
            )
        )

    @property
    def breaking(self) -> list[str]:
        """Changes that invalidate work already done against the old spec.

        A changed value is the serious case: code was written against the old
        number and still compiles against the new one. An added value or a
        closed gap gives the implementer more, and a newly opened conflict
        tells them to stop - both are visible. A silently moved value is not.
        """
        out = [
            f"{c['canonical_name']} changed from {c['from']!r} to {c['to']!r}"
            for c in self.values_changed
        ]
        out += [f"{v['canonical_name']} was removed" for v in self.values_removed]
        out += [f"paper {p} was dropped from the corpus" for p in self.papers_removed]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_run,
            "to": self.to_run,
            "identical": self.identical,
            "papers": {"added": self.papers_added, "removed": self.papers_removed},
            "values": {
                "added": self.values_added,
                "removed": self.values_removed,
                "changed": self.values_changed,
            },
            "conflicts": {"opened": self.conflicts_opened, "closed": self.conflicts_closed},
            "gaps": {"opened": self.gaps_opened, "closed": self.gaps_closed},
            "review": {"from": self.review_from, "to": self.review_to},
            "breaking": self.breaking,
        }


def diff_specs(before: dict[str, Any], after: dict[str, Any]) -> SpecDiff:
    """Compare two emitted specs."""
    result = SpecDiff(
        from_run=str(before.get("run_id", "?")),
        to_run=str(after.get("run_id", "?")),
        review_from=str((before.get("review") or {}).get("status", "")),
        review_to=str((after.get("review") or {}).get("status", "")),
    )

    old_papers = {p["paper_id"] for p in before.get("source_papers") or []}
    new_papers = {p["paper_id"] for p in after.get("source_papers") or []}
    result.papers_added = sorted(new_papers - old_papers)
    result.papers_removed = sorted(old_papers - new_papers)

    old_values = _values(before)
    new_values = _values(after)

    for key in sorted(new_values.keys() - old_values.keys()):
        result.values_added.append(_render(key, new_values[key]))
    for key in sorted(old_values.keys() - new_values.keys()):
        result.values_removed.append(_render(key, old_values[key]))

    for key in sorted(old_values.keys() & new_values.keys()):
        old, new = old_values[key], new_values[key]
        if _normalize(old["value"]) != _normalize(new["value"]):
            result.values_changed.append(
                {
                    "component_id": key[0],
                    "canonical_name": key[1],
                    "condition": key[2] or None,
                    "from": old["value"],
                    "to": new["value"],
                    # Which resolution moved it, when one did. Without this a
                    # reader sees a number change with no way to ask why.
                    "resolved_from": new.get("resolved_from"),
                }
            )

    old_conflicts = {c["contradiction_id"] for c in before.get("open_conflicts") or []}
    new_conflicts = {c["contradiction_id"] for c in after.get("open_conflicts") or []}
    result.conflicts_opened = sorted(new_conflicts - old_conflicts)
    result.conflicts_closed = sorted(old_conflicts - new_conflicts)

    old_gaps = {g["field"] for g in before.get("missing_but_critical") or []}
    new_gaps = {g["field"] for g in after.get("missing_but_critical") or []}
    result.gaps_opened = sorted(new_gaps - old_gaps)
    result.gaps_closed = sorted(old_gaps - new_gaps)

    return result


def _values(spec: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Every emitted hyperparameter, keyed by where it applies.

    Keyed on (component, name, condition) rather than on position, so
    reordering the spec is not a change and a value moving between conditions
    is.
    """
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for component in spec.get("components") or []:
        component_id = str(component.get("component_id", ""))
        for hyperparameter in component.get("hyperparameters") or []:
            key = (
                component_id,
                str(hyperparameter.get("canonical_name", "")),
                str(hyperparameter.get("condition") or ""),
            )
            out[key] = hyperparameter
    return out


def _render(key: tuple[str, str, str], entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "component_id": key[0],
        "canonical_name": key[1],
        "condition": key[2] or None,
        "value": entry.get("value"),
    }


def _normalize(value: Any) -> str:
    """Compare by magnitude, so 0.0001 and 1e-4 are not a change."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | float):
        return f"{float(value):.12g}"
    return str(value).strip().lower()
