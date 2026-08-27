"""Spec validation and the emission gates (section 10.2, 13.5).

Four gates, all hard:

  1. the spec validates against spec.schema.json
  2. every emitted field traces to at least one real, verified claim
  3. components[].depends_on forms a DAG
  4. no BLOCKING contradiction remains unresolved

Gate 2 is the one that makes NFR-01 enforceable rather than aspirational. A
provenance_ref pointing at a claim that does not exist, or at one that was
rejected, means the builder invented something - and the runbook is explicit
that this is a bug to fix, never to bypass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from papersynth.core.errors import (
    BlockingConflictsError,
    CyclicDependencyError,
    ProvenanceIncompleteError,
    SchemaValidationError,
)
from papersynth.core.models import Claim, Contradiction, ReconciliationResult
from papersynth.schemas import validate


@dataclass
class ValidationReport:
    schema_errors: list[str] = field(default_factory=list)
    unclosed_provenance: list[str] = field(default_factory=list)
    dependency_cycle: list[str] = field(default_factory=list)
    blocking_conflicts: list[str] = field(default_factory=list)
    provenance_completeness: float = 1.0

    @property
    def ok(self) -> bool:
        return not (
            self.schema_errors
            or self.unclosed_provenance
            or self.dependency_cycle
            or self.blocking_conflicts
        )

    def raise_first(self) -> None:
        """Raise the most actionable failure, in the order a user should fix them."""
        if self.blocking_conflicts:
            raise BlockingConflictsError(self.blocking_conflicts)
        if self.unclosed_provenance:
            raise ProvenanceIncompleteError(self.unclosed_provenance)
        if self.dependency_cycle:
            raise CyclicDependencyError(self.dependency_cycle)
        if self.schema_errors:
            raise SchemaValidationError("spec.schema.json", self.schema_errors)


class SpecValidator:
    def __init__(self, claims: dict[str, Claim]) -> None:
        self.claims = claims

    def validate(
        self,
        spec: dict[str, Any],
        *,
        contradictions: list[Contradiction] | None = None,
        reconciliation: ReconciliationResult | None = None,
    ) -> ValidationReport:
        report = ValidationReport()
        report.schema_errors = validate(spec, "spec.schema.json")
        self._check_provenance(spec, report)
        self._check_dag(spec, report)
        self._check_blocking(contradictions or [], reconciliation, report)
        return report

    # -- gate 2 ------------------------------------------------------------

    def _check_provenance(self, spec: dict[str, Any], report: ValidationReport) -> None:
        total = 0
        closed = 0

        for path, refs in _iter_provenance_refs(spec):
            total += 1
            if not refs:
                report.unclosed_provenance.append(f"{path}: no provenance_refs")
                continue

            missing = [r for r in refs if r not in self.claims]
            if missing:
                report.unclosed_provenance.append(
                    f"{path}: references unknown claim(s) {', '.join(missing)}"
                )
                continue

            # A rejected claim cannot back an emitted field. Verification
            # exists precisely to keep such claims out of the spec.
            unverified = [r for r in refs if self.claims[r].status != "verified"]
            if unverified:
                report.unclosed_provenance.append(
                    f"{path}: backed only by unverified claim(s) {', '.join(unverified)}"
                )
                continue

            closed += 1

        report.provenance_completeness = round(closed / total, 4) if total else 1.0

    # -- gate 3 ------------------------------------------------------------

    def _check_dag(self, spec: dict[str, Any], report: ValidationReport) -> None:
        graph = {
            c["component_id"]: list(c.get("depends_on", []))
            for c in spec.get("components", [])
            if isinstance(c, dict) and "component_id" in c
        }

        WHITE, GREY, BLACK = 0, 1, 2
        colour = dict.fromkeys(graph, WHITE)

        def visit(node: str, path: list[str]) -> list[str] | None:
            colour[node] = GREY
            for neighbour in graph.get(node, []):
                if neighbour not in colour:
                    continue  # a dangling reference is a schema concern, not a cycle
                if colour[neighbour] == GREY:
                    return [*path, node, neighbour]
                if colour[neighbour] == WHITE:
                    found = visit(neighbour, [*path, node])
                    if found:
                        return found
            colour[node] = BLACK
            return None

        for node in sorted(graph):
            if colour[node] == WHITE:
                cycle = visit(node, [])
                if cycle:
                    report.dependency_cycle = cycle
                    return

    # -- gate 4 ------------------------------------------------------------

    @staticmethod
    def _check_blocking(
        contradictions: list[Contradiction],
        reconciliation: ReconciliationResult | None,
        report: ValidationReport,
    ) -> None:
        for contradiction in contradictions:
            if contradiction.severity != "BLOCKING":
                continue
            resolution = (
                reconciliation.for_contradiction(contradiction.contradiction_id)
                if reconciliation
                else None
            )
            # No resolution, or one that is still open, blocks emission. This
            # is expected behaviour rather than an error (section 15.4).
            if resolution is None or resolution.is_open:
                report.blocking_conflicts.append(contradiction.contradiction_id)


def _iter_provenance_refs(spec: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Every provenance_refs list in the spec, with a readable path."""
    found: list[tuple[str, list[str]]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if "provenance_refs" in node:
                found.append((path or "<root>", list(node["provenance_refs"])))
            for key, value in node.items():
                if key != "provenance_refs":
                    walk(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")

    walk(spec, "")
    return found
