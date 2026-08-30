"""Spec assembly: everything -> implementation_spec.yaml (stage 7).

The deliverable. What reaches it is deliberately narrow: only verified claims,
only conflicts that a policy could not close, and only fields that trace back
to a real span.

Components are derived from what the papers actually said each value
configures (the `applies_to` field), not invented. The MVA has no component
extractor, so a corpus that never names a component yields one global
configuration block - accurate, if coarse. Proper component extraction is a
P1 claim type and slots in without changing this builder.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import papersynth
from papersynth.contradict import normalize_condition
from papersynth.core import ids
from papersynth.core.document import StructuredDocument
from papersynth.core.models import (
    Claim,
    Contradiction,
    Gap,
    ReconciliationResult,
    Resolution,
    utcnow,
)
from papersynth.verify import VerificationReport


class SpecBuilder:
    """Assembles the spec from the artifacts of stages 0-6."""

    def __init__(
        self,
        *,
        run_id: str,
        objective: str,
        documents: list[StructuredDocument],
        claims: dict[str, Claim],
        papers_requested: int | None = None,
    ) -> None:
        self.run_id = run_id
        self.objective = objective
        self.documents = documents
        self.claims = claims
        #: What the user asked for, which is not what arrived when a paper
        #: fails to fetch. Counting only what arrived made a run built from one
        #: of three papers report "1/1" and look complete.
        self.papers_requested = papers_requested or len(documents)

    def build_draft(self, **kwargs: Any) -> dict[str, Any]:
        """Assemble a spec for auditing, without emission gates.

        Deliberately a delegation rather than a second assembly path. Pass B
        audits this draft and reports gaps against it, so if the draft were
        built by different code than the spec that ships, the gaps could
        describe an artifact nobody receives.

        There is nothing to disable here: this builder never validates. The
        BLOCKING-conflict check, provenance closure and schema validation all
        live in SpecValidator and run only on the final emission. Keeping
        assembly and gating in separate objects is what makes the draft
        structurally identical to the emitted spec by construction rather than
        by discipline - and `test_draft_and_final_are_identical` fails if that
        ever stops being true.
        """
        return self.build(**kwargs)

    def build(
        self,
        *,
        contradictions: list[Contradiction] | None = None,
        reconciliation: ReconciliationResult | None = None,
        gaps: list[Gap] | None = None,
        reports: list[VerificationReport] | None = None,
        reviewer: str | None = None,
    ) -> dict[str, Any]:
        contradictions = contradictions or []
        gaps = gaps or []
        reports = reports or []

        resolved_ids = _resolved_ids(reconciliation)
        disputed = self._disputed_scopes(contradictions, resolved_ids)
        superseded = self._superseded_claims(contradictions, reconciliation)

        return {
            "spec_version": papersynth.SPEC_VERSION,
            "run_id": self.run_id,
            "generated_at": utcnow(),
            "source_papers": [
                _paper_entry(d, self._contribution(d.paper_id)) for d in self.documents
            ],
            "objective": self.objective,
            "components": self._components(disputed, superseded, reconciliation),
            "expected_results": self._expected_results(disputed, superseded),
            "open_conflicts": self._open_conflicts(contradictions, reconciliation),
            "resolved_conflicts": _resolved_conflicts(reconciliation),
            "missing_but_critical": [_gap_entry(g) for g in gaps],
            "assumptions": self._assumptions(),
            "verification_summary": self._verification_summary(reports),
            "review": {
                "status": "draft",
                "reviewer": reviewer,
                "approved_at": None,
                "notes": None,
            },
        }

    # -- components --------------------------------------------------------

    def _disputed_scopes(
        self, contradictions: list[Contradiction], resolved_ids: set[str]
    ) -> set[tuple[str, str]]:
        """(canonical_name, condition) pairs still under dispute.

        Scoping by name and condition rather than by claim identity matters: a
        contradiction lists one representative position per distinct value, so
        a third paper repeating one of those values is not itself a listed
        position. Excluding only the representatives would emit that duplicate
        as a settled fact while the very same value was still being argued
        over in open_conflicts.
        """
        disputed: set[tuple[str, str]] = set()
        for contradiction in contradictions:
            if contradiction.contradiction_id in resolved_ids:
                continue
            for claim_id in contradiction.claim_ids:
                claim = self.claims.get(claim_id)
                if claim is None:
                    continue
                disputed.add(
                    (
                        str(claim.payload.get("canonical_name")),
                        normalize_condition(claim.payload.get("condition")),
                    )
                )
        return disputed

    def _superseded_claims(
        self,
        contradictions: list[Contradiction],
        reconciliation: ReconciliationResult | None,
    ) -> set[str]:
        """Claims a resolution decided against.

        Resolving a conflict removes its scope from `disputed`, which would
        otherwise release every claim in that scope back into the spec -
        including the values the resolution rejected. The spec would then state
        both 0.0001 and 0.0003 for the same parameter under the same condition,
        reinstating the exact contradiction a human had just settled.

        Losers are identified by value rather than by claim id, so a third
        paper repeating a rejected value is dropped too, while one repeating
        the winning value is kept and contributes its provenance.
        """
        if reconciliation is None:
            return set()

        superseded: set[str] = set()
        by_id = {c.contradiction_id: c for c in contradictions}

        for resolution in reconciliation.resolutions:
            if resolution.is_open or not resolution.selected_claim_id:
                continue
            contradiction = by_id.get(resolution.contradiction_id)
            winner = self.claims.get(resolution.selected_claim_id)
            if contradiction is None or winner is None:
                continue

            scope = (
                str(winner.payload.get("canonical_name")),
                normalize_condition(winner.payload.get("condition")),
            )
            winning_value = _value_key(winner.payload.get("value"))

            for claim in self.claims.values():
                if claim.claim_id == winner.claim_id:
                    continue
                claim_scope = (
                    str(claim.payload.get("canonical_name")),
                    normalize_condition(claim.payload.get("condition")),
                )
                if claim_scope != scope:
                    continue
                if _value_key(claim.payload.get("value")) != winning_value:
                    superseded.add(claim.claim_id)

        return superseded

    def _components(
        self,
        disputed: set[tuple[str, str]],
        superseded: set[str],
        reconciliation: ReconciliationResult | None,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[Claim]] = defaultdict(list)
        for claim in self.claims.values():
            if claim.status != "verified" or claim.type != "hyperparameter":
                continue
            grouped[str(claim.payload.get("applies_to") or "global")].append(claim)

        components = []
        for target, claims in sorted(grouped.items()):
            # A value still under dispute must not be emitted as settled; it
            # appears in open_conflicts instead, for the implementer.
            settled = [
                c for c in claims if not _is_disputed(c, disputed) and c.claim_id not in superseded
            ]
            hyperparameters = self._merge_agreeing(settled, reconciliation)
            refs = sorted({c.claim_id for c in claims})
            components.append(
                {
                    "component_id": (
                        "cmp_global" if target == "global" else ids.component_id(target)
                    ),
                    "name": "Global configuration" if target == "global" else target,
                    "role": (
                        "Configuration applying across the implementation."
                        if target == "global"
                        else f"Configuration for {target}."
                    ),
                    "depends_on": [],
                    "hyperparameters": hyperparameters,
                    "interfaces": {"inputs": [], "outputs": [], "invariants": []},
                    "provenance_refs": refs,
                }
            )
        return components

    def _merge_agreeing(
        self, claims: list[Claim], reconciliation: ReconciliationResult | None
    ) -> list[dict[str, Any]]:
        """One entry per distinct (name, condition, value), not per claim.

        Three papers each stating dropout 0.1 is one fact corroborated three
        times, not three facts. Emitting it three times gives a coding agent
        noise to disambiguate, and buries the genuinely useful signal - which
        is that the papers agree. The corroboration is preserved where it
        belongs, as multiple provenance_refs on the single entry.
        """
        grouped: dict[tuple[str, str, str], list[Claim]] = defaultdict(list)
        for claim in claims:
            grouped[
                (
                    str(claim.payload.get("canonical_name")),
                    normalize_condition(claim.payload.get("condition")),
                    _value_key(claim.payload.get("value")),
                )
            ].append(claim)

        entries = []
        for key in sorted(grouped):
            members = sorted(grouped[key], key=lambda c: c.claim_id)
            entry = self._hyperparameter_entry(members[0], reconciliation)
            entry["provenance_refs"] = [c.claim_id for c in members]
            entries.append(entry)
        return entries

    def _hyperparameter_entry(
        self, claim: Claim, reconciliation: ReconciliationResult | None
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "canonical_name": claim.payload["canonical_name"],
            "value": claim.payload["value"],
            "unit": claim.payload.get("unit"),
            "condition": claim.payload.get("condition"),
            "resolved_from": None,
            "resolution": None,
            "provenance_refs": [claim.claim_id],
        }

        resolution = _resolution_selecting(reconciliation, claim.claim_id)
        if resolution is not None:
            entry["resolved_from"] = resolution.contradiction_id
            entry["resolution"] = {
                "outcome": resolution.outcome,
                "rule_fired": resolution.rule_fired,
                "resolved_by": resolution.resolved_by,
                "reviewer": resolution.human_note,
            }
        return entry

    # -- conflicts ---------------------------------------------------------

    def _expected_results(
        self, disputed: set[tuple[str, str]], superseded: set[str]
    ) -> list[dict[str, Any]]:
        """Reproduction targets: what the finished code should produce.

        A disputed result is withheld exactly as a disputed hyperparameter is.
        Emitting one of two contested scores as the target would tell an
        implementer their reimplementation is wrong when it is merely matching
        the other paper.

        Tolerance comes from the paper's own reported variance and is left null
        otherwise. Inventing one would be inventing a claim about how
        reproducible the result is, which is precisely what nobody stated.
        """
        grouped: dict[tuple[str, str, str, str], list[Claim]] = defaultdict(list)
        for claim in self.claims.values():
            if claim.status != "verified" or claim.type != "result":
                continue
            if claim.claim_id in superseded:
                continue
            payload = claim.payload
            scope = (
                str(payload.get("metric") or ""),
                str(payload.get("dataset") or ""),
                str(payload.get("split") or ""),
                str(payload.get("model_variant") or ""),
            )
            if (scope[0], normalize_condition(scope[2])) in disputed:
                continue
            grouped[scope].append(claim)

        out = []
        for scope, claims in sorted(grouped.items()):
            members = sorted(claims, key=lambda c: c.claim_id)
            values = {_value_key(c.payload.get("value")) for c in members}
            if len(values) > 1:
                # The corpus disagrees and no resolution closed it, so there is
                # no single target to report.
                continue
            first = members[0]
            out.append(
                {
                    "metric": scope[0],
                    "value": first.payload.get("value"),
                    "tolerance": first.payload.get("reported_variance"),
                    "dataset": scope[1] or None,
                    "split": scope[2] or None,
                    "provenance_refs": [c.claim_id for c in members],
                }
            )
        return out

    def _open_conflicts(
        self,
        contradictions: list[Contradiction],
        reconciliation: ReconciliationResult | None,
    ) -> list[dict[str, Any]]:
        """Unresolved MATERIAL conflicts, annotated for the implementer.

        BLOCKING conflicts never appear here - they halt emission entirely, and
        the schema forbids the severity outright as a second line of defence.
        """
        out = []
        for contradiction in contradictions:
            resolution = (
                reconciliation.for_contradiction(contradiction.contradiction_id)
                if reconciliation
                else None
            )
            if resolution is not None and not resolution.is_open:
                continue
            if contradiction.severity != "MATERIAL":
                continue

            out.append(
                {
                    "contradiction_id": contradiction.contradiction_id,
                    "type": contradiction.type,
                    "severity": "MATERIAL",
                    "summary": contradiction.description,
                    "positions": [
                        {
                            "claim_id": p.claim_id,
                            "paper_id": p.paper_id,
                            "position": p.position,
                            "provenance": _provenance_stub(self.claims.get(p.claim_id)),
                        }
                        for p in contradiction.positions
                    ],
                    "guidance": _guidance(contradiction, resolution),
                }
            )
        return out

    # -- assumptions and summary ------------------------------------------

    def _assumptions(self) -> list[dict[str, Any]]:
        """Values the implementer is accepting that the paper did not state.

        A claim whose value could not be found verbatim in its span was
        inferred or converted. Recording it here means the implementer sees
        what they are inheriting rather than reading it as fact.
        """
        out = []
        for claim in sorted(self.claims.values(), key=lambda c: c.claim_id):
            if claim.status != "verified":
                continue
            if claim.payload.get("stated_explicitly") is not False:
                continue
            out.append(
                {
                    "statement": (
                        f"{claim.payload.get('canonical_name')} = "
                        f"{claim.payload.get('value')!r} was inferred from "
                        f"{claim.paper_id} rather than stated directly."
                    ),
                    "explicit": False,
                    "criticality": "MATERIAL",
                    "provenance_refs": [claim.claim_id],
                }
            )
        return out

    def _contribution(self, paper_id: str) -> int:
        """Verified claims this paper actually put into the spec.

        Listing a paper under source_papers implies the spec synthesizes it. A
        paper whose extraction failed contributed nothing, and saying so is the
        difference between a partial spec that admits it and one that quietly
        claims three sources while reflecting one.
        """
        return sum(
            1
            for claim in self.claims.values()
            if claim.paper_id == paper_id and claim.status == "verified"
        )

    def _verification_summary(self, reports: list[VerificationReport]) -> dict[str, Any]:
        reasons: dict[str, int] = defaultdict(int)
        for report in reports:
            for check, count in report.rejection_reasons.items():
                reasons[check] += count

        total = sum(r.total for r in reports)
        verified = sum(r.verified for r in reports)
        rejected = sum(r.rejected for r in reports)

        return {
            "claims_total": total,
            "verified": verified,
            "rejected": rejected,
            "rejection_reasons": dict(sorted(reasons.items())),
            # Filled in by the validator, which is the component that can
            # actually measure closure across the assembled spec.
            "provenance_completeness": 1.0,
            "papers_requested": self.papers_requested,
            "papers_ingested": len(self.documents),
            "papers_contributing": sum(
                1 for d in self.documents if self._contribution(d.paper_id) > 0
            ),
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _paper_entry(doc: StructuredDocument, claims_contributed: int) -> dict[str, Any]:
    return {
        "paper_id": doc.paper_id,
        "title": doc.title,
        "venue": doc.venue,
        "year": doc.year,
        "ingest_method": doc.ingest_method,
        "sha256": doc.sha256,
        "claims_contributed": claims_contributed,
    }


def _gap_entry(gap: Gap) -> dict[str, Any]:
    return {
        "gap_id": gap.gap_id,
        "component_id": gap.component_id,
        "field": gap.field,
        "question": gap.question,
        "criticality": gap.criticality,
        "searched_papers": gap.searched_papers,
        "suggested_sources": gap.suggested_sources,
    }


def _provenance_stub(claim: Claim | None) -> dict[str, Any]:
    if claim is None:
        return {}
    return {
        "span_id": claim.provenance.span_id,
        "section": claim.provenance.section,
        "page": claim.provenance.page,
    }


def _resolved_ids(reconciliation: ReconciliationResult | None) -> set[str]:
    if reconciliation is None:
        return set()
    return {r.contradiction_id for r in reconciliation.resolutions if not r.is_open}


def _resolution_selecting(
    reconciliation: ReconciliationResult | None, claim_id: str
) -> Resolution | None:
    if reconciliation is None:
        return None
    for resolution in reconciliation.resolutions:
        if not resolution.is_open and resolution.selected_claim_id == claim_id:
            return resolution
    return None


def _resolved_conflicts(reconciliation: ReconciliationResult | None) -> list[dict[str, Any]]:
    """The audit trail: what was closed automatically, and on which rule."""
    if reconciliation is None:
        return []
    return [
        {
            "resolution_id": r.resolution_id,
            "contradiction_id": r.contradiction_id,
            "outcome": r.outcome,
            "selected_claim_id": r.selected_claim_id,
            "rule_fired": r.rule_fired,
            "rationale": r.rationale,
            "resolved_by": r.resolved_by,
            "resolved_at": r.resolved_at,
            "human_note": r.human_note,
        }
        for r in reconciliation.resolutions
        if not r.is_open
    ]


def _guidance(contradiction: Contradiction, resolution: Resolution | None) -> str:
    """What the choice affects, so the implementer can actually decide."""
    if resolution is not None and resolution.outcome == "DEFERRED":
        return (
            f"Policy suggests {resolution.selected_claim_id} via rule "
            f"{resolution.rule_fired!r}, but the rule is low confidence and "
            "needs your confirmation."
        )
    positions = " vs ".join(f"{p.position} ({p.paper_id})" for p in contradiction.positions)
    return (
        f"No policy rule resolved this. Choose per your target setting: {positions}. "
        "Both are implementable; the choice changes behaviour, not correctness."
    )


def _is_disputed(claim: Claim, disputed: set[tuple[str, str]]) -> bool:
    return (
        str(claim.payload.get("canonical_name")),
        normalize_condition(claim.payload.get("condition")),
    ) in disputed


def _value_key(value: Any) -> str:
    """Group by magnitude, so 0.0001 and 1e-4 are one fact, not two."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | float):
        return f"{float(value):.12g}"
    return str(value).strip().lower()
