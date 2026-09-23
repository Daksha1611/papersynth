"""DEFINITION_CONFLICT detection (section 7.6).

Two papers that use the same term to mean different things. No number separates
them and no approach is adopted or rejected, so neither a value detector nor
the method detector can see it - yet it is one of the most dangerous
disagreements a synthesis can carry, because it is silent. Two agent-safety
papers both report an "attack success rate"; one counts any out-of-policy tool
call, the other only a call that reaches a harmful end state. The numbers are
not comparable and nothing in either paper says so. Code that computes one
metric while citing the other's threshold is wrong in a way no test catches.

The comparison is on the operational CRITERION, not the definition prose. Two
definitions worded differently that reduce to the same check are not in
conflict; two worded similarly that reduce to different checks are. A paper
that gives no criterion falls back to its definition text, because a conflict
that can only be seen in prose is still a conflict - it is just one a human
must read to confirm, which is why it lands as MATERIAL rather than BLOCKING.

Primacy is the one signal that can resolve these automatically: the paper that
introduced a term defines it, and a later paper restating it is secondary. That
rule lives in the policy (`prefer_primary_source`) and fires only when primacy
is actually known; until a citation-graph signal exists it does not, and these
escalate. This detector therefore permits auto-resolution rather than
forbidding it - the restraint belongs in the policy, where it is visible and
configurable, not hard-coded here.
"""

from __future__ import annotations

from collections import defaultdict

from rapidfuzz import fuzz

from papersynth.contradict.severity import specificity
from papersynth.core import ids
from papersynth.core.models import (
    Claim,
    ConceptCluster,
    ConceptGraph,
    Contradiction,
    Criticality,
    Position,
    Support,
)

DETECTOR_VERSION = "definition_conflict_detector@1.0.0"

#: Above this token-set similarity, two criteria describe the same check worded
#: differently rather than two different checks. The same constant and the same
#: reasoning as SAME_APPROACH_RATIO in method_conflict: extraction runs over
#: section batches and one paper states one criterion in slightly varying words.
SAME_CRITERION_RATIO = 88


class DefinitionConflictDetector:
    conflict_type = "DEFINITION_CONFLICT"
    claim_type = "definition"
    #: True, so the policy may resolve these when primacy is known. The default
    #: policy escalates them until a primacy signal exists; the restraint is in
    #: the policy, not here. Contrast METHOD_CONFLICT, which is never
    #: auto-resolvable because choosing an approach is always an engineering
    #: decision - whereas a definition has a right answer when its author is
    #: known.
    auto_resolvable = True
    version = DETECTOR_VERSION

    def scan(self, cluster: ConceptCluster, graph: ConceptGraph) -> list[Contradiction]:
        if cluster.concept_type != "definition" or not cluster.is_multi_paper:
            return []

        claims = [
            c
            for c in graph.claims_in(cluster)
            if c.status == "verified" and c.payload.get("attribution", "own") == "own"
        ]
        if len(claims) < 2:
            return []

        out: list[Contradiction] = []
        for condition, group in sorted(_group_by_condition(claims).items()):
            found = self._scan_group(cluster, condition, group)
            if found is not None:
                out.append(found)
        return out

    def _scan_group(
        self, cluster: ConceptCluster, condition: str, claims: list[Claim]
    ) -> Contradiction | None:
        if len({c.paper_id for c in claims}) < 2:
            return None

        criteria = _canonical_criteria(claims)
        distinct = {criteria[c.claim_id] for c in claims}
        if len(distinct) < 2:
            # Same meaning under the same term: agreement, not a conflict. This
            # is the common and desirable case and must not be reported.
            return None

        positions = _positions(claims, criteria)
        if len({p.paper_id for p in positions}) < 2:
            return None

        return Contradiction(
            contradiction_id=ids.contradiction_id(
                cluster.cluster_id, self.conflict_type, [p.claim_id for p in positions]
            ),
            cluster_id=cluster.cluster_id,
            type="DEFINITION_CONFLICT",
            severity=_severity(claims),
            description=_describe(cluster.canonical_name, condition, len(distinct)),
            positions=positions,
            detected_by=self.version,
        )


def _describe(term: str, condition: str, count: int) -> str:
    scope = f" under {condition!r}" if condition else ""
    return (
        f"Papers define {term!r}{scope} incompatibly ({count} distinct criteria): "
        "a metric or check built from one does not measure what the other measures."
    )


def _severity(claims: list[Claim]) -> Criticality:
    """Operational definitions block; prose-only ones are material.

    When the conflicting definitions each reduce to a concrete criterion, code
    computes genuinely different quantities and you cannot write the evaluation
    harness without deciding which - that is BLOCKING by the section 7.6 ladder.
    A conflict visible only in the definition prose, with no operational
    criterion extracted, still matters but needs a human to read both before it
    is even certain to be real, so it escalates as MATERIAL.
    """
    if any(c.payload.get("criterion") for c in claims):
        return "BLOCKING"
    return "MATERIAL"


def _positions(claims: list[Claim], criteria: dict[str, str]) -> list[Position]:
    """One position per distinct criterion, earliest claim id representing it."""
    by_criterion: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        by_criterion[criteria[claim.claim_id]].append(claim)

    positions = []
    for _, group in sorted(by_criterion.items()):
        claim = sorted(group, key=lambda c: c.claim_id)[0]
        payload = claim.payload
        rendered = str(payload.get("criterion") or payload.get("definition", "")).strip()
        positions.append(
            Position(
                claim_id=claim.claim_id,
                paper_id=claim.paper_id,
                position=rendered,
                support=Support(
                    specificity=specificity(payload),
                    stated_explicitly=bool(payload.get("stated_explicitly", True)),
                    has_condition=bool(payload.get("condition")),
                ),
            )
        )
    return positions


def _comparable_text(claim: Claim) -> str:
    """The text two definitions are compared through.

    The operational criterion when the paper gave one, because that is what code
    checks; the definition prose otherwise, so a paper that defined a term
    loosely is still compared rather than silently treated as matching
    everything.
    """
    payload = claim.payload
    text = payload.get("criterion") or payload.get("definition", "")
    return str(text).strip().lower()


def _canonical_criteria(claims: list[Claim]) -> dict[str, str]:
    """Map each claim id to a shared criterion key.

    Criteria whose text is near-identical collapse onto one key, chosen as the
    alphabetically first member's comparable text so the result does not depend
    on claim order. Mirrors _canonical_approaches in method_conflict: the same
    union-find over a token-set similarity, because the same extraction-batch
    wording drift produces the same spurious near-duplicates here.
    """
    text = {c.claim_id: _comparable_text(c) for c in claims}
    parent: dict[str, str] = {cid: cid for cid in text}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    ids_sorted = sorted(text)
    for i, a in enumerate(ids_sorted):
        for b in ids_sorted[i + 1 :]:
            if (
                text[a]
                and text[b]
                and fuzz.token_set_ratio(text[a], text[b]) >= SAME_CRITERION_RATIO
            ):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    return {cid: text[find(cid)] or find(cid) for cid in ids_sorted}


def _group_by_condition(claims: list[Claim]) -> dict[str, list[Claim]]:
    from papersynth.contradict.detectors.value_conflict import normalize_condition

    grouped: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        grouped[normalize_condition(claim.payload.get("condition"))].append(claim)
    return grouped
