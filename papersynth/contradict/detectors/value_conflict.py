"""VALUE_CONFLICT detection (section 8.5).

The condition grouping is the whole point of this detector. Two papers giving
different learning rates for different model sizes are not in conflict; they
are two scoped facts, and reporting them as a disagreement wastes review time
on a non-problem. ER-04 makes this a rule rather than a nicety.

Condition matching is exact on the normalized string, which is deliberately
conservative. "base model" and "base model, WMT14 EN-DE" are treated as
different scopes, so a genuine conflict between them is missed rather than a
false one invented. Section 9 is explicit about that trade: a false
contradiction burns reviewer time and erodes trust in the whole list, while a
missed one is recoverable.
"""

from __future__ import annotations

from collections import defaultdict

from papersynth.contradict.severity import specificity, value_conflict_severity
from papersynth.core import ids
from papersynth.core.models import (
    Claim,
    ConceptCluster,
    ConceptGraph,
    Contradiction,
    Position,
    Support,
)

DETECTOR_VERSION = "value_conflict_detector@1.0.0"

#: Venues whose review process we treat as peer review for policy purposes.
_PEER_REVIEWED = frozenset(
    {
        "neurips",
        "nips",
        "icml",
        "iclr",
        "aaai",
        "acl",
        "emnlp",
        "naacl",
        "cvpr",
        "iccv",
        "eccv",
        "colm",
        "tmlr",
        "jmlr",
        "coling",
        "sigir",
        "kdd",
        "www",
    }
)


class ValueConflictDetector:
    conflict_type = "VALUE_CONFLICT"
    auto_resolvable = True
    version = DETECTOR_VERSION

    def scan(self, cluster: ConceptCluster, graph: ConceptGraph) -> list[Contradiction]:
        if not cluster.is_multi_paper:
            return []

        claims = [c for c in graph.claims_in(cluster) if c.status == "verified"]
        if len(claims) < 2:
            return []

        out: list[Contradiction] = []
        for (condition, unit), group in sorted(_group_by_scope(claims).items()):
            contradiction = self._scan_condition(cluster, condition, unit, group)
            if contradiction is not None:
                out.append(contradiction)
        return out

    def _scan_condition(
        self, cluster: ConceptCluster, condition: str, unit: str, claims: list[Claim]
    ) -> Contradiction | None:
        # A disagreement needs at least two papers. One paper stating two
        # values under one condition is an intra-paper inconsistency, which is
        # addressed to the reader rather than resolved by policy (section 10.1).
        if len({c.paper_id for c in claims}) < 2:
            return None

        by_value: dict[str, list[Claim]] = defaultdict(list)
        for claim in claims:
            by_value[_normalize(claim.payload.get("value"))].append(claim)

        if len(by_value) < 2:
            return None

        positions = [
            _position(sorted(group, key=lambda c: c.claim_id)[0])
            for _, group in sorted(by_value.items())
        ]
        values = [g[0].payload.get("value") for _, g in sorted(by_value.items())]
        severity = value_conflict_severity(values, cluster.canonical_name)

        scope = f" under {condition!r}" if condition else ""
        if unit:
            scope += f" (in {unit})"
        rendered = ", ".join(str(v) for v in values)
        n_papers = len({p.paper_id for p in positions})
        subject = (
            f"{len(positions)} positions across {n_papers} papers"
            if len(positions) != n_papers
            else f"{n_papers} papers"
        )

        return Contradiction(
            contradiction_id=ids.contradiction_id(
                cluster.cluster_id, self.conflict_type, [p.claim_id for p in positions]
            ),
            cluster_id=cluster.cluster_id,
            type="VALUE_CONFLICT",
            severity=severity,
            description=(
                f"{subject} specify different values for "
                f"{cluster.canonical_name}{scope}: {rendered}."
            ),
            positions=positions,
            detected_by=self.version,
        )


def _group_by_scope(claims: list[Claim]) -> dict[tuple[str, str], list[Claim]]:
    """Group by (condition, unit). Both must match before values are compared.

    Unit matters as much as condition and was missed at first. BERT states its
    batch size twice - 256 sequences and 128,000 words - which is one fact in
    two units, since 256 x 512 = 128,000. Compared as bare numbers they look
    like a stark disagreement, and the run reported it as BLOCKING.
    """
    grouped: dict[tuple[str, str], list[Claim]] = defaultdict(list)
    for claim in claims:
        scope = (
            normalize_condition(claim.payload.get("condition")),
            normalize_condition(claim.payload.get("unit")),
        )
        grouped[scope].append(claim)
    return grouped


def normalize_condition(condition: object) -> str:
    if not isinstance(condition, str):
        return ""
    return " ".join(condition.lower().split()).strip(" .,;")


def _normalize(value: object) -> str:
    """Compare values by magnitude, not by how they were typed.

    0.0001 and 1e-4 are the same number, and reporting them as a conflict would
    be a pure artifact of formatting.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | float):
        return f"{float(value):.12g}"
    if isinstance(value, str):
        return " ".join(value.lower().split())
    return str(value)


def _position(claim: Claim) -> Position:
    payload = claim.payload
    rendered = str(payload.get("value"))
    if payload.get("unit"):
        rendered += f" {payload['unit']}"
    if payload.get("stated_explicitly") is False:
        rendered += " (inferred, not stated)"

    return Position(
        claim_id=claim.claim_id,
        paper_id=claim.paper_id,
        position=rendered,
        support=Support(
            specificity=specificity(payload),
            stated_explicitly=bool(payload.get("stated_explicitly", True)),
            has_condition=bool(payload.get("condition")),
        ),
    )


def attach_paper_support(
    contradiction: Contradiction,
    papers: dict[str, tuple[str | None, int | None]],
    primary: dict[str, str] | None = None,
) -> Contradiction:
    """Fill venue, year, and primacy onto each position.

    Kept separate from detection because it needs corpus metadata the detector
    has no business reading, and because these fields drive policy rules that
    must be auditable independently of how the conflict was found.
    """
    primary = primary or {}
    for position in contradiction.positions:
        venue, year = papers.get(position.paper_id, (None, None))
        position.support.venue = venue
        position.support.year = year
        position.support.peer_reviewed = _is_peer_reviewed(venue)
        position.support.is_primary = primary.get(contradiction.cluster_id) == position.paper_id
    return contradiction


def _is_peer_reviewed(venue: str | None) -> bool:
    if not venue:
        return False
    lowered = venue.lower()
    return any(token in lowered for token in _PEER_REVIEWED)
