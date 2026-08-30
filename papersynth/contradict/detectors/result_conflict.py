"""RESULT_CONFLICT detection (sections 7.6, 10.3, ER-06).

Two papers reporting different numbers for the same measurement. These are
never auto-resolved and the detector says so in code, not only in policy:
picking one silently discards a real finding, and a benchmark disagreement is
frequently the most interesting thing a corpus contains.

Almost all the work here is refusing to report. ER-06 makes it a rule that
results measured under different conditions are distinct claims rather than
conflicting ones, and benchmark numbers are unusually easy to compare wrongly:

  DIFFERENT SPLIT    a dev score and a test score differ by design
  DIFFERENT VARIANT  a base model scoring below a large one is expected
  DIFFERENT DATASET  the numbers are not on the same scale at all
  DIFFERENT PROTOCOL beam size and length penalty move BLEU on their own

So the grouping key is every field that scopes the measurement, and a
difference is only a conflict inside an identical group. That is far stricter
than VALUE_CONFLICT, deliberately: a false result conflict sends a reviewer to
adjudicate two numbers that were never comparable.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from papersynth.contradict.severity import specificity
from papersynth.core import ids
from papersynth.core.models import (
    Claim,
    ConceptCluster,
    ConceptGraph,
    Contradiction,
    Position,
    Support,
)

DETECTOR_VERSION = "result_conflict_detector@1.0.0"

#: Relative difference below which two scores are the same number reported to
#: different precision - 27.30 against 27.3 - rather than a disagreement.
ROUNDING_TOLERANCE = 1e-6


class ResultConflictDetector:
    conflict_type = "RESULT_CONFLICT"
    claim_type = "result"
    #: Section 10.3. The policy honours this flag, so no rule can auto-resolve
    #: a result conflict however well its conditions happen to fit.
    auto_resolvable = False
    version = DETECTOR_VERSION

    def scan(self, cluster: ConceptCluster, graph: ConceptGraph) -> list[Contradiction]:
        if cluster.concept_type != "result" or not cluster.is_multi_paper:
            return []

        claims = [c for c in graph.claims_in(cluster) if c.status == "verified"]
        if len(claims) < 2:
            return []

        out: list[Contradiction] = []
        for scope, group in sorted(_group_by_scope(claims).items(), key=lambda kv: str(kv[0])):
            found = self._scan_group(cluster, scope, group)
            if found is not None:
                out.append(found)
        return out

    def _scan_group(
        self, cluster: ConceptCluster, scope: tuple[Any, ...], claims: list[Claim]
    ) -> Contradiction | None:
        if len({c.paper_id for c in claims}) < 2:
            return None

        distinct = _distinct_values(claims)
        if len(distinct) < 2:
            return None

        if _all_intervals_overlap(distinct):
            # The papers report ranges that include each other. Agreement
            # within stated uncertainty is agreement, and calling it a conflict
            # would ask a reviewer to adjudicate noise.
            return None

        positions = [_position(claim) for _, claim in sorted(distinct, key=lambda p: p[1].claim_id)]
        values = ", ".join(
            _render_value(claim) for _, claim in sorted(distinct, key=lambda p: p[1].claim_id)
        )

        return Contradiction(
            contradiction_id=ids.contradiction_id(
                cluster.cluster_id, self.conflict_type, [p.claim_id for p in positions]
            ),
            cluster_id=cluster.cluster_id,
            type="RESULT_CONFLICT",
            # Never BLOCKING. A disagreement about a reported score does not
            # stop anyone writing the code; it stops them knowing whether the
            # code is right, which is a reproduction target rather than an
            # implementation decision (section 7.6).
            severity="MATERIAL",
            description=(
                f"Papers report different {cluster.canonical_name} "
                f"{_describe_scope(claims[0])}: {values}."
            ),
            positions=positions,
            detected_by=self.version,
        )


def _group_by_scope(claims: list[Claim]) -> dict[tuple[Any, ...], list[Claim]]:
    """Group by everything that scopes a measurement (ER-06)."""
    grouped: dict[tuple[Any, ...], list[Claim]] = defaultdict(list)
    for claim in claims:
        payload = claim.payload
        grouped[
            (
                _norm(payload.get("dataset")),
                _norm(payload.get("split")),
                _norm(payload.get("model_variant")),
                _conditions_key(payload.get("conditions")),
            )
        ].append(claim)
    return grouped


def _conditions_key(conditions: Any) -> str:
    """Protocol details as a stable string.

    Compared exactly rather than fuzzily. Beam size and length penalty move a
    BLEU score on their own, so two numbers produced under different decoding
    settings are different measurements even when everything else matches.
    """
    if not isinstance(conditions, dict) or not conditions:
        return ""
    return json.dumps({str(k): conditions[k] for k in sorted(conditions)}, sort_keys=True)


def _norm(value: Any) -> str:
    return " ".join(str(value).lower().split()) if isinstance(value, str) else ""


def _distinct_values(claims: list[Claim]) -> list[tuple[float, Claim]]:
    """One representative claim per distinct value."""
    by_value: dict[str, tuple[float, Claim]] = {}
    for claim in sorted(claims, key=lambda c: c.claim_id):
        raw = claim.payload.get("value")
        if not isinstance(raw, int | float) or isinstance(raw, bool):
            continue
        value = float(raw)
        key = f"{value:.10g}"
        by_value.setdefault(key, (value, claim))

    values = list(by_value.values())
    if len(values) < 2:
        return values

    # Collapse values differing only by reporting precision.
    collapsed: list[tuple[float, Claim]] = []
    for value, claim in sorted(values, key=lambda pair: pair[0]):
        if collapsed and _close(collapsed[-1][0], value):
            continue
        collapsed.append((value, claim))
    return collapsed


def _all_intervals_overlap(values: list[tuple[float, Claim]]) -> bool:
    """Whether every reported interval contains every other reported value.

    Requires at least one paper to have stated a variance; without any, there
    is no basis for calling a difference noise.
    """
    if not any(_variance(claim) is not None for _, claim in values):
        return False

    for value, claim in values:
        spread = _variance(claim)
        if spread is None:
            return False
        for other, _ in values:
            if abs(other - value) > spread:
                return False
    return True


def _variance(claim: Claim) -> float | None:
    raw = claim.payload.get("reported_variance")
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        return abs(float(raw))
    return None


def _position(claim: Claim) -> Position:
    return Position(
        claim_id=claim.claim_id,
        paper_id=claim.paper_id,
        position=_render_value(claim),
        support=Support(
            specificity=specificity(claim.payload),
            stated_explicitly=bool(claim.payload.get("stated_explicitly", True)),
            has_condition=bool(claim.payload.get("conditions")),
        ),
    )


def _render_value(claim: Claim) -> str:
    payload = claim.payload
    rendered = f"{payload.get('value')}"
    spread = _variance(claim)
    if spread is not None:
        rendered += f" +/- {spread}"
    if payload.get("stated_explicitly") is False:
        rendered += " (read from a figure)"
    return rendered


def _describe_scope(claim: Claim) -> str:
    """Render the scope as the paper wrote it.

    Grouping lowercases so that "WMT14" and "wmt14" are one benchmark, but the
    description is read by a person: showing them "wmt14 en-de" when the paper
    says "WMT14 EN-DE" makes a proper noun look like a typo.
    """
    payload = claim.payload
    parts = []
    if payload.get("dataset"):
        parts.append(f"on {payload['dataset']}")
    if payload.get("split"):
        parts.append(f"({payload['split']})")
    if payload.get("model_variant"):
        parts.append(f"for the {payload['model_variant']} model")
    if payload.get("conditions"):
        rendered = ", ".join(
            f"{k}={payload['conditions'][k]}" for k in sorted(payload["conditions"])
        )
        parts.append(f"under {rendered}")
    return " ".join(parts) if parts else "under identical conditions"


def _close(a: float, b: float) -> bool:
    scale = max(abs(a), abs(b))
    return abs(a - b) <= ROUNDING_TOLERANCE * scale if scale else True
