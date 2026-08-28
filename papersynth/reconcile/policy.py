"""Declarative reconciliation policy (section 8.5).

Rules are evaluated in order; the first firing predicate decides. When none
fires, the fallback applies, and the fallback is always ESCALATED - the engine
refuses to construct any other default (DD-03).

Predicates are code rather than prose so that a resolution is deterministic and
reproducible. Which rules exist, their order, their actions, and their
thresholds stay in YAML, which is what R-10 actually needs: a policy whose bias
can be inspected and changed without touching the engine, and where every
auto-resolution names the rule that produced it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from papersynth.core import ids
from papersynth.core.errors import PaperSynthError
from papersynth.core.models import (
    Contradiction,
    Outcome,
    Position,
    ReconciliationResult,
    Resolution,
)

#: A predicate answers: do these positions satisfy this rule's condition?
Predicate = Callable[[Contradiction, dict[str, Any]], bool]
#: A selector answers: given that it fired, which position wins?
Selector = Callable[[Contradiction], Position | None]

_SEVERITY_RANK = {"COSMETIC": 0, "MATERIAL": 1, "BLOCKING": 2}


# ---------------------------------------------------------------------------
# Predicate vocabulary
# ---------------------------------------------------------------------------


def _always(contradiction: Contradiction, params: dict[str, Any]) -> bool:
    return True


def _exactly_one_scoped(contradiction: Contradiction, params: dict[str, Any]) -> bool:
    scoped = [p for p in contradiction.positions if _is_scoped(p)]
    return len(scoped) == 1 and len(contradiction.positions) > 1


def _specificity_gap(contradiction: Contradiction, params: dict[str, Any]) -> bool:
    min_gap = float(params.get("min_gap", 0.3))
    scores = sorted((p.support.specificity for p in contradiction.positions), reverse=True)
    if len(scores) < 2:
        return False
    return (scores[0] - scores[1]) >= min_gap


def _exactly_one_primary(contradiction: Contradiction, params: dict[str, Any]) -> bool:
    return sum(1 for p in contradiction.positions if p.support.is_primary) == 1


def _year_gap_and_recent_peer_reviewed(
    contradiction: Contradiction, params: dict[str, Any]
) -> bool:
    min_years = int(params.get("min_years", 3))
    dated = [p for p in contradiction.positions if p.support.year is not None]
    if len(dated) < 2:
        return False

    newest = max(dated, key=lambda p: p.support.year or 0)
    oldest = min(dated, key=lambda p: p.support.year or 0)
    if (newest.support.year or 0) - (oldest.support.year or 0) < min_years:
        return False
    # Ties on the newest year mean there is no single "more recent" position.
    if sum(1 for p in dated if p.support.year == newest.support.year) > 1:
        return False
    return newest.support.peer_reviewed


def _severity_at_least_material(contradiction: Contradiction, params: dict[str, Any]) -> bool:
    return _SEVERITY_RANK.get(contradiction.severity, 0) >= _SEVERITY_RANK["MATERIAL"]


def _severity_is_cosmetic(contradiction: Contradiction, params: dict[str, Any]) -> bool:
    return contradiction.severity == "COSMETIC"


PREDICATES: dict[str, Predicate] = {
    "always": _always,
    "exactly_one_scoped": _exactly_one_scoped,
    "specificity_gap": _specificity_gap,
    "exactly_one_primary": _exactly_one_primary,
    "year_gap_and_recent_peer_reviewed": _year_gap_and_recent_peer_reviewed,
    "severity_at_least_material": _severity_at_least_material,
    "severity_is_cosmetic": _severity_is_cosmetic,
}


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------


def _only_scoped(contradiction: Contradiction) -> Position | None:
    scoped = [p for p in contradiction.positions if _is_scoped(p)]
    return scoped[0] if len(scoped) == 1 else None


def _highest_specificity(contradiction: Contradiction) -> Position | None:
    return max(
        contradiction.positions,
        key=lambda p: (p.support.specificity, p.claim_id),
        default=None,
    )


def _primary(contradiction: Contradiction) -> Position | None:
    primaries = [p for p in contradiction.positions if p.support.is_primary]
    return primaries[0] if len(primaries) == 1 else None


def _most_recent(contradiction: Contradiction) -> Position | None:
    dated = [p for p in contradiction.positions if p.support.year is not None]
    return max(dated, key=lambda p: (p.support.year or 0, p.claim_id), default=None)


def _first_by_claim_id(contradiction: Contradiction) -> Position | None:
    return min(contradiction.positions, key=lambda p: p.claim_id, default=None)


SELECTORS: dict[str, Selector] = {
    "only_scoped": _only_scoped,
    "highest_specificity": _highest_specificity,
    "primary": _primary,
    "most_recent": _most_recent,
    "first_by_claim_id": _first_by_claim_id,
}


def _is_scoped(position: Position) -> bool:
    """Whether this position states an explicit condition.

    Reads the flag the extractor set rather than a derived specificity score.
    Thresholding the score made "scoped" mean 0.7 and "global" mean 0.6 - a
    0.1 margin assembled from unrelated signals, deciding a conflict that
    neither position was actually scoped for.
    """
    return position.support.has_condition


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyRule:
    id: str
    applies_to: tuple[str, ...]
    when: str
    action: Outcome
    select: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    confidence: str = "medium"
    requires_human_confirmation: bool = False
    description: str = ""

    def matches(self, contradiction: Contradiction) -> bool:
        if contradiction.type not in self.applies_to:
            return False
        predicate = PREDICATES.get(self.when)
        if predicate is None:
            return False
        return predicate(contradiction, self.params)


@dataclass
class Policy:
    """An ordered rule set with a mandatory ESCALATED fallback."""

    policy_version: str = "0.0.0"
    rules: tuple[PolicyRule, ...] = ()

    @classmethod
    def load(cls, path: Path | str) -> Policy:
        path = Path(path)
        if not path.exists():
            raise PaperSynthError(
                f"No reconciliation policy at {path}. Every auto-resolution must "
                "name the rule that produced it, so the engine will not run "
                "without one."
            )
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        rules = []
        for entry in payload.get("rules", []):
            when = entry.get("when")
            if when not in PREDICATES:
                raise PaperSynthError(
                    f"Rule {entry.get('id')!r} uses unknown predicate {when!r}. "
                    f"Available: {', '.join(sorted(PREDICATES))}"
                )
            select = entry.get("select")
            if select is not None and select not in SELECTORS:
                raise PaperSynthError(
                    f"Rule {entry.get('id')!r} uses unknown selector {select!r}. "
                    f"Available: {', '.join(sorted(SELECTORS))}"
                )
            rules.append(
                PolicyRule(
                    id=str(entry["id"]),
                    applies_to=tuple(entry.get("applies_to", [])),
                    when=str(when),
                    action=entry.get("action", "ESCALATED"),
                    select=select,
                    params=entry.get("params") or {},
                    confidence=str(entry.get("confidence", "medium")),
                    requires_human_confirmation=bool(
                        entry.get("requires_human_confirmation", False)
                    ),
                    description=" ".join(str(entry.get("description", "")).split()),
                )
            )

        # The fallback is deliberately not read from the file. Making it
        # configurable would let a policy edit silently turn "ask a human" into
        # "guess", which is the one behaviour this system must never have.
        declared = (payload.get("fallback") or {}).get("action", "ESCALATED")
        if declared != "ESCALATED":
            raise PaperSynthError(
                f"Policy declares fallback action {declared!r}. The fallback is "
                "always ESCALATED (DD-03): an unresolved conflict surfaced to a "
                "human is a correct output, a silently chosen wrong value is a "
                "defect that surfaces months later as an unreproducible result."
            )

        return cls(policy_version=str(payload.get("policy_version", "0.0.0")), rules=tuple(rules))


class PolicyEngine:
    """Applies a policy to contradictions, resolving only what rules justify."""

    def __init__(self, policy: Policy, *, auto_resolvable: dict[str, bool] | None = None) -> None:
        self.policy = policy
        #: Detectors may forbid their own conflicts from auto-resolving
        #: regardless of what a rule says (section 10.3).
        self.auto_resolvable = auto_resolvable or {}

    def resolve(self, contradictions: list[Contradiction]) -> ReconciliationResult:
        return ReconciliationResult(
            policy_version=self.policy.policy_version,
            resolutions=[self.resolve_one(c) for c in contradictions],
        )

    def resolve_one(self, contradiction: Contradiction) -> Resolution:
        for rule in self.policy.rules:
            if not rule.matches(contradiction):
                continue
            return self._apply(rule, contradiction)
        return self._escalate(contradiction, "no rule fired; fallback ESCALATED", rule_fired=None)

    def _apply(self, rule: PolicyRule, contradiction: Contradiction) -> Resolution:
        if rule.action == "ESCALATED":
            return self._escalate(contradiction, rule.description or rule.id, rule_fired=rule.id)

        if contradiction.severity == "BLOCKING":
            # A BLOCKING conflict is defined as one where correct code cannot
            # be written without deciding. Closing that on a heuristic defeats
            # the gate it exists to raise.
            #
            # Observed on BERT/RoBERTa/ALBERT: batch_size 256 against 8000 was
            # auto-resolved to BERT's 256 because RoBERTa's figure had been
            # flagged "inferred", which discards RoBERTa's central finding on
            # an extraction artifact. learning_rate was resolved on a 0.7
            # against 0.6 specificity margin, which is not a reason to prefer
            # one architecture's rate over another's.
            #
            # The rule that fired is still recorded, so the reviewer sees what
            # the policy would have chosen and why (DD-03).
            return self._escalate(
                contradiction,
                f"rule {rule.id!r} would have selected a position, but a BLOCKING "
                "conflict is one where correct code cannot be written without "
                "deciding - that decision is a human's",
                rule_fired=rule.id,
            )

        if not self.auto_resolvable.get(contradiction.type, True):
            return self._escalate(
                contradiction,
                f"{contradiction.type} is never auto-resolved by its detector, "
                f"so rule {rule.id!r} does not apply",
                rule_fired=rule.id,
            )

        selector = SELECTORS.get(rule.select or "")
        selected = selector(contradiction) if selector else None
        if selected is None:
            return self._escalate(
                contradiction,
                f"rule {rule.id!r} fired but its selector chose no position",
                rule_fired=rule.id,
            )

        # ER-07. A value read off a figure cannot decide a conflict, however
        # well the rule's other conditions fit.
        if not selected.support.stated_explicitly:
            return self._escalate(
                contradiction,
                f"rule {rule.id!r} selected a claim that was inferred rather "
                "than stated; such a claim cannot auto-resolve a conflict (ER-07)",
                rule_fired=rule.id,
            )

        if rule.requires_human_confirmation:
            # Deferred is still open. The rule's suggestion is recorded so the
            # reviewer sees the reasoning, but nothing is decided.
            return Resolution(
                resolution_id=ids.resolution_id(contradiction.contradiction_id),
                contradiction_id=contradiction.contradiction_id,
                outcome="DEFERRED",
                selected_claim_id=selected.claim_id,
                rule_fired=rule.id,
                rationale=(
                    f"{rule.description or rule.id} Suggests {selected.claim_id} "
                    f"({selected.position}), but this rule requires human confirmation."
                ),
                resolved_by="policy",
            )

        return Resolution(
            resolution_id=ids.resolution_id(contradiction.contradiction_id),
            contradiction_id=contradiction.contradiction_id,
            outcome=rule.action,
            selected_claim_id=selected.claim_id,
            rule_fired=rule.id,
            rationale=(
                f"{rule.description or rule.id} Selected {selected.claim_id} "
                f"from {selected.paper_id} ({selected.position})."
            ),
            resolved_by="policy",
        )

    def _escalate(
        self, contradiction: Contradiction, reason: str, *, rule_fired: str | None
    ) -> Resolution:
        return Resolution(
            resolution_id=ids.resolution_id(contradiction.contradiction_id),
            contradiction_id=contradiction.contradiction_id,
            outcome="ESCALATED",
            selected_claim_id=None,
            rule_fired=rule_fired,
            rationale=reason,
            resolved_by="policy",
        )
