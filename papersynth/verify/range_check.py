"""Unit and plausibility checking (section 8.3.3).

Rules are declarative and live in config/range_rules.yaml, so adding one needs
no code change (NFR-07).

The split between hard and soft ranges carries the whole judgement of this
module. A hard violation - dropout of 1.7, a negative batch size - is almost
never something a paper actually said; it is a decimal point lost to OCR or a
value read from the wrong table column, so the claim is rejected. A soft
violation is unusual but legitimate often enough that rejecting it would
discard real claims, so it warns and the claim survives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from papersynth.core.models import CheckResult, Claim


@dataclass(frozen=True)
class RangeRule:
    canonical_name: str
    type: str | None = None
    hard_range: tuple[float, float] | None = None
    soft_range: tuple[float, float] | None = None
    must_be_positive: bool = False


@dataclass
class CheckOutcome:
    """Result of one verification check on one claim."""

    result: CheckResult
    reason: str = ""

    @property
    def failed(self) -> bool:
        return self.result == "fail"


@dataclass
class RangeRules:
    version: str = "0.0.0"
    rules: dict[str, RangeRule] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str | None = None) -> RangeRules:
        """Load rules from YAML. A missing file disables the check rather than
        failing the run - the checker is a safety net, not a prerequisite."""
        if path is None:
            return cls()
        path = Path(path)
        if not path.exists():
            return cls()

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rules: dict[str, RangeRule] = {}
        for entry in payload.get("rules", []):
            name = entry.get("canonical_name")
            if not name:
                continue
            rules[name] = RangeRule(
                canonical_name=name,
                type=entry.get("type"),
                hard_range=_pair(entry.get("hard_range")),
                soft_range=_pair(entry.get("soft_range")),
                must_be_positive=bool(entry.get("must_be_positive", False)),
            )
        return cls(version=str(payload.get("version", "0.0.0")), rules=rules)

    def check(self, claim: Claim) -> CheckOutcome:
        """Apply the matching rule, if any."""
        if claim.type != "hyperparameter":
            return CheckOutcome("n/a")

        name = claim.payload.get("canonical_name")
        rule = self.rules.get(str(name))
        if rule is None:
            # No rule is not a failure. The rule set covers common
            # hyperparameters, not every value a paper might configure.
            return CheckOutcome("n/a", f"no range rule for {name!r}")

        value = claim.payload.get("value")

        # Categorical values ("cosine", "Adam") have no numeric range to check.
        if isinstance(value, bool) or not isinstance(value, int | float):
            if rule.type in ("float", "int"):
                return CheckOutcome(
                    "fail",
                    f"{name} should be {rule.type} but got {type(value).__name__} {value!r}",
                )
            return CheckOutcome("n/a")

        if rule.must_be_positive and value <= 0:
            return CheckOutcome("fail", f"{name} must be positive, got {value}")

        if rule.hard_range is not None:
            low, high = rule.hard_range
            if not (low <= value <= high):
                return CheckOutcome(
                    "fail",
                    f"{name}={value} is outside the plausible range [{low}, {high}]; "
                    "this is far more likely an extraction error than a paper error",
                )

        if rule.soft_range is not None:
            low, high = rule.soft_range
            if not (low <= value <= high):
                return CheckOutcome(
                    "warn",
                    f"{name}={value} is unusual (typical range [{low}, {high}]) "
                    "but not impossible; kept for review",
                )

        return CheckOutcome("pass")


def _pair(raw: Any) -> tuple[float, float] | None:
    if not isinstance(raw, list | tuple) or len(raw) != 2:
        return None
    try:
        return (float(raw[0]), float(raw[1]))
    except (TypeError, ValueError):
        return None
