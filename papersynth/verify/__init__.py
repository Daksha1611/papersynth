"""Verification: ClaimSet -> VerifiedClaimSet (stage 2).

A claim reaches `verified` only when no check failed. Anything that failed is
marked `rejected` and excluded from alignment, so a bad extraction cannot
become a contradiction downstream - fabricated conflicts are what erode trust
in the conflict list faster than anything else (section 9).

A `warn` does not block verification. Warnings are for values that are unusual
rather than impossible; rejecting them would discard real claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from papersynth.core.config import Settings, get_settings
from papersynth.core.document import StructuredDocument
from papersynth.core.models import Claim, ClaimSet
from papersynth.llm.base import LLMProvider
from papersynth.verify.citation_trace import TraceResult, check_numeric_literal, trace
from papersynth.verify.range_check import CheckOutcome, RangeRules

__all__ = [
    "CheckOutcome",
    "RangeRules",
    "TraceResult",
    "VerificationReport",
    "Verifier",
    "check_numeric_literal",
    "trace",
]


@dataclass
class VerificationReport:
    """The stage 2 report. Rejection reasons are the interesting part."""

    paper_id: str
    total: int = 0
    verified: int = 0
    rejected: int = 0
    warned: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def record_rejection(self, check: str, reason: str) -> None:
        self.rejected += 1
        self.rejection_reasons[check] = self.rejection_reasons.get(check, 0) + 1
        self.notes.append(f"{check}: {reason}")

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.total if self.total else 0.0


class Verifier:
    """Runs the verification suite over one paper's claims."""

    def __init__(
        self,
        *,
        provider: LLMProvider | None = None,
        range_rules: RangeRules | None = None,
        settings: Settings | None = None,
        entailment: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider
        self.range_rules = range_rules or RangeRules.load(_rules_path(self.settings))
        #: Entailment is the only check that spends a call. Off in dev keeps a
        #: full run inside one free tier's daily allowance (section 6.4.5).
        self.entailment = entailment and provider is not None

    def verify(
        self, claims: ClaimSet, doc: StructuredDocument
    ) -> tuple[ClaimSet, VerificationReport]:
        report = VerificationReport(paper_id=claims.paper_id, total=len(claims.claims))
        verified: list[Claim] = []

        for claim in claims.claims:
            verified.append(self._verify_one(claim, doc, report))

        return (
            ClaimSet(paper_id=claims.paper_id, claims=verified, warnings=claims.warnings),
            report,
        )

    def _verify_one(
        self, claim: Claim, doc: StructuredDocument, report: VerificationReport
    ) -> Claim:
        trace_result = trace(claim, doc, provider=self.provider if self.entailment else None)
        range_result = self.range_rules.check(claim)

        claim.verification.citation_trace = trace_result.outcome.result
        claim.verification.range_check = range_result.result
        claim.verification.notes = [
            note for note in (trace_result.outcome.reason, range_result.reason) if note
        ]

        if trace_result.outcome.failed:
            claim.status = "rejected"
            report.record_rejection("citation_trace", trace_result.outcome.reason)
            return claim

        if range_result.failed:
            claim.status = "rejected"
            report.record_rejection("range_check", range_result.reason)
            return claim

        if trace_result.numeric_literal.result == "warn":
            # The value was not found verbatim in its span, so it was inferred
            # or converted. It stays, but it must not silently carry the
            # authority of a directly stated value (ER-07).
            claim.payload["stated_explicitly"] = False
            claim.confidence = round(claim.confidence * 0.8, 4)

        if "warn" in (trace_result.outcome.result, range_result.result):
            report.warned += 1

        claim.status = "verified"
        report.verified += 1
        return claim


def _rules_path(settings: Settings) -> Path | None:
    path = Path(settings.range_rules)
    if path.exists():
        return path
    # Fall back to the copy bundled beside the package, so an installed wheel
    # verifies the same way a source checkout does.
    bundled = Path(__file__).resolve().parent.parent.parent / "config" / "range_rules.yaml"
    return bundled if bundled.exists() else None
