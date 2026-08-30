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
from papersynth.verify.citation_trace import (
    TraceResult,
    check_entailment_batch,
    check_numeric_literal,
    trace,
)
from papersynth.verify.internal_consistency import ScopeFinding
from papersynth.verify.internal_consistency import review as review_consistency
from papersynth.verify.range_check import CheckOutcome, RangeRules
from papersynth.verify.symbol_check import symbol_check

__all__ = [
    "CheckOutcome",
    "RangeRules",
    "ScopeFinding",
    "TraceResult",
    "VerificationReport",
    "Verifier",
    "check_numeric_literal",
    "review_consistency",
    "symbol_check",
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
    #: Passed every check but did not clear the confidence threshold, usually
    #: because re-extractions disagreed. Neither verified nor rejected.
    low_confidence: int = 0
    #: Sections whose several distinct quantities were scoped together rather
    #: than reported as inconsistent globals (section 10.1).
    scoped_sections: int = 0
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

        # Entailment is judged in batches before the per-claim checks, so the
        # expensive part costs ceil(n / batch_size) calls rather than n. The
        # cheap deterministic checks then run per claim as before.
        entailments = self._batch_entailments(claims.claims, doc)

        verified = [
            self._verify_one(claim, doc, report, entailments.get(claim.claim_id))
            for claim in claims.claims
        ]

        for finding in review_consistency(verified):
            report.scoped_sections += 1
            report.notes.append(finding.note)
            for claim in verified:
                if claim.claim_id in finding.claim_ids:
                    claim.verification.notes.append(
                        f"scope-bound to section {finding.section!r} (section 10.1)"
                    )

        return (
            ClaimSet(paper_id=claims.paper_id, claims=verified, warnings=claims.warnings),
            report,
        )

    def _batch_entailments(
        self, claims: list[Claim], doc: StructuredDocument
    ) -> dict[str, CheckOutcome]:
        """Judge every claim whose span resolves, in batches.

        A claim whose span does not resolve is skipped rather than batched: it
        fails citation_trace on the deterministic check anyway, and spending
        judge tokens on text that could not be located would be paying to
        confirm something already known.
        """
        if not self.entailment or self.provider is None:
            return {}

        pairs = []
        for claim in claims:
            span = doc.resolve_span(claim.provenance.span_id, char_end=claim.provenance.char_end)
            if span is not None:
                pairs.append((claim, span.text))

        outcomes: dict[str, CheckOutcome] = {}
        size = max(1, self.settings.verify_batch_size)
        for start in range(0, len(pairs), size):
            outcomes |= check_entailment_batch(pairs[start : start + size], self.provider)
        return outcomes

    def _verify_one(
        self,
        claim: Claim,
        doc: StructuredDocument,
        report: VerificationReport,
        entailment: CheckOutcome | None = None,
    ) -> Claim:
        # The entailment verdict comes from the batch when one ran, so trace()
        # is asked only for the deterministic checks.
        trace_result = trace(claim, doc, provider=None, entailment=entailment)
        range_result = self.range_rules.check(claim)
        symbol_result = symbol_check(claim)

        claim.verification.citation_trace = trace_result.outcome.result
        claim.verification.range_check = range_result.result
        claim.verification.symbol_check = symbol_result.result
        claim.verification.notes = [
            note
            for note in (
                trace_result.outcome.reason,
                range_result.reason,
                symbol_result.reason,
            )
            if note
        ]

        if trace_result.outcome.failed:
            claim.status = "rejected"
            report.record_rejection("citation_trace", trace_result.outcome.reason)
            return claim

        if range_result.failed:
            claim.status = "rejected"
            report.record_rejection("range_check", range_result.reason)
            return claim

        if symbol_result.failed:
            claim.status = "rejected"
            report.record_rejection("symbol_check", symbol_result.reason)
            return claim

        if claim.confidence < self.settings.confidence_threshold:
            # Not rejected: the checks passed and the claim may well be right.
            # It simply did not survive its own re-extractions consistently
            # enough to drive an automatic decision, so it stays `extracted`
            # and is excluded from alignment and from auto-resolution
            # (section 8.3.4). Rejecting it would discard a plausible claim;
            # promoting it would let a coin-flip settle a conflict.
            claim.status = "extracted"
            claim.verification.notes.append(
                f"confidence {claim.confidence} is below the "
                f"{self.settings.confidence_threshold} threshold; "
                "not promoted to verified"
            )
            report.low_confidence += 1
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
