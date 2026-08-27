"""Verification (section 8.3).

The rule these tests protect: a claim reaches `verified` only when no check
failed, and anything rejected is excluded from alignment - so a bad extraction
can never become a fabricated contradiction downstream.
"""

from __future__ import annotations

import pytest

from papersynth.core.models import ClaimSet
from papersynth.llm.stub import StubProvider
from papersynth.verify import RangeRules, Verifier
from papersynth.verify.citation_trace import check_numeric_literal, trace
from tests.conftest import make_claim


@pytest.fixture(scope="module")
def rules():
    return RangeRules.load("config/range_rules.yaml")


def verify_one(claim, doc, **kwargs):
    verifier = Verifier(entailment=False, **kwargs)
    result, report = verifier.verify(ClaimSet(paper_id=doc.paper_id, claims=[claim]), doc)
    return result.claims[0], report


class TestRangeCheck:
    def test_a_plausible_value_passes(self, rules, doc):
        claim = make_claim(doc, canonical_name="dropout", value=0.1)
        assert rules.check(claim).result == "pass"

    def test_an_impossible_value_is_rejected(self, rules, doc):
        """dropout=1.7 is a lost decimal point, not something a paper said."""
        claim = make_claim(doc, canonical_name="dropout", value=1.7)
        outcome = rules.check(claim)
        assert outcome.result == "fail"
        assert "extraction error" in outcome.reason

    def test_an_unusual_but_possible_value_only_warns(self, rules, doc):
        """Rejecting these would discard real claims."""
        claim = make_claim(doc, canonical_name="learning_rate", value=0.5)
        assert rules.check(claim).result == "warn"

    def test_boundary_values_are_inside_the_hard_range(self, rules, doc):
        assert rules.check(make_claim(doc, canonical_name="dropout", value=0.0)).result != "fail"
        assert rules.check(make_claim(doc, canonical_name="dropout", value=1.0)).result != "fail"

    def test_a_negative_batch_size_is_rejected(self, rules, doc):
        claim = make_claim(doc, canonical_name="batch_size", value=-8)
        assert rules.check(claim).result == "fail"

    def test_an_unknown_hyperparameter_is_not_a_failure(self, rules, doc):
        """The rule set covers common values, not everything a paper configures."""
        claim = make_claim(doc, canonical_name="mystery_knob", value=42)
        assert rules.check(claim).result == "n/a"

    def test_a_categorical_value_where_a_number_is_expected_fails(self, rules, doc):
        claim = make_claim(doc, canonical_name="batch_size", value="large")
        assert rules.check(claim).result == "fail"

    def test_a_missing_rules_file_disables_the_check(self, doc):
        """A safety net, not a prerequisite - absence must not fail the run."""
        empty = RangeRules.load("config/does_not_exist.yaml")
        assert empty.check(make_claim(doc, canonical_name="dropout", value=99.0)).result == "n/a"


class TestNumericLiteral:
    def test_a_value_present_in_the_span_passes(self, doc):
        claim = make_claim(doc, value=0.0001)
        assert check_numeric_literal(claim, "a learning rate of 0.0001 was used").result == "pass"

    def test_a_mis_transcribed_value_warns(self, doc):
        """The most common extraction error, caught with no call spent."""
        claim = make_claim(doc, value=0.001)
        outcome = check_numeric_literal(claim, "a learning rate of 0.0001 was used")
        assert outcome.result == "warn"

    @pytest.mark.parametrize("written", ["1e-4", "0.0001", "1 x 10^-4", "1.0 × 10−4"])
    def test_equivalent_notations_all_match(self, doc, written):
        """Papers write the same quantity many ways; string comparison would
        reject correct claims constantly."""
        claim = make_claim(doc, value=0.0001)
        assert check_numeric_literal(claim, f"we use {written} as the rate").result == "pass"

    def test_thousands_separators_match(self, doc):
        claim = make_claim(doc, value=100000)
        assert check_numeric_literal(claim, "trained for 100,000 steps").result == "pass"

    def test_a_categorical_value_is_not_checked(self, doc):
        claim = make_claim(doc, value="cosine")
        assert check_numeric_literal(claim, "a cosine schedule").result == "n/a"


class TestCitationTrace:
    def test_a_well_anchored_claim_traces(self, doc):
        claim = make_claim(doc, start=0, end=60)
        assert trace(claim, doc).outcome.result in ("pass", "warn")

    def test_an_unresolvable_span_fails(self, doc):
        """ER-01: rejected, never downgraded."""
        claim = make_claim(doc)
        claim.provenance.span_id = "1706.03762#s9.p9.0"
        assert trace(claim, doc).span_resolves.result == "fail"

    def test_a_stale_quote_hash_fails(self, doc):
        """Catches a re-ingested document whose text shifted under old provenance."""
        claim = make_claim(doc)
        claim.provenance.quote_hash = "sha256:" + "0" * 64
        result = trace(claim, doc)
        assert result.quote_hash.result == "fail"
        assert "no longer matches" in result.quote_hash.reason


class TestEntailment:
    def test_an_entailed_claim_passes(self, doc):
        provider = StubProvider([{"entailed": True, "reason": "states it directly"}])
        claim, _ = verify_one(make_claim(doc, start=0, end=80), doc, provider=provider)
        assert claim.status == "verified"

    def test_a_rejected_entailment_rejects_the_claim(self, doc):
        provider = StubProvider([{"entailed": False, "reason": "the span gives a different value"}])
        verifier = Verifier(provider=provider, entailment=True)
        result, report = verifier.verify(
            ClaimSet(paper_id=doc.paper_id, claims=[make_claim(doc, start=0, end=80)]), doc
        )
        assert result.claims[0].status == "rejected"
        assert report.rejection_reasons["citation_trace"] == 1

    def test_an_unreadable_verdict_fails_closed(self, doc):
        """Ties resolve as FAIL; a judge that cannot answer must not wave it through."""
        provider = StubProvider([{"unexpected": "shape"}])
        verifier = Verifier(provider=provider, entailment=True)
        result, _ = verifier.verify(
            ClaimSet(paper_id=doc.paper_id, claims=[make_claim(doc, start=0, end=80)]), doc
        )
        assert result.claims[0].status == "rejected"

    def test_entailment_is_skipped_without_a_provider(self, doc):
        """Keeps a dev run inside one free tier's daily allowance."""
        claim, _ = verify_one(make_claim(doc, start=0, end=80), doc)
        assert claim.verification.citation_trace in ("pass", "warn")
        assert claim.status == "verified"


class TestVerifierOutcomes:
    def test_a_clean_claim_becomes_verified(self, doc):
        claim, report = verify_one(make_claim(doc, start=0, end=80), doc)
        assert claim.status == "verified"
        assert report.verified == 1

    def test_a_warn_does_not_block_verification(self, doc):
        claim = make_claim(doc, canonical_name="learning_rate", value=0.5, start=0, end=80)
        verified, report = verify_one(claim, doc)
        assert verified.status == "verified"
        assert report.warned == 1

    def test_an_inferred_value_loses_explicit_status_and_confidence(self, doc):
        """ER-07: it must not carry the authority of a directly stated value."""
        claim = make_claim(doc, value=0.42, start=0, end=80)
        before = claim.confidence
        verified, _ = verify_one(claim, doc)

        assert verified.status == "verified"
        assert verified.payload["stated_explicitly"] is False
        assert verified.confidence < before

    def test_a_rejected_claim_is_excluded_from_the_verified_set(self, doc):
        """The whole point: a bad extraction must not become a contradiction."""
        good = make_claim(doc, canonical_name="dropout", value=0.1, start=0, end=80)
        bad = make_claim(doc, canonical_name="dropout", value=1.7, start=0, end=80)

        verifier = Verifier(entailment=False)
        result, report = verifier.verify(ClaimSet(paper_id=doc.paper_id, claims=[good, bad]), doc)

        assert len(result.verified) == 1
        assert result.verified[0].payload["value"] == 0.1
        assert report.rejected == 1
        assert report.rejection_reasons == {"range_check": 1}

    def test_the_report_counts_reconcile(self, doc):
        claims = [
            make_claim(doc, canonical_name="dropout", value=0.1, start=0, end=80),
            make_claim(doc, canonical_name="dropout", value=1.7, start=0, end=80),
            make_claim(doc, canonical_name="batch_size", value=64, start=0, end=80),
        ]
        _, report = Verifier(entailment=False).verify(
            ClaimSet(paper_id=doc.paper_id, claims=claims), doc
        )
        assert report.total == 3
        assert report.verified + report.rejected == 3
