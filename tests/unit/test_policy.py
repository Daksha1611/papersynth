"""Reconciliation policy (section 8.5, DD-03).

The invariant under test throughout: nothing is ever resolved without a named
rule, and when no rule fires the answer is "ask a human" rather than a guess.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from papersynth.core.errors import PaperSynthError
from papersynth.core.models import Contradiction, Position, Support
from papersynth.reconcile import Policy, PolicyEngine

POLICY_PATH = "config/reconcile_policy.yaml"

#: Loaded once. The property tests below generate hundreds of examples, and
#: re-reading the policy per example makes the suite slow for no added coverage.
_SHARED_POLICY = Policy.load(POLICY_PATH)


@pytest.fixture(scope="module")
def policy():
    return Policy.load(POLICY_PATH)


@pytest.fixture
def engine(policy):
    return PolicyEngine(policy)


def contradiction(positions, *, ctype="VALUE_CONFLICT", severity="MATERIAL"):
    return Contradiction(
        contradiction_id="ctr_test01",
        cluster_id="cnc_hype_learning_rate",
        type=ctype,
        severity=severity,
        description="test",
        positions=positions,
        detected_by="value_conflict_detector@1.0.0",
    )


def position(claim_id, paper_id, value, **support):
    return Position(
        claim_id=claim_id,
        paper_id=paper_id,
        position=str(value),
        support=Support(**support),
    )


class TestFallback:
    def test_no_rule_fired_escalates(self, engine):
        """DD-03. There is no best-guess path."""
        result = engine.resolve_one(
            contradiction(
                [
                    position("clm_aaaaaa", "p1", 0.0001, specificity=0.5),
                    position("clm_bbbbbb", "p2", 0.0003, specificity=0.5),
                ]
            )
        )
        assert result.outcome == "ESCALATED"
        assert result.rule_fired is None
        assert result.is_open

    def test_a_policy_declaring_another_fallback_is_rejected(self, tmp_path):
        """A policy edit must not be able to turn 'ask a human' into 'guess'."""
        path = tmp_path / "bad.yaml"
        path.write_text("policy_version: '1.0.0'\nrules: []\nfallback:\n  action: SELECTED\n")
        with pytest.raises(PaperSynthError, match="always ESCALATED"):
            Policy.load(path)

    def test_a_missing_policy_file_is_an_error(self, tmp_path):
        with pytest.raises(PaperSynthError, match="No reconciliation policy"):
            Policy.load(tmp_path / "absent.yaml")

    def test_an_unknown_predicate_is_rejected_at_load(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            "policy_version: '1.0.0'\n"
            "rules:\n"
            "  - id: r\n    applies_to: [VALUE_CONFLICT]\n    when: no_such_predicate\n"
            "    action: SELECTED\n"
            "fallback: {action: ESCALATED}\n"
        )
        with pytest.raises(PaperSynthError, match="unknown predicate"):
            Policy.load(path)


class TestRules:
    def test_a_scoped_position_beats_an_unconditional_default(self, engine):
        result = engine.resolve_one(
            contradiction(
                [
                    position("clm_aaaaaa", "p1", 0.0001, specificity=0.9),
                    position("clm_bbbbbb", "p2", 0.0003, specificity=0.5),
                ]
            )
        )
        assert result.outcome == "SCOPED"
        assert result.rule_fired == "prefer_scoped_over_global"
        assert result.selected_claim_id == "clm_aaaaaa"
        assert not result.is_open

    def test_a_clear_specificity_gap_selects(self, engine):
        result = engine.resolve_one(
            contradiction(
                [
                    position("clm_aaaaaa", "p1", 0.0001, specificity=0.95),
                    position("clm_bbbbbb", "p2", 0.0003, specificity=0.6),
                    position("clm_cccccc", "p3", 0.0005, specificity=0.6),
                ]
            )
        )
        assert result.rule_fired in ("prefer_scoped_over_global", "prefer_specific_over_general")
        assert result.selected_claim_id == "clm_aaaaaa"

    def test_a_narrow_specificity_gap_does_not_fire(self, engine):
        result = engine.resolve_one(
            contradiction(
                [
                    position("clm_aaaaaa", "p1", 0.0001, specificity=0.65),
                    position("clm_bbbbbb", "p2", 0.0003, specificity=0.6),
                ]
            )
        )
        assert result.outcome == "ESCALATED"

    def test_a_primary_source_wins_when_it_is_the_only_one(self, engine):
        result = engine.resolve_one(
            contradiction(
                [
                    position("clm_aaaaaa", "p1", 0.0001, specificity=0.5, is_primary=True),
                    position("clm_bbbbbb", "p2", 0.0003, specificity=0.5),
                ]
            )
        )
        assert result.rule_fired == "prefer_primary_source"
        assert result.selected_claim_id == "clm_aaaaaa"

    def test_two_primary_sources_do_not_fire_the_rule(self, engine):
        result = engine.resolve_one(
            contradiction(
                [
                    position("clm_aaaaaa", "p1", 0.0001, specificity=0.5, is_primary=True),
                    position("clm_bbbbbb", "p2", 0.0003, specificity=0.5, is_primary=True),
                ]
            )
        )
        assert result.outcome == "ESCALATED"

    def test_a_low_confidence_rule_defers_instead_of_deciding(self, engine):
        """Recorded so the reviewer sees the reasoning; nothing is decided."""
        result = engine.resolve_one(
            contradiction(
                [
                    position(
                        "clm_aaaaaa", "p1", 0.0001, specificity=0.5, year=2017, peer_reviewed=True
                    ),
                    position(
                        "clm_bbbbbb", "p2", 0.0003, specificity=0.5, year=2026, peer_reviewed=True
                    ),
                ]
            )
        )
        assert result.outcome == "DEFERRED"
        assert result.is_open, "a deferred conflict still needs a human"
        assert result.selected_claim_id == "clm_bbbbbb"
        assert "requires human confirmation" in result.rationale

    def test_result_conflicts_always_escalate(self, engine):
        """Two papers reporting different scores ran different experiments;
        picking one silently discards a real finding."""
        result = engine.resolve_one(
            contradiction(
                [
                    position("clm_aaaaaa", "p1", 27.3, specificity=0.95),
                    position("clm_bbbbbb", "p2", 28.4, specificity=0.5),
                ],
                ctype="RESULT_CONFLICT",
            )
        )
        assert result.outcome == "ESCALATED"
        assert result.rule_fired == "never_auto_resolve_results"

    def test_method_conflicts_escalate(self, engine):
        result = engine.resolve_one(
            contradiction(
                [
                    position("clm_aaaaaa", "p1", "sinusoidal", specificity=0.95),
                    position("clm_bbbbbb", "p2", "learned", specificity=0.5),
                ],
                ctype="METHOD_CONFLICT",
            )
        )
        assert result.outcome == "ESCALATED"

    def test_cosmetic_conflicts_resolve_without_a_human(self, engine):
        result = engine.resolve_one(
            contradiction(
                [
                    position("clm_aaaaaa", "p1", 0.1, specificity=0.5),
                    position("clm_bbbbbb", "p2", 0.1, specificity=0.5),
                ],
                severity="COSMETIC",
            )
        )
        assert result.outcome == "SELECTED"
        assert not result.is_open


class TestGuards:
    def test_an_inferred_claim_cannot_auto_resolve(self, engine):
        """ER-07: a value read off a figure cannot decide a conflict, however
        well the rule's other conditions fit."""
        result = engine.resolve_one(
            contradiction(
                [
                    position("clm_aaaaaa", "p1", 0.0001, specificity=0.9, stated_explicitly=False),
                    position("clm_bbbbbb", "p2", 0.0003, specificity=0.5),
                ]
            )
        )
        assert result.outcome == "ESCALATED"
        assert "ER-07" in result.rationale
        assert result.rule_fired == "prefer_scoped_over_global", "the rule fired but was vetoed"

    def test_a_detector_can_forbid_auto_resolution_outright(self, policy):
        """section 10.3: auto_resolvable=False overrides any matching rule."""
        engine = PolicyEngine(policy, auto_resolvable={"VALUE_CONFLICT": False})
        result = engine.resolve_one(
            contradiction(
                [
                    position("clm_aaaaaa", "p1", 0.0001, specificity=0.9),
                    position("clm_bbbbbb", "p2", 0.0003, specificity=0.5),
                ]
            )
        )
        assert result.outcome == "ESCALATED"
        assert "never auto-resolved" in result.rationale


class TestDeterminism:
    def test_the_same_contradiction_resolves_identically(self, engine):
        positions = [
            position("clm_aaaaaa", "p1", 0.0001, specificity=0.9),
            position("clm_bbbbbb", "p2", 0.0003, specificity=0.5),
        ]
        first = engine.resolve_one(contradiction(list(positions)))
        second = engine.resolve_one(contradiction(list(positions)))
        assert (first.outcome, first.selected_claim_id, first.rule_fired) == (
            second.outcome,
            second.selected_claim_id,
            second.rule_fired,
        )


@st.composite
def contradictions(draw):
    """Arbitrary well-formed contradictions, for the invariant check."""
    n = draw(st.integers(min_value=2, max_value=4))
    positions = [
        position(
            f"clm_{i:06x}",
            f"p{i}",
            draw(st.floats(min_value=1e-6, max_value=1e3, allow_nan=False)),
            specificity=draw(st.floats(min_value=0.0, max_value=1.0)),
            is_primary=draw(st.booleans()),
            year=draw(st.one_of(st.none(), st.integers(min_value=1990, max_value=2030))),
            peer_reviewed=draw(st.booleans()),
            stated_explicitly=draw(st.booleans()),
        )
        for i in range(n)
    ]
    return contradiction(
        positions,
        ctype=draw(
            st.sampled_from(
                [
                    "VALUE_CONFLICT",
                    "METHOD_CONFLICT",
                    "RESULT_CONFLICT",
                    "DEFINITION_CONFLICT",
                    "SCOPE_CONFLICT",
                ]
            )
        ),
        severity=draw(st.sampled_from(["BLOCKING", "MATERIAL", "COSMETIC"])),
    )


class TestInvariants:
    @settings(max_examples=300, deadline=None)
    @given(contradictions())
    def test_nothing_resolves_without_a_named_rule(self, contra):
        """The property section 14.2 asks for, and the one that makes every
        auto-resolution auditable after the fact."""
        engine = PolicyEngine(_SHARED_POLICY)
        result = engine.resolve_one(contra)

        if not result.is_open:
            assert result.rule_fired is not None
            assert result.selected_claim_id is not None

    @settings(max_examples=300, deadline=None)
    @given(contradictions())
    def test_a_resolved_conflict_always_selects_a_real_position(self, contra):
        engine = PolicyEngine(_SHARED_POLICY)
        result = engine.resolve_one(contra)

        if result.selected_claim_id is not None:
            assert result.selected_claim_id in {p.claim_id for p in contra.positions}

    @settings(max_examples=300, deadline=None)
    @given(contradictions())
    def test_an_inferred_claim_is_never_auto_selected(self, contra):
        """ER-07 must hold across every rule, not just the ones tested above."""
        engine = PolicyEngine(_SHARED_POLICY)
        result = engine.resolve_one(contra)

        if not result.is_open and result.selected_claim_id:
            chosen = next(p for p in contra.positions if p.claim_id == result.selected_claim_id)
            assert chosen.support.stated_explicitly

    @settings(max_examples=200, deadline=None)
    @given(contradictions())
    def test_result_conflicts_never_auto_resolve(self, contra):
        engine = PolicyEngine(_SHARED_POLICY)
        result = engine.resolve_one(contra)

        if contra.type == "RESULT_CONFLICT":
            assert result.is_open
