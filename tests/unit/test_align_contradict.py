"""Alignment and contradiction detection (sections 8.4, 8.5).

Precision is weighted above recall throughout, per section 9: a noisy conflict
list gets abandoned by reviewers, which defeats the tool. Several tests here
assert that something is NOT reported.
"""

from __future__ import annotations

from papersynth.align import Aligner
from papersynth.contradict import ContradictionScan, attach_paper_support
from papersynth.contradict.severity import specificity, value_conflict_severity
from papersynth.core.models import ClaimSet
from tests.conftest import make_claim, make_doc


def cluster_and_scan(*claim_groups):
    sets = [ClaimSet(paper_id=g[0].paper_id, claims=list(g)) for g in claim_groups if g]
    graph, report = Aligner().align(sets)
    return graph, ContradictionScan().run(graph), report


def paper_a(**kwargs):
    return make_claim(make_doc("1706.03762"), **kwargs)


def paper_b(**kwargs):
    doc = make_doc("2504.17192")
    return make_claim(doc, **kwargs)


class TestAlignment:
    def test_the_same_parameter_across_papers_forms_one_cluster(self):
        graph, _, _ = cluster_and_scan(
            [paper_a(canonical_name="learning_rate", value=0.0001)],
            [paper_b(canonical_name="learning_rate", value=0.0003)],
        )
        assert len(graph.clusters) == 1
        assert graph.clusters[0].papers == ["1706.03762", "2504.17192"]

    def test_different_parameters_stay_separate(self):
        graph, _, _ = cluster_and_scan(
            [paper_a(canonical_name="learning_rate", value=0.0001)],
            [paper_b(canonical_name="batch_size", value=64)],
        )
        assert len(graph.clusters) == 2

    def test_a_single_paper_cluster_is_a_singleton(self):
        graph, _, _ = cluster_and_scan([paper_a(canonical_name="dropout", value=0.1)])
        assert graph.clusters[0].agreement == "singleton"

    def test_agreement_is_unanimous_when_values_match(self):
        graph, _, _ = cluster_and_scan(
            [paper_a(canonical_name="dropout", value=0.1)],
            [paper_b(canonical_name="dropout", value=0.1)],
        )
        assert graph.clusters[0].agreement == "unanimous"

    def test_agreement_is_conflicting_when_values_differ(self):
        graph, _, _ = cluster_and_scan(
            [paper_a(canonical_name="dropout", value=0.1)],
            [paper_b(canonical_name="dropout", value=0.3)],
        )
        assert graph.clusters[0].agreement == "conflicting"

    def test_rejected_claims_never_reach_a_cluster(self):
        """A bad extraction becoming a contradiction is the failure mode
        verification exists to prevent."""
        good = paper_a(canonical_name="dropout", value=0.1)
        bad = paper_b(canonical_name="dropout", value=1.7, status="rejected")

        graph, contradictions, _ = cluster_and_scan([good], [bad])

        assert len(graph.claims) == 1
        assert contradictions == []

    def test_different_units_are_not_merged(self):
        """warmup in steps and warmup in epochs are not one concept."""
        a = paper_a(canonical_name="warmup_steps", value=4000)
        a.payload["unit"] = "steps"
        b = paper_b(canonical_name="warmup_period", value=2)
        b.payload["unit"] = "epochs"

        graph, _, _ = cluster_and_scan([a], [b])

        assert len(graph.clusters) == 2

    def test_cluster_ids_are_stable_across_runs(self):
        """NFR-02: identical inputs must produce identical artifacts."""
        args = (
            [paper_a(canonical_name="learning_rate", value=0.0001)],
            [paper_b(canonical_name="learning_rate", value=0.0003)],
        )
        first, _, _ = cluster_and_scan(*args)
        second, _, _ = cluster_and_scan(*args)

        assert [c.cluster_id for c in first.clusters] == [c.cluster_id for c in second.clusters]

    def test_split_check_is_recorded_as_not_run(self):
        """No SplitterAgent in the MVA; the artifact must not overstate it."""
        graph, _, _ = cluster_and_scan(
            [paper_a(canonical_name="dropout", value=0.1)],
            [paper_b(canonical_name="dropout", value=0.3)],
        )
        assert graph.clusters[0].split_check == "n/a"


class TestValueConflict:
    def test_a_genuine_disagreement_is_detected(self):
        _, contradictions, _ = cluster_and_scan(
            [paper_a(canonical_name="learning_rate", value=0.0001, condition="base model")],
            [paper_b(canonical_name="learning_rate", value=0.0003, condition="base model")],
        )
        assert len(contradictions) == 1
        assert contradictions[0].type == "VALUE_CONFLICT"
        assert len(contradictions[0].positions) == 2

    def test_agreeing_papers_produce_no_conflict(self):
        _, contradictions, _ = cluster_and_scan(
            [paper_a(canonical_name="dropout", value=0.1, condition="base model")],
            [paper_b(canonical_name="dropout", value=0.1, condition="base model")],
        )
        assert contradictions == []

    def test_different_conditions_are_not_a_conflict(self):
        """ER-04. Two scoped facts, not a disagreement - reporting it would
        waste review time on a non-problem."""
        _, contradictions, _ = cluster_and_scan(
            [paper_a(canonical_name="learning_rate", value=0.0001, condition="base model")],
            [paper_b(canonical_name="learning_rate", value=0.0003, condition="large model")],
        )
        assert contradictions == []

    def test_the_same_value_written_differently_is_not_a_conflict(self):
        """A pure formatting artifact must never reach the review list."""
        _, contradictions, _ = cluster_and_scan(
            [paper_a(canonical_name="learning_rate", value=0.0001, condition="base")],
            [paper_b(canonical_name="learning_rate", value=1e-4, condition="base")],
        )
        assert contradictions == []

    def test_one_paper_contradicting_itself_is_not_a_cross_paper_conflict(self):
        """An intra-paper inconsistency is addressed to the reader, not
        resolved by policy (section 10.1)."""
        doc = make_doc("1706.03762")
        a = make_claim(doc, canonical_name="dropout", value=0.1, condition="base", start=0, end=40)
        b = make_claim(doc, canonical_name="dropout", value=0.3, condition="base", start=5, end=45)

        _, contradictions, _ = cluster_and_scan([a, b])

        assert contradictions == []

    def test_a_singleton_cluster_yields_no_conflict(self):
        _, contradictions, _ = cluster_and_scan([paper_a(canonical_name="dropout", value=0.1)])
        assert contradictions == []

    def test_positions_carry_provenance_back_to_claims(self):
        graph, contradictions, _ = cluster_and_scan(
            [paper_a(canonical_name="learning_rate", value=0.0001, condition="base")],
            [paper_b(canonical_name="learning_rate", value=0.0003, condition="base")],
        )
        for position in contradictions[0].positions:
            assert position.claim_id in graph.claims
            assert graph.claims[position.claim_id].provenance.span_id

    def test_contradiction_ids_are_stable(self):
        args = (
            [paper_a(canonical_name="learning_rate", value=0.0001, condition="base")],
            [paper_b(canonical_name="learning_rate", value=0.0003, condition="base")],
        )
        first = cluster_and_scan(*args)[1][0]
        second = cluster_and_scan(*args)[1][0]
        assert first.contradiction_id == second.contradiction_id


class TestSeverity:
    def test_an_order_of_magnitude_apart_blocks(self):
        """One model trains, the other diverges."""
        assert value_conflict_severity([0.0001, 0.1]) == "BLOCKING"

    def test_a_tuning_difference_is_material(self):
        assert value_conflict_severity([0.0001, 0.0003]) == "MATERIAL"

    def test_a_structural_parameter_always_blocks(self):
        """6 layers against 12 is a different model, whatever the ratio."""
        assert value_conflict_severity([6, 12], "num_layers") == "BLOCKING"

    def test_a_categorical_structural_disagreement_blocks(self):
        assert value_conflict_severity(["Adam", "SGD"], "optimizer") == "BLOCKING"

    def test_identical_values_are_cosmetic(self):
        assert value_conflict_severity([0.1, 0.1 + 1e-15]) == "COSMETIC"

    def test_blocking_conflicts_sort_first(self):
        """The review list must lead with what actually halts emission."""
        _, contradictions, _ = cluster_and_scan(
            [
                paper_a(canonical_name="learning_rate", value=0.0001, condition="base"),
                paper_a(canonical_name="dropout", value=0.1, condition="base", start=5, end=45),
            ],
            [
                paper_b(canonical_name="learning_rate", value=0.1, condition="base"),
                paper_b(canonical_name="dropout", value=0.3, condition="base", start=5, end=45),
            ],
        )
        severities = [c.severity for c in contradictions]
        assert severities == sorted(severities, key=lambda s: {"BLOCKING": 0, "MATERIAL": 1}[s])


class TestSpecificity:
    def test_a_scoped_claim_outranks_a_global_default(self):
        scoped = specificity({"condition": "base model", "applies_to": "cmp_encoder"})
        general = specificity({"condition": None, "applies_to": "global"})
        assert scoped > general

    def test_a_figure_derived_claim_is_penalized(self):
        """ER-07: it cannot auto-resolve a conflict, so it must not rank as if
        it could."""
        stated = specificity({"condition": "base", "stated_explicitly": True})
        inferred = specificity({"condition": "base", "stated_explicitly": False})
        assert inferred < stated


class TestPaperSupport:
    def test_venue_and_year_are_attached_for_policy_use(self):
        _, contradictions, _ = cluster_and_scan(
            [paper_a(canonical_name="learning_rate", value=0.0001, condition="base")],
            [paper_b(canonical_name="learning_rate", value=0.0003, condition="base")],
        )
        attach_paper_support(
            contradictions[0],
            {"1706.03762": ("NeurIPS", 2017), "2504.17192": ("ICLR", 2026)},
        )
        support = {p.paper_id: p.support for p in contradictions[0].positions}
        assert support["1706.03762"].year == 2017
        assert support["1706.03762"].peer_reviewed is True
        assert support["2504.17192"].venue == "ICLR"

    def test_an_unknown_venue_is_not_treated_as_peer_reviewed(self):
        _, contradictions, _ = cluster_and_scan(
            [paper_a(canonical_name="learning_rate", value=0.0001, condition="base")],
            [paper_b(canonical_name="learning_rate", value=0.0003, condition="base")],
        )
        attach_paper_support(contradictions[0], {})
        assert all(not p.support.peer_reviewed for p in contradictions[0].positions)
