"""RESULT_CONFLICT (sections 7.6, 10.3, ER-06).

Most of these assert that something is NOT reported. Benchmark numbers are
unusually easy to compare wrongly, and a false result conflict sends a reviewer
to adjudicate two numbers that were never comparable.
"""

from __future__ import annotations

import pytest

from papersynth.align import Aligner
from papersynth.contradict import ContradictionScan, ResultConflictDetector
from papersynth.core import ids
from papersynth.core.models import Claim, ClaimSet, Provenance
from papersynth.extract.extractors.result import ResultExtractor
from papersynth.llm.stub import StubProvider
from papersynth.reconcile import Policy, PolicyEngine
from tests.conftest import make_doc

POLICY = Policy.load("config/reconcile_policy.yaml")


def result_claim(
    paper,
    value,
    *,
    metric="bleu",
    dataset="WMT14 EN-DE",
    split="newstest2014",
    variant="base",
    conditions=None,
    variance=None,
    explicit=True,
):
    payload = {
        "metric": metric,
        "value": value,
        "dataset": dataset,
        "split": split,
        "model_variant": variant,
        "conditions": conditions or {},
        "reported_variance": variance,
        "stated_explicitly": explicit,
    }
    claim = Claim.build(
        paper_id=paper,
        claim_type="result",
        provenance=Provenance(
            paper_id=paper,
            span_id=f"{paper}#s1.p0.0",
            section="Results",
            page=5,
            char_start=0,
            char_end=40,
            quote_hash=ids.quote_hash(str(value)),
            extraction_method="llm",
            extractor_version="result@1.0.0",
            confidence=0.9,
        ),
        payload=payload,
    )
    claim.status = "verified"
    return claim


def scan(*claims):
    sets = [ClaimSet(paper_id=c.paper_id, claims=[c]) for c in claims]
    graph, _ = Aligner().align(sets)
    return ContradictionScan().run(graph)


class TestGenuineConflicts:
    def test_the_same_measurement_with_different_scores_conflicts(self):
        found = scan(result_claim("p1", 27.3), result_claim("p2", 28.4))

        assert len(found) == 1
        assert found[0].type == "RESULT_CONFLICT"
        assert {p.position for p in found[0].positions} == {"27.3", "28.4"}

    def test_agreement_produces_no_conflict(self):
        assert scan(result_claim("p1", 27.3), result_claim("p2", 27.3)) == []

    def test_the_scope_is_named_in_the_description(self):
        """A reviewer needs to see what was held constant."""
        found = scan(result_claim("p1", 27.3), result_claim("p2", 28.4))
        description = found[0].description

        assert "WMT14 EN-DE" in description
        assert "newstest2014" in description
        assert "base" in description

    def test_a_result_conflict_is_never_blocking(self):
        """It does not stop anyone writing the code; it stops them knowing
        whether the code is right (section 7.6)."""
        assert scan(result_claim("p1", 27.3), result_claim("p2", 28.4))[0].severity == "MATERIAL"


class TestER06:
    """Results measured differently are distinct claims, not conflicting ones."""

    def test_different_splits_are_not_compared(self):
        """A dev score and a test score differ by design."""
        assert (
            scan(
                result_claim("p1", 27.3, split="dev"),
                result_claim("p2", 24.1, split="test"),
            )
            == []
        )

    def test_different_model_variants_are_not_compared(self):
        """A base model scoring below a large one is expected."""
        assert (
            scan(
                result_claim("p1", 27.3, variant="base"),
                result_claim("p2", 28.4, variant="large"),
            )
            == []
        )

    def test_different_datasets_are_not_compared(self):
        """The numbers are not on the same scale at all."""
        assert (
            scan(
                result_claim("p1", 27.3, dataset="WMT14 EN-DE"),
                result_claim("p2", 41.8, dataset="WMT14 EN-FR"),
            )
            == []
        )

    def test_different_protocols_are_not_compared(self):
        """Beam size moves BLEU on its own."""
        assert (
            scan(
                result_claim("p1", 27.3, conditions={"beam_size": 4}),
                result_claim("p2", 26.9, conditions={"beam_size": 1}),
            )
            == []
        )

    def test_identical_protocols_are_compared(self):
        found = scan(
            result_claim("p1", 27.3, conditions={"beam_size": 4, "length_penalty": 0.6}),
            result_claim("p2", 28.4, conditions={"length_penalty": 0.6, "beam_size": 4}),
        )
        assert len(found) == 1, "key order must not make identical protocols differ"

    def test_different_metrics_never_meet(self):
        assert (
            scan(
                result_claim("p1", 27.3, metric="bleu"),
                result_claim("p2", 91.2, metric="accuracy"),
            )
            == []
        )


class TestVarianceAndPrecision:
    def test_overlapping_intervals_are_not_a_conflict(self):
        """Agreement within stated uncertainty is agreement; reporting it
        would ask a reviewer to adjudicate noise."""
        assert (
            scan(
                result_claim("p1", 27.3, variance=0.5),
                result_claim("p2", 27.6, variance=0.4),
            )
            == []
        )

    def test_non_overlapping_intervals_still_conflict(self):
        found = scan(
            result_claim("p1", 27.3, variance=0.1),
            result_claim("p2", 29.0, variance=0.1),
        )
        assert len(found) == 1

    def test_variance_is_shown_to_the_reviewer(self):
        found = scan(
            result_claim("p1", 27.3, variance=0.1),
            result_claim("p2", 29.0, variance=0.1),
        )
        assert any("+/-" in p.position for p in found[0].positions)

    def test_a_difference_without_any_variance_is_a_conflict(self):
        """With no stated uncertainty there is no basis for calling it noise."""
        assert len(scan(result_claim("p1", 27.3), result_claim("p2", 27.9))) == 1

    def test_reporting_precision_is_not_a_disagreement(self):
        assert scan(result_claim("p1", 27.3), result_claim("p2", 27.30)) == []

    def test_a_figure_derived_result_is_marked(self):
        found = scan(
            result_claim("p1", 27.3),
            result_claim("p2", 28.4, explicit=False),
        )
        assert any("read from a figure" in p.position for p in found[0].positions)


class TestNeverAutoResolved:
    def test_the_detector_forbids_it(self):
        assert ResultConflictDetector.auto_resolvable is False

    def test_the_policy_escalates_even_a_clear_case(self):
        """Picking one silently discards a real finding."""
        found = scan(
            result_claim("p1", 27.3, conditions={"runs": 5}),
            result_claim("p2", 28.4),
        )
        if not found:
            pytest.skip("scoped apart, which is also correct")

        engine = PolicyEngine(POLICY, auto_resolvable={"RESULT_CONFLICT": False})
        assert engine.resolve_one(found[0]).is_open

    def test_one_paper_alone_is_not_a_conflict(self):
        assert scan(result_claim("p1", 27.3), result_claim("p1", 28.4)) == []


class TestExtraction:
    ITEM = {
        "metric": "BLEU score",
        "value": 27.3,
        "dataset": "WMT14 EN-DE",
        "split": "newstest2014",
        "model_variant": "base",
        "conditions": {"beam_size": 4},
        "reported_variance": None,
        "stated_explicitly": True,
        "quote": "learning rate of 0.0001",
    }

    def extract(self, **overrides):
        return ResultExtractor(StubProvider([[{**self.ITEM, **overrides}]])).extract(make_doc())

    def test_a_result_is_extracted(self):
        claim = self.extract().claims[0]
        assert claim.type == "result"
        assert claim.payload["value"] == 27.3

    def test_metric_names_are_canonicalized(self):
        """Two papers naming the same measurement must reach the same key."""
        assert self.extract(metric="BLEU score").claims[0].payload["metric"] == "bleu"
        assert self.extract(metric="Top-1 Accuracy").claims[0].payload["metric"] == "top_1_accuracy"

    def test_an_absent_split_stays_null(self):
        """A guessed split makes two incomparable numbers look comparable."""
        assert self.extract(split="").claims[0].payload["split"] is None

    def test_variance_is_kept_as_a_magnitude(self):
        assert self.extract(reported_variance=-0.4).claims[0].payload["reported_variance"] == 0.4

    def test_a_fabricated_quote_is_rejected(self):
        result = self.extract(quote="a sentence nowhere in this paper")
        assert result.claims == [] and result.rejected


class TestExpectedResults:
    def build(self, claims):
        from papersynth.synth import SpecBuilder

        doc = make_doc("p1")
        return SpecBuilder(
            run_id="r",
            objective="o",
            documents=[doc],
            claims={c.claim_id: c for c in claims},
        ).build(contradictions=[], gaps=[])

    def test_an_agreed_result_becomes_a_reproduction_target(self):
        spec = self.build([result_claim("p1", 27.3), result_claim("p2", 27.3)])
        targets = spec["expected_results"]

        assert len(targets) == 1
        assert targets[0]["metric"] == "bleu"
        assert targets[0]["value"] == 27.3
        assert len(targets[0]["provenance_refs"]) == 2

    def test_a_disputed_result_is_not_emitted_as_a_target(self):
        """Emitting one of two contested scores would tell an implementer
        their reimplementation is wrong when it matches the other paper."""
        spec = self.build([result_claim("p1", 27.3), result_claim("p2", 28.4)])
        assert spec["expected_results"] == []

    def test_tolerance_comes_from_the_paper(self):
        spec = self.build([result_claim("p1", 27.3, variance=0.4)])
        assert spec["expected_results"][0]["tolerance"] == 0.4

    def test_tolerance_is_null_when_unstated(self):
        """Inventing one would invent a claim about reproducibility."""
        spec = self.build([result_claim("p1", 27.3)])
        assert spec["expected_results"][0]["tolerance"] is None

    def test_differently_scoped_results_are_separate_targets(self):
        spec = self.build(
            [
                result_claim("p1", 27.3, split="dev"),
                result_claim("p1", 24.1, split="test"),
            ]
        )
        assert {t["split"] for t in spec["expected_results"]} == {"dev", "test"}

    def test_the_spec_still_validates(self):
        from papersynth.schemas import validate

        spec = self.build([result_claim("p1", 27.3, variance=0.4)])
        assert validate(spec, "spec.schema.json") == []
