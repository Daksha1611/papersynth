"""Draft/final parity, Pass A+B merging, ER-02 enforcement, and the harness.

The through-line: a Gap records a QUESTION. The moment it also records a guess,
a coding agent reading quickly implements the guess, which is the exact failure
the gap mechanism exists to prevent.
"""

from __future__ import annotations

import pytest

from papersynth.core.models import Gap
from papersynth.eval import ABLATABLE_FIELDS, COMPLETE_SPEC, ablate, evaluate_gaps
from papersynth.gapcheck import AdversarialGapAgent, Checklist
from papersynth.gapcheck.adversarial import strip_suggestion
from papersynth.llm.stub import StubProvider
from papersynth.synth import SpecBuilder
from tests.conftest import make_claim, make_doc


class TestDraftAndFinalParity:
    """Pass B audits the draft and reports gaps against it. If the draft came
    from different code than the spec that ships, those gaps could describe an
    artifact nobody receives."""

    @pytest.fixture
    def builder(self):
        doc = make_doc()
        claims = {c.claim_id: c for c in [make_claim(doc, canonical_name="dropout", value=0.1)]}
        return SpecBuilder(run_id="run_parity", objective="Test.", documents=[doc], claims=claims)

    def test_draft_and_final_are_identical(self, builder):
        """With no conflicts and no gaps there is nothing for the two to
        differ about, so any difference is a forked assembly path."""
        draft = builder.build_draft(contradictions=[], gaps=[])
        final = builder.build(contradictions=[], gaps=[])

        draft.pop("generated_at")
        final.pop("generated_at")
        assert draft == final

    def test_the_draft_carries_the_components_pass_b_audits(self, builder):
        draft = builder.build_draft(contradictions=[], gaps=[])
        assert draft["components"], "an empty draft would make every gap a false positive"
        assert draft["objective"]

    def test_build_draft_applies_no_gates(self, builder):
        """Gating lives in SpecValidator, so a draft assembles even when the
        result would not be emittable."""
        draft = builder.build_draft(contradictions=[], gaps=[])
        assert draft["review"]["status"] == "draft"


class TestPassAPassBMerge:
    """Both passes will flag a missing learning_rate. That is one gap."""

    def agent(self, gaps):
        return AdversarialGapAgent(StubProvider([{"gaps": gaps}]))

    @pytest.fixture
    def pass_a_gap(self):
        return Gap(
            gap_id="gap_lr",
            component_id=None,
            field="learning_rate",
            question="What learning rate should be used?",
            criticality="BLOCKING",
            searched_papers=["p1"],
        )

    def test_an_overlapping_gap_produces_one_entry(self, pass_a_gap):
        found = self.agent(
            [
                {
                    "field": "learning_rate",
                    "question": "Which peak learning rate?",
                    "criticality": "COSMETIC",
                }
            ]
        ).audit(COMPLETE_SPEC, claims=[], existing=[pass_a_gap], paper_ids=["p1"])

        assert found == [], "the merged gap stays in Pass A's list, not duplicated"

    def test_pass_a_criticality_wins(self, pass_a_gap):
        """It comes from config a human wrote, not from a model."""
        self.agent(
            [
                {
                    "field": "learning_rate",
                    "question": "Which peak learning rate?",
                    "criticality": "COSMETIC",
                }
            ]
        ).audit(COMPLETE_SPEC, claims=[], existing=[pass_a_gap], paper_ids=["p1"])

        assert pass_a_gap.criticality == "BLOCKING"

    def test_pass_b_phrasing_wins(self, pass_a_gap):
        """It names the specific thing that is missing."""
        self.agent(
            [{"field": "learning_rate", "question": "Which peak learning rate after warmup?"}]
        ).audit(COMPLETE_SPEC, claims=[], existing=[pass_a_gap], paper_ids=["p1"])

        assert pass_a_gap.question == "Which peak learning rate after warmup?"

    def test_an_abbreviated_field_name_merges(self, pass_a_gap):
        """Token similarity cannot see abbreviation: "weight init scheme"
        against "weight initialization" scores below "num steps" against "num
        epochs", which must never merge. No threshold separates them."""
        gap = Gap(
            gap_id="gap_wi",
            component_id=None,
            field="weight_initialization",
            question="How are weights initialized?",
            criticality="MATERIAL",
            searched_papers=["p1"],
        )
        found = self.agent(
            [{"field": "weight_init_scheme", "question": "Which init scheme?"}]
        ).audit(COMPLETE_SPEC, claims=[], existing=[gap], paper_ids=["p1"])

        assert found == []
        assert gap.question == "Which init scheme?"

    def test_similarly_named_but_distinct_fields_stay_separate(self):
        """num_steps and num_epochs are different quantities."""
        gap = Gap(
            gap_id="gap_ns",
            component_id=None,
            field="num_steps",
            question="How many steps?",
            criticality="MATERIAL",
            searched_papers=["p1"],
        )
        found = self.agent([{"field": "num_epochs", "question": "How many epochs?"}]).audit(
            COMPLETE_SPEC, claims=[], existing=[gap], paper_ids=["p1"]
        )

        assert [g.field for g in found] == ["num_epochs"]

    def test_a_near_duplicate_field_name_still_merges(self, pass_a_gap):
        self.agent([{"field": "learning_rate_peak", "question": "Peak LR?"}]).audit(
            COMPLETE_SPEC, claims=[], existing=[pass_a_gap], paper_ids=["p1"]
        )

        assert pass_a_gap.question == "Peak LR?"


class TestGapsNeverCarryAnswers:
    """ER-02 at emission time."""

    @pytest.mark.parametrize(
        ("question", "must_not_contain"),
        [
            ("How are weights initialized? Probably Xavier.", "Xavier"),
            ("What clipping norm? We recommend 1.0.", "1.0"),
            ("Which optimizer? Likely Adam.", "Adam"),
            ("What batch size? Default to 256.", "256"),
            ("How long to train? Assume 100000 steps.", "100000"),
        ],
    )
    def test_a_proposed_answer_is_stripped(self, question, must_not_contain):
        assert must_not_contain not in strip_suggestion(question)

    def test_context_explaining_why_it_matters_is_kept(self):
        """Stakes are not an answer. A reviewer needs them to prioritise."""
        question = (
            "Which optimizer is used? Adam and SGD with momentum give "
            "materially different training dynamics."
        )
        assert strip_suggestion(question) == question

    def test_a_gap_that_only_proposes_an_answer_is_dropped(self):
        """The exact failure mode: a coding agent implements the guess."""
        found = AdversarialGapAgent(
            StubProvider(
                [{"gaps": [{"field": "weight_initialization", "question": "probably Xavier"}]}]
            )
        ).audit(COMPLETE_SPEC, claims=[], existing=[], paper_ids=["p1"])

        assert found == []

    def test_the_question_survives_when_only_the_guess_is_removed(self):
        found = AdversarialGapAgent(
            StubProvider(
                [
                    {
                        "gaps": [
                            {
                                "field": "weight_initialization",
                                "question": "How are weights initialized? Probably Xavier.",
                            }
                        ]
                    }
                ]
            )
        ).audit(COMPLETE_SPEC, claims=[], existing=[], paper_ids=["p1"])

        assert len(found) == 1
        assert "Xavier" not in found[0].question
        assert "How are weights initialized?" in found[0].question


class TestAblationHarness:
    def test_ablation_removes_only_the_named_field(self):
        ablated = ablate(COMPLETE_SPEC, "learning_rate")
        names = {h["canonical_name"] for c in ablated["components"] for h in c["hyperparameters"]}
        assert "learning_rate" not in names
        assert "batch_size" in names

    def test_the_reference_spec_is_untouched_by_ablation(self):
        before = len(COMPLETE_SPEC["components"][0]["hyperparameters"])
        ablate(COMPLETE_SPEC, "learning_rate")
        assert len(COMPLETE_SPEC["components"][0]["hyperparameters"]) == before

    def test_checklist_only_recall_is_perfect_and_silent(self):
        """Pass A is deterministic, so this is a fixed expectation rather than
        a threshold: every ablated field is recovered, and an untouched spec
        raises nothing."""
        report = evaluate_gaps(StubProvider([]), adversarial=False)

        assert report.recall == 1.0, report.render()
        assert report.false_positive_count == 0, report.render()

    def test_a_one_of_requirement_ablates_its_whole_group(self):
        """Deleting num_steps while num_epochs remains proves nothing, because
        the requirement is genuinely still satisfied."""
        entry = next(e for e in ABLATABLE_FIELDS if e[0] == "num_steps_or_epochs")
        assert set(entry[1]) == {"num_steps", "num_epochs"}

    def test_the_report_shows_both_numbers(self):
        """Recall alone would reward a detector that reports everything."""
        rendered = evaluate_gaps(StubProvider([]), adversarial=False).render()
        assert "recall" in rendered
        assert "false positives" in rendered

    def test_false_positives_are_counted_per_trial(self):
        from papersynth.eval import GapEvalReport

        report = GapEvalReport(detected=["a", "b"], missed=[], false_positives=["x"])
        assert report.false_positive_rate == 0.5


class TestChecklistStillSeesEverything:
    def test_pass_a_alone_finds_a_deleted_field(self):
        checklist = Checklist.load("config/implementability_checklist.yaml")
        from papersynth.eval.gap_ablation import claims_for

        ablated = ablate(COMPLETE_SPEC, "learning_rate")
        gaps = checklist.audit(claims_for(ablated), paper_ids=["p1"])

        assert "learning_rate" in {g.field for g in gaps}
