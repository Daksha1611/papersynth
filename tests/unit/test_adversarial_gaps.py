"""Pass B: the adversarial implementability audit (section 8.6).

A model asked to find problems will find some that are not there, so most of
these tests are about what the pass must NOT report. A padded gap list is worse
than a short one: a reviewer who stops reading gets nothing from it at all.
"""

from __future__ import annotations

import pytest

from papersynth.core.models import Gap
from papersynth.gapcheck import AdversarialGapAgent, render_spec
from papersynth.llm.stub import StubProvider
from tests.conftest import make_claim, make_doc

SPEC = {
    "objective": "Implement a masked language model.",
    "components": [
        {
            "component_id": "cmp_global",
            "name": "Global configuration",
            "role": "Configuration applying across the implementation.",
            "hyperparameters": [
                {
                    "canonical_name": "learning_rate",
                    "value": 0.0001,
                    "unit": None,
                    "condition": "base model",
                },
                {"canonical_name": "dropout", "value": 0.1, "unit": None, "condition": None},
            ],
            "equations": [],
        }
    ],
    "open_conflicts": [{"summary": "batch size 256 against 8000"}],
    "missing_but_critical": [{"field": "optimizer"}],
}


def agent(gaps):
    return AdversarialGapAgent(StubProvider([{"gaps": gaps}]))


def audit(gaps, *, claims=None, existing=None, disputed=None):
    return agent(gaps).audit(
        SPEC,
        claims=claims if claims is not None else [],
        existing=existing or [],
        disputed=disputed,
        paper_ids=["p1", "p2"],
    )


class TestGapsFound:
    def test_a_genuine_gap_is_reported(self):
        found = audit(
            [
                {
                    "field": "weight_initialization",
                    "question": "How are weights initialized?",
                    "criticality": "MATERIAL",
                }
            ]
        )
        assert [g.field for g in found] == ["weight_initialization"]

    def test_criticality_is_carried_through(self):
        found = audit(
            [
                {
                    "field": "gradient_clipping",
                    "question": "Is gradient clipping applied?",
                    "criticality": "BLOCKING",
                }
            ]
        )
        assert found[0].criticality == "BLOCKING"

    def test_what_it_blocks_reaches_the_reader(self):
        """The reason it matters is the part that lets a reviewer prioritise."""
        found = audit(
            [
                {
                    "field": "padding_strategy",
                    "question": "How is padding handled?",
                    "blocks": "the collate function",
                }
            ]
        )
        assert "the collate function" in found[0].question

    def test_every_gap_records_which_papers_were_searched(self):
        found = audit([{"field": "warmup", "question": "Is warmup used?"}])
        assert found[0].searched_papers == ["p1", "p2"]

    def test_gaps_point_back_at_the_paper_text(self):
        """The likeliest cause is still an extraction miss."""
        found = audit([{"field": "warmup", "question": "Is warmup used?"}])
        assert any("extraction missed" in src for src in found[0].suggested_sources)

    def test_an_unknown_criticality_defaults_to_material(self):
        found = audit([{"field": "x_setting", "question": "q?", "criticality": "URGENT"}])
        assert found[0].criticality == "MATERIAL"

    def test_gap_ids_are_stable(self):
        entry = [{"field": "weight_initialization", "question": "How?"}]
        assert audit(entry)[0].gap_id == audit(entry)[0].gap_id


class TestFalseGapsSuppressed:
    def test_a_value_the_corpus_supplies_is_not_a_gap(self):
        """Listing an answered question wastes the reviewer's attention."""
        doc = make_doc()
        claims = [make_claim(doc, canonical_name="learning_rate", value=0.0001)]

        found = audit(
            [{"field": "learning_rate", "question": "What learning rate?"}], claims=claims
        )
        assert found == []

    def test_a_rejected_claim_does_not_suppress_a_gap(self):
        """It never established its value, so the gap is real."""
        doc = make_doc()
        claims = [make_claim(doc, canonical_name="learning_rate", value=0.0001, status="rejected")]

        found = audit(
            [{"field": "learning_rate", "question": "What learning rate?"}], claims=claims
        )
        assert [g.field for g in found] == ["learning_rate"]

    def test_a_gap_pass_a_already_found_is_not_repeated(self):
        existing = [
            Gap(
                gap_id="gap_1",
                field="optimizer",
                question="Which optimizer?",
                criticality="BLOCKING",
                searched_papers=["p1"],
            )
        ]
        found = audit(
            [{"field": "optimizer", "question": "Which optimizer is used?"}], existing=existing
        )
        assert found == []

    def test_a_near_duplicate_name_is_still_a_duplicate(self):
        existing = [
            Gap(
                gap_id="gap_1",
                field="weight_initialization",
                question="How?",
                criticality="MATERIAL",
                searched_papers=["p1"],
            )
        ]
        found = audit([{"field": "weight_initialisation", "question": "How?"}], existing=existing)
        assert found == []

    def test_a_contested_value_is_not_reported_as_missing(self):
        """The papers answer this, they just answer it differently. The spec
        already surfaces that as a conflict, and calling it missing tells the
        reader something false."""
        found = audit(
            [{"field": "num_steps", "question": "How many steps, given the conflict?"}],
            disputed={"num_steps"},
        )
        assert found == []

    def test_a_contested_field_still_allows_unrelated_gaps(self):
        found = audit(
            [
                {"field": "num_steps", "question": "How many steps?"},
                {"field": "gradient_clipping", "question": "Is clipping applied?"},
            ],
            disputed={"num_steps"},
        )
        assert [g.field for g in found] == ["gradient_clipping"]

    def test_the_same_gap_twice_in_one_response_collapses(self):
        found = audit(
            [
                {"field": "weight_initialization", "question": "How?"},
                {"field": "weight_initialization", "question": "How exactly?"},
            ]
        )
        assert len(found) == 1

    def test_an_entry_without_a_question_is_dropped(self):
        assert audit([{"field": "something"}]) == []

    def test_an_entry_without_a_field_is_dropped(self):
        assert audit([{"question": "What about the thing?"}]) == []

    def test_an_empty_audit_is_a_valid_outcome(self):
        """A sufficient spec is a real result, not a failure."""
        assert audit([]) == []


class TestSpecRendering:
    def test_the_agent_sees_values_it_can_check_against(self):
        rendered = render_spec(SPEC)
        assert "learning_rate = 0.0001" in rendered
        assert "dropout = 0.1" in rendered

    def test_conditions_are_shown(self):
        """Without the scope, a scoped value looks like a global default."""
        assert "[base model]" in render_spec(SPEC)

    def test_unresolved_conflicts_are_shown(self):
        assert "batch size 256 against 8000" in render_spec(SPEC)

    def test_known_gaps_are_shown_so_they_are_not_restated(self):
        rendered = render_spec(SPEC)
        assert "ALREADY KNOWN TO BE MISSING" in rendered
        assert "optimizer" in rendered

    def test_the_papers_are_never_shown(self):
        """The pass simulates an implementer who cannot read them; showing the
        papers would let the agent answer its own questions from context a real
        implementer will not have."""
        rendered = render_spec(SPEC)
        assert "Recurrent models" not in rendered
        assert "span_id" not in rendered

    def test_an_empty_spec_still_renders(self):
        assert render_spec({}) != ""


class TestFailureIsNotFatal:
    def test_a_failed_audit_does_not_lose_the_run(self):
        """It costs the gaps this pass would have found; aborting would cost
        the whole spec."""
        from papersynth.core.errors import ProviderError
        from papersynth.core.run import Pipeline, RunResult

        pipeline = Pipeline(StubProvider(error=ProviderError("down")), adversarial_gaps=True)
        result = RunResult(run_id="r")

        found = pipeline._adversarial_pass(SPEC, result)

        assert found == []
        assert any("adversarial gap pass failed" in w for w in result.warnings)

    def test_an_unreadable_response_yields_no_gaps(self):
        agent_ = AdversarialGapAgent(StubProvider([{"unexpected": "shape"}]))
        assert agent_.audit(SPEC, claims=[], existing=[], paper_ids=["p1"]) == []


class TestPipelineIntegration:
    @pytest.fixture
    def three_papers(self):
        from pathlib import Path

        from papersynth.ingest.latex import LatexIngestor

        fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "three_paper"
        return [
            LatexIngestor().ingest(str(fixtures / f"paper_{p}.tex"), paper_id=f"paper_{p}")
            for p in "abc"
        ]

    def test_pass_b_gaps_reach_the_emitted_spec(self, three_papers):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from e2e.test_mva_acceptance import ENTAILED, EXTRACTIONS  # noqa: F401

        from papersynth.core.run import Pipeline

        def respond(prompt: str):
            if "implement this specification TODAY" in prompt:
                return {
                    "gaps": [
                        {
                            "field": "gradient_accumulation",
                            "question": "Is gradient accumulation used?",
                            "criticality": "MATERIAL",
                        }
                    ]
                }
            if "SAME configurable quantity" in prompt:
                return {"assignments": [], "reason": "n/a"}
            for paper_id, items in EXTRACTIONS.items():
                marker = {
                    "paper_a": "batch size of 128",
                    "paper_b": "batch size of 256",
                    "paper_c": "larger stack is less stable",
                }[paper_id]
                if marker in prompt:
                    return items
            return []

        result = Pipeline(
            StubProvider(respond),
            extractors=["hyperparameter"],
            entailment=False,
            adversarial_gaps=True,
        ).run(three_papers, objective="Test.")

        fields = {g["field"] for g in result.spec["missing_but_critical"]}
        assert "gradient_accumulation" in fields, "Pass B gap must survive the rebuild"
        assert "optimizer" in fields, "Pass A gaps must survive alongside it"
