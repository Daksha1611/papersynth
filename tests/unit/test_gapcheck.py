"""Gap check, Pass A (section 8.6).

A gap is a claim about what the corpus does not supply, so the tests here are
mostly about not overclaiming: no gap for something a paper did state, no gap
for a category the corpus was never going to cover.
"""

from __future__ import annotations

import pytest

from papersynth.gapcheck import Checklist, summarize
from tests.conftest import make_claim, make_doc

CHECKLIST_PATH = "config/implementability_checklist.yaml"


@pytest.fixture(scope="module")
def checklist():
    return Checklist.load(CHECKLIST_PATH)


def claims(*specs, status="verified"):
    doc = make_doc("paper_a")
    return [
        make_claim(doc, canonical_name=name, value=value, status=status) for name, value in specs
    ]


def fields(gaps):
    return {g.field for g in gaps}


class TestChecklistAudit:
    def test_a_missing_required_field_becomes_a_gap(self, checklist):
        gaps = checklist.audit(claims(("dropout", 0.1)), paper_ids=["paper_a"])
        assert "learning_rate" in fields(gaps)

    def test_a_supplied_field_is_not_a_gap(self, checklist):
        gaps = checklist.audit(claims(("learning_rate", 0.0001)), paper_ids=["paper_a"])
        assert "learning_rate" not in fields(gaps)

    def test_one_paper_supplying_a_value_satisfies_the_corpus(self, checklist):
        """Reporting it missing because a different paper omitted it is false."""
        a = make_claim(make_doc("paper_a"), canonical_name="learning_rate", value=0.0001)
        b = make_claim(make_doc("paper_b"), canonical_name="dropout", value=0.1)

        gaps = checklist.audit([a, b], paper_ids=["paper_a", "paper_b"])

        assert "learning_rate" not in fields(gaps)
        assert "dropout" not in fields(gaps)

    def test_a_one_of_requirement_is_met_by_either_alternative(self, checklist):
        with_steps = checklist.audit(claims(("num_steps", 100000)), paper_ids=["p"])
        with_epochs = checklist.audit(claims(("num_epochs", 30)), paper_ids=["p"])

        assert "num_steps_or_epochs" not in fields(with_steps)
        assert "num_steps_or_epochs" not in fields(with_epochs)

    def test_a_one_of_requirement_is_a_gap_when_neither_is_present(self, checklist):
        gaps = checklist.audit(claims(("dropout", 0.1)), paper_ids=["p"])
        assert "num_steps_or_epochs" in fields(gaps)

    def test_a_rejected_claim_does_not_count_as_coverage(self, checklist):
        """It never established its value. Counting it would tell the
        implementer nothing is missing when the only source for it failed."""
        gaps = checklist.audit(
            claims(("learning_rate", 0.0001), status="rejected"), paper_ids=["paper_a"]
        )
        assert "learning_rate" in fields(gaps)

    def test_an_inapplicable_group_raises_no_gaps(self, checklist):
        """A corpus describing no training must not be asked for a batch size."""
        gaps = checklist.audit([], paper_ids=["paper_a"])
        assert gaps == []

    def test_every_gap_records_which_papers_were_searched(self, checklist):
        gaps = checklist.audit(claims(("dropout", 0.1)), paper_ids=["p_b", "p_a"])
        assert all(g.searched_papers == ["p_a", "p_b"] for g in gaps)

    def test_gaps_lead_with_what_blocks_an_implementer(self, checklist):
        gaps = checklist.audit(claims(("dropout", 0.1)), paper_ids=["p"])
        rank = {"BLOCKING": 0, "MATERIAL": 1, "COSMETIC": 2}
        assert [rank[g.criticality] for g in gaps] == sorted(rank[g.criticality] for g in gaps)

    def test_gap_ids_are_stable(self, checklist):
        first = checklist.audit(claims(("dropout", 0.1)), paper_ids=["p"])
        second = checklist.audit(claims(("dropout", 0.1)), paper_ids=["p"])
        assert [g.gap_id for g in first] == [g.gap_id for g in second]

    def test_a_missing_checklist_reports_no_gaps(self):
        """Nothing to check against must not imply the corpus is complete."""
        assert (
            Checklist.load("config/does_not_exist.yaml").audit(
                claims(("dropout", 0.1)), paper_ids=["p"]
            )
            == []
        )


class TestWording:
    def test_no_question_asserts_the_papers_are_silent(self, checklist):
        """A gap means no VERIFIED CLAIM supplies the field, which is not the
        same as "no paper states it" - extraction can miss a stated value.
        Telling an implementer the papers omit something that was printed on
        page four sends them off to invent a number that already existed."""
        overclaiming = ("no paper", "the corpus names neither", "not stated anywhere")

        for group in checklist.groups:
            for required in group.required:
                lowered = required.question.lower()
                for phrase in overclaiming:
                    assert phrase not in lowered, (
                        f"{required.field} overclaims corpus-wide absence: {required.question!r}"
                    )

    def test_blocking_gaps_point_back_at_the_paper_text(self, checklist):
        """The likeliest cause of a BLOCKING gap is an extraction miss, so the
        first place to look is the paper itself."""
        for group in checklist.groups:
            for required in group.required:
                if required.criticality != "BLOCKING":
                    continue
                assert any(
                    "extraction missed" in source for source in required.suggested_sources
                ), f"{required.field} does not suggest re-checking the paper text"

    def test_every_requirement_has_an_actionable_question(self, checklist):
        for group in checklist.groups:
            for required in group.required:
                assert required.question.endswith(".") or "?" in required.question
                assert required.suggested_sources, f"{required.field} suggests nowhere to look"


class TestSummary:
    def test_counts_by_criticality(self, checklist):
        gaps = checklist.audit(claims(("dropout", 0.1)), paper_ids=["p"])
        summary = summarize(gaps)
        assert summary["total"] == len(gaps)
        assert sum(summary["by_criticality"].values()) == len(gaps)


class TestPipelineIntegration:
    def test_gaps_reach_the_emitted_spec(self):
        """missing_but_critical is a required spec field; leaving it always
        empty would let the spec overstate its own completeness."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from e2e.test_mva_acceptance import scripted_provider

        from papersynth.core.run import Pipeline
        from papersynth.ingest.latex import LatexIngestor

        fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "three_paper"
        docs = [
            LatexIngestor().ingest(str(fixtures / f"paper_{p}.tex"), paper_id=f"paper_{p}")
            for p in "abc"
        ]
        result = Pipeline(scripted_provider(), extractors=["hyperparameter"]).run(
            docs, objective="Regression check."
        )

        assert result.gaps, "the fixture corpus states no optimizer, so a gap is expected"
        emitted = {g["field"] for g in result.spec["missing_but_critical"]}
        assert emitted == {g.field for g in result.gaps}
        assert all(g["searched_papers"] for g in result.spec["missing_but_critical"])
