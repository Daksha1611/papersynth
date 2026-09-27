"""DEFINITION_CONFLICT and the definition claim type (section 7.6).

The most dangerous disagreement a synthesis can carry, because it is silent.
Two agent-safety papers both report an "attack success" rate; one counts any
out-of-policy tool call, the other only a call that reaches a harmful end
state. No number separates them and no approach is adopted or rejected, so
until this claim type and detector existed nothing in the pipeline could see
it. Code that computes one metric while citing the other's threshold is wrong
in a way no test catches.
"""

from __future__ import annotations

import pytest

from papersynth.align import Aligner
from papersynth.contradict import ContradictionScan, DefinitionConflictDetector
from papersynth.core import ids
from papersynth.core.models import Claim, ClaimSet, Provenance
from papersynth.extract.extractors.definition import DefinitionExtractor
from papersynth.llm.stub import StubProvider
from papersynth.reconcile import Policy, PolicyEngine
from tests.conftest import make_doc

POLICY = Policy.load("config/reconcile_policy.yaml")


def definition_claim(
    paper,
    term="attack_success",
    criterion=None,
    definition="an attack succeeds",
    condition=None,
    attribution="own",
):
    payload = {
        "term": term,
        "definition": definition,
        "criterion": criterion,
        "attribution": attribution,
        "applies_to": "global",
        "condition": condition,
        "stated_explicitly": True,
    }
    provenance = Provenance(
        paper_id=paper,
        span_id=f"{paper}#s1.p0.0",
        section="Threat Model",
        page=1,
        char_start=0,
        char_end=40,
        quote_hash=ids.quote_hash(criterion or definition),
        extraction_method="llm",
        extractor_version="definition@1.0.0",
        confidence=0.9,
    )
    claim = Claim.build(
        paper_id=paper, claim_type="definition", provenance=provenance, payload=payload
    )
    claim.status = "verified"
    return claim


def scan(*claims):
    sets = [ClaimSet(paper_id=c.paper_id, claims=[c]) for c in claims]
    graph, _ = Aligner(semantic_merges=False).align(sets)
    return ContradictionScan().run(graph)


class TestCriterionNotProse:
    """The comparison is on the operational criterion, not the definition
    wording. This is the whole point: numbers computed from different checks
    are not comparable however similarly the papers describe them."""

    def test_different_criteria_under_one_term_conflict(self):
        a = definition_claim("agentdojo", criterion="any tool call outside the granted set")
        b = definition_claim("toolemu", criterion="the agent reaches a harmful end state")

        found = scan(a, b)

        assert len(found) == 1
        assert found[0].type == "DEFINITION_CONFLICT"
        assert len(found[0].positions) == 2

    def test_same_criterion_worded_differently_is_agreement(self):
        """Different prose reducing to one check is not a conflict; reporting it
        would ask a reviewer to adjudicate paraphrase."""
        a = definition_claim("p1", criterion="the agent calls a tool outside the granted set")
        b = definition_claim("p2", criterion="agent calls a tool outside the granted set")
        assert scan(a, b) == []

    def test_identical_prose_with_different_criteria_still_conflicts(self):
        """The definition text matches; the operational check does not. Prose
        agreement must not mask a criterion disagreement."""
        a = definition_claim("p1", definition="an attack succeeds", criterion="out-of-policy call")
        b = definition_claim("p2", definition="an attack succeeds", criterion="harmful end state")
        assert len(scan(a, b)) == 1

    def test_prose_fallback_when_no_criterion(self):
        """A paper that states no operational criterion is still compared, on
        its definition text, rather than silently matching everything."""
        a = definition_claim("p1", criterion=None, definition="any policy rule is broken")
        b = definition_claim("p2", criterion=None, definition="the task outcome is harmful")
        assert len(scan(a, b)) == 1


class TestSeverity:
    """Operational definitions block; prose-only ones are material (the 7.6
    ladder: BLOCKING means code cannot be written without deciding)."""

    def test_operational_criteria_block(self):
        a = definition_claim("p1", criterion="any out-of-policy tool call")
        b = definition_claim("p2", criterion="a harmful end state is reached")
        assert scan(a, b)[0].severity == "BLOCKING"

    def test_prose_only_is_material(self):
        a = definition_claim("p1", criterion=None, definition="any policy rule is broken")
        b = definition_claim("p2", criterion=None, definition="the task outcome is harmful")
        assert scan(a, b)[0].severity == "MATERIAL"

    def test_one_operational_side_is_enough_to_block(self):
        """If either side reduces to a concrete check, code diverges."""
        a = definition_claim("p1", criterion="any out-of-policy tool call")
        b = definition_claim("p2", criterion=None, definition="the task outcome is harmful")
        assert scan(a, b)[0].severity == "BLOCKING"


class TestWhatIsNotAConflict:
    def test_different_terms_never_compare(self):
        a = definition_claim("p1", term="attack_success", criterion="out-of-policy call")
        b = definition_claim("p2", term="task_failure", criterion="harmful end state")
        assert scan(a, b) == []

    def test_one_paper_alone_is_not_a_conflict(self):
        a = definition_claim("p1", criterion="out-of-policy call")
        b = definition_claim("p1", criterion="harmful end state")
        assert scan(a, b) == []

    def test_different_conditions_are_not_compared(self):
        """ER-04 applies to definitions too: a term defined for one setting does
        not contradict the same term defined for another."""
        a = definition_claim("p1", criterion="out-of-policy call", condition="training")
        b = definition_claim("p2", criterion="harmful end state", condition="deployment")
        assert scan(a, b) == []

    def test_a_prior_work_definition_is_not_a_position(self):
        """A paper restating a predecessor's definition is not asserting its
        own, the same way background method descriptions are excluded."""
        a = definition_claim("p1", criterion="out-of-policy call")
        b = definition_claim("p2", criterion="harmful end state", attribution="prior_work")
        assert scan(a, b) == []

    def test_wording_drift_in_one_paper_is_one_position(self):
        """Section-batch extraction states one criterion in varying words; those
        must collapse, exactly as approaches do in method conflicts."""
        a = definition_claim("p1", criterion="the agent calls a tool outside the granted set")
        b = definition_claim("p1", criterion="agent calls a tool outside the granted set")
        c = definition_claim("p2", criterion="the agent reaches a harmful end state")

        found = scan(a, b, c)

        assert len(found) == 1
        papers = [p.paper_id for p in found[0].positions]
        assert papers.count("p1") == 1, "one criterion, however many times it was worded"


class TestResolutionLivesInPolicy:
    def test_the_detector_permits_auto_resolution(self):
        """Unlike METHOD_CONFLICT: a definition has a right answer when its
        author is known, so the detector does not forbid resolution - the
        restraint lives in the policy, visible and configurable."""
        assert DefinitionConflictDetector.auto_resolvable is True

    def test_the_policy_escalates_with_a_named_rule(self):
        """Primacy would decide it, but primacy is unknown until a citation
        graph exists, so a named rule escalates rather than the bare fallback -
        so the escalation is auditable, not implicit."""
        a = definition_claim("p1", criterion="any out-of-policy tool call")
        b = definition_claim("p2", criterion="a harmful end state is reached")
        found = scan(a, b)[0]

        engine = PolicyEngine(POLICY, auto_resolvable={"DEFINITION_CONFLICT": True})
        res = engine.resolve_one(found)

        assert res.is_open
        assert res.outcome == "ESCALATED"
        assert res.rule_fired == "definition_conflicts_escalate"


@pytest.fixture
def doc():
    return make_doc()


class TestDefinitionExtraction:
    @staticmethod
    def extract(doc, items):
        return DefinitionExtractor(StubProvider([items])).extract(doc)

    def test_a_definition_is_extracted(self, doc):
        result = self.extract(
            doc,
            [
                {
                    "term": "attack_success",
                    "definition": "an attack succeeds when a tool call is out of policy",
                    "criterion": "any tool call outside the granted capability set",
                    "quote": "learning rate of 0.0001",
                }
            ],
        )
        assert len(result.claims) == 1
        assert result.claims[0].type == "definition"

    def test_term_names_are_canonicalized(self, doc):
        """Two papers must reach the same key or their disagreement about what
        counts as a successful attack is invisible."""
        result = self.extract(
            doc,
            [
                {
                    "term": "Attack Success Rate",
                    "definition": "fraction of attempts that succeed",
                    "quote": "learning rate of 0.0001",
                }
            ],
        )
        assert result.claims[0].payload["term"] == "attack_success"

    def test_an_empty_criterion_is_normalized_to_none(self, doc):
        """An empty string is not a criterion; left as "" it would read as a
        distinct check that conflicts with every real one."""
        result = self.extract(
            doc,
            [
                {
                    "term": "robustness",
                    "definition": "resistance to perturbation",
                    "criterion": "   ",
                    "quote": "learning rate of 0.0001",
                }
            ],
        )
        assert result.claims[0].payload["criterion"] is None

    def test_own_is_the_default_attribution(self, doc):
        result = self.extract(
            doc,
            [
                {
                    "term": "utility",
                    "definition": "task completion rate",
                    "quote": "learning rate of 0.0001",
                }
            ],
        )
        assert result.claims[0].payload["attribution"] == "own"

    def test_a_fabricated_quote_is_rejected(self, doc):
        result = self.extract(
            doc,
            [
                {
                    "term": "utility",
                    "definition": "task completion rate",
                    "quote": "a sentence that is nowhere in this paper",
                }
            ],
        )
        assert result.claims == []
        assert result.rejected
