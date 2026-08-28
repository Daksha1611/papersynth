"""Design decisions and METHOD_CONFLICT (section 7.6).

The disagreements here carry no number, so nothing else in the pipeline can
see them. BERT using next-sentence prediction and RoBERTa removing it is one
of the sharpest disagreements in that literature and was invisible until this.
"""

from __future__ import annotations

import pytest

from papersynth.align import Aligner
from papersynth.contradict import ContradictionScan, MethodConflictDetector
from papersynth.core import ids
from papersynth.core.models import Claim, ClaimSet, Provenance
from papersynth.extract.extractors.method import MethodExtractor
from papersynth.llm.stub import StubProvider
from papersynth.reconcile import Policy, PolicyEngine
from tests.conftest import make_doc

POLICY = Policy.load("config/reconcile_policy.yaml")


def method_claim(
    paper,
    sub_problem,
    approach,
    adopted=True,
    condition=None,
    rationale=None,
    attribution="own",
):
    payload = {
        "sub_problem": sub_problem,
        "approach": approach,
        "adopted": adopted,
        "alternatives_rejected": [],
        "rationale": rationale,
        "attribution": attribution,
        "applies_to": "global",
        "condition": condition,
        "stated_explicitly": True,
    }
    provenance = Provenance(
        paper_id=paper,
        span_id=f"{paper}#s1.p0.0",
        section="Method",
        page=1,
        char_start=0,
        char_end=40,
        quote_hash=ids.quote_hash("text"),
        extraction_method="llm",
        extractor_version="method@1.0.0",
        confidence=0.9,
    )
    claim = Claim.build(paper_id=paper, claim_type="method", provenance=provenance, payload=payload)
    claim.status = "verified"
    return claim


def scan(*claims):
    sets = [ClaimSet(paper_id=c.paper_id, claims=[c]) for c in claims]
    graph, _ = Aligner().align(sets)
    return ContradictionScan().run(graph)


class TestAdoptVersusReject:
    def test_removing_what_another_paper_uses_is_a_conflict(self):
        """The case that motivated this detector. Comparing approach strings
        alone would call these two papers in agreement."""
        bert = method_claim("bert", "sentence_level_objective", "next sentence prediction (NSP)")
        roberta = method_claim(
            "roberta",
            "sentence_level_objective",
            "next sentence prediction (NSP)",
            adopted=False,
            rationale="removing NSP matches or improves downstream performance",
        )

        found = scan(bert, roberta)

        assert len(found) == 1
        assert found[0].type == "METHOD_CONFLICT"

    def test_the_removal_is_rendered_as_a_removal(self):
        bert = method_claim("bert", "sentence_level_objective", "NSP")
        roberta = method_claim("roberta", "sentence_level_objective", "NSP", adopted=False)

        positions = {p.paper_id: p.position for p in scan(bert, roberta)[0].positions}

        assert positions["roberta"].startswith("removes")
        assert not positions["bert"].startswith("removes")

    def test_an_acronym_and_its_expansion_are_one_approach(self):
        """Otherwise adopting the spelled-out form and removing the acronym
        looks like rival approaches rather than a direct contradiction."""
        bert = method_claim("bert", "sentence_level_objective", "next sentence prediction (NSP)")
        roberta = method_claim("roberta", "sentence_level_objective", "NSP", adopted=False)

        found = scan(bert, roberta)
        assert len(found) == 1
        assert "one adopts it, another explicitly removes it" in found[0].description

    def test_a_disputed_approach_blocks(self):
        bert = method_claim("bert", "sentence_level_objective", "NSP")
        roberta = method_claim("roberta", "sentence_level_objective", "NSP", adopted=False)
        assert scan(bert, roberta)[0].severity == "BLOCKING"

    def test_the_rationale_reaches_the_reviewer(self):
        """It is the evidence a human needs to actually decide."""
        bert = method_claim("bert", "sentence_level_objective", "NSP")
        roberta = method_claim(
            "roberta",
            "sentence_level_objective",
            "NSP",
            adopted=False,
            rationale="removing it improves downstream performance",
        )
        positions = {p.paper_id: p.position for p in scan(bert, roberta)[0].positions}
        assert "improves downstream" in positions["roberta"]


class TestRivalApproaches:
    def test_two_different_adopted_approaches_conflict(self):
        bert = method_claim("bert", "sentence_level_objective", "next sentence prediction")
        albert = method_claim("albert", "sentence_level_objective", "sentence order prediction")

        found = scan(bert, albert)

        assert len(found) == 1
        assert "incompatible approaches" in found[0].description

    def test_agreement_produces_no_conflict(self):
        a = method_claim("p1", "positional_encoding", "learned absolute embeddings")
        b = method_claim("p2", "positional_encoding", "learned absolute embeddings")
        assert scan(a, b) == []

    def test_one_paper_declining_an_alternative_is_not_a_conflict(self):
        """Both adopt the same thing; one merely notes what it did not take."""
        a = method_claim("p1", "positional_encoding", "learned embeddings")
        b = method_claim("p2", "positional_encoding", "learned embeddings")
        b.payload["alternatives_rejected"] = ["sinusoidal"]
        assert scan(a, b) == []

    def test_different_sub_problems_never_compare(self):
        a = method_claim("p1", "positional_encoding", "sinusoidal")
        b = method_claim("p2", "tokenizer", "byte-pair encoding")
        assert scan(a, b) == []

    def test_one_paper_alone_is_not_a_conflict(self):
        a = method_claim("p1", "positional_encoding", "sinusoidal")
        b = method_claim("p1", "positional_encoding", "learned")
        assert scan(a, b) == []

    def test_different_conditions_are_not_compared(self):
        """ER-04 applies to methods as much as to values."""
        a = method_claim("p1", "masking_strategy", "static masking", condition="pretraining")
        b = method_claim("p2", "masking_strategy", "dynamic masking", condition="fine-tuning")
        assert scan(a, b) == []

    def test_a_non_structural_conflict_is_material(self):
        a = method_claim("p1", "learning_rate_schedule", "linear decay")
        b = method_claim("p2", "learning_rate_schedule", "cosine decay")
        assert scan(a, b)[0].severity == "MATERIAL"


class TestDuplicateWording:
    """Extraction runs over section batches, so one paper describes a single
    decision several times in slightly different words. Each of those became
    its own position in the conflict until approaches were matched fuzzily."""

    def test_one_decision_worded_twice_is_one_position(self):
        a = method_claim(
            "bert", "masking_strategy", "80% mask token, 10% random token, 10% unchanged"
        )
        b = method_claim(
            "bert", "masking_strategy", "80% [MASK] token, 10% random token, 10% unchanged token"
        )
        albert = method_claim("albert", "masking_strategy", "n-gram masking")

        found = scan(a, b, albert)

        assert len(found) == 1
        papers = [p.paper_id for p in found[0].positions]
        assert papers.count("bert") == 1, "one decision, however many times it was worded"

    def test_a_qualifier_does_not_make_a_rival_approach(self):
        a = method_claim("bert", "tokenizer", "case-preserving WordPiece")
        b = method_claim("bert", "tokenizer", "WordPiece")
        albert = method_claim("albert", "tokenizer", "SentencePiece")

        found = scan(a, b, albert)
        assert len(found[0].positions) == 2

    def test_genuinely_different_approaches_stay_apart(self):
        """The dedup must not swallow a real disagreement."""
        a = method_claim("p1", "tokenizer", "WordPiece")
        b = method_claim("p2", "tokenizer", "SentencePiece")
        assert len(scan(a, b)[0].positions) == 2

    def test_similar_wording_with_a_different_head_word_stays_apart(self):
        a = method_claim("p1", "positional_encoding", "sinusoidal positional encoding")
        b = method_claim("p2", "positional_encoding", "learned positional encoding")
        assert len(scan(a, b)[0].positions) == 2


class TestAttribution:
    """A background section explaining a predecessor's design is not a decision
    by the paper containing it. RoBERTa describes BERT's NSP at length before
    removing it, and reading that as RoBERTa's own choice made RoBERTa look
    like an NSP proponent - the opposite of its argument."""

    def test_a_prior_work_description_is_not_a_position(self):
        bert = method_claim("bert", "sentence_level_objective", "NSP")
        roberta_background = method_claim(
            "roberta", "sentence_level_objective", "NSP", attribution="prior_work"
        )
        albert = method_claim("albert", "sentence_level_objective", "SOP")

        found = scan(bert, roberta_background, albert)

        assert len(found) == 1
        assert "roberta" not in {p.paper_id for p in found[0].positions}

    def test_a_paper_is_not_made_to_contradict_itself_by_background(self):
        """RoBERTa describing NSP and removing NSP is one coherent position."""
        describes = method_claim(
            "roberta", "sentence_level_objective", "NSP", attribution="prior_work"
        )
        removes = method_claim("roberta", "sentence_level_objective", "NSP", adopted=False)
        bert = method_claim("bert", "sentence_level_objective", "NSP")

        found = scan(describes, removes, bert)

        assert len(found) == 1
        roberta = [p for p in found[0].positions if p.paper_id == "roberta"]
        assert len(roberta) == 1
        assert roberta[0].position.startswith("removes")

    def test_background_alone_yields_no_conflict(self):
        """Two papers both describing a third paper's design do not disagree."""
        a = method_claim("p1", "tokenizer", "WordPiece", attribution="prior_work")
        b = method_claim("p2", "tokenizer", "SentencePiece", attribution="prior_work")
        assert scan(a, b) == []

    def test_own_is_the_default(self):
        """A decision wrongly kept is visible and dismissable; one wrongly
        discarded as background is simply absent."""
        from papersynth.extract.extractors.method import MethodExtractor
        from papersynth.llm.stub import StubProvider

        result = MethodExtractor(
            StubProvider(
                [
                    [
                        {
                            "sub_problem": "tokenizer",
                            "approach": "wordpiece",
                            "adopted": True,
                            "quote": "learning rate of 0.0001",
                        }
                    ]
                ]
            )
        ).extract(make_doc())

        assert result.claims[0].payload["attribution"] == "own"


class TestReferenceTrace:
    """Section 10.2: a claim whose span cites another work carries that
    reference, so a borrowed method is not credited to the citing paper."""

    def test_a_cited_span_records_the_reference(self):
        from papersynth.extract.base import reference_trace
        from papersynth.ingest.latex import LatexIngestor

        doc = LatexIngestor().ingest("tests/fixtures/sample_paper.tex", paper_id="t.1")
        secondary = reference_trace("following [bahdanau2014], we use attention", doc)

        assert secondary is not None
        assert secondary.cited_ref == "bahdanau2014"
        assert secondary.resolved_paper_id == "1409.0473"

    def test_an_uncited_span_records_nothing(self):
        from papersynth.extract.base import reference_trace
        from papersynth.ingest.latex import LatexIngestor

        doc = LatexIngestor().ingest("tests/fixtures/sample_paper.tex", paper_id="t.1")
        assert reference_trace("we use a learning rate of 0.0001", doc) is None

    def test_a_special_token_is_not_a_citation(self):
        """BERT writes [MASK], [CLS] and [SEP] constantly, and reading those as
        references attributed BERT's own masking decisions to a cited work."""
        from papersynth.extract.base import reference_trace
        from papersynth.ingest.latex import LatexIngestor

        doc = LatexIngestor().ingest("tests/fixtures/sample_paper.tex", paper_id="t.1")
        for token in ("[MASK]", "[CLS]", "[SEP]"):
            text = f"we replace the word with the {token} token during training"
            assert reference_trace(text, doc) is None, token

    def test_a_citation_after_a_special_token_is_still_found(self):
        from papersynth.extract.base import reference_trace
        from papersynth.ingest.latex import LatexIngestor

        doc = LatexIngestor().ingest("tests/fixtures/sample_paper.tex", paper_id="t.1")
        secondary = reference_trace("the [MASK] token, following [bahdanau2014], is used here", doc)
        assert secondary is not None
        assert secondary.cited_ref == "bahdanau2014"

    def test_an_unresolvable_key_is_still_recorded(self):
        """The citation is evidence even when the bibliography did not parse."""
        from papersynth.extract.base import reference_trace
        from papersynth.ingest.latex import LatexIngestor

        doc = LatexIngestor().ingest("tests/fixtures/sample_paper.tex", paper_id="t.1")
        secondary = reference_trace("as shown in [unknownkey2020] the method works", doc)

        assert secondary is not None
        assert secondary.resolved_paper_id is None


class TestNeverAutoResolved:
    def test_the_detector_forbids_auto_resolution(self):
        """Section 10.3: choosing between incompatible approaches is an
        engineering decision, not a lookup."""
        assert MethodConflictDetector.auto_resolvable is False

    @pytest.mark.parametrize("severity", ["BLOCKING", "MATERIAL"])
    def test_the_policy_escalates_method_conflicts(self, severity):
        a = method_claim("p1", "learning_rate_schedule", "linear decay", condition="pretraining")
        b = method_claim("p2", "learning_rate_schedule", "cosine decay", condition="pretraining")
        found = scan(a, b)[0]
        found.severity = severity

        engine = PolicyEngine(POLICY, auto_resolvable={"METHOD_CONFLICT": False})
        assert engine.resolve_one(found).is_open


@pytest.fixture
def doc():
    return make_doc()


class TestMethodExtraction:
    @staticmethod
    def extract(doc, items):
        return MethodExtractor(StubProvider([items])).extract(doc)

    def test_a_decision_is_extracted(self, doc):
        result = self.extract(
            doc,
            [
                {
                    "sub_problem": "sentence_level_objective",
                    "approach": "next sentence prediction",
                    "adopted": True,
                    "quote": "learning rate of 0.0001",
                }
            ],
        )
        assert len(result.claims) == 1
        assert result.claims[0].type == "method"

    def test_sub_problem_names_are_canonicalized(self, doc):
        """Two papers must reach the same key or their disagreement is
        invisible."""
        result = self.extract(
            doc,
            [
                {
                    "sub_problem": "Inter-Sentence Objective",
                    "approach": "NSP",
                    "adopted": True,
                    "quote": "learning rate of 0.0001",
                }
            ],
        )
        assert result.claims[0].payload["sub_problem"] == "sentence_level_objective"

    def test_adopted_defaults_to_true(self, doc):
        """A rejection is never the silent default."""
        result = self.extract(
            doc,
            [
                {
                    "sub_problem": "tokenizer",
                    "approach": "wordpiece",
                    "quote": "learning rate of 0.0001",
                }
            ],
        )
        assert result.claims[0].payload["adopted"] is True

    def test_a_removal_survives_as_a_claim(self, doc):
        result = self.extract(
            doc,
            [
                {
                    "sub_problem": "sentence_level_objective",
                    "approach": "NSP",
                    "adopted": False,
                    "quote": "learning rate of 0.0001",
                }
            ],
        )
        assert result.claims[0].payload["adopted"] is False

    def test_a_fabricated_quote_is_rejected(self, doc):
        result = self.extract(
            doc,
            [
                {
                    "sub_problem": "tokenizer",
                    "approach": "wordpiece",
                    "adopted": True,
                    "quote": "a sentence that is nowhere in this paper",
                }
            ],
        )
        assert result.claims == []
        assert result.rejected
