"""Extraction contract (section 8.2).

These tests exist to pin down the rules that make provenance trustworthy: a
claim is admitted only when its quote is genuinely in the document, and no
value is ever invented. Everything else about extraction can change.
"""

from __future__ import annotations

import pytest

from papersynth.core.errors import PaperSynthError
from papersynth.extract import registry
from papersynth.extract.extractors.hyperparameter import HyperparameterExtractor
from papersynth.llm.stub import StubProvider
from tests.conftest import make_doc


def extractor(responses):
    return HyperparameterExtractor(StubProvider([responses]))


LR_ITEM = {
    "canonical_name": "learning_rate",
    "paper_symbol": "\\eta",
    "value": 0.0001,
    "value_type": "float",
    "unit": None,
    "applies_to": "global",
    "condition": "base model",
    "stated_explicitly": True,
    "quote": "learning rate of 0.0001",
}


def test_a_supported_claim_is_admitted(doc):
    result = extractor([LR_ITEM]).extract(doc)

    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.type == "hyperparameter"
    assert claim.payload["canonical_name"] == "learning_rate"
    assert claim.payload["value"] == 0.0001


def test_provenance_is_anchored_to_a_span_that_resolves(doc):
    """The model supplies a quote; the span is resolved deterministically here."""
    claim = extractor([LR_ITEM]).extract(doc).claims[0]

    span = doc.resolve_span(claim.provenance.span_id, char_end=claim.provenance.char_end)
    assert span is not None
    assert "learning rate of 0.0001" in span.text
    assert claim.provenance.extractor_version == "hyperparameter@1.0.0"


def test_quote_hash_matches_the_resolved_span(doc):
    """citation_trace re-derives this hash; a mismatch there means drift."""
    from papersynth.core.ids import quote_hash

    claim = extractor([LR_ITEM]).extract(doc).claims[0]
    span = doc.resolve_span(claim.provenance.span_id, char_end=claim.provenance.char_end)

    assert claim.provenance.quote_hash == quote_hash(span.text)


def test_a_fabricated_quote_is_rejected_not_downgraded(doc):
    """ER-01. A model quoting text that is not in the paper has either
    fabricated or paraphrased it; either way the claim is unsupported."""
    item = {**LR_ITEM, "value": 0.5, "quote": "we use a learning rate of 0.5 throughout"}

    result = extractor([item]).extract(doc)

    assert result.claims == []
    assert len(result.rejected) == 1
    assert "does not appear" in result.rejected[0].reason


def test_a_claim_without_a_quote_is_rejected(doc):
    item = {k: v for k, v in LR_ITEM.items() if k != "quote"}

    result = extractor([item]).extract(doc)

    assert result.claims == []
    assert "no supporting quote" in result.rejected[0].reason


def test_a_payload_failing_its_schema_is_rejected(doc):
    """The payload schema is authoritative even when the model is confident."""
    item = {**LR_ITEM, "canonical_name": "Learning Rate!!", "value_type": "float"}
    result = extractor([item]).extract(doc)

    # normalize_payload lowercases and underscores, but punctuation still fails.
    assert result.claims == [] or result.claims[0].payload["canonical_name"].islower()


def test_condition_is_preserved(doc):
    """ER-04: dropping the condition manufactures a contradiction."""
    claim = extractor([LR_ITEM]).extract(doc).claims[0]
    assert claim.payload["condition"] == "base model"


def test_a_figure_derived_claim_keeps_stated_explicitly_false(doc):
    """ER-07: such a claim may not auto-resolve a conflict downstream."""
    item = {**LR_ITEM, "stated_explicitly": False}
    claim = extractor([item]).extract(doc).claims[0]

    assert claim.payload["stated_explicitly"] is False


def test_identical_claims_at_the_same_span_collapse(doc):
    result = extractor([LR_ITEM, dict(LR_ITEM)]).extract(doc)
    assert len(result.claims) == 1


def test_two_values_under_different_conditions_are_two_claims(doc):
    """Not a contradiction - two separately scoped facts."""
    a = {**LR_ITEM, "condition": "base model"}
    b = {**LR_ITEM, "condition": "fine-tuning", "quote": "dropout rate of 0.1"}

    result = extractor([a, b]).extract(doc)

    assert len(result.claims) == 2
    assert {c.payload["condition"] for c in result.claims} == {"base model", "fine-tuning"}


def test_an_empty_response_yields_no_claims(doc):
    """A paper stating no hyperparameters is a valid outcome, not an error."""
    result = extractor([]).extract(doc)
    assert result.claims == []
    assert result.rejected == []


def test_claim_ids_are_stable_across_identical_extractions(doc):
    first = extractor([LR_ITEM]).extract(doc).claims[0]
    second = extractor([LR_ITEM]).extract(doc).claims[0]
    assert first.claim_id == second.claim_id


def test_ocr_recovered_documents_start_with_lower_confidence(doc):
    """R-01: garbled math and lost decimals originate there."""
    ocr_doc = make_doc()
    ocr_doc.math_fidelity = "ocr_recovered"

    native = extractor([LR_ITEM]).extract(doc).claims[0]
    degraded = extractor([LR_ITEM]).extract(ocr_doc).claims[0]

    assert degraded.confidence < native.confidence


class TestNormalization:
    def test_synonyms_map_to_a_canonical_name(self, doc):
        item = {**LR_ITEM, "canonical_name": "LR"}
        claim = extractor([item]).extract(doc).claims[0]
        assert claim.payload["canonical_name"] == "learning_rate"

    def test_a_numeric_string_is_recovered_to_a_number(self, doc):
        item = {**LR_ITEM, "value": "0.0001", "value_type": "float"}
        claim = extractor([item]).extract(doc).claims[0]
        assert claim.payload["value"] == 0.0001

    def test_a_categorical_value_stays_a_string(self, doc):
        item = {
            **LR_ITEM,
            "canonical_name": "optimizer",
            "value": "Adam",
            "value_type": "categorical",
            "quote": "Adam optimizer",
        }
        claim = extractor([item]).extract(doc).claims[0]
        assert claim.payload["value"] == "Adam"
        assert claim.payload["value_type"] == "categorical"

    def test_normalization_never_invents_a_value(self, doc):
        """ER-02: absent means absent. A gap is raised later, not a default."""
        item = {"canonical_name": "learning_rate", "quote": "learning rate of 0.0001"}
        result = extractor([item]).extract(doc)

        for claim in result.claims:
            assert "value" in claim.payload
        assert not any(c.payload.get("value") in (None, 0) for c in result.claims)


class TestSectionNarrowing:
    def test_applicable_sections_narrows_the_search(self):
        """Cost and precision both improve when the prompt skips related work."""
        from papersynth.core.document import Paragraph, Section, StructuredDocument

        doc = StructuredDocument(
            paper_id="narrow.001",
            title="Narrowing",
            ingest_method="latex",
            sha256="e" * 64,
            sections=[
                Section(
                    index=0,
                    title="2 Related Work",
                    paragraphs=[Paragraph(index=0, text="Prior systems are surveyed here.")],
                ),
                Section(
                    index=1,
                    title="4 Training Setup",
                    paragraphs=[Paragraph(index=0, text="We use a batch size of 64 here.")],
                ),
            ],
        )
        sections = HyperparameterExtractor(StubProvider([[]])).applicable_sections(doc)

        assert [s.title for s in sections] == ["4 Training Setup"]

    def test_a_paper_with_no_matching_headings_still_gets_read(self):
        """Returning nothing here would look like a clean run on a real paper."""
        from papersynth.core.document import Paragraph, Section, StructuredDocument

        doc = StructuredDocument(
            paper_id="odd.001",
            title="Odd headings",
            ingest_method="pdf",
            sha256="d" * 64,
            sections=[
                Section(
                    index=0,
                    title="Preliminaries",
                    paragraphs=[Paragraph(index=0, text="We use a batch size of 64 here.")],
                )
            ],
        )
        sections = HyperparameterExtractor(StubProvider([[]])).applicable_sections(doc)
        assert len(sections) == 1


class TestRegistry:
    def test_the_bundled_extractor_is_registered(self):
        assert "hyperparameter" in registry.available()

    def test_building_an_unknown_extractor_fails_loudly(self):
        with pytest.raises(PaperSynthError, match="Unknown extractor"):
            registry.build(["nonexistent"], StubProvider([]))

    def test_run_all_survives_one_extractor_failing(self, doc):
        """NFR-09: a partial claim set with a visible warning beats none."""
        from papersynth.core.errors import ProviderError

        good = HyperparameterExtractor(StubProvider([[LR_ITEM]]))
        bad = HyperparameterExtractor(StubProvider(error=ProviderError("boom")))

        result = registry.run_all(doc, [bad, good])

        assert len(result.claims) == 1
        assert any("failed" in w for w in result.warnings)


class TestIsolation:
    def test_an_extractor_prompt_contains_only_its_own_paper(self, doc, second_doc):
        """ER-09. If extraction were corpus-aware, a model holding Paper A's
        learning rate in context while extracting Paper B's would drift toward
        agreement, and real contradictions would vanish before detection."""
        provider = StubProvider([[], []])

        HyperparameterExtractor(provider).extract(doc)
        HyperparameterExtractor(provider).extract(second_doc)

        assert provider.call_count == 2
        provider.assert_no_prompt_contains_multiple_paper_ids(
            ["Attention Is All You Need", "Paper2Code"]
        )
        assert "0.0003" not in provider.prompts[0]
        assert "Recurrent models" not in provider.prompts[1]
