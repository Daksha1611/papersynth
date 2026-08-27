"""Equation and algorithm extraction.

Both anchor to objects located at ingestion rather than to a model quote, and
both refuse to let the model author content it should only be describing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from papersynth.core.models import ClaimSet
from papersynth.extract.extractors.algorithm import AlgorithmExtractor
from papersynth.extract.extractors.equation import EquationExtractor
from papersynth.ingest.latex import LatexIngestor
from papersynth.llm.stub import StubProvider
from papersynth.verify import Verifier
from papersynth.verify.symbol_check import symbol_check

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample_paper.tex"


@pytest.fixture(scope="module")
def doc():
    return LatexIngestor().ingest(str(FIXTURE), paper_id="test.0001")


def equation_claim(doc, symbols):
    provider = StubProvider([[{"index": 0, "symbols": symbols}]])
    return EquationExtractor(provider).extract(doc)


DEFINED = [
    {
        "sym": "Q",
        "role": "query matrix",
        "shape": "[n, d_k]",
        "defined_by": "Attention is computed over queries, keys, and values",
    },
    {
        "sym": "d_k",
        "role": "key dimension",
        "shape": None,
        "defined_by": "where d_k is the key dimension",
    },
]


class TestEquationExtraction:
    def test_the_latex_comes_from_ingestion_not_the_model(self, doc):
        """A model asked to reproduce an equation will occasionally 'correct'
        it, and a silently corrected equation is the R-01 failure arriving
        through the component meant to catch it."""
        claim = equation_claim(doc, DEFINED).claims[0]
        assert claim.payload["latex"] == doc.equations[0].latex

    def test_source_fidelity_is_carried_through(self, doc):
        claim = equation_claim(doc, DEFINED).claims[0]
        assert claim.payload["source_fidelity"] == "latex_native"

    def test_provenance_needs_no_model_quote(self, doc):
        """The equation was located at parse time, so its span cannot be wrong
        the way a hallucinated quote can."""
        claim = equation_claim(doc, DEFINED).claims[0]
        assert doc.resolve_span(claim.provenance.span_id) is not None

    def test_a_definition_that_resolves_becomes_a_span(self, doc):
        claim = equation_claim(doc, DEFINED).claims[0]
        for symbol in claim.payload["symbols"]:
            assert symbol["defined_at"] is not None
            assert doc.resolve_span(symbol["defined_at"]) is not None

    def test_a_definition_the_model_invented_marks_the_symbol_undefined(self, doc):
        """A model claiming a symbol is defined is not evidence that it is."""
        symbols = [
            *DEFINED,
            {"sym": "Z", "role": "unexplained", "defined_by": "no such sentence in this paper"},
        ]
        claim = equation_claim(doc, symbols).claims[0]

        assert claim.payload["undefined_symbols"] == ["Z"]
        assert next(s for s in claim.payload["symbols"] if s["sym"] == "Z")["defined_at"] is None

    def test_a_symbol_with_no_definition_is_undefined(self, doc):
        symbols = [*DEFINED, {"sym": "W", "role": "weights", "defined_by": None}]
        claim = equation_claim(doc, symbols).claims[0]
        assert "W" in claim.payload["undefined_symbols"]

    def test_operators_are_not_treated_as_symbols(self, doc):
        """softmax is notation, not a quantity to implement, and listing it
        would flag it undefined forever."""
        symbols = [*DEFINED, {"sym": "softmax", "role": "normalizer", "defined_by": None}]
        claim = equation_claim(doc, symbols).claims[0]

        assert "softmax" not in [s["sym"] for s in claim.payload["symbols"]]
        assert "softmax" not in claim.payload["undefined_symbols"]

    def test_a_document_with_no_equations_extracts_nothing(self):
        from papersynth.core.document import Paragraph, Section, StructuredDocument

        empty = StructuredDocument(
            paper_id="none.001",
            title="No math",
            ingest_method="pdf",
            sha256="f" * 64,
            sections=[
                Section(
                    index=0,
                    title="Body",
                    paragraphs=[Paragraph(index=0, text="Prose only, no equations here.")],
                )
            ],
        )
        provider = StubProvider([])
        result = EquationExtractor(provider).extract(empty)

        assert result.claims == []
        assert provider.call_count == 0, "no equations means no call to spend"

    def test_an_out_of_range_index_is_rejected(self, doc):
        result = equation_claim(doc, DEFINED)
        bad = EquationExtractor(StubProvider([[{"index": 99, "symbols": DEFINED}]])).extract(doc)

        assert result.claims and not bad.claims


class TestSymbolCheck:
    def test_a_fully_defined_equation_passes(self, doc):
        claim = equation_claim(doc, DEFINED).claims[0]
        assert symbol_check(claim).result == "pass"

    def test_an_undefined_symbol_fails(self, doc):
        """An equation with undefined symbols is not implementable."""
        symbols = [*DEFINED, {"sym": "W", "role": "weights", "defined_by": None}]
        claim = equation_claim(doc, symbols).claims[0]

        outcome = symbol_check(claim)
        assert outcome.result == "fail"
        assert "W" in outcome.reason

    def test_mostly_undefined_symbols_read_as_corruption(self, doc):
        """Garbled OCR reliably produces phantom symbols nothing defines."""
        symbols = [{"sym": s, "role": "?", "defined_by": None} for s in ("W", "X", "Y", "Z")]
        claim = equation_claim(doc, symbols).claims[0]

        assert "mangled during" in symbol_check(claim).reason

    def test_a_non_equation_claim_is_not_checked(self, doc):
        from tests.conftest import make_claim

        assert symbol_check(make_claim(doc)).result == "n/a"

    def test_the_verifier_rejects_an_unclosed_equation(self, doc):
        symbols = [*DEFINED, {"sym": "W", "role": "weights", "defined_by": None}]
        claim = equation_claim(doc, symbols).claims[0]

        result, report = Verifier(entailment=False).verify(
            ClaimSet(paper_id=doc.paper_id, claims=[claim]), doc
        )

        assert result.claims[0].status == "rejected"
        assert report.rejection_reasons == {"symbol_check": 1}

    def test_the_verifier_accepts_a_closed_equation(self, doc):
        claim = equation_claim(doc, DEFINED).claims[0]
        result, _ = Verifier(entailment=False).verify(
            ClaimSet(paper_id=doc.paper_id, claims=[claim]), doc
        )
        assert result.claims[0].status == "verified"
        assert result.claims[0].verification.symbol_check == "pass"


def algorithm_claim(doc, **overrides):
    item = {
        "index": 0,
        "name": "Training loop",
        "inputs": [{"name": "D", "type": "Dataset", "description": "training data"}],
        "outputs": [{"name": "theta", "type": "Parameters", "description": None}],
        "steps": [
            {"index": 1, "text": "Initialize parameters"},
            {"index": 2, "text": "Sample a minibatch"},
        ],
        "complexity": None,
        "preconditions": [],
    }
    item.update(overrides)
    return AlgorithmExtractor(StubProvider([[item]])).extract(doc)


class TestAlgorithmExtraction:
    def test_steps_are_transcribed_in_order(self, doc):
        claim = algorithm_claim(doc).claims[0]
        assert [s["text"] for s in claim.payload["steps"]] == [
            "Initialize parameters",
            "Sample a minibatch",
        ]

    def test_step_numbering_is_made_contiguous(self, doc):
        """A gap in the numbering reads as a dropped step to an implementer."""
        claim = algorithm_claim(
            doc,
            steps=[
                {"index": 1, "text": "First"},
                {"index": 7, "text": "Second"},
                {"index": 9, "text": "Third"},
            ],
        ).claims[0]
        assert [s["index"] for s in claim.payload["steps"]] == [1, 2, 3]

    def test_the_label_comes_from_ingestion(self, doc):
        claim = algorithm_claim(doc).claims[0]
        assert claim.payload["label"] == "alg:train"

    def test_provenance_needs_no_model_quote(self, doc):
        claim = algorithm_claim(doc).claims[0]
        assert doc.resolve_span(claim.provenance.span_id) is not None

    def test_an_unstated_complexity_stays_null(self, doc):
        """A plausible-looking complexity the authors never claimed is a
        fabrication, not a helpful default."""
        claim = algorithm_claim(doc, complexity={"time": None, "space": None}).claims[0]
        assert claim.payload["complexity"] is None

    def test_a_stated_complexity_is_kept(self, doc):
        claim = algorithm_claim(doc, complexity={"time": "O(n^2 d)", "space": None}).claims[0]
        assert claim.payload["complexity"]["time"] == "O(n^2 d)"

    def test_an_algorithm_with_no_steps_is_rejected(self, doc):
        """The payload schema requires at least one step; an algorithm with
        none looks complete and is not."""
        result = algorithm_claim(doc, steps=[])
        assert result.claims == []
        assert result.rejected

    def test_empty_step_text_is_dropped(self, doc):
        claim = algorithm_claim(
            doc,
            steps=[{"index": 1, "text": "Real step"}, {"index": 2, "text": "   "}],
        ).claims[0]
        assert len(claim.payload["steps"]) == 1

    def test_ports_without_a_name_are_dropped(self, doc):
        claim = algorithm_claim(doc, inputs=[{"name": "D"}, {"type": "Nameless"}]).claims[0]
        assert [p["name"] for p in claim.payload["inputs"]] == ["D"]

    def test_a_document_with_no_algorithms_spends_no_call(self):
        from papersynth.core.document import Paragraph, Section, StructuredDocument

        empty = StructuredDocument(
            paper_id="none.002",
            title="No algorithms",
            ingest_method="pdf",
            sha256="e" * 64,
            sections=[
                Section(
                    index=0,
                    title="Body",
                    paragraphs=[Paragraph(index=0, text="Prose only, nothing to transcribe.")],
                )
            ],
        )
        provider = StubProvider([])
        result = AlgorithmExtractor(provider).extract(empty)

        assert result.claims == []
        assert provider.call_count == 0
