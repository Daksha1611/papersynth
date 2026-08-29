"""The math layer (section 8.1), which is the R-01 mitigation.

Corrupted equations producing silently wrong specs is the highest
likelihood-and-impact risk in the register. Three behaviours respond to it -
the confidence penalty, the symbol_check corruption heuristic, and the
document fidelity flag - and all three keyed off a value nothing produced until
this module existed. These tests exist to keep it reachable.
"""

from __future__ import annotations

import pytest

from papersynth.core.document import RawEquation
from papersynth.ingest.math_layer import (
    NullRecoverer,
    assess,
    build_recoverer,
    review,
)


def equation(latex: str, **kwargs) -> RawEquation:
    return RawEquation(latex=latex, section_index=0, paragraph_index=0, **kwargs)


class TestDetection:
    @pytest.mark.parametrize(
        "latex",
        [
            r"E = mc^2",
            r"\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(\frac{QK^{\top}}{\sqrt{d_k}})V",
            r"\left( \sum_{i=1}^{n} x_i \right)",
            r"\alpha \beta \gamma \int \partial \nabla",
        ],
    )
    def test_well_formed_math_is_reliable(self, latex):
        """Real display math is symbol-dense. Flagging it would make the
        penalty meaningless by applying it everywhere."""
        assert assess(latex).reliable, assess(latex).reasons

    def test_unbalanced_braces_are_caught(self):
        result = assess(r"\frac{QK^{\top}{\sqrt{d_k}}")
        assert not result.reliable
        assert any("unbalanced" in r for r in result.reasons)

    def test_unbalanced_left_right_is_caught(self):
        result = assess(r"\left( a + b")
        assert not result.reliable
        assert any("left" in r for r in result.reasons)

    def test_replacement_glyphs_are_caught(self):
        """What a text layer emits when it cannot map a subsetted math font."""
        result = assess("x = � + ��")
        assert not result.reliable
        assert any("unmappable" in r for r in result.reasons)

    def test_private_use_codepoints_are_caught(self):
        """Where a subsetted font lands when its encoding is missing."""
        result = assess("y =    plus more text here")
        assert not result.reliable

    def test_a_fragment_is_caught(self):
        result = assess("x =")
        assert not result.reliable
        assert any("fragment" in r for r in result.reasons)

    def test_empty_math_is_caught(self):
        assert not assess("   ").reliable

    def test_escaped_delimiters_do_not_count_as_unbalanced(self):
        """A literal brace in LaTeX is escaped and is not a grouping brace."""
        assert assess(r"\{ a, b \} \cup \{ c \}").reliable


class TestReview:
    def test_reliable_equations_are_left_alone(self):
        reviewed, warnings = review([equation(r"E = mc^2")])
        assert reviewed[0].source_fidelity == "text_layer"
        assert warnings == []

    def test_damaged_math_without_a_backend_is_marked_suspect(self):
        """Not ocr_recovered: claiming OCR ran when it did not would
        misdescribe how the text was obtained, and the two situations need
        different follow-up."""
        reviewed, warnings = review([equation(r"\frac{a}{b")])

        assert reviewed[0].source_fidelity == "text_layer_suspect"
        assert "no OCR backend installed" in warnings[0]

    def test_the_warning_says_what_was_wrong(self):
        _, warnings = review([equation(r"\left( a + b", label="eq:3")])
        assert "eq:3" in warnings[0]
        assert "unbalanced" in warnings[0]

    def test_latex_native_equations_are_never_reviewed(self):
        """They came from the author's source; there is no text layer to have
        mangled them."""
        reviewed, warnings = review([equation(r"\frac{a}{b", source_fidelity="latex_native")])
        assert reviewed[0].source_fidelity == "latex_native"
        assert warnings == []

    def test_a_working_backend_produces_ocr_recovered(self):
        class Recoverer:
            available = True

            def recover(self, eq, pdf_path):
                return r"\frac{a}{b}"

        reviewed, warnings = review([equation(r"\frac{a}{b")], recoverer=Recoverer())

        assert reviewed[0].source_fidelity == "ocr_recovered"
        assert reviewed[0].latex == r"\frac{a}{b}"
        assert "re-recognized by OCR" in warnings[0]

    def test_a_failing_backend_is_distinguished_from_an_absent_one(self):
        """One wants the OCR checked, the other wants the extra installed."""

        class Failing:
            available = True

            def recover(self, eq, pdf_path):
                return None

        _, warnings = review([equation(r"\frac{a}{b")], recoverer=Failing())
        assert "OCR failed" in warnings[0]

    def test_the_default_backend_detects_without_recovering(self):
        """Detection must run on every install; recovery needs torch."""
        assert build_recoverer(enabled=False).available is False
        assert isinstance(build_recoverer(enabled=False), NullRecoverer)


class TestMitigationIsReachable:
    """The point of the module. Each of the three R-01 behaviours must
    actually fire on a document with degraded math."""

    def build_doc(self, latex: str):
        from papersynth.ingest.base import DocumentBuilder

        builder = DocumentBuilder(
            paper_id="degraded.001",
            title="Degraded",
            ingest_method="pdf",
            sha256="a" * 64,
            math_fidelity="text_layer",
        )
        index = builder.add_section("Method")
        builder.add_paragraph(index, "The attention weights are computed as shown below.")
        doc = builder.build()

        reviewed, warnings = review([equation(latex)])
        doc.equations = reviewed
        doc.warnings.extend(warnings)
        degraded = {e.source_fidelity for e in reviewed} & {"ocr_recovered", "text_layer_suspect"}
        if degraded:
            doc.math_fidelity = sorted(degraded)[0]
        return doc

    def test_document_fidelity_degrades(self):
        assert self.build_doc(r"\frac{a}{b").math_fidelity == "text_layer_suspect"

    def test_the_confidence_penalty_actually_fires(self):
        """The behaviour that was inert. A claim from a document with damaged
        math must carry less confidence than one from clean math."""
        from papersynth.extract.extractors.hyperparameter import HyperparameterExtractor
        from papersynth.llm.stub import StubProvider

        item = {
            "canonical_name": "dropout",
            "value": 0.1,
            "value_type": "float",
            "quote": "The attention weights are computed as shown below.",
            "stated_explicitly": True,
        }

        clean = self.build_doc(r"E = mc^2")
        damaged = self.build_doc(r"\frac{a}{b")

        clean_claim = HyperparameterExtractor(StubProvider([[item]])).extract(clean).claims[0]
        damaged_claim = HyperparameterExtractor(StubProvider([[item]])).extract(damaged).claims[0]

        assert clean_claim.confidence == 0.85, "clean text layer"
        assert damaged_claim.confidence == 0.6, "damaged math must cost confidence"
        assert damaged_claim.confidence < clean_claim.confidence

    def test_symbol_check_treats_suspect_math_as_corruption(self):
        """Garbled math reliably produces phantom symbols nothing defines, so
        one undefined symbol is enough when the source is already suspect."""
        from papersynth.core import ids
        from papersynth.core.models import Claim, Provenance
        from papersynth.verify.symbol_check import symbol_check

        def claim_with(fidelity: str) -> Claim:
            claim = Claim.build(
                paper_id="p",
                claim_type="equation",
                provenance=Provenance(
                    paper_id="p",
                    span_id="p#s0.p0.0",
                    section="Method",
                    page=1,
                    char_start=0,
                    char_end=10,
                    quote_hash=ids.quote_hash("x"),
                    extraction_method="llm",
                    extractor_version="equation@1.0.0",
                    confidence=0.9,
                ),
                payload={
                    "label": None,
                    "latex": "x = y + z",
                    "symbols": [
                        {"sym": "x", "role": "a", "defined_at": "p#s0.p0.0"},
                        {"sym": "y", "role": "b", "defined_at": "p#s0.p0.0"},
                        {"sym": "z", "role": "c", "defined_at": None},
                    ],
                    "undefined_symbols": ["z"],
                    "source_fidelity": fidelity,
                },
            )
            claim.status = "verified"
            return claim

        clean = symbol_check(claim_with("text_layer"))
        suspect = symbol_check(claim_with("text_layer_suspect"))

        assert clean.result == "fail" and "mangled" not in clean.reason
        assert suspect.result == "fail" and "mangled" in suspect.reason
