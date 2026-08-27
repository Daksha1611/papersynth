"""Equation annotation.

The load-bearing decision: the LaTeX is never sent back through the model. It
comes from ingestion - the paper's own source on the LaTeX path - and the model
only annotates what the symbols mean. A model asked to reproduce an equation
will occasionally 'correct' it, and a silently corrected equation is exactly
the R-01 failure this system exists to catch, arriving through the component
meant to catch it.

Provenance is likewise deterministic. An equation already has a known position
in the document, so its span comes from ingestion rather than from a model
quote - there is nothing for the model to get wrong.
"""

from __future__ import annotations

from typing import Any, ClassVar

from papersynth.core.document import RawEquation, Section, Span, StructuredDocument
from papersynth.extract.base import LLMExtractor, render_sections
from papersynth.extract.prompts import render
from papersynth.extract.registry import register

#: Notation rather than quantities to implement; a symbol table listing these
#: is noise, and they would each be flagged as undefined.
OPERATORS = frozenset(
    {
        "softmax",
        "exp",
        "log",
        "ln",
        "sum",
        "prod",
        "max",
        "min",
        "argmax",
        "argmin",
        "sin",
        "cos",
        "tan",
        "tanh",
        "sigmoid",
        "relu",
        "gelu",
        "sqrt",
        "abs",
        "det",
        "tr",
        "diag",
        "mean",
        "var",
        "std",
        "cov",
        "kl",
        "mathrm",
        "text",
        "frac",
        "left",
        "right",
        "cdot",
        "times",
        "top",
        "mathbb",
        "mathcal",
        "operatorname",
    }
)

_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "index": {"type": "integer"},
        "symbols": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sym": {"type": "string"},
                    "role": {"type": "string"},
                    "shape": {"type": ["string", "null"]},
                    "defined_by": {"type": ["string", "null"]},
                },
                "required": ["sym", "role"],
            },
        },
    },
    "required": ["index", "symbols"],
}


@register
class EquationExtractor(LLMExtractor):
    claim_type: ClassVar[str] = "equation"
    version: ClassVar[str] = "1.0.0"
    payload_schema_name: ClassVar[str] = "payload.equation.json"
    output_schema: ClassVar[dict[str, Any]] = {"type": "array", "items": _ITEM_SCHEMA}
    section_pattern: ClassVar[str] = ""
    system_prompt: ClassVar[str] = (
        "You annotate mathematical notation from research papers. You never "
        "alter the mathematics itself. You quote definitions verbatim, and you "
        "say plainly when a symbol is never defined."
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        #: Equations by index, populated per document so anchoring and payload
        #: assembly can reach the ingested object rather than trusting the model.
        self._equations: list[RawEquation] = []

    def extract(self, doc: StructuredDocument, sections: list[Section] | None = None) -> Any:
        self._equations = list(doc.equations)
        if not self._equations:
            from papersynth.extract.base import ExtractionResult

            return ExtractionResult()
        return super().extract(doc, sections)

    def build_prompt(self, doc: StructuredDocument, sections: list[Section]) -> str:
        listing = "\n\n".join(
            f"[{i}] {eq.label or 'unlabelled'}\n{eq.latex}" for i, eq in enumerate(self._equations)
        )
        return render(
            "extract_equation.md",
            sections=render_sections(doc, sections),
            equations=listing,
        )

    def anchor(
        self,
        item: dict[str, Any],
        quote: str | None,
        doc: StructuredDocument,
        section_indices: list[int],
    ) -> Span | None:
        """Anchor to the equation's own position, recorded at ingestion.

        No model quote is involved: the equation was located when the document
        was parsed, so its provenance cannot be wrong in the way a hallucinated
        quote can.
        """
        equation = self._equation_for(item)
        if equation is None:
            return None
        try:
            section = doc.sections[equation.section_index]
            paragraph = section.paragraphs[
                min(equation.paragraph_index, len(section.paragraphs) - 1)
            ]
        except (IndexError, ValueError):
            return None
        return doc.make_span(section.index, paragraph.index, 0, len(paragraph.text))

    def normalize_payload(self, payload: dict[str, Any], doc: StructuredDocument) -> dict[str, Any]:
        """Assemble the payload around the ingested LaTeX, not the model's text."""
        equation = self._equation_for(payload)
        if equation is None:
            # Without a matching ingested equation there is nothing to anchor
            # the annotation to; the schema check below will reject it.
            return {
                "latex": "",
                "symbols": [],
                "undefined_symbols": [],
                "source_fidelity": "text_layer",
            }

        symbols: list[dict[str, Any]] = []
        undefined: list[str] = []
        for raw in payload.get("symbols") or []:
            if not isinstance(raw, dict):
                continue
            sym = str(raw.get("sym", "")).strip()
            if not sym or _is_operator(sym):
                continue
            symbols.append(
                {
                    "sym": sym,
                    "role": str(raw.get("role", "")).strip(),
                    "shape": raw.get("shape") or None,
                    # Resolved against the document by the extract step below;
                    # a definition the model cannot point to does not count.
                    "defined_at": raw.get("defined_by") or None,
                }
            )

        payload = {
            "label": equation.label,
            "latex": equation.latex,
            "symbols": symbols,
            "undefined_symbols": undefined,
            "source_fidelity": equation.source_fidelity,
        }
        self.resolve_definitions(payload, doc)
        return payload

    def resolve_definitions(self, payload: dict[str, Any], doc: StructuredDocument) -> None:
        """Turn each symbol's quoted definition into a real span, or mark it undefined.

        A model claiming a symbol is defined is not evidence that it is. The
        quote has to resolve in the document, otherwise the symbol joins
        undefined_symbols and symbol_check fails - which is precisely how
        garbled math gets caught, since OCR corruption reliably produces
        phantom symbols nothing defines.
        """
        undefined: list[str] = []
        for symbol in payload.get("symbols", []):
            quote = symbol.get("defined_at")
            span = doc.find_span(str(quote)) if quote else None
            symbol["defined_at"] = span.span_id if span else None
            if span is None:
                undefined.append(symbol["sym"])
        payload["undefined_symbols"] = undefined

    def _equation_for(self, item: dict[str, Any]) -> RawEquation | None:
        index = item.get("index")
        if not isinstance(index, int) or not (0 <= index < len(self._equations)):
            return None
        return self._equations[index]


def _is_operator(symbol: str) -> bool:
    return symbol.lstrip("\\").split("_")[0].lower() in OPERATORS
