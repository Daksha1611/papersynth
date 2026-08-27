"""Algorithm transcription.

Like equations, algorithms are ingested as located objects, so their provenance
comes from the parser rather than from a model quote.

The rule that shapes the prompt: an algorithm missing a step is worse than no
algorithm at all, because it looks complete. An implementer reading a
seven-step transcription of an eight-step procedure has no signal that anything
is absent, whereas an absent algorithm sends them to the paper.
"""

from __future__ import annotations

from typing import Any, ClassVar

from papersynth.core.document import RawAlgorithm, Section, Span, StructuredDocument
from papersynth.extract.base import ExtractionResult, LLMExtractor, render_sections
from papersynth.extract.prompts import render
from papersynth.extract.registry import register

_PORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "type": {"type": ["string", "null"]},
        "description": {"type": ["string", "null"]},
    },
    "required": ["name"],
}

_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "index": {"type": "integer"},
        "name": {"type": "string"},
        "inputs": {"type": "array", "items": _PORT_SCHEMA},
        "outputs": {"type": "array", "items": _PORT_SCHEMA},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "text": {"type": "string"},
                    "refs_equations": {"type": "array", "items": {"type": "string"}},
                    "refs_symbols": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["index", "text"],
            },
        },
        "complexity": {"type": ["object", "null"]},
        "preconditions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["index", "name", "steps"],
}


@register
class AlgorithmExtractor(LLMExtractor):
    claim_type: ClassVar[str] = "algorithm"
    version: ClassVar[str] = "1.0.0"
    payload_schema_name: ClassVar[str] = "payload.algorithm.json"
    output_schema: ClassVar[dict[str, Any]] = {"type": "array", "items": _ITEM_SCHEMA}
    section_pattern: ClassVar[str] = ""
    system_prompt: ClassVar[str] = (
        "You transcribe pseudocode from research papers exactly. You never add "
        "a step the paper leaves implicit, and you never claim a complexity "
        "the authors did not state."
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._algorithms: list[RawAlgorithm] = []

    def extract(
        self, doc: StructuredDocument, sections: list[Section] | None = None
    ) -> ExtractionResult:
        self._algorithms = list(doc.algorithms_raw)
        if not self._algorithms:
            return ExtractionResult()
        return super().extract(doc, sections)

    def build_prompt(self, doc: StructuredDocument, sections: list[Section]) -> str:
        listing = "\n\n".join(
            f"[{i}] {alg.label or 'unlabelled'}"
            + (f" - {alg.caption}" if alg.caption else "")
            + f"\n{alg.body}"
            for i, alg in enumerate(self._algorithms)
        )
        return render(
            "extract_algorithm.md",
            sections=render_sections(doc, sections),
            algorithms=listing,
        )

    def anchor(
        self,
        item: dict[str, Any],
        quote: str | None,
        doc: StructuredDocument,
        section_indices: list[int],
    ) -> Span | None:
        """Anchor to the algorithm block's own position, recorded at ingestion."""
        algorithm = self._algorithm_for(item)
        if algorithm is None:
            return None
        try:
            section = doc.sections[algorithm.section_index]
            index = min(algorithm.paragraph_index, len(section.paragraphs) - 1)
            paragraph = section.paragraphs[index]
        except (IndexError, ValueError):
            return None
        return doc.make_span(section.index, paragraph.index, 0, len(paragraph.text))

    def normalize_payload(self, payload: dict[str, Any], doc: StructuredDocument) -> dict[str, Any]:
        algorithm = self._algorithm_for(payload)

        steps: list[dict[str, Any]] = []
        for raw in payload.get("steps") or []:
            if not isinstance(raw, dict) or not str(raw.get("text", "")).strip():
                continue
            steps.append(
                {
                    "index": int(raw.get("index", len(steps) + 1)),
                    "text": str(raw["text"]).strip(),
                    "refs_equations": list(raw.get("refs_equations") or []),
                    "refs_symbols": list(raw.get("refs_symbols") or []),
                }
            )
        # Renumber contiguously from 1. A gap in the numbering would read as a
        # dropped step to anyone implementing this.
        for position, step in enumerate(steps, start=1):
            step["index"] = position

        complexity = payload.get("complexity")
        if isinstance(complexity, dict):
            complexity = {
                "time": complexity.get("time") or None,
                "space": complexity.get("space") or None,
            }
            if complexity["time"] is None and complexity["space"] is None:
                complexity = None
        else:
            complexity = None

        return {
            "name": str(payload.get("name", "")).strip()
            or (algorithm.caption if algorithm and algorithm.caption else "Unnamed algorithm"),
            "label": algorithm.label if algorithm else None,
            "inputs": _ports(payload.get("inputs")),
            "outputs": _ports(payload.get("outputs")),
            "steps": steps,
            "complexity": complexity,
            "preconditions": [str(p) for p in (payload.get("preconditions") or [])],
        }

    def _algorithm_for(self, item: dict[str, Any]) -> RawAlgorithm | None:
        index = item.get("index")
        if not isinstance(index, int) or not (0 <= index < len(self._algorithms)):
            return None
        return self._algorithms[index]


def _ports(raw: Any) -> list[dict[str, Any]]:
    out = []
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        out.append(
            {
                "name": name,
                "type": entry.get("type") or None,
                "description": entry.get("description") or None,
            }
        )
    return out
