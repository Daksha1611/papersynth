"""Definition extraction.

The claim type that makes DEFINITION_CONFLICT detectable. A method claim
disagrees over which approach to take; a definition claim disagrees over what a
shared term means. The two are genuinely different: an engineer resolves a
method conflict by picking an approach, and resolves a definition conflict by
discovering that two papers' "attack success" rates were never comparable in
the first place.

The load-bearing field is `term`: the WORD being defined, not its meaning. Two
papers both defining `attack_success` must align on that key even when they
mean different things by it, because that shared term under conflicting
criteria is precisely the disagreement the detector looks for. Named after the
meaning instead, the two definitions would land in different clusters and the
conflict would never be seen - the same failure mode `sub_problem` guards
against for method claims.
"""

from __future__ import annotations

from typing import Any, ClassVar

from papersynth.core.document import Section, StructuredDocument
from papersynth.extract.base import LLMExtractor, render_sections
from papersynth.extract.prompts import render
from papersynth.extract.registry import register

#: Term names seen in the wild, mapped to a canonical form. Alignment happens
#: on this field, so a paper writing "attack success rate" and one writing
#: "successful attack" must arrive at the same key or their disagreement about
#: what counts as a successful attack is invisible.
CANONICAL_TERMS: dict[str, str] = {
    "attack_success_rate": "attack_success",
    "successful_attack": "attack_success",
    "attack_success": "attack_success",
    "task_failure": "task_failure",
    "failure": "task_failure",
    "task_success": "task_success",
    "success": "task_success",
    "harmful_action": "harmful_action",
    "harm": "harmful_action",
    "unsafe_action": "harmful_action",
    "safety_violation": "safety_violation",
    "violation": "safety_violation",
    "robustness": "robustness",
    "utility": "utility",
}

_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "term": {"type": "string"},
        "definition": {"type": "string"},
        "criterion": {"type": ["string", "null"]},
        "attribution": {"type": "string"},
        "applies_to": {"type": "string"},
        "condition": {"type": ["string", "null"]},
        "stated_explicitly": {"type": "boolean"},
        "quote": {"type": "string"},
    },
    "required": ["term", "definition", "quote"],
}


@register
class DefinitionExtractor(LLMExtractor):
    claim_type: ClassVar[str] = "definition"
    version: ClassVar[str] = "1.0.0"
    payload_schema_name: ClassVar[str] = "payload.definition.json"
    output_schema: ClassVar[dict[str, Any]] = {"type": "array", "items": _ITEM_SCHEMA}
    looks_for: ClassVar[str] = (
        "definitions - what a term means or how a quantity is measured: what "
        "counts as a success, a failure, an attack, a violation, or a metric"
    )
    section_pattern: ClassVar[str] = (
        r"definition|threat\s*model|problem|preliminar|notation|metric|"
        r"evaluation|setup|formulation|terminology|background"
    )
    system_prompt: ClassVar[str] = (
        "You extract definitions from research papers: what each term is taken "
        "to mean, and the operational test it reduces to. You record the term "
        "being defined, never a paraphrase of its meaning, and you never infer "
        "a definition a paper does not state."
    )

    def build_prompt(self, doc: StructuredDocument, sections: list[Section]) -> str:
        return render("extract_definition.md", sections=render_sections(doc, sections))

    def normalize_payload(self, payload: dict[str, Any], doc: StructuredDocument) -> dict[str, Any]:
        raw = str(payload.get("term", "")).strip().lower()
        raw = raw.replace(" ", "_").replace("-", "_")
        payload["term"] = CANONICAL_TERMS.get(raw, raw)

        payload["definition"] = str(payload.get("definition", "")).strip()
        payload.setdefault("applies_to", "global")
        payload.setdefault("condition", None)
        payload.setdefault("stated_explicitly", True)

        # A criterion is optional, but an empty string is not a criterion. The
        # detector treats a missing criterion as "falls back to the definition
        # text", so normalize both absence and emptiness to None rather than
        # letting "" read as a distinct criterion that conflicts with every
        # real one.
        criterion = payload.get("criterion")
        payload["criterion"] = (
            criterion.strip() if isinstance(criterion, str) and criterion.strip() else None
        )

        # "own" is the safe default, matching the method extractor: a definition
        # wrongly kept is visible in the conflict list and dismissable, while
        # one wrongly discarded as background is simply absent.
        attribution = str(payload.get("attribution", "own")).strip().lower()
        payload["attribution"] = attribution if attribution == "prior_work" else "own"
        return payload
