"""Pass B: the adversarial implementability audit (section 8.6).

Pass A can only find what its checklist names. This asks an engineer who has to
write the code today, and who cannot see the papers, what they would be forced
to guess. That catches what a static list cannot - a paper saying "standard
augmentation" without defining it, or an algorithm step invoking a function
nothing defines.

The agent sees the assembled spec rather than the papers, which is the whole
point: if the spec is insufficient, the questions it raises are real by
construction.

Two guards sit around it, because a model asked to find problems will find
them. A gap naming something the corpus already supplies is dropped, and a gap
duplicating one Pass A already found is dropped. Both are deterministic. A
padded gap list is worse than a short one: a reviewer who stops reading gets
nothing from it at all.
"""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from papersynth.core import ids
from papersynth.core.models import Claim, Criticality, Gap
from papersynth.llm.base import LLMProvider

GAP_AGENT_SYSTEM = (
    "You are an engineer implementing a specification with no access to the "
    "papers behind it. You report only what would genuinely stop you, and you "
    "never pad the list."
)

#: Above this similarity between field names, a Pass B gap is the same gap Pass
#: A already reported under a slightly different name.
DUPLICATE_RATIO = 85

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "question": {"type": "string"},
                    "criticality": {
                        "type": "string",
                        "enum": ["BLOCKING", "MATERIAL", "COSMETIC"],
                    },
                    "blocks": {"type": "string"},
                },
                "required": ["field", "question"],
            },
        }
    },
    "required": ["gaps"],
}


class AdversarialGapAgent:
    """Asks what an implementer would have to guess."""

    version = "gap_agent@1.0.0"

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def audit(
        self,
        spec: dict[str, Any],
        *,
        claims: list[Claim],
        existing: list[Gap],
        paper_ids: list[str],
        disputed: set[str] | None = None,
    ) -> list[Gap]:
        from papersynth.extract.prompts import render

        prompt = render("gap_audit.md", spec=render_spec(spec))

        kwargs: dict[str, Any] = {
            "schema": _SCHEMA,
            "temperature": 0.0,
            "system": GAP_AGENT_SYSTEM,
        }
        if hasattr(self.provider, "chain"):
            kwargs |= {
                "stage": "gapcheck",
                "extractor": self.version,
                "template_id": self.version,
            }

        completion = self.provider.complete(prompt, **kwargs)
        payload = completion.parsed if isinstance(completion.parsed, dict) else {}

        supplied = {
            str(c.payload.get("canonical_name") or c.payload.get("sub_problem") or "")
            for c in claims
            if c.status == "verified"
        }
        supplied.discard("")
        known = [g.field for g in existing]
        contested = list(disputed or ())

        out: list[Gap] = []
        seen: set[str] = set()

        for entry in payload.get("gaps") or []:
            if not isinstance(entry, dict):
                continue
            field = _normalize(entry.get("field"))
            question = str(entry.get("question", "")).strip()
            if not field or not question or field in seen:
                continue

            # The corpus already answers this, so it is not missing. A model
            # asked to find problems will find some that are not there.
            if _matches_any(field, supplied):
                continue
            if _matches_any(field, known):
                continue

            # The papers answer this, they just answer it differently. That is
            # a conflict the spec already surfaces, and listing it again as
            # missing doubles the review list while telling the reader
            # something false: the value is not absent, it is contested.
            if _matches_any(field, contested, text=question):
                continue

            seen.add(field)
            blocks = str(entry.get("blocks", "")).strip()
            out.append(
                Gap(
                    gap_id=ids.gap_id(None, field),
                    component_id=None,
                    field=field,
                    question=question + (f" Blocks: {blocks}" if blocks else ""),
                    criticality=_criticality(entry.get("criticality")),
                    searched_papers=sorted(paper_ids),
                    suggested_sources=[
                        "the paper text itself, if extraction missed it",
                        "the authors' reference implementation",
                    ],
                )
            )
        return out


def render_spec(spec: dict[str, Any]) -> str:
    """What the implementer can see. Deliberately only the spec.

    Showing the papers would let the agent answer its own questions from
    context an implementer will not have, which is exactly the situation this
    pass exists to simulate.
    """
    lines = [f"OBJECTIVE: {spec.get('objective', '(none stated)')}", ""]

    for component in spec.get("components") or []:
        lines.append(f"COMPONENT {component.get('name')} ({component.get('component_id')})")
        lines.append(f"  role: {component.get('role')}")
        for hyperparameter in component.get("hyperparameters") or []:
            scope = hyperparameter.get("condition")
            unit = hyperparameter.get("unit")
            rendered = f"{hyperparameter['canonical_name']} = {hyperparameter['value']!r}"
            if unit:
                rendered += f" {unit}"
            if scope:
                rendered += f"  [{scope}]"
            lines.append(f"  - {rendered}")
        for equation in component.get("equations") or []:
            lines.append(f"  - equation {equation.get('label') or ''}: {equation.get('latex', '')}")
        lines.append("")

    conflicts = spec.get("open_conflicts") or []
    if conflicts:
        lines.append("UNRESOLVED, the implementer must choose:")
        lines += [f"  - {c.get('summary', '')}" for c in conflicts]
        lines.append("")

    known_gaps = spec.get("missing_but_critical") or []
    if known_gaps:
        # Shown so the agent does not spend its list restating them.
        lines.append("ALREADY KNOWN TO BE MISSING, do not repeat these:")
        lines += [f"  - {g.get('field')}" for g in known_gaps]
        lines.append("")

    return "\n".join(lines).strip() or "(the specification is empty)"


def _normalize(field: Any) -> str:
    text = str(field or "").strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _matches_any(field: str, candidates: list[str] | set[str], *, text: str = "") -> bool:
    """Whether `field` names the same thing as any candidate.

    Underscores become spaces first, so token-set similarity can actually see
    the words. Compared as single tokens, "tokenizer_type" and "tokenizer"
    scored below threshold and a contested tokenizer choice was reported as a
    missing one.

    `text` lets a gap's question be checked too, since a model restating a
    conflict often invents a field name unrelated to the disputed one while
    quoting the options in the question itself.
    """
    spaced = field.replace("_", " ")
    haystack = f"{spaced} {text}".strip().lower()

    for candidate in candidates:
        if not candidate:
            continue
        needle = candidate.replace("_", " ").lower()
        if fuzz.token_set_ratio(spaced, needle) >= DUPLICATE_RATIO:
            return True
        # A question naming the disputed quantity is restating the conflict.
        if text and needle in haystack:
            return True
    return False


def _criticality(value: Any) -> Criticality:
    text = str(value or "").strip().upper()
    return text if text in ("BLOCKING", "MATERIAL", "COSMETIC") else "MATERIAL"  # type: ignore[return-value]
