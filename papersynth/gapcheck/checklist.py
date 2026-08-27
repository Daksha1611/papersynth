"""Pass A: the deterministic implementability checklist (section 8.6).

Every required field is either satisfied by some claim in the corpus or it
becomes a Gap. No model is involved, which makes this pass free, exactly
reproducible, and the one part of gap detection that cannot hallucinate a
problem.

The `searched_papers` field on every Gap records which papers were examined, so
a reader can see the absence spans the whole corpus rather than one document.

One limit worth stating plainly: a gap means no VERIFIED CLAIM supplies the
field, which is not the same as "no paper states it". Extraction can miss a
value the paper does state, and a claim can fail verification. Gap questions
are therefore worded around what was verified, never around what the papers
contain - telling an implementer "the papers do not specify this" when the
value was in fact printed on page four sends them off to invent a number that
already existed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from papersynth.core import ids
from papersynth.core.models import Claim, Criticality, Gap


@dataclass(frozen=True)
class RequiredField:
    field: str
    kind: str
    criticality: Criticality
    question: str
    suggested_sources: tuple[str, ...] = ()
    one_of: tuple[str, ...] = ()

    def satisfied_by(self, present: set[str]) -> bool:
        if self.kind == "one_of":
            return any(name in present for name in self.one_of)
        return self.field in present


@dataclass(frozen=True)
class ChecklistGroup:
    id: str
    description: str
    applies_when: str
    required: tuple[RequiredField, ...]

    def applies(self, claims: list[Claim]) -> bool:
        """Do not ask for training details of a corpus that describes no training.

        Manufacturing a gap for a field the papers were never going to state is
        the same failure as manufacturing a contradiction: it wastes review
        time and teaches the reader to skim the list.
        """
        if self.applies_when == "always":
            return True
        if self.applies_when == "any_hyperparameter":
            return any(c.type == "hyperparameter" for c in claims)
        return True


@dataclass
class Checklist:
    version: str = "0.0.0"
    groups: tuple[ChecklistGroup, ...] = ()
    notes: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str | None) -> Checklist:
        """Load the checklist. A missing file yields an empty one.

        An absent checklist means no gaps are reported, which is honest: the
        run has nothing to check against and must not imply the corpus is
        complete either way.
        """
        if path is None:
            return cls()
        path = Path(path)
        if not path.exists():
            return cls()

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        groups = []
        for raw_group in payload.get("groups", []):
            required = tuple(
                RequiredField(
                    field=str(entry["field"]),
                    kind=str(entry.get("kind", "hyperparameter")),
                    criticality=entry.get("criticality", "MATERIAL"),
                    question=" ".join(str(entry.get("question", "")).split()),
                    suggested_sources=tuple(entry.get("suggested_sources", [])),
                    one_of=tuple(entry.get("one_of", [])),
                )
                for entry in raw_group.get("required", [])
                if entry.get("field")
            )
            groups.append(
                ChecklistGroup(
                    id=str(raw_group.get("id", "unnamed")),
                    description=" ".join(str(raw_group.get("description", "")).split()),
                    applies_when=str(raw_group.get("applies_when", "always")),
                    required=required,
                )
            )
        return cls(version=str(payload.get("version", "0.0.0")), groups=tuple(groups))

    def audit(
        self,
        claims: list[Claim],
        *,
        paper_ids: list[str],
        component_id: str | None = None,
    ) -> list[Gap]:
        """Every required field the corpus does not supply."""
        present = _present_fields(claims)
        searched = sorted(paper_ids)

        gaps: list[Gap] = []
        for group in self.groups:
            if not group.applies(claims):
                continue
            for required in group.required:
                if required.satisfied_by(present):
                    continue
                gaps.append(
                    Gap(
                        gap_id=ids.gap_id(component_id, required.field),
                        component_id=component_id,
                        field=required.field,
                        question=required.question,
                        criticality=required.criticality,
                        searched_papers=searched,
                        suggested_sources=list(required.suggested_sources),
                    )
                )
        # Deterministic order, worst first, so the review list leads with what
        # actually stops an implementer.
        rank = {"BLOCKING": 0, "MATERIAL": 1, "COSMETIC": 2}
        return sorted(gaps, key=lambda g: (rank.get(g.criticality, 3), g.field))


def _present_fields(claims: list[Claim]) -> set[str]:
    """Which required fields the corpus actually supplies.

    Only verified claims count. A rejected claim did not establish its value,
    so treating it as coverage would suppress a gap that genuinely exists -
    the implementer would be told nothing is missing when the only source for
    it failed verification.
    """
    present: set[str] = set()
    for claim in claims:
        if claim.status != "verified":
            continue
        name = claim.payload.get("canonical_name")
        if isinstance(name, str) and name:
            present.add(name)
    return present


def summarize(gaps: list[Gap]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for gap in gaps:
        counts[gap.criticality] = counts.get(gap.criticality, 0) + 1
    return {"total": len(gaps), "by_criticality": counts}
