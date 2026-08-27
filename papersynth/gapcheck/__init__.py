"""Gap check: what is missing but required to implement (stage 6).

Pass A is the deterministic checklist below. Pass B - the adversarial
"could you write this file today?" audit - is a P1 addition; until it lands,
gap recall is bounded by whatever the checklist names.
"""

from __future__ import annotations

from papersynth.gapcheck.checklist import (
    Checklist,
    ChecklistGroup,
    RequiredField,
    summarize,
)

__all__ = ["Checklist", "ChecklistGroup", "RequiredField", "summarize"]
