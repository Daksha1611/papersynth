"""Gap check: what is missing but required to implement (stage 6).

Pass A is the deterministic checklist, which finds only what it names. Pass B
asks an implementer who cannot see the papers what they would have to guess,
which is what catches the omissions no static list anticipated.
"""

from __future__ import annotations

from papersynth.gapcheck.adversarial import AdversarialGapAgent, render_spec
from papersynth.gapcheck.checklist import (
    Checklist,
    ChecklistGroup,
    RequiredField,
    summarize,
)

__all__ = [
    "AdversarialGapAgent",
    "Checklist",
    "ChecklistGroup",
    "RequiredField",
    "render_spec",
    "summarize",
]
