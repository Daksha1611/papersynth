"""Evaluation harnesses (section 13)."""

from __future__ import annotations

from papersynth.eval.gap_ablation import (
    ABLATABLE_FIELDS,
    COMPLETE_SPEC,
    GapEvalReport,
    ablate,
    evaluate_gaps,
)

__all__ = [
    "ABLATABLE_FIELDS",
    "COMPLETE_SPEC",
    "GapEvalReport",
    "ablate",
    "evaluate_gaps",
]
