"""Extraction: StructuredDocument -> Claim[] (stage 1)."""

from __future__ import annotations

from papersynth.extract.base import (
    ExtractionResult,
    Extractor,
    LLMExtractor,
    RejectedClaim,
    render_sections,
)
from papersynth.extract.prompts import load_prompt, render
from papersynth.extract.registry import EXTRACTORS, available, build, describe, register, run_all

__all__ = [
    "EXTRACTORS",
    "ExtractionResult",
    "Extractor",
    "LLMExtractor",
    "RejectedClaim",
    "available",
    "build",
    "describe",
    "load_prompt",
    "register",
    "render",
    "render_sections",
    "run_all",
]
