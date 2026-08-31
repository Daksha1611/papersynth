"""Alignment: verified claims across papers -> ConceptGraph (stage 3)."""

from __future__ import annotations

from papersynth.align.cluster import Aligner, AlignmentReport
from papersynth.align.semantic import MergeCandidate, propose_merges
from papersynth.align.splitter import SplitterAgent

__all__ = [
    "Aligner",
    "AlignmentReport",
    "MergeCandidate",
    "SplitterAgent",
    "propose_merges",
]
