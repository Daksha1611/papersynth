"""Alignment: verified claims across papers -> ConceptGraph (stage 3)."""

from __future__ import annotations

from papersynth.align.cluster import Aligner, AlignmentReport
from papersynth.align.embed import Embedder, HashEmbedder, build_embedder, cosine

__all__ = [
    "Aligner",
    "AlignmentReport",
    "Embedder",
    "HashEmbedder",
    "build_embedder",
    "cosine",
]
