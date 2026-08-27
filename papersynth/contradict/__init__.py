"""Contradiction detection: ConceptGraph -> Contradiction[] (stage 4).

Detectors self-register, so adding a domain-specific one is a module drop
(section 10.3, 19.2).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from papersynth.contradict.detectors.value_conflict import (
    ValueConflictDetector,
    attach_paper_support,
    normalize_condition,
)
from papersynth.contradict.severity import specificity, value_conflict_severity
from papersynth.core.models import ConceptCluster, ConceptGraph, Contradiction

__all__ = [
    "DETECTORS",
    "ContradictionScan",
    "Detector",
    "ValueConflictDetector",
    "attach_paper_support",
    "detect",
    "normalize_condition",
    "register_detector",
    "specificity",
    "value_conflict_severity",
]


@runtime_checkable
class Detector(Protocol):
    conflict_type: str
    #: Honoured by the reconciliation policy; a detector may forbid its own
    #: conflicts from ever being auto-resolved (section 10.3).
    auto_resolvable: bool

    def scan(self, cluster: ConceptCluster, graph: ConceptGraph) -> list[Contradiction]: ...


DETECTORS: dict[str, Detector] = {}


def register_detector(cls: type[Detector]) -> type[Detector]:
    """Class decorator. Instantiates once and self-registers on import."""
    DETECTORS[cls.conflict_type] = cls()
    return cls


register_detector(ValueConflictDetector)


class ContradictionScan:
    """Runs every registered detector over every multi-paper cluster."""

    def __init__(self, detectors: list[Detector] | None = None) -> None:
        self.detectors = detectors or list(DETECTORS.values())

    def run(self, graph: ConceptGraph) -> list[Contradiction]:
        found: list[Contradiction] = []
        for cluster in graph.clusters:
            if not cluster.is_multi_paper:
                continue
            for detector in self.detectors:
                found.extend(detector.scan(cluster, graph))

        # Deterministic order: severity first so the review list leads with what
        # actually blocks, then by ID so two identical runs agree byte for byte.
        rank = {"BLOCKING": 0, "MATERIAL": 1, "COSMETIC": 2}
        return sorted(found, key=lambda c: (rank.get(c.severity, 3), c.contradiction_id))


def detect(graph: ConceptGraph) -> list[Contradiction]:
    return ContradictionScan().run(graph)
