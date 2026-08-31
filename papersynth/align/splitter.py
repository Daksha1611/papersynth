"""The split gate (sections 6.2, 8.4).

The adversary to alignment. The Aligner maximizes correct merges; this rejects
merges of things that were only superficially similar. Neither can admit a
cluster on its own.

It exists because of an asymmetry: a false merge fabricates a contradiction
that does not exist, or silently averages two unrelated quantities, while a
false split yields two singleton clusters and no contradiction - visible and
recoverable. So the gate is deliberately biased toward splitting, and a single
confident distinction is enough.

Observed need: on BERT/RoBERTa/ALBERT, hidden_dim aligned ALBERT's 4096 and
2048 with RoBERTa's 768. Those are three model variants, not a disagreement,
and nothing before this stage could tell the difference.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from papersynth.core import ids
from papersynth.core.models import Agreement, Claim, ConceptCluster
from papersynth.extract.prompts import render
from papersynth.llm.base import LLMProvider

SPLITTER_SYSTEM = (
    "You distinguish configurable quantities in research papers. You are "
    "adversarial: you look for reasons two similarly-named values are actually "
    "different quantities. You never split merely because two values disagree - "
    "a disagreement about one quantity is the finding, not a reason to separate."
)

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "concept": {"type": "string"},
                },
                "required": ["claim_id", "concept"],
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["assignments"],
}


@dataclass
class SplitReport:
    reviewed: int = 0
    split: int = 0
    notes: list[str] = field(default_factory=list)


class SplitterAgent:
    """Reviews a multi-paper cluster and splits it if the members differ."""

    version = "splitter@1.0.0"

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def review(
        self, cluster: ConceptCluster, claims: list[Claim]
    ) -> tuple[list[ConceptCluster], str]:
        """Return the clusters this one should become, plus a note.

        A cluster that survives is returned unchanged with split_check "pass".
        """
        if not cluster.is_multi_paper or len(claims) < 2:
            # Nothing to adjudicate: a single-paper cluster cannot host a
            # cross-paper contradiction, so splitting it changes nothing.
            return [cluster.model_copy(update={"split_check": "n/a"})], ""

        prompt = render(
            "split_check.md",
            canonical_name=cluster.canonical_name,
            claims=_render_claims(claims),
        )

        kwargs: dict[str, Any] = {
            "schema": _SCHEMA,
            "temperature": 0.0,
            "system": SPLITTER_SYSTEM,
        }
        if hasattr(self.provider, "chain"):
            kwargs |= {
                "stage": "align",
                "extractor": self.version,
                "template_id": self.version,
            }

        completion = self.provider.complete(prompt, **kwargs)
        verdict = completion.parsed if isinstance(completion.parsed, dict) else {}

        assignments = [
            entry
            for entry in (verdict.get("assignments") or [])
            if isinstance(entry, dict) and entry.get("claim_id")
        ]
        if not assignments:
            # Section 8.4 splits on a *confident* NO. An unreadable verdict is
            # not one, and treating it as a split would shatter the cluster
            # into singletons and quietly delete a real disagreement. The
            # cluster stands, recorded as unreviewed rather than as passing.
            return [cluster.model_copy(update={"split_check": "n/a"})], ""

        groups = _merge_identical(_groups_from(verdict, claims))

        if len(groups) <= 1:
            return [cluster.model_copy(update={"split_check": "pass"})], ""

        reason = str(verdict.get("reason", "")).strip() or "splitter found distinct concepts"
        return _rebuild(cluster, groups), (
            f"split {cluster.cluster_id} into {len(groups)} concepts: {reason}"
        )


def _render_claims(claims: list[Claim]) -> str:
    lines = []
    for claim in sorted(claims, key=lambda c: c.claim_id):
        payload = claim.payload
        parts = [
            f"- claim_id: {claim.claim_id}",
            f"  paper: {claim.paper_id}",
            f"  name as written: {payload.get('canonical_name')}",
            f"  value: {payload.get('value')!r}",
        ]
        for field_name, label in (
            ("unit", "unit"),
            ("condition", "stated condition"),
            ("applies_to", "applies to"),
            ("paper_symbol", "symbol"),
        ):
            value = payload.get(field_name)
            if value and value != "global":
                parts.append(f"  {label}: {value}")
        parts.append(f"  found in section: {claim.provenance.section}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def _groups_from(verdict: dict[str, Any], claims: list[Claim]) -> list[list[Claim]]:
    """Group claims by the label the splitter assigned.

    A claim the splitter did not mention keeps its own group rather than being
    dropped or lumped in. Silently discarding a claim here would remove
    evidence from the corpus on the strength of an incomplete answer.
    """
    by_id = {c.claim_id: c for c in claims}
    labelled: dict[str, list[Claim]] = defaultdict(list)
    seen: set[str] = set()

    for entry in verdict.get("assignments") or []:
        if not isinstance(entry, dict):
            continue
        claim_id = str(entry.get("claim_id", ""))
        claim = by_id.get(claim_id)
        if claim is None or claim_id in seen:
            continue
        seen.add(claim_id)
        labelled[str(entry.get("concept", "")).strip().lower() or claim_id].append(claim)

    for claim in claims:
        if claim.claim_id not in seen:
            labelled[f"unassigned_{claim.claim_id}"].append(claim)

    return [group for _, group in sorted(labelled.items())]


def _merge_identical(groups: list[list[Claim]]) -> list[list[Claim]]:
    """Re-merge groups holding the same value in the same unit.

    Two papers both stating 512 tokens are reporting one measurement, and
    separating them cannot be right. It is also free to enforce: identical
    values produce no contradiction either way, so this only removes noise
    from the artifact rather than changing any finding.

    Observed: the splitter separated three identical max_sequence_length
    values of 512 across the BERT corpus.

    It applies only to claims that carry a value. A method claim carries none,
    so every method group would key on the same empty value and the re-merge
    would silently undo every split the gate made - on exactly the clusters
    semantic alignment creates, where the gate is the only review there is.
    """
    if len(groups) <= 1:
        return groups

    by_value: dict[tuple[str, str], list[Claim]] = defaultdict(list)
    ungrouped: list[list[Claim]] = []

    for group in groups:
        if any(c.payload.get("value") is None for c in group):
            ungrouped.append(group)
            continue
        values = {_value_key(c) for c in group}
        if len(values) == 1:
            by_value[values.pop()].extend(group)
        else:
            ungrouped.append(group)

    return [*ungrouped, *by_value.values()]


def _value_key(claim: Claim) -> tuple[str, str]:
    value = claim.payload.get("value")
    rendered = f"{float(value):.12g}" if isinstance(value, int | float) else str(value).lower()
    return rendered, str(claim.payload.get("unit") or "").lower()


def _rebuild(cluster: ConceptCluster, groups: list[list[Claim]]) -> list[ConceptCluster]:
    """Turn one rejected cluster into one cluster per distinct concept."""
    out = []
    for index, group in enumerate(sorted(groups, key=lambda g: g[0].claim_id)):
        members = sorted(group, key=lambda c: c.claim_id)
        papers = sorted({c.paper_id for c in members})
        out.append(
            ConceptCluster(
                cluster_id=f"{cluster.cluster_id}_{index}",
                canonical_name=cluster.canonical_name,
                concept_type=cluster.concept_type,
                member_claims=[c.claim_id for c in members],
                symbol_aliases=sorted(
                    {
                        str(c.payload["paper_symbol"])
                        for c in members
                        if c.payload.get("paper_symbol")
                    }
                ),
                papers=papers,
                agreement=_agreement(members),
                # The gate rejected the parent, so every child records that a
                # split happened rather than claiming a clean pass.
                split_check="fail",
            )
        )
    return out


def _agreement(members: list[Claim]) -> Agreement:
    # Imported lazily: cluster.py owns the split gate and imports this module,
    # so a top-level import here would close the cycle.
    from papersynth.align.cluster import _agreement as base_agreement

    return base_agreement(members)


__all__ = ["SplitReport", "SplitterAgent", "ids"]
