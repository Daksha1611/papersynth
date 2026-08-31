"""Semantic merge proposal for alignment (section 8.4).

Exact-name blocking is the right primary key and stays the primary key: for
hyperparameters `learning_rate` is `learning_rate` in every paper, and the
match is free and near-certain. For method claims it is not a key at all. It
requires two papers to independently invent the same snake_case name for the
same question, and on the M8 corpus they never did:

    CaMeL   security_mechanism, data_flow_security, capability_tagging
    NeMo    rail_specification_language, canonical_form_definition

Same question, no shared key, and so zero of 37 clusters spanned more than one
paper. No detector fired, and the run reported "0 contradictions" - a number
indistinguishable from the one an empty corpus produces.

Embeddings cannot close that gap. Measured on the same corpus the best
cross-paper pair scored 0.401 against a 0.82 threshold and the next best
0.107, while on BERT/RoBERTa/ALBERT surface similarity fired five times and
was wrong five times (num_steps with warmup_steps). Hyperparameter names are
composed from shared words, so surface similarity tracks naming convention
rather than meaning: too blunt to find the real merges and sharp enough to
invent false ones. That is not a threshold to tune, which is why this asks a
model the question directly.

The proposal is never admitted on its own. Section 8.4's asymmetry still
holds - a false merge fabricates a contradiction, a false split only yields
two singletons - so every group proposed here goes to the SplitterAgent
before it becomes a cluster. This is the gate's designed job, and until now it
had nothing to review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from papersynth.llm.base import LLMProvider

#: Bumped when the prompt changes (ER-10).
SEMANTIC_ALIGN_VERSION = "semantic_align@1.0.0"

#: Below two candidates there is nothing to pair. Below two papers any merge
#: found would be within one paper, which alignment does not want: collapsing
#: two quantities a single paper stated separately is how the M8 field-experiment
#: funnel became a contradiction in the first place.
MIN_CANDIDATES = 2

#: One call has to hold every unmatched key for a claim type. Past this the
#: prompt stops fitting a free-tier request, and the truncation is recorded
#: rather than silent - the M8 lesson is that a partial read must never look
#: like a whole one.
MAX_CANDIDATES = 60

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "members": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["members"],
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["groups"],
}

SEMANTIC_ALIGN_SYSTEM = (
    "You match up concepts across research papers that describe the same thing "
    "under different names. You group by the QUESTION being answered, never by "
    "the answer given: two papers that answer one question differently belong "
    "in the same group, because that disagreement is exactly what the grouping "
    "exists to surface."
)


@dataclass(frozen=True)
class MergeCandidate:
    """One unmatched alignment key, offered for pairing."""

    key: str
    paper_id: str
    description: str


def propose_merges(
    claim_type: str,
    candidates: list[MergeCandidate],
    *,
    provider: LLMProvider | None,
) -> tuple[list[list[str]], list[str]]:
    """Group keys that name the same concept. Returns groups and notes.

    Each returned group holds two or more distinct keys spanning two or more
    papers. Keys never appear in more than one group. An empty result means no
    merge was proposed, which is the safe outcome - the caller keeps the exact
    name blocking it already had.
    """
    notes: list[str] = []
    if provider is None:
        return [], []

    # Deterministic ordering before anything else. The prompt, the indices the
    # model answers with, and therefore the cluster IDs downstream all depend
    # on this being stable across runs (NFR-02).
    ordered = sorted(candidates, key=lambda c: (c.key, c.paper_id))

    if len(ordered) < MIN_CANDIDATES:
        return [], []
    if len({c.paper_id for c in ordered}) < 2:
        return [], []

    if len(ordered) > MAX_CANDIDATES:
        notes.append(
            f"semantic alignment for {claim_type}: {len(ordered)} unmatched keys exceeds "
            f"the {MAX_CANDIDATES} that fit one call; the remainder stayed unaligned"
        )
        ordered = ordered[:MAX_CANDIDATES]

    listing = "\n".join(
        f"[{i}] paper={c.paper_id} key={c.key} :: {c.description}" for i, c in enumerate(ordered)
    )
    prompt = (
        f"Below are {claim_type} concepts extracted from several papers. Each was "
        "named independently by its own paper, so the same underlying concept may "
        "appear under different names.\n\n"
        "Group the entries that address the SAME underlying question or quantity.\n\n"
        "Rules:\n"
        "- Group across papers only. Two entries from the same paper are two things "
        "that paper chose to state separately; leave them separate.\n"
        "- Group by the question, not the answer. If two papers answer the same "
        "question with different approaches, they belong together - that "
        "disagreement is the point.\n"
        "- Do not group things that are merely related, adjacent, or from the same "
        "area of the paper. A wrong grouping invents a disagreement that does not "
        "exist, which is worse than leaving two entries apart.\n"
        "- Leave an entry out entirely if nothing matches it. Most entries will have "
        "no match, and that is a correct answer.\n\n"
        f"{listing}\n\n"
        'Answer JSON: {"groups": [{"concept": "<short snake_case name>", '
        '"members": [<index>, ...]}], "reason": "<one line>"}'
    )

    kwargs: dict[str, Any] = {
        "schema": _SCHEMA,
        "temperature": 0.0,
        "system": SEMANTIC_ALIGN_SYSTEM,
    }
    if hasattr(provider, "chain"):
        kwargs |= {
            "stage": "align",
            "extractor": SEMANTIC_ALIGN_VERSION,
            "template_id": SEMANTIC_ALIGN_VERSION,
        }

    try:
        completion = provider.complete(prompt, **kwargs)
    except Exception as exc:
        # Alignment without this call is the behaviour that shipped before it.
        # Failing the run over a merge proposal would trade a missed conflict
        # for no spec at all.
        return [], [f"semantic alignment for {claim_type} failed ({exc}); exact names only"]

    payload = completion.parsed if isinstance(completion.parsed, dict) else {}
    groups, dropped = _validate(payload, ordered)

    for reason, count in sorted(dropped.items()):
        notes.append(f"semantic alignment for {claim_type}: dropped {count} group(s), {reason}")
    if groups:
        notes.append(
            f"semantic alignment for {claim_type}: proposed {len(groups)} cross-paper "
            f"merge(s) from {len(ordered)} unmatched keys"
        )
    return groups, notes


def _validate(
    payload: dict[str, Any], ordered: list[MergeCandidate]
) -> tuple[list[list[str]], dict[str, int]]:
    """Keep only groups that are safe to act on.

    Everything the model returns is treated as a proposal about indices we
    supplied, not as an answer to trust. A group survives only if it names two
    or more distinct keys spanning two or more papers, and no key may be
    claimed twice - the first group to claim it wins, ordered as the model
    returned them.
    """
    claimed: set[str] = set()
    groups: list[list[str]] = []
    dropped: dict[str, int] = {}

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    for entry in payload.get("groups") or []:
        if not isinstance(entry, dict):
            continue
        indices = [
            i
            for i in entry.get("members") or []
            if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(ordered)
        ]
        members = [ordered[i] for i in dict.fromkeys(indices)]

        keys = sorted({m.key for m in members if m.key not in claimed})
        if len(keys) < 2:
            drop("fewer than two unclaimed keys")
            continue
        if len({m.paper_id for m in members if m.key in keys}) < 2:
            # A single-paper group is the M8 funnel failure in miniature:
            # separate quantities one paper stated separately, collapsed.
            drop("all members from one paper")
            continue

        claimed.update(keys)
        groups.append(keys)

    return groups, dropped
