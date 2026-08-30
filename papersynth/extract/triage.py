"""Section triage: which sections an extractor should read (section 8.1).

The regex approach this replaces was BERT-shaped. Its title list -
"Training Setup", "Experiments", "Pre-training" - matched the fixtures it was
tuned on and missed everything else, and the fallback only fires when NOTHING
matches, so a partial match left 79-99% of the M8 papers unread in silence.
Raising the fallback threshold would only widen when the system gives up on a
guess that was wrong to begin with.

So triage is now a judgement, made once per extractor per paper: given this
extractor's claim type and the paper's actual section titles, which sections
are relevant. The regex list is kept only as a zero-cost pre-filter - when it
already matches most of the paper, the paper is BERT-shaped and the call would
be wasted.

Chosen per-extractor rather than once-per-paper: "relevant to hyperparameter
extraction" and "relevant to method extraction" do not carve a paper up the
same way, and a shared map would force the looser union on both. It is one
extra call per extractor per paper - 35 in the M8 corpus, against ~46 for the
whole run - and it fixes the cause rather than the symptom. Merge to a shared
map only if that cost becomes real at scale.
"""

from __future__ import annotations

from typing import Any

from papersynth.core.document import Section, StructuredDocument
from papersynth.llm.base import LLMProvider

#: When the static regex already selects at least this fraction of sections,
#: the paper is shaped like the fixtures and semantic triage is skipped.
REGEX_CONFIDENT_COVERAGE = 0.60

#: Below this many sections there is nothing to narrow - reading all of them
#: costs less than the triage call would. Real papers have 30+; this only
#: spares tiny documents and test fixtures a pointless round trip.
MIN_SECTIONS_FOR_TRIAGE = 6

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relevant_sections": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"},
    },
    "required": ["relevant_sections"],
}


def triage(
    doc: StructuredDocument,
    *,
    claim_type: str,
    what: str,
    regex_selection: list[Section],
    provider: LLMProvider | None,
) -> tuple[list[Section], str]:
    """Return the sections to read, and a one-line explanation of why.

    `what` is a short description of what this extractor looks for, e.g.
    "configured numeric or categorical values an implementer must set".
    """
    total = len(doc.sections)
    if total == 0:
        return [], "document has no sections"

    if total < MIN_SECTIONS_FOR_TRIAGE:
        return list(doc.sections), f"only {total} sections; reading all"

    if len(regex_selection) / total >= REGEX_CONFIDENT_COVERAGE:
        return (
            regex_selection,
            f"regex pre-filter already covers {len(regex_selection)}/{total} "
            "sections; paper is fixture-shaped, semantic triage skipped",
        )

    if provider is None:
        # No model to ask. The regex selection stands, but say plainly it may
        # be BERT-shaped and miss this paper's vocabulary.
        return (
            regex_selection or list(doc.sections),
            f"no provider for semantic triage; regex selected {len(regex_selection)}/"
            f"{total} sections and may miss this paper's terminology",
        )

    listing = "\n".join(f"[{s.index}] {s.title}" for s in doc.sections)
    prompt = (
        f"An extractor is looking for: {what}.\n\n"
        "Here are the section titles of one paper. Return the indices of the "
        "sections most likely to contain that, and only those - reading an "
        "irrelevant section wastes budget and adds noise, but missing a "
        "relevant one loses real content.\n\n"
        f"{listing}\n\n"
        'Answer JSON: {"relevant_sections": [<int>, ...], "reason": "<one line>"}'
    )

    kwargs: dict[str, Any] = {"schema": _SCHEMA, "temperature": 0.0}
    if hasattr(provider, "chain"):
        kwargs |= {
            "stage": "extract",
            "paper_id": doc.paper_id,
            "extractor": f"triage:{claim_type}",
        }

    try:
        completion = provider.complete(prompt, **kwargs)
    except Exception:
        return (
            regex_selection or list(doc.sections),
            f"semantic triage failed; fell back to regex ({len(regex_selection)}/{total})",
        )

    payload = completion.parsed if isinstance(completion.parsed, dict) else {}
    picked = {
        i for i in payload.get("relevant_sections") or [] if isinstance(i, int) and 0 <= i < total
    }
    if not picked:
        # The model chose nothing. Trust the regex over an empty answer rather
        # than reading zero sections.
        return (
            regex_selection or list(doc.sections),
            "semantic triage returned no sections; kept regex selection",
        )

    selected = [s for s in doc.sections if s.index in picked]
    reason = str(payload.get("reason", "")).strip() or "selected by semantic triage"
    return selected, f"semantic triage picked {len(selected)}/{total}: {reason}"
