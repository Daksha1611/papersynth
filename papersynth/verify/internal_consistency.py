"""Intra-paper consistency (section 10.1).

The cross-paper detectors ask whether two papers disagree. This asks whether
one paper's claims are being read wrongly - as independent global settings when
they are stages of a described procedure.

The motivating failure: Kunzel's field experiment reports 68,378 registered
voters, 1,295 households randomized, 913 in the treatment arm, 501 in the final
sample. Read as configuration those numbers contradict - 913 treated out of a
501 sample is impossible - and a coding agent handed the flattened spec said
so. Read as what they are, a study's funnel, they are perfectly coherent.

The resolution is not to report a conflict. It is to attach the scope the
values share, so downstream they are presented together as a described context
rather than apart as knobs. This module detects the situation and records it;
`SpecBuilder` does the scoping.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from papersynth.core.models import Claim

#: canonical_names known to be genuine settings. Kept in sync with
#: SpecBuilder._KNOWN_SETTINGS by test; a value outside this set that clusters
#: within one section is a described quantity, not a knob.
KNOWN_SETTINGS = frozenset(
    {
        "learning_rate",
        "batch_size",
        "dropout",
        "weight_decay",
        "optimizer",
        "num_layers",
        "num_heads",
        "hidden_dim",
        "embed_dim",
        "num_epochs",
        "num_steps",
        "warmup_steps",
        "sequence_length",
        "vocab_size",
        "beam_size",
        "temperature",
        "top_p",
        "gradient_clip",
        "momentum",
        "label_smoothing",
        "weight_initialization",
        "random_seed",
        "activation_function",
        "layernorm_epsilon",
        "attention_heads",
    }
)


@dataclass
class ScopeFinding:
    scope_id: str
    section: str
    claim_ids: list[str]
    note: str


def review(claims: list[Claim]) -> list[ScopeFinding]:
    """Findings for one paper's claims. Empty when nothing needs scoping.

    A finding is raised when a single section contributes two or more
    unrecognized numeric hyperparameter claims with distinct names. That
    pattern - several different named quantities from one passage - is a
    described procedure, and presenting the quantities flat is what turns a
    funnel into an apparent contradiction.
    """
    by_scope: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        if claim.status not in ("verified", "extracted"):
            continue
        if claim.type != "hyperparameter":
            continue
        name = str(claim.payload.get("canonical_name") or "")
        if name in KNOWN_SETTINGS:
            continue
        value = claim.payload.get("value")
        if not isinstance(value, int | float) or isinstance(value, bool):
            continue
        by_scope[claim.scope_id or claim.paper_id].append(claim)

    findings = []
    for scope_id, group in sorted(by_scope.items()):
        names = {str(c.payload.get("canonical_name")) for c in group}
        if len(group) < 2 or len(names) < 2:
            continue
        section = group[0].provenance.section or scope_id
        findings.append(
            ScopeFinding(
                scope_id=scope_id,
                section=section,
                claim_ids=sorted(c.claim_id for c in group),
                note=(
                    f"{len(group)} distinct quantities extracted from one section "
                    f"({section!r}): {', '.join(sorted(names))}. Read as independent "
                    "settings these can look inconsistent; they describe one "
                    "context and are scoped to it rather than reported as a conflict."
                ),
            )
        )
    return findings
