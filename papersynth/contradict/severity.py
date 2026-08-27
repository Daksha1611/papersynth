"""Severity assignment (section 7.6).

The ladder decides what halts a spec and what merely annotates it:

  BLOCKING  you cannot write correct code without deciding; blocks emission
  MATERIAL  affects behaviour but is implementable either way; rides along
  COSMETIC  notation or formatting only; auto-resolved and logged

For a value conflict, magnitude is the honest proxy. Two papers saying 0.0001
and 0.0003 give a model that trains either way; 0.0001 against 0.1 gives one
that trains and one that diverges. An order of magnitude is where "a different
tuning" becomes "a different algorithm", so that is the BLOCKING line.
"""

from __future__ import annotations

from papersynth.core.models import Criticality

#: Ratio between extreme values at which a disagreement stops being a matter
#: of tuning. Configurable rather than absolute - it is a heuristic, not a law.
BLOCKING_RATIO = 10.0

#: Values differing by less than this are the same number written differently.
COSMETIC_TOLERANCE = 1e-9

#: Getting these wrong changes what the code does, not just how well it works.
STRUCTURAL_PARAMETERS = frozenset(
    {
        "num_layers",
        "num_heads",
        "hidden_dim",
        "embed_dim",
        "vocab_size",
        "sequence_length",
        "optimizer",
        "loss_function",
        "activation",
        "normalization",
        "positional_encoding",
    }
)


def value_conflict_severity(values: list[object], canonical_name: str = "") -> Criticality:
    """Severity for a set of disagreeing values under one condition."""
    numeric = [float(v) for v in values if isinstance(v, int | float) and not isinstance(v, bool)]

    if len(numeric) != len(values):
        # A categorical disagreement - "Adam" against "SGD". There is no
        # midpoint to split, and picking wrong changes behaviour outright.
        return "BLOCKING" if canonical_name in STRUCTURAL_PARAMETERS else "MATERIAL"

    if not numeric:
        return "MATERIAL"

    low, high = min(numeric), max(numeric)

    if abs(high - low) <= COSMETIC_TOLERANCE:
        return "COSMETIC"

    if canonical_name in STRUCTURAL_PARAMETERS:
        # A layer count of 6 against 12 is not a tuning difference; it is a
        # different model, however small the ratio looks.
        return "BLOCKING"

    if low <= 0 or high <= 0:
        # A sign disagreement, or one side at zero, means the values are not
        # on a comparable scale at all.
        return "BLOCKING"

    return "BLOCKING" if (high / low) >= BLOCKING_RATIO else "MATERIAL"


def specificity(payload: dict[str, object]) -> float:
    """How narrowly scoped a claim is, in [0, 1].

    Drives `prefer_specific_over_general` during reconciliation. A value stated
    for one named variant under one named condition is better evidence for that
    variant than a paper-wide default.
    """
    score = 0.5
    if payload.get("condition"):
        score += 0.3
    applies_to = payload.get("applies_to")
    if applies_to and applies_to != "global":
        score += 0.1
    if payload.get("stated_explicitly") is False:
        # Read off a figure rather than stated. It cannot auto-resolve a
        # conflict (ER-07), and its specificity should not help it try.
        score -= 0.3
    else:
        score += 0.1
    return round(min(1.0, max(0.0, score)), 2)
