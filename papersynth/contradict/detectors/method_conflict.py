"""METHOD_CONFLICT detection (section 7.6).

Two papers solving the same sub-problem with incompatible approaches. No number
separates them, so no value detector can see it - which is why BERT using
next-sentence prediction and RoBERTa removing it went undetected until this
existed, despite being one of the sharpest disagreements in that literature.

Two shapes of disagreement, and the second is easy to miss:

  RIVAL APPROACHES  both papers adopt something, but different things.
                    NSP against sentence order prediction.

  ADOPT vs REJECT   both papers name the SAME approach and disagree about
                    whether to use it. RoBERTa removing NSP agrees with BERT
                    on what NSP is and contradicts it on whether to use it.
                    Comparing approach strings alone would call that agreement.

These are never auto-resolved. Choosing between incompatible approaches is an
engineering decision about a target setting, not a lookup, and the policy's
method_conflicts_escalate rule reflects that.
"""

from __future__ import annotations

import re
from collections import defaultdict

from rapidfuzz import fuzz

from papersynth.contradict.severity import specificity
from papersynth.core import ids
from papersynth.core.models import (
    Claim,
    ConceptCluster,
    ConceptGraph,
    Contradiction,
    Criticality,
    Position,
    Support,
)

DETECTOR_VERSION = "method_conflict_detector@1.0.0"

#: Sub-problems where picking the wrong approach changes what the code computes
#: rather than how well it performs.
STRUCTURAL_SUB_PROBLEMS = frozenset(
    {
        "sentence_level_objective",
        "pretraining_objective",
        "positional_encoding",
        "attention_mechanism",
        "tokenizer",
        "parameter_sharing",
        "normalization_placement",
        "masking_strategy",
    }
)


class MethodConflictDetector:
    conflict_type = "METHOD_CONFLICT"
    #: Section 10.3: the policy honours this and never auto-resolves these,
    #: whatever rule might otherwise match.
    auto_resolvable = False
    version = DETECTOR_VERSION

    def scan(self, cluster: ConceptCluster, graph: ConceptGraph) -> list[Contradiction]:
        if cluster.concept_type != "method" or not cluster.is_multi_paper:
            return []

        claims = [
            c
            for c in graph.claims_in(cluster)
            if c.status == "verified" and c.payload.get("attribution", "own") == "own"
        ]
        if len(claims) < 2:
            return []

        out: list[Contradiction] = []
        for condition, group in sorted(_group_by_condition(claims).items()):
            found = self._scan_group(cluster, condition, group)
            if found is not None:
                out.append(found)
        return out

    def _scan_group(
        self, cluster: ConceptCluster, condition: str, claims: list[Claim]
    ) -> Contradiction | None:
        if len({c.paper_id for c in claims}) < 2:
            return None

        approaches = _canonical_approaches(claims)
        stances = {(approaches[c.claim_id], bool(c.payload.get("adopted", True))) for c in claims}
        if len(stances) < 2:
            return None

        adopted = {a for a, used in stances if used}
        rejected = {a for a, used in stances if not used}

        # Rival approaches only count when more than one is actually adopted.
        # Two papers both adopting NSP, one of which also lists an alternative
        # it declined, are in agreement.
        disputed = adopted & rejected
        if len(adopted) < 2 and not disputed:
            return None

        positions = _positions(claims, approaches)
        if len(positions) < 2:
            return None

        return Contradiction(
            contradiction_id=ids.contradiction_id(
                cluster.cluster_id, self.conflict_type, [p.claim_id for p in positions]
            ),
            cluster_id=cluster.cluster_id,
            type="METHOD_CONFLICT",
            severity=_severity(cluster.canonical_name, disputed),
            description=_describe(cluster.canonical_name, condition, adopted, disputed),
            positions=positions,
            detected_by=self.version,
        )


def _describe(sub_problem: str, condition: str, adopted: set[str], disputed: set[str]) -> str:
    scope = f" under {condition!r}" if condition else ""
    if disputed:
        name = sorted(disputed)[0]
        return (
            f"Papers disagree on whether to use {name} for {sub_problem}{scope}: "
            "one adopts it, another explicitly removes it."
        )
    return (
        f"Papers take incompatible approaches to {sub_problem}{scope}: "
        + ", ".join(sorted(adopted))
        + "."
    )


def _severity(sub_problem: str, disputed: set[str]) -> Criticality:
    """Method conflicts escalate regardless, so severity only sets urgency.

    Structural sub-problems block: you cannot write the training loop without
    knowing whether a sentence-level objective exists at all. The rest are
    material - implementable either way, with different results.
    """
    if sub_problem in STRUCTURAL_SUB_PROBLEMS or disputed:
        return "BLOCKING"
    return "MATERIAL"


def _positions(claims: list[Claim], approaches: dict[str, str]) -> list[Position]:
    """One position per distinct stance, earliest claim id representing it."""
    by_stance: dict[tuple[str, bool], list[Claim]] = defaultdict(list)
    for claim in claims:
        stance = (approaches[claim.claim_id], bool(claim.payload.get("adopted", True)))
        by_stance[stance].append(claim)

    positions = []
    for _, group in sorted(by_stance.items()):
        claim = sorted(group, key=lambda c: c.claim_id)[0]
        payload = claim.payload
        rendered = str(payload.get("approach", "")).strip()
        if not payload.get("adopted", True):
            rendered = f"removes {rendered}"
        if payload.get("rationale"):
            rendered += f" - {payload['rationale']}"
        positions.append(
            Position(
                claim_id=claim.claim_id,
                paper_id=claim.paper_id,
                position=rendered,
                support=Support(
                    specificity=specificity(payload),
                    stated_explicitly=bool(payload.get("stated_explicitly", True)),
                    has_condition=bool(payload.get("condition")),
                ),
            )
        )
    return positions


def _approach_forms(text: str) -> set[str]:
    """Every string this approach might legitimately be written as.

    A paper naming "next sentence prediction (NSP)" and one naming "NSP" are
    describing the same thing. Matching on the literal string would make them
    rival approaches instead of a direct contradiction about one, which is
    exactly backwards - and the adopt-versus-reject case, the sharpest kind of
    method conflict, depends entirely on recognising them as the same.
    """
    lowered = text.lower()
    forms: set[str] = set()

    parenthetical = re.findall(r"\(([^)]*)\)", lowered)
    stripped = _words(re.sub(r"\([^)]*\)", " ", lowered))

    if stripped:
        forms.add(stripped)
        initials = "".join(word[0] for word in stripped.split() if word)
        if len(initials) >= 2:
            forms.add(initials)

    for inner in parenthetical:
        cleaned = _words(inner)
        if cleaned:
            forms.add(cleaned)

    return forms or {_words(lowered)}


def _words(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text).split())


#: Above this token-set similarity, two approach descriptions are the same
#: decision worded differently rather than rival approaches.
SAME_APPROACH_RATIO = 90


def _near_identical(left: set[str], right: set[str]) -> bool:
    """Whether two approach descriptions name the same decision.

    Exact form matching is not enough in practice. Extraction runs over section
    batches, so one paper describes a single decision several times in slightly
    different words - BERT's masking strategy came back as "80% mask token, 10%
    random token, 10% unchanged" and "80% [MASK] token, 10% random token, 10%
    unchanged token" from two batches, and each became its own position in the
    conflict.

    Token-set similarity is the right comparison here: it ignores word order
    and tolerates one description carrying extra qualifiers, so
    "case-preserving WordPiece" matches "WordPiece" while "WordPiece" and
    "SentencePiece" stay apart.
    """
    return any(fuzz.token_set_ratio(a, b) >= SAME_APPROACH_RATIO for a in left for b in right)


def _canonical_approaches(claims: list[Claim]) -> dict[str, str]:
    """Map each claim id to a shared approach key.

    Approaches whose written forms overlap collapse onto one key, chosen as the
    alphabetically first form so the result does not depend on claim order.
    """
    forms = {c.claim_id: _approach_forms(str(c.payload.get("approach", ""))) for c in claims}

    parent: dict[str, str] = {cid: cid for cid in forms}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    ids_sorted = sorted(forms)
    for i, a in enumerate(ids_sorted):
        for b in ids_sorted[i + 1 :]:
            if forms[a] & forms[b] or _near_identical(forms[a], forms[b]):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    grouped: dict[str, set[str]] = defaultdict(set)
    for cid in ids_sorted:
        grouped[find(cid)] |= forms[cid]

    return {cid: sorted(grouped[find(cid)])[0] for cid in ids_sorted}


def _group_by_condition(claims: list[Claim]) -> dict[str, list[Claim]]:
    from papersynth.contradict.detectors.value_conflict import normalize_condition

    grouped: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        grouped[normalize_condition(claim.payload.get("condition"))].append(claim)
    return grouped
