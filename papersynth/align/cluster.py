"""Cross-paper alignment: VerifiedClaimSet[] -> ConceptGraph (stage 3).

The asymmetry in section 8.4 drives every choice here: a false merge is far
more damaging than a false split. Merging two distinct parameters fabricates a
contradiction that does not exist, or silently averages unrelated values.
Splitting one concept in two just yields two singleton clusters and no
contradiction - visible, recoverable, and harmless.

So the primary key is the canonical name, which for hyperparameters is a
strong and near-exact signal.

It is not a key at all for method claims, which align on `sub_problem` - the
question a design decision answers - and that requires two papers to
independently invent the same snake_case name. On the M8 corpus they never
did, and zero of 37 clusters spanned more than one paper. A run in which
nothing aligns reports "0 contradictions" for the same reason an empty run
does, so this failure is invisible in every artifact the system produces.

What closes the gap is `semantic.propose_merges`: one call per claim type over
the keys exact-name blocking left unmatched, asking which of them name the
same question. Embeddings were tried first and cannot do it - the best M8
cross-paper pair scored 0.401 against a 0.82 threshold, while on
BERT/RoBERTa/ALBERT surface similarity proposed five merges and all five were
wrong (num_steps with warmup_steps, next_sentence_positive_ratio with
next_sentence_negative_ratio), fabricating two of the three contradictions
that run reported. Names composed from shared words make surface similarity
track naming convention rather than meaning, so the embedding path is gone
rather than merely defaulted off.

No semantic merge is admitted unreviewed. Every cluster one creates goes to
the SplitterAgent regardless of `split_all`, which is the gate's designed job
(section 8.4) and what it was waiting for. Exact-name clusters are not
reviewed by default, on measurement: on BERT/RoBERTa/ALBERT reviewing them
split batch_size into three concepts and lost the genuine 256-against-8000
disagreement, the corpus's headline finding, while gaining nothing that
condition grouping had not already kept apart.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from papersynth.align.semantic import MergeCandidate, propose_merges
from papersynth.align.splitter import SplitterAgent
from papersynth.core import ids
from papersynth.core.models import (
    Agreement,
    Claim,
    ClaimSet,
    ConceptCluster,
    ConceptGraph,
)
from papersynth.llm.base import LLMProvider


@dataclass
class AlignmentReport:
    clusters: int = 0
    multi_paper_clusters: int = 0
    merged_by_name: int = 0
    merged_by_semantic: int = 0
    split_reviewed: int = 0
    split_rejected: int = 0
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []


class _UnionFind:
    def __init__(self, keys: list[str]) -> None:
        self.parent = {k: k for k in keys}

    def find(self, key: str) -> str:
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # Deterministic ordering, so the same inputs always produce the same
        # representative and therefore the same cluster IDs (NFR-02).
        lo, hi = sorted((ra, rb))
        self.parent[hi] = lo
        return True


class Aligner:
    """Groups semantically equivalent claims across papers."""

    def __init__(
        self,
        *,
        provider: LLMProvider | None = None,
        splitter: SplitterAgent | None = None,
        semantic_merges: bool = True,
        split_all: bool = False,
    ) -> None:
        self.provider = provider
        self.splitter = splitter or (SplitterAgent(provider) if provider else None)
        #: On by default when a provider exists. This is the only mechanism
        #: that aligns differently-named concepts, and without it a corpus
        #: whose papers do not share vocabulary produces no cross-paper
        #: clusters and therefore no findings at all.
        self.semantic_merges = semantic_merges
        #: Review every multi-paper cluster, not only the semantically merged
        #: ones. Off by default because on BERT/RoBERTa/ALBERT it split
        #: batch_size into three concepts and lost the corpus's headline
        #: disagreement. Semantic merges are reviewed either way.
        self.split_all = split_all

    def align(self, claim_sets: list[ClaimSet]) -> tuple[ConceptGraph, AlignmentReport]:
        report = AlignmentReport()

        # Only verified claims align. A rejected claim reaching a cluster could
        # become a contradiction, which is exactly the failure mode
        # verification exists to prevent.
        claims = [c for cs in claim_sets for c in cs.claims if c.status == "verified"]
        graph = ConceptGraph(claims={c.claim_id: c for c in claims})

        by_type: dict[str, list[Claim]] = defaultdict(list)
        for claim in claims:
            by_type[claim.type].append(claim)

        #: Clusters a semantic merge created. They are reviewed by the split
        #: gate whatever `split_all` says: a proposed merge that no adversary
        #: has looked at is exactly the false merge section 8.4 warns about.
        review_required: set[str] = set()
        for claim_type, group in sorted(by_type.items()):
            for cluster, was_proposed in self._align_block(claim_type, group, report):
                graph.clusters.append(cluster)
                if was_proposed:
                    review_required.add(cluster.cluster_id)

        graph.clusters = self._apply_split_gate(graph, report, review_required)

        for cluster in graph.clusters:
            for alias in cluster.symbol_aliases:
                graph.symbol_map[alias] = cluster.canonical_name

        report.clusters = len(graph.clusters)
        report.multi_paper_clusters = sum(1 for c in graph.clusters if c.is_multi_paper)
        return graph, report

    def _apply_split_gate(
        self,
        graph: ConceptGraph,
        report: AlignmentReport,
        review_required: set[str],
    ) -> list[ConceptCluster]:
        """Let the splitter reject merges the aligner proposed.

        Only multi-paper clusters are reviewed: a single-paper cluster cannot
        host a cross-paper contradiction, so splitting it changes nothing and
        would spend a call to learn that. Among those, semantically merged
        clusters are always reviewed and exact-name clusters only under
        `split_all` - the gate earns its keep on the former and measurably
        costs findings on the latter.

        A splitter failure leaves the cluster intact rather than dropping it.
        Losing a cluster because a review call failed would silently remove a
        real disagreement from the corpus, which is worse than leaving an
        unreviewed merge visible in the artifact as split_check "n/a".
        """
        if self.splitter is None:
            return graph.clusters

        from papersynth.core.errors import PaperSynthError

        out: list[ConceptCluster] = []
        for cluster in graph.clusters:
            if not cluster.is_multi_paper:
                out.append(cluster)
                continue
            if not (self.split_all or cluster.cluster_id in review_required):
                out.append(cluster)
                continue

            report.split_reviewed += 1
            try:
                replacements, note = self.splitter.review(cluster, graph.claims_in(cluster))
            except PaperSynthError as exc:
                report.notes.append(f"split gate failed on {cluster.cluster_id}: {exc}")
                out.append(cluster)
                continue

            if len(replacements) > 1:
                report.split_rejected += 1
                report.notes.append(note)
            out.extend(replacements)
        return out

    def _align_block(
        self, claim_type: str, claims: list[Claim], report: AlignmentReport
    ) -> list[tuple[ConceptCluster, bool]]:
        """Align within one claim type. Never across - an equation is not a dataset.

        Returns each cluster with a flag saying whether a semantic merge built
        it, so the caller knows which ones must face the split gate.
        """
        buckets: dict[str, list[Claim]] = defaultdict(list)
        for claim in claims:
            buckets[_alignment_key(claim)].append(claim)

        keys = sorted(buckets)
        if len(keys) > 1:
            report.merged_by_name += len(claims) - len(keys)

        union = _UnionFind(keys)
        merged_keys: set[str] = set()
        if self.semantic_merges and len(keys) > 1:
            merged_keys = self._merge_semantic(claim_type, keys, buckets, union, report)

        grouped: dict[str, list[Claim]] = defaultdict(list)
        for key in keys:
            grouped[union.find(key)].extend(buckets[key])

        proposed_roots = {union.find(k) for k in merged_keys}
        return [
            (self._build_cluster(claim_type, members), root in proposed_roots)
            for root, members in sorted(grouped.items())
        ]

    def _merge_semantic(
        self,
        claim_type: str,
        keys: list[str],
        buckets: dict[str, list[Claim]],
        union: _UnionFind,
        report: AlignmentReport,
    ) -> set[str]:
        """Ask a model which unmatched keys name the same concept.

        Only keys that exact-name blocking left inside a single paper are
        offered. A key already spanning papers has found its match, and
        putting it up for re-grouping could only take it away from one it
        already earned by exact agreement.
        """
        candidates = [
            MergeCandidate(
                key=key,
                paper_id=buckets[key][0].paper_id,
                description=_role_description(buckets[key][0]),
            )
            for key in keys
            if len({c.paper_id for c in buckets[key]}) == 1
        ]

        groups, notes = propose_merges(claim_type, candidates, provider=self.provider)
        report.notes.extend(notes)

        merged: set[str] = set()
        for group in groups:
            # The unit and value-type guards still apply. A model calling two
            # things one concept does not make a warmup in steps and a warmup
            # in epochs comparable, and merging them would invent a value
            # conflict out of a unit mismatch.
            head = group[0]
            for other in group[1:]:
                if not _safe_to_merge(buckets[head][0], buckets[other][0]):
                    report.notes.append(
                        f"rejected semantic merge of {head!r} with {other!r}: "
                        "incompatible unit or value type"
                    )
                    continue
                if union.union(head, other):
                    report.merged_by_semantic += 1
                    merged.update((head, other))
                    report.notes.append(f"merged {head!r} with {other!r} on semantic alignment")
        return merged

    def _build_cluster(self, claim_type: str, members: list[Claim]) -> ConceptCluster:
        members = sorted(members, key=lambda c: c.claim_id)
        canonical = _canonical_name(members)
        aliases = sorted(
            {str(c.payload["paper_symbol"]) for c in members if c.payload.get("paper_symbol")}
        )
        return ConceptCluster(
            cluster_id=ids.cluster_id(claim_type, canonical),
            canonical_name=canonical,
            concept_type=claim_type,  # type: ignore[arg-type]
            member_claims=[c.claim_id for c in members],
            symbol_aliases=aliases,
            papers=sorted({c.paper_id for c in members}),
            agreement=_agreement(members),
            # Not reviewed yet. The gate overwrites this with pass or fail on
            # the clusters it looks at; the rest keep "n/a" so the artifact
            # never records a review that did not happen.
            split_check="n/a",
        )


def _alignment_key(claim: Claim) -> str:
    """The blocking key. Exact for hyperparameters, best-effort otherwise."""
    payload = claim.payload
    # sub_problem first: a method claim aligns on the question it answers, not
    # on the approach it names, or two papers proposing rival answers would
    # never be compared.
    for field in ("canonical_name", "sub_problem", "component_id", "name", "label", "metric"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return claim.claim_id


#: Fields that say what a claim is ABOUT. `value`, `approach` and the like are
#: deliberately absent: they say what a paper concluded, and two papers must
#: land in one cluster precisely when they concluded differently.
_ROLE_FIELDS = ("role", "unit", "applies_to", "component_id", "metric", "dataset")


def _role_description(claim: Claim) -> str:
    """One line describing what a claim is about, for the merge proposer.

    Deliberately excludes the value and the chosen approach. Two papers
    disagreeing about a learning rate, or answering one design question with
    rival methods, must still land in one cluster - describing them by their
    answers would push them apart at exactly the moment the disagreement
    becomes worth finding.
    """
    payload = claim.payload
    parts = [_alignment_key(claim).replace("_", " ")]
    for field in _ROLE_FIELDS:
        value = payload.get(field)
        if isinstance(value, str) and value.strip() and value != "global":
            parts.append(value.strip())
    # The section a claim came from is often the only hint of what a bare
    # snake_case key means, and it costs nothing to include.
    section = claim.provenance.section
    if section:
        parts.append(f"(section: {section})")
    return " ".join(dict.fromkeys(parts))


def _safe_to_merge(a: Claim, b: Claim) -> bool:
    """Cheap guards against the merges that hurt most.

    Two hyperparameters with different units are different quantities however
    similar their names read - "warmup" in steps and "warmup" in epochs are not
    one concept, and merging them would invent a value conflict out of a unit
    mismatch.
    """
    unit_a, unit_b = a.payload.get("unit"), b.payload.get("unit")
    if unit_a and unit_b and unit_a != unit_b:
        return False

    type_a, type_b = a.payload.get("value_type"), b.payload.get("value_type")
    if type_a and type_b:
        numeric = {"float", "int"}
        if (type_a in numeric) != (type_b in numeric):
            return False

    return True


def _canonical_name(members: list[Claim]) -> str:
    """Most common name in the cluster; ties break alphabetically for stability."""
    counts: dict[str, int] = defaultdict(int)
    for claim in members:
        counts[_alignment_key(claim)] += 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _agreement(members: list[Claim]) -> Agreement:
    if len({c.paper_id for c in members}) <= 1:
        return "singleton"

    values = [_normalized_value(c) for c in members]
    distinct = set(values)
    if len(distinct) == 1:
        return "unanimous"

    top = max(distinct, key=values.count)
    return "majority" if values.count(top) > len(values) / 2 else "conflicting"


def _normalized_value(claim: Claim) -> str:
    value = claim.payload.get("value")
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, str):
        return value.strip().lower()
    return str(value)
