"""Cross-paper alignment: VerifiedClaimSet[] -> ConceptGraph (stage 3).

The asymmetry in section 8.4 drives every choice here: a false merge is far
more damaging than a false split. Merging two distinct parameters fabricates a
contradiction that does not exist, or silently averages unrelated values.
Splitting one concept in two just yields two singleton clusters and no
contradiction - visible, recoverable, and harmless.

So the primary key is the canonical name, which for hyperparameters is a
strong and near-exact signal.

Embedding-proposed merges are OFF by default, and that is an empirical finding
rather than caution. On BERT/RoBERTa/ALBERT they fired five times and were
wrong five times: num_steps merged with warmup_steps, and
next_sentence_positive_ratio with next_sentence_negative_ratio. A
bag-of-ngrams embedder scores those pairs highly because they share most of
their characters, which is exactly the wrong signal - hyperparameter names are
built by composing shared words, so surface similarity tracks naming
convention rather than meaning. Two of the three contradictions that run
reported were fabricated by these merges.

They can be re-enabled once the SplitterAgent exists to reject a proposed merge
(section 8.4), which is the gate the design always intended to sit behind them.
Until then a false split costs a missed conflict, while a false merge invents
one and burns the reviewer's trust in the whole list (section 9).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from papersynth.align.embed import Embedder, HashEmbedder, cosine
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
    merged_by_embedding: int = 0
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
        embedder: Embedder | None = None,
        threshold: float = 0.82,
        embedding_merges: bool | None = None,
        splitter: SplitterAgent | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        self.embedder = embedder or HashEmbedder()
        self.threshold = threshold
        self.splitter = splitter or (SplitterAgent(provider) if provider else None)
        #: Off by default even with the gate present. Measured on
        #: BERT/RoBERTa/ALBERT: embedding merges proposed five merges and the
        #: gate rejected all five, so they contributed nothing except calls
        #: spent undoing them. The gate earns its keep on same-name clusters
        #: instead - hidden_dim aligning three model variants - which does not
        #: depend on embedding merges at all. Opt in explicitly if a corpus
        #: uses genuinely divergent naming.
        self.embedding_merges = bool(embedding_merges)

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

        for claim_type, group in sorted(by_type.items()):
            graph.clusters.extend(self._align_block(claim_type, group, report))

        graph.clusters = self._apply_split_gate(graph, report)

        for cluster in graph.clusters:
            for alias in cluster.symbol_aliases:
                graph.symbol_map[alias] = cluster.canonical_name

        report.clusters = len(graph.clusters)
        report.multi_paper_clusters = sum(1 for c in graph.clusters if c.is_multi_paper)
        return graph, report

    def _apply_split_gate(
        self, graph: ConceptGraph, report: AlignmentReport
    ) -> list[ConceptCluster]:
        """Let the splitter reject merges the aligner proposed.

        Only multi-paper clusters are reviewed: a single-paper cluster cannot
        host a cross-paper contradiction, so splitting it changes nothing and
        would spend a call to learn that.

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
    ) -> list[ConceptCluster]:
        """Align within one claim type. Never across - an equation is not a dataset."""
        buckets: dict[str, list[Claim]] = defaultdict(list)
        for claim in claims:
            buckets[_alignment_key(claim)].append(claim)

        keys = sorted(buckets)
        if len(keys) > 1:
            report.merged_by_name += len(claims) - len(keys)

        union = _UnionFind(keys)
        if self.embedding_merges and len(keys) > 1:
            self._merge_similar(keys, buckets, union, report)

        grouped: dict[str, list[Claim]] = defaultdict(list)
        for key in keys:
            grouped[union.find(key)].extend(buckets[key])

        return [self._build_cluster(claim_type, members) for _, members in sorted(grouped.items())]

    def _merge_similar(
        self,
        keys: list[str],
        buckets: dict[str, list[Claim]],
        union: _UnionFind,
        report: AlignmentReport,
    ) -> None:
        """Propose merges between differently named claims, above threshold."""
        descriptions = [_role_description(buckets[k][0]) for k in keys]
        vectors = self.embedder.embed(descriptions)

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if cosine(vectors[i], vectors[j]) < self.threshold:
                    continue
                if not _safe_to_merge(buckets[keys[i]][0], buckets[keys[j]][0]):
                    continue
                if union.union(keys[i], keys[j]):
                    report.merged_by_embedding += 1
                    report.notes.append(
                        f"merged {keys[i]!r} with {keys[j]!r} on embedding similarity"
                    )

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
            # No SplitterAgent in the MVA; the gate is recorded as not run
            # rather than as passed, so the artifact never overstates it.
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


def _role_description(claim: Claim) -> str:
    """Text an embedder can compare. Deliberately excludes the value.

    Two papers disagreeing about a learning rate must still land in one
    cluster; embedding the value would push them apart precisely when the
    disagreement is the thing worth finding.
    """
    payload = claim.payload
    parts = [_alignment_key(claim).replace("_", " ")]
    for field in ("role", "unit", "applies_to"):
        value = payload.get(field)
        if isinstance(value, str) and value and value != "global":
            parts.append(value)
    return " ".join(parts)


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
