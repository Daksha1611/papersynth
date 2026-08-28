"""The split gate (sections 6.2, 8.4).

The Aligner maximizes merges; this rejects the ones that were only
superficially similar. The bias is deliberate and asymmetric: a false merge
fabricates a contradiction or averages two unrelated quantities, while a false
split yields singletons and no contradiction - visible and recoverable.
"""

from __future__ import annotations

from papersynth.align import Aligner
from papersynth.align.splitter import SplitterAgent
from papersynth.core.errors import ProviderError
from papersynth.core.models import ClaimSet
from papersynth.llm.stub import StubProvider
from tests.conftest import make_claim, make_doc


def claim(paper, name, value, **kwargs):
    return make_claim(make_doc(paper), canonical_name=name, value=value, **kwargs)


def cluster_of(*claims):
    sets = [ClaimSet(paper_id=c.paper_id, claims=[c]) for c in claims]
    graph, _ = Aligner().align(sets)
    return graph, graph.clusters[0]


def assignments(*pairs):
    return {
        "assignments": [{"claim_id": cid, "concept": concept} for cid, concept in pairs],
        "reason": "test",
    }


class TestSplitterVerdicts:
    def test_a_genuine_disagreement_stays_merged(self):
        """Differing values are the point, not a reason to separate."""
        a = claim("p1", "learning_rate", 0.0001)
        b = claim("p2", "learning_rate", 0.0003)
        graph, cluster = cluster_of(a, b)

        provider = StubProvider([assignments((a.claim_id, "lr"), (b.claim_id, "lr"))])
        out, note = SplitterAgent(provider).review(cluster, graph.claims_in(cluster))

        assert len(out) == 1
        assert out[0].split_check == "pass"
        assert note == ""

    def test_distinct_concepts_are_separated(self):
        """hidden_dim 4096 and 768 are two model variants, not a disagreement."""
        a = claim("p1", "hidden_dim", 4096)
        b = claim("p2", "hidden_dim", 768)
        graph, cluster = cluster_of(a, b)

        provider = StubProvider(
            [assignments((a.claim_id, "hidden_dim_xxlarge"), (b.claim_id, "hidden_dim_base"))]
        )
        out, note = SplitterAgent(provider).review(cluster, graph.claims_in(cluster))

        assert len(out) == 2
        assert all(c.split_check == "fail" for c in out)
        assert "split" in note

    def test_a_split_cluster_can_no_longer_host_a_conflict(self):
        """The whole point: separated concepts stop being compared."""
        a = claim("p1", "hidden_dim", 4096)
        b = claim("p2", "hidden_dim", 768)
        graph, cluster = cluster_of(a, b)

        provider = StubProvider([assignments((a.claim_id, "large"), (b.claim_id, "base"))])
        out, _ = SplitterAgent(provider).review(cluster, graph.claims_in(cluster))

        assert all(not c.is_multi_paper for c in out)

    def test_split_clusters_keep_every_claim(self):
        """Evidence must not vanish through the gate."""
        a = claim("p1", "hidden_dim", 4096)
        b = claim("p2", "hidden_dim", 768)
        graph, cluster = cluster_of(a, b)

        provider = StubProvider([assignments((a.claim_id, "x"), (b.claim_id, "y"))])
        out, _ = SplitterAgent(provider).review(cluster, graph.claims_in(cluster))

        assert {m for c in out for m in c.member_claims} == {a.claim_id, b.claim_id}

    def test_split_cluster_ids_stay_unique(self):
        a = claim("p1", "hidden_dim", 4096)
        b = claim("p2", "hidden_dim", 768)
        graph, cluster = cluster_of(a, b)

        provider = StubProvider([assignments((a.claim_id, "x"), (b.claim_id, "y"))])
        out, _ = SplitterAgent(provider).review(cluster, graph.claims_in(cluster))

        assert len({c.cluster_id for c in out}) == len(out)


class TestFailSafe:
    def test_an_unreadable_verdict_leaves_the_cluster_intact(self):
        """A confident NO splits; an unparseable answer is not one. Shattering
        a cluster here would quietly delete a real disagreement."""
        a = claim("p1", "learning_rate", 0.0001)
        b = claim("p2", "learning_rate", 0.0003)
        graph, cluster = cluster_of(a, b)

        out, _ = SplitterAgent(StubProvider([{"nonsense": True}])).review(
            cluster, graph.claims_in(cluster)
        )

        assert len(out) == 1
        assert out[0].split_check == "n/a", "unreviewed, not passed"

    def test_an_empty_assignment_list_leaves_the_cluster_intact(self):
        a = claim("p1", "learning_rate", 0.0001)
        b = claim("p2", "learning_rate", 0.0003)
        graph, cluster = cluster_of(a, b)

        out, _ = SplitterAgent(StubProvider([{"assignments": []}])).review(
            cluster, graph.claims_in(cluster)
        )
        assert len(out) == 1

    def test_a_claim_the_splitter_ignored_is_not_dropped(self):
        """An incomplete answer must not remove evidence from the corpus."""
        a = claim("p1", "learning_rate", 0.0001)
        b = claim("p2", "learning_rate", 0.0003)
        graph, cluster = cluster_of(a, b)

        provider = StubProvider([assignments((a.claim_id, "lr"))])
        out, _ = SplitterAgent(provider).review(cluster, graph.claims_in(cluster))

        assert {m for c in out for m in c.member_claims} == {a.claim_id, b.claim_id}

    def test_a_single_paper_cluster_is_not_reviewed(self):
        """It cannot host a cross-paper contradiction, so a call would buy
        nothing."""
        a = claim("p1", "learning_rate", 0.0001)
        graph, cluster = cluster_of(a)

        provider = StubProvider([])
        out, _ = SplitterAgent(provider).review(cluster, graph.claims_in(cluster))

        assert provider.call_count == 0
        assert out[0].split_check == "n/a"


class TestAlignerIntegration:
    def test_the_gate_runs_only_on_multi_paper_clusters(self):
        a = claim("p1", "learning_rate", 0.0001)
        b = claim("p2", "learning_rate", 0.0003)
        c = claim("p1", "dropout", 0.1)

        provider = StubProvider(lambda _p: assignments((a.claim_id, "lr"), (b.claim_id, "lr")))
        sets = [
            ClaimSet(paper_id="p1", claims=[a, c]),
            ClaimSet(paper_id="p2", claims=[b]),
        ]
        _, report = Aligner(splitter=SplitterAgent(provider)).align(sets)

        assert report.split_reviewed == 1, "only the learning_rate cluster spans papers"
        assert provider.call_count == 1

    def test_a_splitter_failure_leaves_the_cluster_intact(self):
        """Losing a cluster to a failed review call would silently remove a
        real disagreement - worse than an unreviewed merge."""
        a = claim("p1", "learning_rate", 0.0001)
        b = claim("p2", "learning_rate", 0.0003)

        provider = StubProvider(error=ProviderError("splitter unavailable"))
        sets = [ClaimSet(paper_id="p1", claims=[a]), ClaimSet(paper_id="p2", claims=[b])]
        graph, report = Aligner(splitter=SplitterAgent(provider)).align(sets)

        assert len(graph.clusters) == 1
        assert graph.clusters[0].is_multi_paper
        assert any("split gate failed" in n for n in report.notes)

    def test_embedding_merges_are_off_without_a_gate(self):
        """They fabricated two of three contradictions on the first real
        corpus; the gate is what makes them safe."""
        assert Aligner().embedding_merges is False

    def test_embedding_merges_stay_off_even_with_a_gate(self):
        """Measured: they proposed five merges on the first real corpus and
        the gate rejected all five, so they only bought calls spent undoing
        them. Opt in explicitly for a corpus with divergent naming."""
        assert Aligner(splitter=SplitterAgent(StubProvider([]))).embedding_merges is False

    def test_an_explicit_setting_still_wins(self):
        assert Aligner(embedding_merges=True).embedding_merges is True
        assert (
            Aligner(
                splitter=SplitterAgent(StubProvider([])), embedding_merges=False
            ).embedding_merges
            is False
        )

    def test_a_rejected_merge_removes_the_fabricated_conflict(self):
        """End to end: the gate is what stops a false merge becoming a
        contradiction in the spec."""
        from papersynth.contradict import ContradictionScan

        a = claim("p1", "hidden_dim", 4096)
        b = claim("p2", "hidden_dim", 768)
        sets = [ClaimSet(paper_id="p1", claims=[a]), ClaimSet(paper_id="p2", claims=[b])]

        ungated, _ = Aligner().align(sets)
        assert len(ContradictionScan().run(ungated)) == 1, "merged, so it looks like a conflict"

        provider = StubProvider(
            lambda _p: assignments((a.claim_id, "xxlarge"), (b.claim_id, "base"))
        )
        gated, report = Aligner(splitter=SplitterAgent(provider)).align(sets)

        assert ContradictionScan().run(gated) == []
        assert report.split_rejected == 1
