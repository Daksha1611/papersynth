"""Semantic merge proposal (section 8.4).

The M8 failure this exists for: five papers, 37 clusters, zero spanning more
than one paper. Method claims align on `sub_problem`, and no two papers chose
the same snake_case name for the same question, so no detector could fire and
the run reported "0 contradictions" - the number an empty corpus produces.
"""

from __future__ import annotations

from papersynth.align import Aligner
from papersynth.align.semantic import MergeCandidate, propose_merges
from papersynth.align.splitter import SplitterAgent
from papersynth.core import ids
from papersynth.core.models import Claim, ClaimSet, Provenance
from papersynth.llm.stub import StubProvider


def method_claim(paper, sub_problem, approach, section="Method"):
    provenance = Provenance(
        paper_id=paper,
        span_id=f"{paper}#s1.p0.0",
        section=section,
        page=1,
        char_start=0,
        char_end=40,
        quote_hash=ids.quote_hash("text"),
        extraction_method="llm",
        extractor_version="method@1.0.0",
        confidence=0.9,
    )
    claim = Claim.build(
        paper_id=paper,
        claim_type="method",
        provenance=provenance,
        payload={
            "sub_problem": sub_problem,
            "approach": approach,
            "adopted": True,
            "alternatives_rejected": [],
            "rationale": None,
            "attribution": "own",
            "applies_to": "global",
            "condition": None,
            "stated_explicitly": True,
        },
    )
    claim.status = "verified"
    return claim


def groups(*specs):
    return {
        "groups": [{"concept": name, "members": list(members)} for name, members in specs],
        "reason": "test",
    }


def candidates(*pairs):
    return [MergeCandidate(key=k, paper_id=p, description=k.replace("_", " ")) for k, p in pairs]


class TestProposalValidation:
    """Everything the model returns is a proposal about indices we supplied,
    not an answer to trust."""

    def test_a_cross_paper_group_survives(self):
        provider = StubProvider([groups(("policy_enforcement", [0, 1]))])
        out, notes = propose_merges(
            "method",
            candidates(("data_flow_security", "camel"), ("rail_specification", "nemo")),
            provider=provider,
        )
        assert out == [["data_flow_security", "rail_specification"]]
        assert any("proposed 1 cross-paper merge" in n for n in notes)

    def test_a_single_paper_group_is_rejected(self):
        """The M8 funnel failure in miniature: quantities one paper stated
        separately, collapsed into one concept."""
        # Candidates are sorted by (key, paper) before the prompt is built, so
        # indices 1 and 2 are the two Kunzel keys.
        provider = StubProvider([groups(("sample_counts", [1, 2]))])
        out, notes = propose_merges(
            "hyperparameter",
            candidates(
                ("treatment_group_size", "kunzel"),
                ("final_sample_size", "kunzel"),
                ("batch_size", "nemo"),
            ),
            provider=provider,
        )
        assert out == []
        assert any("all members from one paper" in n for n in notes)

    def test_a_key_is_never_claimed_twice(self):
        provider = StubProvider([groups(("a", [0, 1]), ("b", [1, 2]))])
        out, _ = propose_merges(
            "method",
            candidates(("alpha", "p1"), ("beta", "p2"), ("gamma", "p3")),
            provider=provider,
        )
        assert out == [["alpha", "beta"]], "the second group has one unclaimed key left"

    def test_out_of_range_indices_are_dropped(self):
        provider = StubProvider([groups(("a", [0, 99, -3]))])
        out, _ = propose_merges(
            "method", candidates(("alpha", "p1"), ("beta", "p2")), provider=provider
        )
        assert out == []

    def test_no_call_without_two_papers(self):
        provider = StubProvider([groups(("a", [0, 1]))])
        out, _ = propose_merges(
            "method", candidates(("alpha", "p1"), ("beta", "p1")), provider=provider
        )
        assert out == []
        assert provider.call_count == 0, "nothing a cross-paper merge could join"

    def test_no_provider_is_not_a_failure(self):
        out, notes = propose_merges(
            "method", candidates(("alpha", "p1"), ("beta", "p2")), provider=None
        )
        assert (out, notes) == ([], [])

    def test_a_provider_failure_falls_back_to_exact_names(self):
        """Alignment without this call is what shipped before it. Failing the
        run over a merge proposal trades a missed conflict for no spec."""
        provider = StubProvider(error=RuntimeError("boom"))
        out, notes = propose_merges(
            "method", candidates(("alpha", "p1"), ("beta", "p2")), provider=provider
        )
        assert out == []
        assert any("exact names only" in n for n in notes)

    def test_the_prompt_never_shows_the_answer(self):
        """Grouping is by the question. Showing the approach would let a model
        separate two papers precisely because they disagree."""
        a = method_claim("camel", "data_flow_security", "capability tags on values")
        b = method_claim("nemo", "rail_specification", "colang canonical forms")
        provider = StubProvider([groups(("policy", [0, 1]))])
        Aligner(provider=provider, splitter=None).align(
            [ClaimSet(paper_id="camel", claims=[a]), ClaimSet(paper_id="nemo", claims=[b])]
        )
        prompt = provider.prompts[0]
        assert "data_flow_security" in prompt
        assert "capability tags on values" not in prompt
        assert "colang canonical forms" not in prompt


class TestAlignerIntegration:
    def test_differently_named_concepts_reach_one_cluster(self):
        """The M8 pair. Same question, no shared key, and before this they
        were two singletons no detector could see."""
        a = method_claim("camel", "data_flow_security", "capability tags")
        b = method_claim("nemo", "rail_specification_language", "colang forms")
        sets = [ClaimSet(paper_id="camel", claims=[a]), ClaimSet(paper_id="nemo", claims=[b])]

        merge = StubProvider(lambda _p: groups(("policy_enforcement", [0, 1])))
        keep = StubProvider(
            lambda _p: {
                "assignments": [
                    {"claim_id": a.claim_id, "concept": "policy"},
                    {"claim_id": b.claim_id, "concept": "policy"},
                ],
                "reason": "one question",
            }
        )
        graph, report = Aligner(provider=merge, splitter=SplitterAgent(keep)).align(sets)

        assert report.merged_by_semantic == 1
        assert len(graph.clusters) == 1
        assert graph.clusters[0].is_multi_paper

    def test_a_semantic_merge_is_always_split_gated(self):
        """Even with split_all off. A proposed merge no adversary has looked
        at is exactly the false merge section 8.4 warns about."""
        a = method_claim("camel", "data_flow_security", "capability tags")
        b = method_claim("nemo", "rail_specification_language", "colang forms")
        sets = [ClaimSet(paper_id="camel", claims=[a]), ClaimSet(paper_id="nemo", claims=[b])]

        merge = StubProvider(lambda _p: groups(("policy_enforcement", [0, 1])))
        reject = StubProvider(
            lambda _p: {
                "assignments": [
                    {"claim_id": a.claim_id, "concept": "data_flow"},
                    {"claim_id": b.claim_id, "concept": "dialogue_rails"},
                ],
                "reason": "different questions",
            }
        )
        graph, report = Aligner(
            provider=merge, splitter=SplitterAgent(reject), split_all=False
        ).align(sets)

        assert report.split_reviewed == 1
        assert report.split_rejected == 1
        assert len(graph.clusters) == 2, "the gate undid the merge"

    def test_keys_that_already_span_papers_are_not_offered(self):
        """An exact-name match has found its match; re-grouping could only
        take it away from one it earned outright."""
        a = method_claim("p1", "optimizer_choice", "adam")
        b = method_claim("p2", "optimizer_choice", "lamb")
        c = method_claim("p3", "gradient_clipping", "1.0")
        sets = [ClaimSet(paper_id=c.paper_id, claims=[c]) for c in (a, b, c)]

        provider = StubProvider(lambda _p: groups())
        Aligner(provider=provider, splitter=None).align(sets)

        assert provider.call_count == 0, "one unmatched key left; nothing to pair it with"

    def test_unit_mismatch_still_blocks_a_proposed_merge(self):
        """A model calling two things one concept does not make warmup in
        steps and warmup in epochs comparable."""
        from tests.conftest import make_claim, make_doc

        a = make_claim(make_doc("p1"), canonical_name="warmup", value=10000.0)
        b = make_claim(make_doc("p2"), canonical_name="warmup_period", value=6.0)
        a.payload["unit"] = "steps"
        b.payload["unit"] = "epochs"
        sets = [ClaimSet(paper_id="p1", claims=[a]), ClaimSet(paper_id="p2", claims=[b])]

        provider = StubProvider(lambda _p: groups(("warmup", [0, 1])))
        graph, report = Aligner(provider=provider, splitter=None).align(sets)

        assert report.merged_by_semantic == 0
        assert len(graph.clusters) == 2
        assert any("incompatible unit" in n for n in report.notes)
