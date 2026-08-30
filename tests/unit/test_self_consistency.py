"""Self-consistency, the confidence threshold, and batched entailment.

All three were configured and read by nothing. A knob that implies a capability
is worse than an absent one, because someone sets it and believes the behaviour
is on.
"""

from __future__ import annotations

import pytest

from papersynth.core.config import Settings
from papersynth.core.models import ClaimSet
from papersynth.extract.extractors.hyperparameter import HyperparameterExtractor
from papersynth.llm.stub import StubProvider
from papersynth.verify import Verifier
from tests.conftest import make_claim, make_doc

LR = {
    "canonical_name": "learning_rate",
    "value": 0.0001,
    "value_type": "float",
    "condition": "base model",
    "stated_explicitly": True,
    "quote": "learning rate of 0.0001",
}
DROPOUT = {
    "canonical_name": "dropout",
    "value": 0.1,
    "value_type": "float",
    "condition": "base model",
    "stated_explicitly": True,
    "quote": "dropout rate of 0.1",
}


def multi_section_doc():
    """Several sections, so rotation actually reorders something."""
    from papersynth.core.document import Paragraph, Section, StructuredDocument

    return StructuredDocument(
        paper_id="sc.001",
        title="Self consistency",
        ingest_method="latex",
        # Stated rather than defaulted: a LaTeX ingest yields native math, and
        # the base confidence of 0.95 follows from that.
        math_fidelity="latex_native",
        sha256="a" * 64,
        sections=[
            Section(
                index=0,
                title="3 Training Setup",
                paragraphs=[
                    Paragraph(
                        index=0, text="We train with a learning rate of 0.0001 for the base model."
                    )
                ],
            ),
            Section(
                index=1,
                title="4 Implementation Details",
                paragraphs=[
                    Paragraph(index=0, text="We apply a dropout rate of 0.1 across all sub-layers.")
                ],
            ),
        ],
    )


class TestSelfConsistency:
    def test_n_of_one_makes_one_pass(self):
        """The default must cost exactly what it always did."""
        provider = StubProvider([[LR]])
        HyperparameterExtractor(provider, self_consistency_n=1).extract(multi_section_doc())
        assert provider.call_count == 1

    def test_n_of_three_makes_three_passes(self):
        provider = StubProvider([[LR], [LR], [LR]])
        HyperparameterExtractor(provider, self_consistency_n=3).extract(multi_section_doc())
        assert provider.call_count == 3

    def test_unanimous_agreement_keeps_full_confidence(self):
        provider = StubProvider([[LR], [LR], [LR]])
        result = HyperparameterExtractor(provider, self_consistency_n=3).extract(
            multi_section_doc()
        )

        claim = result.claims[0]
        assert claim.verification.self_consistency == "3/3"
        assert claim.confidence == pytest.approx(0.95)

    def test_disagreement_scales_confidence(self):
        """A fact found once in three is reported, and reported as such."""
        provider = StubProvider([[LR], [], []])
        result = HyperparameterExtractor(provider, self_consistency_n=3).extract(
            multi_section_doc()
        )

        claim = result.claims[0]
        assert claim.verification.self_consistency == "1/3"
        assert claim.confidence == pytest.approx(0.95 / 3, abs=1e-3)

    def test_a_claim_seen_once_is_not_discarded(self):
        """It may still be correct, and dropping it trades a visible
        low-confidence entry for an invisible absence."""
        provider = StubProvider([[LR], [], []])
        result = HyperparameterExtractor(provider, self_consistency_n=3).extract(
            multi_section_doc()
        )
        assert len(result.claims) == 1

    def test_disagreement_is_explained_in_the_notes(self):
        provider = StubProvider([[LR], [], []])
        claim = (
            HyperparameterExtractor(provider, self_consistency_n=3)
            .extract(multi_section_doc())
            .claims[0]
        )

        assert any("1 of 3" in n for n in claim.verification.notes)

    def test_prompts_differ_between_passes(self):
        """Identical prompts at temperature 0 return identical cached answers,
        so a second opinion needs a different question."""
        provider = StubProvider([[LR], [LR], [LR]])
        HyperparameterExtractor(provider, self_consistency_n=3).extract(multi_section_doc())

        assert len(set(provider.prompts)) > 1, "rotation must change the prompt"

    def test_the_same_fact_from_a_different_span_still_agrees(self):
        """claim_id folds in the span; scoring on it would penalise exactly the
        claims that were found twice."""
        elsewhere = {**LR, "quote": "learning rate of 0.0001 for the base model"}
        provider = StubProvider([[LR], [elsewhere]])
        result = HyperparameterExtractor(provider, self_consistency_n=2).extract(
            multi_section_doc()
        )

        assert len(result.claims) == 1
        assert result.claims[0].verification.self_consistency == "2/2"

    def test_different_values_do_not_agree(self):
        provider = StubProvider([[LR], [{**LR, "value": 0.0003}]])
        result = HyperparameterExtractor(provider, self_consistency_n=2).extract(
            multi_section_doc()
        )

        assert len(result.claims) == 2
        assert all(c.verification.self_consistency == "1/2" for c in result.claims)


class TestConfidenceThreshold:
    def verify(self, claim, doc, threshold=0.6):
        verifier = Verifier(entailment=False, settings=Settings(confidence_threshold=threshold))
        result, report = verifier.verify(ClaimSet(paper_id=doc.paper_id, claims=[claim]), doc)
        return result.claims[0], report

    def test_a_low_confidence_claim_is_not_promoted(self, doc):
        """Section 8.3.4: it stays `extracted`, which excludes it from
        alignment and from auto-resolution."""
        claim = make_claim(doc, start=0, end=80)
        claim.confidence = 0.3

        verified, report = self.verify(claim, doc)

        assert verified.status == "extracted"
        assert report.low_confidence == 1

    def test_a_low_confidence_claim_is_not_rejected_either(self):
        """The checks passed; it may well be right. Rejecting would discard a
        plausible claim, promoting would let a coin flip settle a conflict."""
        doc = make_doc()
        claim = make_claim(doc, start=0, end=80)
        claim.confidence = 0.3

        verified, report = self.verify(claim, doc)

        assert verified.status != "rejected"
        assert report.rejected == 0

    def test_the_reason_is_recorded(self):
        doc = make_doc()
        claim = make_claim(doc, start=0, end=80)
        claim.confidence = 0.3

        verified, _ = self.verify(claim, doc)
        assert any("below the" in n for n in verified.verification.notes)

    def test_a_confident_claim_is_promoted(self):
        doc = make_doc()
        claim = make_claim(doc, start=0, end=80)
        claim.confidence = 0.95

        verified, report = self.verify(claim, doc)

        assert verified.status == "verified"
        assert report.low_confidence == 0

    def test_an_unpromoted_claim_never_reaches_a_cluster(self):
        """Which is the point: it cannot drive an auto-resolution."""
        from papersynth.align import Aligner

        doc = make_doc()
        claim = make_claim(doc, start=0, end=80)
        claim.confidence = 0.3
        verified, _ = self.verify(claim, doc)

        graph, _ = Aligner().align([ClaimSet(paper_id=doc.paper_id, claims=[verified])])
        assert graph.clusters == []


class TestBatchedEntailment:
    def test_many_claims_cost_one_call(self):
        """One call per claim is the largest single source of calls in a run."""
        doc = make_doc()
        claims = [
            make_claim(doc, start=0, end=80, canonical_name=n, value=0.1)
            for n in ("dropout", "learning_rate", "weight_decay")
        ]

        provider = StubProvider(
            [
                {
                    "verdicts": [
                        {"claim_id": c.claim_id, "entailed": True, "reason": "stated"}
                        for c in claims
                    ]
                }
            ]
        )
        verifier = Verifier(
            provider=provider, entailment=True, settings=Settings(verify_batch_size=10)
        )
        result, _ = verifier.verify(ClaimSet(paper_id=doc.paper_id, claims=claims), doc)

        assert provider.call_count == 1
        assert all(c.status == "verified" for c in result.claims)

    def test_the_batch_size_is_respected(self):
        doc = make_doc()
        claims = [
            make_claim(doc, start=0, end=80, canonical_name=n, value=0.1)
            for n in ("a_one", "b_two", "c_three", "d_four", "e_five")
        ]

        def respond(prompt):
            import re

            return {
                "verdicts": [
                    {"claim_id": cid, "entailed": True, "reason": "ok"}
                    for cid in re.findall(r"--- CLAIM (clm_[0-9a-f]{6}) ---", prompt)
                ]
            }

        provider = StubProvider(respond)
        Verifier(provider=provider, entailment=True, settings=Settings(verify_batch_size=2)).verify(
            ClaimSet(paper_id=doc.paper_id, claims=claims), doc
        )

        assert provider.call_count == 3, "5 claims at batch size 2"

    def test_a_claim_the_judge_skipped_fails_closed(self):
        """A truncated response must not silently verify what it ran out of
        room to consider."""
        doc = make_doc()
        claims = [
            make_claim(doc, start=0, end=80, canonical_name=n, value=0.1)
            for n in ("dropout", "learning_rate")
        ]

        provider = StubProvider(
            [{"verdicts": [{"claim_id": claims[0].claim_id, "entailed": True, "reason": "ok"}]}]
        )
        result, report = Verifier(
            provider=provider, entailment=True, settings=Settings(verify_batch_size=10)
        ).verify(ClaimSet(paper_id=doc.paper_id, claims=claims), doc)

        statuses = {c.payload["canonical_name"]: c.status for c in result.claims}
        assert statuses["dropout"] == "verified"
        assert statuses["learning_rate"] == "rejected"
        assert report.rejection_reasons.get("citation_trace") == 1

    def test_a_rejected_verdict_rejects_that_claim_only(self):
        doc = make_doc()
        claims = [
            make_claim(doc, start=0, end=80, canonical_name=n, value=0.1)
            for n in ("dropout", "learning_rate")
        ]

        provider = StubProvider(
            [
                {
                    "verdicts": [
                        {"claim_id": claims[0].claim_id, "entailed": True, "reason": "ok"},
                        {
                            "claim_id": claims[1].claim_id,
                            "entailed": False,
                            "reason": "different value",
                        },
                    ]
                }
            ]
        )
        result, _ = Verifier(
            provider=provider, entailment=True, settings=Settings(verify_batch_size=10)
        ).verify(ClaimSet(paper_id=doc.paper_id, claims=claims), doc)

        statuses = {c.payload["canonical_name"]: c.status for c in result.claims}
        assert statuses == {"dropout": "verified", "learning_rate": "rejected"}


class TestParallelPapers:
    """Concurrency must not change the answer, only the wall clock."""

    def documents(self):
        from pathlib import Path

        from papersynth.ingest.latex import LatexIngestor

        fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "three_paper"
        return [
            LatexIngestor().ingest(str(fixtures / f"paper_{p}.tex"), paper_id=f"paper_{p}")
            for p in "abc"
        ]

    def responder(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from e2e.test_mva_acceptance import EXTRACTIONS

        def respond(prompt: str):
            for paper_id, items in EXTRACTIONS.items():
                marker = {
                    "paper_a": "batch size of 128",
                    "paper_b": "batch size of 256",
                    "paper_c": "larger stack is less stable",
                }[paper_id]
                if marker in prompt:
                    return items
            return []

        return respond

    def run(self, workers, tmp_path, name):
        from papersynth.core.run import Pipeline, Workspace

        return Pipeline(
            StubProvider(self.responder()),
            settings=Settings(max_parallel_papers=workers),
            workspace=Workspace(tmp_path, name),
            extractors=["hyperparameter"],
            entailment=False,
        ).run(self.documents(), objective="Parallel test.", run_id=name)

    def test_parallel_and_sequential_produce_the_same_spec(self, tmp_path):
        """Merging in completion order would make cluster and contradiction
        IDs depend on which paper finished first (NFR-02)."""
        import yaml

        sequential = self.run(1, tmp_path, "seq").spec
        parallel = self.run(3, tmp_path, "par").spec

        for spec in (sequential, parallel):
            spec.pop("generated_at")
            spec.pop("run_id")

        assert yaml.safe_dump(sequential, sort_keys=True) == yaml.safe_dump(
            parallel, sort_keys=True
        )

    def test_papers_are_merged_in_input_order(self, tmp_path):
        result = self.run(3, tmp_path, "order")
        assert [p["paper_id"] for p in result.spec["source_papers"]] == [
            "paper_a",
            "paper_b",
            "paper_c",
        ]

    def test_every_paper_still_contributes(self, tmp_path):
        result = self.run(3, tmp_path, "contrib")
        summary = result.spec["verification_summary"]
        assert summary["papers_contributing"] == 3

    def test_artifacts_are_written_for_every_paper(self, tmp_path):
        self.run(3, tmp_path, "artifacts")
        root = tmp_path / "artifacts"
        for paper in ("paper_a", "paper_b", "paper_c"):
            assert (root / "02_verified" / f"{paper}.yaml").exists()
        assert (root / "02_verified" / "verification_report.json").exists()

    def test_the_default_is_sequential(self):
        """Three concurrent papers put roughly 9,000 tokens in flight against
        an 8,000 per-minute cap, so the default must not be concurrent."""
        assert Settings().max_parallel_papers == 1
