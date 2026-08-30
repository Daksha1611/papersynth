"""The MVA acceptance criterion (section 17.1).

    Given three papers with one hand-planted conflicting hyperparameter, the
    pipeline emits a spec that (a) contains that conflict in open_conflicts
    with both positions and provenance, and (b) contains no fabricated
    conflicts.

Nothing else needs to work for the concept to be validated, so this file is the
one that says whether the core idea holds.

The planted conflict: papers A and B give different learning rates for the SAME
condition ("base model"). Paper C agrees with A, and separately gives a
different rate for the "large model" - which must NOT be reported, because a
differently scoped value is not a disagreement (ER-04).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from papersynth.core.run import Pipeline, Workspace
from papersynth.ingest.latex import LatexIngestor
from papersynth.llm.stub import StubProvider

FIXTURES = Path(__file__).parent.parent / "fixtures" / "three_paper"

# What a faithful extractor returns for each paper. Scripted rather than live
# so the suite is deterministic, offline, and free (section 14.4).
EXTRACTIONS = {
    "paper_a": [
        {
            "canonical_name": "learning_rate",
            "value": 0.0001,
            "value_type": "float",
            "condition": "base model",
            "stated_explicitly": True,
            "quote": "learning rate of 0.0001 for the base\nmodel",
        },
        {
            "canonical_name": "dropout",
            "value": 0.1,
            "value_type": "float",
            "condition": "base model",
            "stated_explicitly": True,
            "quote": "dropout rate of 0.1",
        },
        {
            "canonical_name": "batch_size",
            "value": 128,
            "value_type": "int",
            "condition": "base model",
            "stated_explicitly": True,
            "quote": "batch size of 128 sequences",
        },
        {
            "canonical_name": "num_layers",
            "value": 6,
            "value_type": "int",
            "condition": None,
            "stated_explicitly": True,
            "quote": "We use 6 layers throughout",
        },
    ],
    "paper_b": [
        {
            "canonical_name": "learning_rate",
            "value": 0.0003,
            "value_type": "float",
            "condition": "base model",
            "stated_explicitly": True,
            "quote": "learning rate of 0.0003 for the base model",
        },
        {
            "canonical_name": "dropout",
            "value": 0.1,
            "value_type": "float",
            "condition": "base model",
            "stated_explicitly": True,
            "quote": "dropout rate of 0.1 to remain\ncomparable",
        },
        {
            "canonical_name": "batch_size",
            "value": 256,
            "value_type": "int",
            "condition": "our runs",
            "stated_explicitly": True,
            "quote": "batch size of 256 sequences",
        },
        {
            "canonical_name": "num_layers",
            "value": 6,
            "value_type": "int",
            "condition": None,
            "stated_explicitly": True,
            "quote": "using 6 layers",
        },
    ],
    "paper_c": [
        {
            "canonical_name": "learning_rate",
            "value": 0.0001,
            "value_type": "float",
            "condition": "base model",
            "stated_explicitly": True,
            "quote": "For the base model we use a learning rate of\n0.0001",
        },
        {
            "canonical_name": "learning_rate",
            "value": 0.00005,
            "value_type": "float",
            "condition": "large model",
            "stated_explicitly": True,
            "quote": "For the large model we use a learning rate of 0.00005",
        },
        {
            "canonical_name": "dropout",
            "value": 0.1,
            "value_type": "float",
            "condition": "base model",
            "stated_explicitly": True,
            "quote": "dropout rate of 0.1 for the base model",
        },
    ],
}

#: Every verification entailment call answers yes; the extractions are faithful.
ENTAILED = {"entailed": True, "reason": "the passage states this directly"}


def scripted_provider() -> StubProvider:
    """Serves extraction responses by paper, and entailment for everything else."""

    def respond(prompt: str):
        if "Your job" in prompt and "SAME configurable quantity" in prompt:
            # Split gate: these fixtures describe one shared encoder, so every
            # aligned claim really is the same quantity.
            return {
                "assignments": [
                    {"claim_id": cid, "concept": "same"}
                    for cid in re.findall(r"claim_id: (clm_[0-9a-f]{6})", prompt)
                ],
                "reason": "one quantity, values genuinely disagree",
            }
        if "Judge each claim below" in prompt:
            # Batched entailment: every claim in the batch is supported.
            return {
                "verdicts": [
                    {"claim_id": cid, "entailed": True, "reason": "stated directly"}
                    for cid in re.findall(r"--- CLAIM (clm_[0-9a-f]{6}) ---", prompt)
                ]
            }
        if "Does the passage state" in prompt:
            return ENTAILED
        for paper_id, items in EXTRACTIONS.items():
            marker = {
                "paper_a": "Baseline Sequence Encoder",
                "paper_b": "Improved Sequence Encoder",
                "paper_c": "Scaling the Sequence Encoder",
            }[paper_id]
            if marker in prompt or _body_marker(paper_id) in prompt:
                return items
        return []

    return StubProvider(respond)


def _body_marker(paper_id: str) -> str:
    return {
        "paper_a": "batch size of 128",
        "paper_b": "batch size of 256",
        "paper_c": "larger stack is less stable",
    }[paper_id]


@pytest.fixture(scope="module")
def documents():
    return [
        LatexIngestor().ingest(str(FIXTURES / f"paper_{p}.tex"), paper_id=f"paper_{p}")
        for p in "abc"
    ]


@pytest.fixture(scope="module")
def run(documents, tmp_path_factory):
    workspace = Workspace(tmp_path_factory.mktemp("runs"), "run_mva_test")
    pipeline = Pipeline(
        scripted_provider(),
        workspace=workspace,
        extractors=["hyperparameter"],
        entailment=True,
    )
    result = pipeline.run(
        documents,
        objective="Implement the baseline sequence encoder shared by these three papers.",
    )
    return result, workspace


class TestAcceptanceCriterion:
    def test_the_planted_conflict_is_detected(self, run):
        result, _ = run
        learning_rate_conflicts = [
            c
            for c in result.contradictions
            if "learning_rate" in c.cluster_id and c.type == "VALUE_CONFLICT"
        ]
        assert len(learning_rate_conflicts) == 1, (
            "expected exactly the planted conflict, got: "
            f"{[c.description for c in result.contradictions]}"
        )

    def test_the_conflict_carries_both_positions(self, run):
        result, _ = run
        conflict = next(c for c in result.contradictions if "learning_rate" in c.cluster_id)
        values = sorted(p.position for p in conflict.positions)

        assert values == ["0.0001", "0.0003"]
        assert {p.paper_id for p in conflict.positions} == {"paper_a", "paper_b"}

    def test_each_position_traces_back_to_a_real_span(self, run):
        """Criterion (a): with provenance, not just values."""
        result, _ = run
        conflict = next(c for c in result.contradictions if "learning_rate" in c.cluster_id)
        documents = {d.paper_id: d for d in result.documents}

        for position in conflict.positions:
            claim = result.claims[position.claim_id]
            span = documents[claim.paper_id].resolve_span(
                claim.provenance.span_id, char_end=claim.provenance.char_end
            )
            assert span is not None, f"{position.claim_id} has unresolvable provenance"
            assert str(claim.payload["value"]) in span.text.replace("\n", " ")

    def test_the_conflict_appears_in_the_emitted_spec(self, run):
        """Criterion (a): in open_conflicts, where the implementer will see it."""
        result, _ = run
        entry = next((c for c in result.open_conflicts if c["type"] == "VALUE_CONFLICT"), None)
        assert entry is not None, f"open_conflicts was {result.open_conflicts}"
        assert len(entry["positions"]) == 2
        assert all(p["provenance"].get("span_id") for p in entry["positions"])
        assert entry["guidance"]

    def test_no_fabricated_conflicts(self, run):
        """Criterion (b). The papers agree on dropout and num_layers, and
        differ on batch_size and the large-model rate only under different
        conditions. None of those may be reported."""
        result, _ = run
        reported = {c.cluster_id for c in result.contradictions}

        assert not any("dropout" in cid for cid in reported), "dropout is unanimous"
        assert not any("num_layers" in cid for cid in reported), "num_layers is unanimous"
        assert not any("batch_size" in cid for cid in reported), (
            "batch sizes are stated under different conditions (ER-04)"
        )
        assert len(result.contradictions) == 1, (
            f"exactly one conflict expected, got {[c.description for c in result.contradictions]}"
        )

    def test_a_differently_scoped_value_is_not_a_conflict(self, run):
        """Paper C's large-model rate differs from every base-model rate, and
        must not be reported - it is a separate scoped fact (ER-04)."""
        result, _ = run
        conflict = next(c for c in result.contradictions if "learning_rate" in c.cluster_id)

        assert "0.00005" not in [p.position for p in conflict.positions]
        assert "large" not in conflict.description


class TestEmittedSpec:
    def test_the_spec_validates_against_its_schema(self, run):
        from papersynth.schemas import validate

        result, _ = run
        assert validate(result.spec, "spec.schema.json") == []

    def test_provenance_is_fully_closed(self, run):
        """NFR-01, the hard gate: 100% or the spec does not emit."""
        result, _ = run
        assert result.provenance_completeness == 1.0

    def test_no_blocking_conflicts_remain(self, run):
        result, _ = run
        assert result.blocking == []
        assert result.status == "ready"

    def test_the_disputed_value_is_not_emitted_as_settled(self, run):
        """It belongs in open_conflicts, not in a component as fact."""
        result, _ = run
        emitted = [
            hp["value"]
            for component in result.spec["components"]
            for hp in component["hyperparameters"]
            if hp["canonical_name"] == "learning_rate"
        ]
        assert 0.0003 not in emitted
        assert 0.0001 not in emitted

    def test_undisputed_values_are_emitted(self, run):
        result, _ = run
        names = {
            hp["canonical_name"]
            for component in result.spec["components"]
            for hp in component["hyperparameters"]
        }
        assert "dropout" in names
        assert "num_layers" in names

    def test_every_source_paper_is_recorded(self, run):
        result, _ = run
        assert {p["paper_id"] for p in result.spec["source_papers"]} == {
            "paper_a",
            "paper_b",
            "paper_c",
        }
        assert all(p["sha256"] for p in result.spec["source_papers"])

    def test_the_spec_starts_as_a_draft(self, run):
        """Human approval is a required stage, not an optional one (DD-01)."""
        result, _ = run
        assert result.spec["review"]["status"] == "draft"
        assert result.spec["review"]["approved_at"] is None


class TestArtifacts:
    def test_every_stage_wrote_its_artifact(self, run):
        _, workspace = run
        for name in (
            "manifest.yaml",
            "03_concept_graph.json",
            "04_contradictions.yaml",
            "05_reconciliation.yaml",
            "06_gaps.yaml",
            "implementation_spec.yaml",
            "SPEC_REVIEW.md",
        ):
            assert (workspace.root / name).exists(), f"{name} missing"

    def test_per_paper_artifacts_exist(self, run):
        _, workspace = run
        for paper in ("paper_a", "paper_b", "paper_c"):
            assert (workspace.root / "00_documents" / f"{paper}.json").exists()
            assert (workspace.root / "01_claims" / f"{paper}.yaml").exists()
            assert (workspace.root / "02_verified" / f"{paper}.yaml").exists()

    def test_the_manifest_records_what_could_change_the_output(self, run):
        """A spec differing from a prior run must be attributable to config
        rather than silent model drift (R-06)."""
        _, workspace = run
        manifest = yaml.safe_load((workspace.root / "manifest.yaml").read_text())

        assert manifest["config"]["temperature"] == 0.0
        assert manifest["config"]["align_threshold"]
        assert manifest["extractors"] == ["hyperparameter"]
        assert manifest["spec_version"]

    def test_the_review_document_names_the_conflict(self, run):
        _, workspace = run
        review = (workspace.root / "SPEC_REVIEW.md").read_text()

        assert "Open conflicts (1)" in review
        assert "learning_rate" in review
        assert "paper_a" in review and "paper_b" in review

    def test_the_emitted_spec_is_readable_yaml(self, run):
        _, workspace = run
        spec = yaml.safe_load((workspace.root / "implementation_spec.yaml").read_text())
        assert spec["spec_version"]
        assert spec["objective"]


class TestDeterminism:
    def test_two_identical_runs_produce_identical_specs(self, documents, tmp_path):
        """NFR-02. A spec that differs run to run cannot be diffed or trusted."""
        specs = []
        for i in range(2):
            pipeline = Pipeline(
                scripted_provider(),
                workspace=Workspace(tmp_path, f"run_{i}"),
                extractors=["hyperparameter"],
                entailment=True,
            )
            result = pipeline.run(documents, objective="Same objective.", run_id="run_fixed")
            spec = dict(result.spec)
            spec.pop("generated_at")
            specs.append(yaml.safe_dump(spec, sort_keys=True))

        assert specs[0] == specs[1]
