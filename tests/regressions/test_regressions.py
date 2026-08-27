"""Permanent fixtures for bugs that actually shipped (section 14.5).

Each test here corresponds to a defect found during development. They are kept
separate from the feature suites because their value is specifically that they
fail if the old behaviour returns - the e2e suite covers some of the same
ground incidentally, but incidental coverage is not a guarantee.
"""

from __future__ import annotations

import pytest

from papersynth.core.models import (
    Claim,
    Contradiction,
    Position,
    Provenance,
    ReconciliationResult,
    Support,
)
from papersynth.reconcile import Policy, PolicyEngine
from papersynth.synth import SpecBuilder
from tests.conftest import make_doc

POLICY = Policy.load("config/reconcile_policy.yaml")


def hyperparameter_claim(
    paper_id: str, name: str, value: object, condition: str | None, claim_id: str
) -> Claim:
    """A verified claim, built directly so the fixture states exactly one thing."""
    return Claim(
        claim_id=claim_id,
        paper_id=paper_id,
        type="hyperparameter",
        status="verified",
        provenance=Provenance(
            paper_id=paper_id,
            span_id=f"{paper_id}#s1.p0.0",
            section="Training Setup",
            page=1,
            char_start=0,
            char_end=40,
            quote_hash="sha256:" + "a" * 64,
            extraction_method="llm",
            extractor_version="hyperparameter@1.0.0",
            confidence=0.95,
        ),
        payload={
            "canonical_name": name,
            "paper_symbol": None,
            "value": value,
            "value_type": "float" if isinstance(value, float) else "int",
            "unit": None,
            "applies_to": "global",
            "condition": condition,
            "stated_explicitly": True,
        },
        confidence=0.95,
    )


def build_spec(claims: list[Claim], contradictions: list[Contradiction]) -> dict:
    docs = [make_doc(pid) for pid in sorted({c.paper_id for c in claims})]
    builder = SpecBuilder(
        run_id="run_regression",
        objective="Regression fixture.",
        documents=docs,
        claims={c.claim_id: c for c in claims},
    )
    return builder.build(
        contradictions=contradictions,
        reconciliation=ReconciliationResult(policy_version="1.0.0", resolutions=[]),
    )


def emitted(spec: dict, name: str) -> list[dict]:
    return [
        hp
        for component in spec["components"]
        for hp in component["hyperparameters"]
        if hp["canonical_name"] == name
    ]


class TestR001PrimacyBias:
    """Primacy was inferred as "earliest paper in the cluster", which silently
    auto-resolved genuine value conflicts in favour of whichever paper was
    older. A hyperparameter is not a concept anyone introduces, so age says
    nothing about whose value is authoritative."""

    def test_the_pipeline_does_not_assign_primacy(self):
        """The heuristic is gone and must not come back without a real signal."""
        from papersynth.core.run import _primary_sources

        docs = [make_doc("paper_a"), make_doc("paper_b")]

        class _Graph:
            clusters = [
                type(
                    "C",
                    (),
                    {"cluster_id": "cnc_hype_learning_rate", "papers": ["paper_a", "paper_b"]},
                )()
            ]

        assert _primary_sources(_Graph(), docs) == {}

    def test_an_age_difference_alone_does_not_resolve_a_value_conflict(self):
        """The end-to-end symptom: without primacy set, this must escalate."""
        contradiction = Contradiction(
            contradiction_id="ctr_r001",
            cluster_id="cnc_hype_learning_rate",
            type="VALUE_CONFLICT",
            severity="MATERIAL",
            description="two papers disagree",
            positions=[
                Position(
                    claim_id="clm_aaaaaa",
                    paper_id="paper_a",
                    position="0.0001",
                    support=Support(specificity=0.9, year=2017, is_primary=False),
                ),
                Position(
                    claim_id="clm_bbbbbb",
                    paper_id="paper_b",
                    position="0.0003",
                    support=Support(specificity=0.9, year=2021, is_primary=False),
                ),
            ],
            detected_by="value_conflict_detector@1.0.0",
        )

        resolution = PolicyEngine(POLICY).resolve_one(contradiction)

        assert resolution.is_open, "an age gap alone must not decide a hyperparameter"
        assert resolution.selected_claim_id is None


class TestR002DisputedValueLeak:
    """Disputed values were excluded from components by claim identity. A
    contradiction lists one representative position per distinct value, so a
    third paper repeating one of those values was not itself a listed position
    and leaked into the spec as settled fact - while that same value sat
    unresolved in open_conflicts."""

    @pytest.fixture
    def claims(self):
        return [
            hyperparameter_claim("paper_a", "learning_rate", 0.0001, "base model", "clm_aaaaaa"),
            hyperparameter_claim("paper_b", "learning_rate", 0.0003, "base model", "clm_bbbbbb"),
            # The duplicate. Agrees with paper_a, and is NOT a listed position.
            hyperparameter_claim("paper_c", "learning_rate", 0.0001, "base model", "clm_cccccc"),
        ]

    @pytest.fixture
    def contradiction(self):
        return Contradiction(
            contradiction_id="ctr_r002",
            cluster_id="cnc_hype_learning_rate",
            type="VALUE_CONFLICT",
            severity="MATERIAL",
            description="two papers disagree",
            positions=[
                Position(claim_id="clm_aaaaaa", paper_id="paper_a", position="0.0001"),
                Position(claim_id="clm_bbbbbb", paper_id="paper_b", position="0.0003"),
            ],
            detected_by="value_conflict_detector@1.0.0",
        )

    def test_a_non_representative_duplicate_is_not_emitted(self, claims, contradiction):
        spec = build_spec(claims, [contradiction])
        assert emitted(spec, "learning_rate") == [], (
            "a value under active dispute must not appear as a settled fact"
        )

    def test_the_conflict_is_still_reported(self, claims, contradiction):
        """The value is withheld from components because it is in dispute -
        that only holds together if the dispute is actually surfaced."""
        spec = build_spec(claims, [contradiction])
        assert len(spec["open_conflicts"]) == 1
        assert spec["open_conflicts"][0]["contradiction_id"] == "ctr_r002"

    def test_a_differently_scoped_value_is_still_emitted(self, claims, contradiction):
        """The exclusion is scoped to the disputed condition, not the whole
        parameter - over-excluding would lose undisputed facts."""
        claims = [
            *claims,
            hyperparameter_claim("paper_c", "learning_rate", 0.00005, "large model", "clm_dddddd"),
        ]
        spec = build_spec(claims, [contradiction])

        values = [hp["value"] for hp in emitted(spec, "learning_rate")]
        assert values == [0.00005]


class TestR003DuplicatedAgreement:
    """Agreeing claims were emitted once per paper, so three papers stating
    dropout 0.1 produced three identical entries. That is one fact corroborated
    three times, not three facts, and the duplication buried the useful signal
    in noise a coding agent then has to disambiguate."""

    @pytest.fixture
    def claims(self):
        return [
            hyperparameter_claim("paper_a", "dropout", 0.1, "base model", "clm_aaaaaa"),
            hyperparameter_claim("paper_b", "dropout", 0.1, "base model", "clm_bbbbbb"),
            hyperparameter_claim("paper_c", "dropout", 0.1, "base model", "clm_cccccc"),
        ]

    def test_agreeing_claims_collapse_to_one_entry(self, claims):
        spec = build_spec(claims, [])
        assert len(emitted(spec, "dropout")) == 1

    def test_the_corroboration_is_preserved_as_provenance(self, claims):
        """Collapsing must not lose which papers agreed - that is the signal."""
        spec = build_spec(claims, [])
        assert emitted(spec, "dropout")[0]["provenance_refs"] == [
            "clm_aaaaaa",
            "clm_bbbbbb",
            "clm_cccccc",
        ]

    def test_the_same_value_written_differently_still_collapses(self):
        """0.0001 and 1e-4 are one fact; grouping on the literal would split them."""
        claims = [
            hyperparameter_claim("paper_a", "learning_rate", 0.0001, None, "clm_aaaaaa"),
            hyperparameter_claim("paper_b", "learning_rate", 1e-4, None, "clm_bbbbbb"),
        ]
        spec = build_spec(claims, [])

        entries = emitted(spec, "learning_rate")
        assert len(entries) == 1
        assert entries[0]["provenance_refs"] == ["clm_aaaaaa", "clm_bbbbbb"]

    def test_genuinely_different_values_stay_separate(self):
        """Collapsing must key on the value, not just the name - merging these
        would hide a real difference behind one arbitrary number."""
        claims = [
            hyperparameter_claim("paper_a", "batch_size", 128, "base model", "clm_aaaaaa"),
            hyperparameter_claim("paper_b", "batch_size", 256, "our runs", "clm_bbbbbb"),
        ]
        spec = build_spec(claims, [])

        assert sorted(hp["value"] for hp in emitted(spec, "batch_size")) == [128, 256]

    def test_different_conditions_stay_separate(self):
        claims = [
            hyperparameter_claim("paper_a", "learning_rate", 0.0001, "base model", "clm_aaaaaa"),
            hyperparameter_claim("paper_b", "learning_rate", 0.0001, "large model", "clm_bbbbbb"),
        ]
        spec = build_spec(claims, [])

        conditions = sorted(hp["condition"] for hp in emitted(spec, "learning_rate"))
        assert conditions == ["base model", "large model"]
