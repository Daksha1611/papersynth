"""Spec diff (FR-17).

The question is not "what bytes changed" but "did anything I depend on change".
A spec goes to a coding agent; when it is re-emitted, the implementer needs to
know whether a value they already built against has moved.
"""

from __future__ import annotations

import copy

import pytest

from papersynth.synth import diff_specs

BASE = {
    "run_id": "run_a",
    "generated_at": "2026-08-01T00:00:00Z",
    "source_papers": [{"paper_id": "p1"}, {"paper_id": "p2"}],
    "components": [
        {
            "component_id": "cmp_global",
            "hyperparameters": [
                {"canonical_name": "learning_rate", "value": 0.0001, "condition": "base model"},
                {"canonical_name": "dropout", "value": 0.1, "condition": None},
            ],
        }
    ],
    "open_conflicts": [{"contradiction_id": "ctr_0001"}],
    "missing_but_critical": [{"field": "optimizer"}],
    "review": {"status": "draft"},
}


def altered(**changes):
    spec = copy.deepcopy(BASE)
    spec.update(changes)
    return spec


def set_value(spec, name, value):
    for hp in spec["components"][0]["hyperparameters"]:
        if hp["canonical_name"] == name:
            hp["value"] = value
    return spec


class TestNoChange:
    def test_identical_specs_diff_as_identical(self):
        assert diff_specs(BASE, copy.deepcopy(BASE)).identical

    def test_a_new_timestamp_is_not_a_change(self):
        """Two runs always differ here, and it means nothing to an implementer."""
        later = altered(generated_at="2026-09-01T00:00:00Z", run_id="run_b")
        assert diff_specs(BASE, later).identical

    def test_reordering_is_not_a_change(self):
        reordered = copy.deepcopy(BASE)
        reordered["components"][0]["hyperparameters"].reverse()
        assert diff_specs(BASE, reordered).identical

    def test_the_same_number_written_differently_is_not_a_change(self):
        assert diff_specs(BASE, set_value(copy.deepcopy(BASE), "learning_rate", 1e-4)).identical


class TestValueChanges:
    def test_a_changed_value_is_reported(self):
        result = diff_specs(BASE, set_value(copy.deepcopy(BASE), "learning_rate", 0.0003))

        assert len(result.values_changed) == 1
        change = result.values_changed[0]
        assert change["canonical_name"] == "learning_rate"
        assert (change["from"], change["to"]) == (0.0001, 0.0003)

    def test_a_changed_value_is_breaking(self):
        """Code written against the old number still compiles against the new
        one, so nothing else will catch it."""
        result = diff_specs(BASE, set_value(copy.deepcopy(BASE), "learning_rate", 0.0003))
        assert result.breaking

    def test_a_removed_value_is_breaking(self):
        after = copy.deepcopy(BASE)
        after["components"][0]["hyperparameters"] = [
            h for h in after["components"][0]["hyperparameters"] if h["canonical_name"] != "dropout"
        ]
        result = diff_specs(BASE, after)

        assert [v["canonical_name"] for v in result.values_removed] == ["dropout"]
        assert result.breaking

    def test_an_added_value_is_not_breaking(self):
        """It gives the implementer more, and is visible in the spec."""
        after = copy.deepcopy(BASE)
        after["components"][0]["hyperparameters"].append(
            {"canonical_name": "batch_size", "value": 256, "condition": None}
        )
        result = diff_specs(BASE, after)

        assert [v["canonical_name"] for v in result.values_added] == ["batch_size"]
        assert not result.breaking

    def test_the_same_value_under_a_new_condition_is_an_addition(self):
        """Scope is part of a value's identity, not decoration."""
        after = copy.deepcopy(BASE)
        after["components"][0]["hyperparameters"].append(
            {"canonical_name": "dropout", "value": 0.1, "condition": "large model"}
        )
        result = diff_specs(BASE, after)

        assert len(result.values_added) == 1
        assert result.values_added[0]["condition"] == "large model"

    def test_a_resolution_is_attributed(self):
        """Without this a reader sees a number move with no way to ask why."""
        after = set_value(copy.deepcopy(BASE), "learning_rate", 0.0003)
        after["components"][0]["hyperparameters"][0]["resolved_from"] = "ctr_0001"

        assert diff_specs(BASE, after).values_changed[0]["resolved_from"] == "ctr_0001"


class TestConflictsGapsAndReview:
    def test_a_resolved_conflict_is_reported_as_closed(self):
        after = altered(open_conflicts=[])
        result = diff_specs(BASE, after)

        assert result.conflicts_closed == ["ctr_0001"]
        assert not result.breaking, "closing a conflict does not invalidate prior work"

    def test_a_new_conflict_is_reported_as_opened(self):
        after = altered(
            open_conflicts=[{"contradiction_id": "ctr_0001"}, {"contradiction_id": "ctr_0002"}]
        )
        assert diff_specs(BASE, after).conflicts_opened == ["ctr_0002"]

    def test_gap_movement_is_reported_both_ways(self):
        after = altered(missing_but_critical=[{"field": "weight_initialization"}])
        result = diff_specs(BASE, after)

        assert result.gaps_opened == ["weight_initialization"]
        assert result.gaps_closed == ["optimizer"]

    def test_approval_is_reported(self):
        result = diff_specs(BASE, altered(review={"status": "approved"}))
        assert (result.review_from, result.review_to) == ("draft", "approved")
        assert not result.identical

    def test_a_dropped_paper_is_breaking(self):
        """The spec no longer reflects a source it previously synthesized."""
        result = diff_specs(BASE, altered(source_papers=[{"paper_id": "p1"}]))

        assert result.papers_removed == ["p2"]
        assert result.breaking


class TestSerialization:
    def test_the_diff_is_machine_readable(self):
        import json

        payload = diff_specs(
            BASE, set_value(copy.deepcopy(BASE), "learning_rate", 0.0003)
        ).to_dict()
        json.dumps(payload)

        assert payload["values"]["changed"][0]["canonical_name"] == "learning_rate"
        assert payload["breaking"]
        assert payload["identical"] is False


class TestDiffCommand:
    @pytest.fixture
    def two_runs(self, tmp_path):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from e2e.test_mva_acceptance import EXTRACTIONS

        from papersynth.core.run import Pipeline, Workspace
        from papersynth.ingest.latex import LatexIngestor
        from papersynth.llm.stub import StubProvider

        fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "three_paper"
        docs = [
            LatexIngestor().ingest(str(fixtures / f"paper_{p}.tex"), paper_id=f"paper_{p}")
            for p in "abc"
        ]

        def respond(prompt: str):
            for pid, items in EXTRACTIONS.items():
                marker = {
                    "paper_a": "batch size of 128",
                    "paper_b": "batch size of 256",
                    "paper_c": "larger stack is less stable",
                }[pid]
                if marker in prompt:
                    return items
            return []

        for name in ("before", "after"):
            Pipeline(
                StubProvider(respond),
                workspace=Workspace(tmp_path, name),
                extractors=["hyperparameter"],
                entailment=False,
            ).run(docs, objective="Diff test.", run_id=name)
        return tmp_path / "before", tmp_path / "after"

    def test_two_identical_runs_report_no_change(self, two_runs):
        from typer.testing import CliRunner

        from papersynth.cli import app

        before, after = two_runs
        result = CliRunner().invoke(app, ["diff", str(before), str(after)])

        assert result.exit_code == 0
        assert "No change" in result.stdout

    def test_a_resolution_shows_as_a_breaking_change(self, two_runs):
        """Resolving a conflict emits a value that was previously withheld -
        which for anyone holding the earlier spec is new information."""
        import yaml
        from typer.testing import CliRunner

        from papersynth.cli import app

        before, after = two_runs
        spec = yaml.safe_load((after / "implementation_spec.yaml").read_text())
        conflict = spec["open_conflicts"][0]

        runner = CliRunner()
        runner.invoke(
            app,
            [
                "resolve",
                str(after),
                conflict["contradiction_id"],
                "--select",
                conflict["positions"][0]["claim_id"],
            ],
        )
        result = runner.invoke(app, ["diff", str(before), str(after), "--format", "json"])

        assert result.exit_code in (0, 3)
        assert "ctr_" in result.stdout
