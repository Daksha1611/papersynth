"""CLI surface (section 11.4).

Exit codes are part of the contract, not decoration: scripts/run.sh branches on
`conflicts --quiet` to decide whether emitting a spec is even allowed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from papersynth.cli import app
from papersynth.core.run import Pipeline, Workspace
from papersynth.ingest.latex import LatexIngestor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e2e.test_mva_acceptance import scripted_provider

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "three_paper"
runner = CliRunner()

#: Rich decides on colour and wrapping from the environment, so raw CLI output
#: is not a stable string to assert against.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """CLI output with colour and line wrapping removed."""
    return " ".join(_ANSI.sub("", text).split())


@pytest.fixture
def run_dir(tmp_path):
    """A completed run on disk, built offline with scripted responses."""
    docs = [
        LatexIngestor().ingest(str(FIXTURES / f"paper_{p}.tex"), paper_id=f"paper_{p}")
        for p in "abc"
    ]
    workspace = Workspace(tmp_path, "demo")
    Pipeline(scripted_provider(), workspace=workspace, extractors=["hyperparameter"]).run(
        docs, objective="Implement the baseline sequence encoder.", run_id="demo"
    )
    return workspace.root


def spec_of(run_dir: Path) -> dict:
    return yaml.safe_load((run_dir / "implementation_spec.yaml").read_text())


def learning_rates(run_dir: Path, condition: str) -> list:
    return [
        hp["value"]
        for component in spec_of(run_dir)["components"]
        for hp in component["hyperparameters"]
        if hp["canonical_name"] == "learning_rate" and hp["condition"] == condition
    ]


class TestBasics:
    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "papersynth" in plain(result.stdout)

    def test_validate_schemas(self):
        result = runner.invoke(app, ["validate-schemas"])
        assert result.exit_code == 0
        assert "schemas valid" in plain(result.stdout)

    def test_extractors_lists_the_registry(self):
        result = runner.invoke(app, ["extractors"])
        assert result.exit_code == 0
        assert "hyperparameter" in plain(result.stdout)

    def test_a_non_run_directory_fails_clearly(self, tmp_path):
        result = runner.invoke(app, ["conflicts", str(tmp_path)])
        assert result.exit_code == 1
        assert "not a PaperSynth run directory" in plain(result.output)

    def test_ingest_requires_a_source(self):
        result = runner.invoke(app, ["ingest"])
        assert result.exit_code == 2


class TestConflicts:
    def test_open_conflicts_exit_non_zero(self, run_dir):
        """scripts/run.sh relies on this to refuse to emit."""
        result = runner.invoke(app, ["conflicts", str(run_dir), "--status", "open"])
        assert result.exit_code == 2
        assert "VALUE_CONFLICT" in plain(result.stdout)

    def test_quiet_prints_nothing(self, run_dir):
        result = runner.invoke(app, ["conflicts", str(run_dir), "--quiet"])
        assert result.exit_code == 2
        assert result.stdout.strip() == ""

    def test_no_blocking_conflicts_exits_zero(self, run_dir):
        result = runner.invoke(
            app, ["conflicts", str(run_dir), "--severity", "BLOCKING", "--quiet"]
        )
        assert result.exit_code == 0

    def test_positions_show_provenance(self, run_dir):
        result = runner.invoke(app, ["conflicts", str(run_dir)])
        assert "paper_a" in plain(result.stdout) and "paper_b" in plain(result.stdout)
        assert "specificity" in plain(result.stdout)

    def test_the_fallback_verdict_is_not_stuttered(self, run_dir):
        """rule_fired is None exactly when the fallback applied, and the
        rationale already says so."""
        result = runner.invoke(app, ["conflicts", str(run_dir)])
        assert "no rule fired - no rule fired" not in plain(result.stdout)


class TestResolveFlow:
    def test_resolving_closes_the_conflict(self, run_dir):
        conflict_id = spec_of(run_dir)["open_conflicts"][0]["contradiction_id"]
        claim_id = spec_of(run_dir)["open_conflicts"][0]["positions"][0]["claim_id"]

        result = runner.invoke(
            app,
            ["resolve", str(run_dir), conflict_id, "--select", claim_id, "--note", "testing"],
        )

        assert result.exit_code == 0
        assert runner.invoke(app, ["conflicts", str(run_dir), "--quiet"]).exit_code == 0

    def test_an_unknown_claim_id_is_rejected(self, run_dir):
        conflict_id = spec_of(run_dir)["open_conflicts"][0]["contradiction_id"]
        result = runner.invoke(
            app, ["resolve", str(run_dir), conflict_id, "--select", "clm_zzzzzz"]
        )
        assert result.exit_code == 2
        assert "not a position" in plain(result.output)

    def test_an_unknown_contradiction_is_rejected(self, run_dir):
        result = runner.invoke(app, ["resolve", str(run_dir), "ctr_nope", "--select", "clm_aaaaaa"])
        assert result.exit_code == 1

    def test_resolving_is_idempotent(self, run_dir):
        conflict_id = spec_of(run_dir)["open_conflicts"][0]["contradiction_id"]
        claim_id = spec_of(run_dir)["open_conflicts"][0]["positions"][0]["claim_id"]
        args = ["resolve", str(run_dir), conflict_id, "--select", claim_id]

        runner.invoke(app, args)
        runner.invoke(app, args)

        reconciliation = yaml.safe_load((run_dir / "05_reconciliation.yaml").read_text())
        matching = [
            r for r in reconciliation["resolutions"] if r["contradiction_id"] == conflict_id
        ]
        assert len(matching) == 1, "resolving twice must replace, not append"

    def test_only_the_selected_value_survives(self, run_dir):
        """The rejected position must not return to the spec as settled fact."""
        conflict = spec_of(run_dir)["open_conflicts"][0]
        winner = next(p for p in conflict["positions"] if p["position"] == "0.0001")

        runner.invoke(
            app,
            [
                "resolve",
                str(run_dir),
                conflict["contradiction_id"],
                "--select",
                winner["claim_id"],
            ],
        )

        assert learning_rates(run_dir, "base model") == [0.0001]

    def test_a_differently_scoped_value_is_untouched(self, run_dir):
        conflict = spec_of(run_dir)["open_conflicts"][0]
        runner.invoke(
            app,
            [
                "resolve",
                str(run_dir),
                conflict["contradiction_id"],
                "--select",
                conflict["positions"][0]["claim_id"],
            ],
        )
        assert learning_rates(run_dir, "large model") == [5e-05]

    def test_the_audit_trail_records_the_human(self, run_dir):
        conflict_id = spec_of(run_dir)["open_conflicts"][0]["contradiction_id"]
        claim_id = spec_of(run_dir)["open_conflicts"][0]["positions"][0]["claim_id"]
        runner.invoke(
            app,
            ["resolve", str(run_dir), conflict_id, "--select", claim_id, "--note", "because"],
        )

        resolved = spec_of(run_dir)["resolved_conflicts"]
        assert len(resolved) == 1
        assert resolved[0]["resolved_by"] == "human"
        assert resolved[0]["human_note"] == "because"
        assert resolved[0]["rule_fired"] is None


class TestApprove:
    def test_approving_stamps_the_reviewer(self, run_dir):
        result = runner.invoke(app, ["approve", str(run_dir), "--reviewer", "daksha"])
        assert result.exit_code == 0

        review = spec_of(run_dir)["review"]
        assert review["status"] == "approved"
        assert review["reviewer"] == "daksha"
        assert review["approved_at"]

    def test_a_spec_starts_unapproved(self, run_dir):
        """Human approval is a required stage, not a default (DD-01)."""
        assert spec_of(run_dir)["review"]["status"] == "draft"


class TestGapsAndCost:
    def test_gaps_are_listed(self, run_dir):
        result = runner.invoke(app, ["gaps", str(run_dir)])
        assert result.exit_code == 0
        assert "optimizer" in plain(result.stdout)

    def test_gaps_output_does_not_overclaim(self, run_dir):
        result = runner.invoke(app, ["gaps", str(run_dir)])
        assert "no verified claim" in plain(result.stdout).lower()

    def test_cost_reports_zero_on_the_free_chain(self, run_dir):
        result = runner.invoke(app, ["cost", str(run_dir)])
        assert result.exit_code == 0
        assert "$0.00" in plain(result.stdout)


class TestSpecCommand:
    def test_spec_prints_yaml(self, run_dir):
        result = runner.invoke(app, ["spec", str(run_dir)])
        assert result.exit_code == 0
        assert "spec_version" in plain(result.stdout)

    def test_spec_json_format(self, run_dir):
        result = runner.invoke(app, ["spec", str(run_dir), "--format", "json"])
        assert result.exit_code == 0
        assert "spec_version" in plain(result.stdout)

    def test_rebuild_reproduces_the_same_spec(self, run_dir):
        """Rebuilding from artifacts must not drift from what the run emitted."""
        before = spec_of(run_dir)
        runner.invoke(app, ["spec", str(run_dir), "--rebuild"])
        after = spec_of(run_dir)

        for key in ("components", "open_conflicts", "missing_but_critical"):
            assert before[key] == after[key]
