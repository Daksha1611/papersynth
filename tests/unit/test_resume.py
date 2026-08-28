"""Resumable runs (FR-15, NFR-05) and routing observability (section 6.4.3).

Both were specified and neither had ever been exercised. The exhaustion
message told users to run `papersynth run --resume`, which did not exist - an
instruction that fails when followed is worse than a missing feature.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

from papersynth.core.errors import AllProvidersExhausted, RateLimitError
from papersynth.core.ledger import Ledger
from papersynth.core.run import Pipeline, Workspace
from papersynth.ingest.latex import LatexIngestor
from papersynth.llm.base import Usage
from papersynth.llm.cache import PromptCache
from papersynth.llm.router import FallbackRouter
from papersynth.llm.stub import StubProvider
from papersynth.llm.usage_tracker import UsageTracker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from e2e.test_mva_acceptance import EXTRACTIONS

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "three_paper"


def docs():
    return [
        LatexIngestor().ingest(str(FIXTURES / f"paper_{p}.tex"), paper_id=f"paper_{p}")
        for p in "abc"
    ]


def extraction_responder():
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


class TestResume:
    @pytest.fixture
    def completed(self, tmp_path):
        workspace = Workspace(tmp_path, "run")
        Pipeline(
            StubProvider(extraction_responder()),
            workspace=workspace,
            extractors=["hyperparameter"],
            entailment=False,
        ).run(docs(), objective="Test.", run_id="run")
        return workspace

    def test_a_resumed_run_makes_no_extraction_calls(self, completed):
        """Extract and verify are where a run's calls go. Reusing them is the
        whole point of resuming after a quota reset."""
        provider = StubProvider(extraction_responder())
        Pipeline(
            provider,
            workspace=completed,
            extractors=["hyperparameter"],
            entailment=False,
            resume=True,
        ).run(docs(), objective="Test.", run_id="run")

        assert provider.call_count == 0

    def test_a_resumed_run_reproduces_the_same_claims(self, completed):
        before = yaml.safe_load((completed.root / "implementation_spec.yaml").read_text())[
            "verification_summary"
        ]["claims_total"]

        result = Pipeline(
            StubProvider([]),
            workspace=completed,
            extractors=["hyperparameter"],
            entailment=False,
            resume=True,
        ).run(docs(), objective="Test.", run_id="run")

        assert result.spec["verification_summary"]["claims_total"] == before

    def test_resume_says_which_papers_were_reused(self, completed):
        result = Pipeline(
            StubProvider([]),
            workspace=completed,
            extractors=["hyperparameter"],
            entailment=False,
            resume=True,
        ).run(docs(), objective="Test.", run_id="run")

        reused = [w for w in result.warnings if "reused verified claims" in w]
        assert len(reused) == 3

    def test_a_paper_with_no_artifact_is_extracted(self, completed):
        """The interrupted case: some papers done, others untouched."""
        (completed.root / "02_verified" / "paper_c.yaml").unlink()

        provider = StubProvider(extraction_responder())
        Pipeline(
            provider,
            workspace=completed,
            extractors=["hyperparameter"],
            entailment=False,
            resume=True,
        ).run(docs(), objective="Test.", run_id="run")

        assert provider.call_count > 0, "paper_c must be re-extracted"

    def test_a_truncated_artifact_is_re_extracted(self, completed):
        """Trusting a half-written file would put half a paper's claims into
        the spec and call it complete."""
        (completed.root / "02_verified" / "paper_a.yaml").write_text("claims: [")

        provider = StubProvider(extraction_responder())
        Pipeline(
            provider,
            workspace=completed,
            extractors=["hyperparameter"],
            entailment=False,
            resume=True,
        ).run(docs(), objective="Test.", run_id="run")

        assert provider.call_count > 0

    def test_without_resume_everything_is_recomputed(self, completed):
        provider = StubProvider(extraction_responder())
        Pipeline(
            provider,
            workspace=completed,
            extractors=["hyperparameter"],
            entailment=False,
            resume=False,
        ).run(docs(), objective="Test.", run_id="run")

        assert provider.call_count == 3


class TestExhaustionMessage:
    def test_the_message_names_a_flag_that_exists(self):
        """It previously pointed at `papersynth run --resume <run>`, which was
        never implemented."""
        message = str(AllProvidersExhausted())
        assert "--resume" in message

        from typer.testing import CliRunner

        from papersynth.cli import app

        result = CliRunner().invoke(app, ["run", "--help"])
        # Rich colours and wraps help text, so neither is stable to match on.
        plain = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", result.output).split())
        assert "--resume" in plain


class TestSkipObservability:
    def make_router(self, chain, usage, ledger):
        return FallbackRouter(
            chain,
            usage=usage,
            ledger=ledger,
            cache=PromptCache(None, enabled=False),
            sleep=lambda _s: None,
        )

    def test_a_proactively_skipped_provider_is_recorded(self):
        """Section 6.4.3 promises the ledger explains which provider served
        each call. A silent skip made the second leg look unexplained."""
        usage = UsageTracker(limits={"groq": 10}, safety_margin=0.9, persist=False)
        for _ in range(9):
            usage.record("groq", Usage(1, 1))

        ledger = Ledger()
        router = self.make_router(
            [
                StubProvider([{"a": 1}], provider_id="groq"),
                StubProvider([{"a": 2}], provider_id="gemini"),
            ],
            usage,
            ledger,
        )
        router.complete("x", schema={"type": "object"}, stage="extract")

        skipped = [e for e in ledger.entries if e.error and "skipped" in e.error]
        assert len(skipped) == 1
        assert skipped[0].provider_id == "groq"

    def test_the_serving_provider_is_still_recorded(self):
        usage = UsageTracker(limits={"groq": 10}, safety_margin=0.9, persist=False)
        for _ in range(9):
            usage.record("groq", Usage(1, 1))

        ledger = Ledger()
        self.make_router(
            [
                StubProvider([{"a": 1}], provider_id="groq"),
                StubProvider([{"a": 2}], provider_id="gemini"),
            ],
            usage,
            ledger,
        ).complete("x", schema={"type": "object"})

        served = [e for e in ledger.entries if not e.error]
        assert [e.provider_id for e in served] == ["gemini"]


class TestUsageCountsStayTruthful:
    def test_a_daily_block_does_not_fabricate_requests(self):
        """gemini reported 1350 requests after roughly ten, and
        `papersynth cost --by-provider` published that number."""
        tracker = UsageTracker(limits={"gemini": 1500}, persist=False)
        tracker.record("gemini", Usage(10, 5))
        tracker.mark_exhausted("gemini", retry_after=None)

        snapshot = tracker.snapshot(["gemini"])[0]
        assert snapshot.requests == 1, "the count must reflect calls actually made"

    def test_a_blocked_provider_reports_no_headroom(self):
        """Whatever the counter says, a blocked provider has nothing usable."""
        tracker = UsageTracker(limits={"gemini": 1500}, persist=False)
        tracker.mark_exhausted("gemini", retry_after=None)

        assert tracker.headroom("gemini") == 0
        assert tracker.snapshot(["gemini"])[0].exhausted

    def test_an_unblocked_provider_reports_real_headroom(self):
        tracker = UsageTracker(limits={"groq": 100}, safety_margin=0.9, persist=False)
        tracker.record("groq", Usage(1, 1))
        assert tracker.headroom("groq") == 89

    def test_a_rate_limit_hit_is_still_counted(self):
        tracker = UsageTracker(limits={"groq": 100}, persist=False)
        tracker.mark_exhausted("groq", retry_after=RateLimitError("groq", 5.0).retry_after)
        assert tracker.snapshot(["groq"])[0].rate_limit_hits == 1
