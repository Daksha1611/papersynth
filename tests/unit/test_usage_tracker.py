"""Quota accounting. The persistence tests matter most: a daily counter that
resets when the process restarts would let six debugging runs each believe the
free tier is untouched (section 6.4.1)."""

from __future__ import annotations

from papersynth.llm.base import Usage
from papersynth.llm.usage_tracker import UsageTracker


def test_counts_survive_a_new_tracker_instance(tmp_path):
    state = tmp_path / "usage.json"
    first = UsageTracker(state, limits={"groq": 100})
    for _ in range(5):
        first.record("groq", Usage(10, 5))

    second = UsageTracker(state, limits={"groq": 100})

    assert second.snapshot(["groq"])[0].requests == 5
    assert second.snapshot(["groq"])[0].input_tokens == 50


def test_a_rate_limit_block_survives_a_restart(tmp_path):
    state = tmp_path / "usage.json"
    UsageTracker(state, limits={"groq": 100}).mark_exhausted("groq", retry_after=3600)

    assert UsageTracker(state, limits={"groq": 100}).likely_exhausted("groq")


def test_a_corrupt_state_file_does_not_stop_a_run(tmp_path):
    """Undercounting risks one wasted call; refusing to start costs the run."""
    state = tmp_path / "usage.json"
    state.write_text("{ this is not json")

    tracker = UsageTracker(state, limits={"groq": 100})

    assert tracker.likely_exhausted("groq") is False


def test_a_429_without_retry_after_is_treated_as_a_daily_cap(tmp_path):
    """With no header there is no way to tell a per-minute limit from a
    per-day one, so headroom reporting assumes the worse case and says so."""
    tracker = UsageTracker(tmp_path / "u.json", limits={"groq": 1000})
    tracker.mark_exhausted("groq", retry_after=None)

    snapshot = tracker.snapshot(["groq"])[0]
    assert tracker.likely_exhausted("groq")
    assert snapshot.headroom == 0
    assert snapshot.rate_limit_hits == 1


def test_a_provider_with_no_known_limit_is_never_predicted_exhausted():
    tracker = UsageTracker(limits={"vllm": None}, persist=False)
    for _ in range(10_000):
        tracker.record("vllm", Usage(1, 1))

    assert tracker.likely_exhausted("vllm") is False
    assert tracker.headroom("vllm") is None


def test_counters_are_scoped_to_a_day(tmp_path):
    state = tmp_path / "usage.json"
    tracker = UsageTracker(state, limits={"groq": 10})
    tracker.record("groq", Usage(1, 1))

    # Simulate the stored day being yesterday.
    import json

    payload = json.loads(state.read_text())
    payload["groq"]["day"] = "2020-01-01"
    state.write_text(json.dumps(payload))

    assert UsageTracker(state, limits={"groq": 10}).snapshot(["groq"])[0].requests == 0


def test_spend_stays_zero_on_the_free_chain():
    tracker = UsageTracker(persist=False)
    tracker.record("groq", Usage(100, 50))
    tracker.record_spend("groq", 0.0)

    assert tracker.snapshot(["groq"])[0].spend_usd == 0.0
