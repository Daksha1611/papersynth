"""FallbackRouter behaviour (section 6.4.3).

The distinction these tests exist to protect: rate-limit and capacity errors
fall through to the next free tier; schema, content-policy, and model-not-found
errors do not, because a different provider coincidentally succeeding would
hide a real bug indefinitely.
"""

from __future__ import annotations

import pytest

from papersynth.core.errors import (
    AllProvidersExhausted,
    CapacityError,
    ContentPolicyError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    SchemaValidationError,
)
from papersynth.core.ledger import Ledger
from papersynth.llm.base import Usage
from papersynth.llm.cache import PromptCache
from papersynth.llm.router import FallbackRouter
from papersynth.llm.stub import StubProvider
from papersynth.llm.usage_tracker import UsageTracker

SCHEMA = {"type": "object", "properties": {"value": {"type": "number"}}}


def make_router(chain, **kwargs):
    # No real waiting in tests; the retry path is still exercised.
    kwargs.setdefault("sleep", lambda _seconds: None)
    kwargs.setdefault("usage", UsageTracker(persist=False))
    kwargs.setdefault("ledger", Ledger())
    kwargs.setdefault("cache", PromptCache(None, enabled=False))
    return FallbackRouter(chain, **kwargs)


def test_first_healthy_provider_serves_the_call():
    primary = StubProvider([{"value": 1}], provider_id="groq")
    secondary = StubProvider([{"value": 2}], provider_id="gemini")
    router = make_router([primary, secondary])

    result = router.complete("extract", schema=SCHEMA)

    assert result.provider_id == "groq"
    assert secondary.call_count == 0


def test_rate_limit_falls_through_to_the_next_provider():
    primary = StubProvider(provider_id="groq", error=RateLimitError("groq", retry_after=30))
    secondary = StubProvider([{"value": 2}], provider_id="gemini")
    router = make_router([primary, secondary])

    result = router.complete("extract", schema=SCHEMA)

    assert result.provider_id == "gemini"
    assert result.parsed == {"value": 2}


def test_capacity_error_falls_through():
    primary = StubProvider(provider_id="groq", error=CapacityError("groq overloaded"))
    secondary = StubProvider([{"value": 7}], provider_id="gemini")

    assert make_router([primary, secondary]).complete("x", schema=SCHEMA).provider_id == "gemini"


@pytest.mark.parametrize(
    "error",
    [
        SchemaValidationError("groq response", ["not valid JSON"]),
        ContentPolicyError("groq refused the request"),
        ModelNotFoundError("groq does not serve model 'stale:free'"),
    ],
    ids=["schema", "content_policy", "model_not_found"],
)
def test_real_errors_surface_instead_of_falling_through(error):
    """Falling through here would hide the bug behind a lucky second provider."""
    primary = StubProvider(provider_id="groq", error=error)
    secondary = StubProvider([{"value": 2}], provider_id="gemini")
    router = make_router([primary, secondary])

    with pytest.raises(type(error)):
        router.complete("extract", schema=SCHEMA)

    assert secondary.call_count == 0, "a real error must not reach the next provider"


def test_a_provider_known_to_be_exhausted_is_skipped_without_a_call():
    """Proactive, not just reactive - do not spend a call to rediscover a cap."""
    usage = UsageTracker(limits={"groq": 10}, safety_margin=0.9, persist=False)
    for _ in range(9):
        usage.record("groq", Usage(1, 1))

    primary = StubProvider([{"value": 1}], provider_id="groq")
    secondary = StubProvider([{"value": 2}], provider_id="gemini")
    router = make_router([primary, secondary], usage=usage)

    result = router.complete("extract", schema=SCHEMA)

    assert primary.call_count == 0
    assert result.provider_id == "gemini"


def test_rate_limit_marks_the_provider_exhausted_for_later_calls():
    usage = UsageTracker(persist=False)
    primary = StubProvider(provider_id="groq", error=RateLimitError("groq", retry_after=300))
    secondary = StubProvider([{"a": 1}, {"a": 2}], provider_id="gemini")
    router = make_router([primary, secondary], usage=usage)

    router.complete("first", schema=SCHEMA)
    router.complete("second", schema=SCHEMA)

    assert usage.likely_exhausted("groq")
    assert primary.call_count == 1, "the second call should skip groq entirely"


def test_all_exhausted_raises_a_pause_not_a_crash():
    primary = StubProvider(provider_id="groq", error=RateLimitError("groq"))
    secondary = StubProvider(provider_id="gemini", error=RateLimitError("gemini"))
    router = make_router([primary, secondary])

    with pytest.raises(AllProvidersExhausted) as exc:
        router.complete("extract", schema=SCHEMA)

    assert "resume" in str(exc.value).lower(), "the message should point at --resume"


def test_empty_chain_is_rejected_with_an_actionable_message():
    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        FallbackRouter([])


class TestLedger:
    def test_every_call_is_recorded(self):
        ledger = Ledger()
        router = make_router([StubProvider([{"a": 1}], provider_id="groq")], ledger=ledger)
        router.complete("extract", schema=SCHEMA, stage="extract", paper_id="1706.03762")

        entry = ledger.entries[0]
        assert entry.provider_id == "groq"
        assert entry.stage == "extract"
        assert entry.paper_id == "1706.03762"
        assert entry.prompt_hash

    def test_a_fallback_is_attributed_in_the_ledger(self):
        """You must be able to see which provider actually served a call."""
        ledger = Ledger()
        router = make_router(
            [
                StubProvider(provider_id="groq", error=RateLimitError("groq")),
                StubProvider([{"a": 1}], provider_id="gemini"),
            ],
            ledger=ledger,
        )
        router.complete("extract", schema=SCHEMA, stage="reconcile")

        served = [e for e in ledger.entries if not e.error][-1]
        assert served.provider_id == "gemini"
        assert served.fallback_from == "groq"
        assert served.fallback_reason == "RateLimitError"
        assert ledger.summary().fallbacks == 1

    def test_free_tier_chain_never_accrues_cost(self):
        ledger = Ledger()
        router = make_router(
            [StubProvider([{"a": 1}, {"b": 2}], provider_id="groq")], ledger=ledger
        )
        router.complete("one", schema=SCHEMA)
        router.complete("two", schema=SCHEMA)

        assert ledger.summary().cost_usd == 0.0


class TestCache:
    def test_a_repeated_prompt_is_served_from_cache(self, tmp_path):
        provider = StubProvider([{"value": 1}], provider_id="groq")
        cache = PromptCache(tmp_path / "prompts")
        router = make_router([provider], cache=cache)

        first = router.complete("identical", schema=SCHEMA, template_id="t@1")
        second = router.complete("identical", schema=SCHEMA, template_id="t@1")

        assert provider.call_count == 1, "the second call must not reach the provider"
        assert second.cached is True
        assert first.text == second.text

    def test_a_changed_template_invalidates_the_cache(self, tmp_path):
        """ER-10: a prompt change must not silently reuse old claims."""
        provider = StubProvider([{"value": 1}, {"value": 2}], provider_id="groq")
        router = make_router([provider], cache=PromptCache(tmp_path / "prompts"))

        router.complete("same text", schema=SCHEMA, template_id="extract@1.0.0")
        router.complete("same text", schema=SCHEMA, template_id="extract@1.1.0")

        assert provider.call_count == 2

    def test_a_cache_hit_does_not_consume_quota(self, tmp_path):
        usage = UsageTracker(limits={"groq": 100}, persist=False)
        router = make_router(
            [StubProvider([{"value": 1}], provider_id="groq")],
            cache=PromptCache(tmp_path / "prompts"),
            usage=usage,
        )
        router.complete("identical", schema=SCHEMA, template_id="t@1")
        before = usage.headroom("groq")
        router.complete("identical", schema=SCHEMA, template_id="t@1")

        assert usage.headroom("groq") == before


class TestThreeLegChain:
    """Section 6.4.2's full chain. Two legs were exercised in production; the
    third had never run, so its behaviour was an assumption. Exhaustion is
    faked here rather than burning real quota to reach it."""

    def chain(self):
        return [
            StubProvider(provider_id="groq", error=RateLimitError("groq", retry_after=None)),
            StubProvider(provider_id="gemini", error=RateLimitError("gemini", retry_after=None)),
            StubProvider([{"value": 3}], provider_id="openrouter"),
        ]

    def test_the_third_leg_serves_when_the_first_two_are_rate_limited(self):
        result = make_router(self.chain()).complete("extract", schema=SCHEMA)

        assert result.provider_id == "openrouter"
        assert result.parsed == {"value": 3}

    def test_the_chain_is_tried_in_configured_order(self):
        chain = self.chain()
        make_router(chain).complete("extract", schema=SCHEMA)

        assert chain[0].call_count == 1, "groq tried first"
        assert chain[1].call_count == 1, "gemini tried second"
        assert chain[2].call_count == 1, "openrouter served"

    def test_the_third_leg_is_attributed_to_the_first_failure(self):
        """Which provider a call fell through FROM is what makes a ledger
        readable when three legs are involved."""
        ledger = Ledger()
        make_router(self.chain(), ledger=ledger).complete("extract", schema=SCHEMA)

        served = [e for e in ledger.entries if not e.error][-1]
        assert served.provider_id == "openrouter"
        assert served.fallback_from == "groq"

    def test_every_leg_appears_in_the_ledger(self):
        ledger = Ledger()
        make_router(self.chain(), ledger=ledger).complete("extract", schema=SCHEMA)

        assert {e.provider_id for e in ledger.entries} == {"groq", "gemini", "openrouter"}

    def test_all_three_exhausted_pauses_rather_than_crashing(self):
        chain = [
            StubProvider(provider_id=p, error=RateLimitError(p, retry_after=None))
            for p in ("groq", "gemini", "openrouter")
        ]
        with pytest.raises(AllProvidersExhausted) as exc:
            make_router(chain).complete("extract", schema=SCHEMA)

        assert "--resume" in str(exc.value)

    def test_a_proactively_exhausted_third_leg_is_skipped_not_called(self):
        usage = UsageTracker(limits={"openrouter": 50}, safety_margin=0.9, persist=False)
        for _ in range(45):
            usage.record("openrouter", Usage(1, 1))

        chain = self.chain()
        with pytest.raises(AllProvidersExhausted):
            make_router(chain, usage=usage).complete("extract", schema=SCHEMA)

        assert chain[2].call_count == 0, "a known-tapped leg must not be called"

    def test_a_terminal_error_on_the_first_leg_never_reaches_the_third(self):
        """Section 6.4.3: a schema bug hidden behind a third provider
        coincidentally succeeding would never be found."""
        chain = self.chain()
        chain[0] = StubProvider(provider_id="groq", error=SchemaValidationError("groq", ["bad"]))

        with pytest.raises(SchemaValidationError):
            make_router(chain).complete("extract", schema=SCHEMA)

        assert chain[1].call_count == 0 and chain[2].call_count == 0
