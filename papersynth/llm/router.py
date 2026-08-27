"""The fallback router (section 6.4.3).

Tries providers in chain order. The rule that gives this module its shape is
which errors fall through and which do not:

  - 429 and capacity errors fall through. The next free tier is an independent
    quota, so a stalled run becomes a slightly slower one.
  - Schema, content-policy, and model-not-found errors do not. Those are bugs
    or misconfiguration, and letting a different provider "coincidentally"
    succeed would hide the actual problem indefinitely - you would never find
    out that a prompt is malformed, only that one provider dislikes it.

Nothing in the default chain can spend money; every leg is free tier.
"""

from __future__ import annotations

from typing import Any

from papersynth.core import ids
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
from papersynth.llm.base import Completion, LLMProvider
from papersynth.llm.cache import PromptCache
from papersynth.llm.usage_tracker import UsageTracker

#: Errors that mean "this provider cannot serve me right now" - try the next.
_FALL_THROUGH = (RateLimitError, CapacityError)

#: Errors that mean "something is genuinely wrong" - stop and surface it.
_TERMINAL = (SchemaValidationError, ContentPolicyError, ModelNotFoundError)


class FallbackRouter:
    """An LLMProvider that delegates to a chain of real providers."""

    provider_id = "router"

    def __init__(
        self,
        chain: list[LLMProvider],
        *,
        usage: UsageTracker | None = None,
        ledger: Ledger | None = None,
        cache: PromptCache | None = None,
    ) -> None:
        if not chain:
            raise ProviderError(
                "No LLM providers are configured. Set at least one of GROQ_API_KEY, "
                "GOOGLE_API_KEY, or OPENROUTER_API_KEY in .env, or point "
                "PAPERSYNTH_PROVIDER_CHAIN at a local vllm endpoint."
            )
        self.chain = chain
        self.usage = usage or UsageTracker()
        self.ledger = ledger or Ledger()
        self.cache = cache or PromptCache(None, enabled=False)

    @property
    def model(self) -> str:
        return self.chain[0].model

    def complete(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        system: str | None = None,
        max_tokens: int = 4096,
        stage: str = "unknown",
        paper_id: str | None = None,
        extractor: str | None = None,
        template_id: str = "",
    ) -> Completion:
        last_error: Exception | None = None
        skipped: list[str] = []
        first_attempted: str | None = None

        for provider in self.chain:
            cache_key = ids.prompt_hash(template_id, f"{system or ''}\n{prompt}", provider.model)

            cached = self.cache.get(cache_key)
            if cached is not None:
                self._record(
                    cached, stage, cache_key, paper_id, extractor, fallback_from=None, reason=None
                )
                return cached

            if self.usage.likely_exhausted(provider.provider_id):
                skipped.append(provider.provider_id)
                continue

            if first_attempted is None:
                first_attempted = provider.provider_id
            fallback_from = first_attempted if provider.provider_id != first_attempted else None

            try:
                completion = provider.complete(
                    prompt,
                    schema=schema,
                    temperature=temperature,
                    system=system,
                    max_tokens=max_tokens,
                )
            except _TERMINAL:
                raise
            except _FALL_THROUGH as exc:
                retry_after = getattr(exc, "retry_after", None)
                self.usage.mark_exhausted(provider.provider_id, retry_after=retry_after)
                last_error = exc
                self.ledger.record(
                    stage=stage,
                    call_id=cache_key[:12],
                    provider_id=provider.provider_id,
                    model=provider.model,
                    prompt_hash=cache_key,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    latency_ms=0.0,
                    paper_id=paper_id,
                    extractor=extractor,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue

            self.usage.record(provider.provider_id, completion.usage)
            if completion.cost_usd > 0:
                self.usage.record_spend(provider.provider_id, completion.cost_usd)

            self.cache.put(cache_key, completion)
            self._record(
                completion,
                stage,
                cache_key,
                paper_id,
                extractor,
                fallback_from=fallback_from,
                reason=type(last_error).__name__ if last_error else None,
            )
            return completion

        raise AllProvidersExhausted(
            last_error
            or ProviderError(
                "every provider was skipped as already exhausted today: "
                + ", ".join(skipped or [p.provider_id for p in self.chain])
            )
        )

    def _record(
        self,
        completion: Completion,
        stage: str,
        cache_key: str,
        paper_id: str | None,
        extractor: str | None,
        *,
        fallback_from: str | None,
        reason: str | None,
    ) -> None:
        self.ledger.record(
            stage=stage,
            call_id=cache_key[:12],
            provider_id=completion.provider_id,
            model=completion.model,
            prompt_hash=cache_key,
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
            cost_usd=completion.cost_usd,
            latency_ms=completion.latency_ms,
            cached=completion.cached,
            paper_id=paper_id,
            extractor=extractor,
            fallback_from=fallback_from,
            fallback_reason=reason,
        )
