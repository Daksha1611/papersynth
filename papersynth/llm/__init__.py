"""LLM provider layer.

Nothing outside this package constructs a provider. Stages receive a router and
stay indifferent to which free tier served any given call (section 3.3).
"""

from __future__ import annotations

from pathlib import Path

from papersynth.core.config import Settings, get_settings
from papersynth.core.ledger import Ledger
from papersynth.llm.base import Completion, LLMProvider, Usage, parse_json_response
from papersynth.llm.cache import PromptCache
from papersynth.llm.providers import build_chain, build_provider
from papersynth.llm.router import FallbackRouter
from papersynth.llm.stub import StubProvider
from papersynth.llm.usage_tracker import UsageTracker

__all__ = [
    "Completion",
    "FallbackRouter",
    "LLMProvider",
    "PromptCache",
    "StubProvider",
    "Usage",
    "UsageTracker",
    "build_chain",
    "build_provider",
    "build_router",
    "parse_json_response",
]


def build_router(
    settings: Settings | None = None,
    *,
    ledger: Ledger | None = None,
    workspace: Path | None = None,
) -> FallbackRouter:
    """Assemble the configured chain with quota tracking and caching wired in."""
    settings = settings or get_settings()

    return FallbackRouter(
        build_chain(settings),
        usage=UsageTracker(
            state_path=settings.cache_dir / "usage.json",
            limits={p: settings.rpd_limit(p) for p in settings.provider_chain},
            safety_margin=settings.rpd_safety_margin,
        ),
        ledger=ledger or Ledger(workspace / "ledger.jsonl" if workspace else None),
        cache=PromptCache(
            settings.cache_dir / "prompts",
            enabled=settings.cache_by_prompt_hash,
        ),
    )
