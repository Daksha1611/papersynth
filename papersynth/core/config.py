"""Configuration.

Everything is env-overridable with a ``PAPERSYNTH_`` prefix, and every setting
that changes model output is recorded in the run manifest - a run you cannot
reproduce is a run you cannot debug (NFR-02, NFR-06).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderId = Literal["groq", "gemini", "openrouter", "vllm", "stub"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PAPERSYNTH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- providers ---------------------------------------------------------
    # Free tiers only, tried in order. A run cannot spend money on this chain.
    provider_chain: list[ProviderId] = Field(default=["groq", "gemini"])
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # API keys are declared here, with their conventional un-prefixed names, so
    # that a key written into .env is actually picked up. Reading them through
    # os.getenv instead would ignore .env entirely and fail with "no providers
    # configured" while the key sits right there in the file.
    groq_api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    openrouter_api_key: str | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")

    # Free-tier model lineups rotate without notice (R-13): llama-3.3-70b-versatile
    # was the documented default and has since been delisted. Verify with
    # `papersynth models` rather than assuming this ID is still served.
    groq_model: str = "openai/gpt-oss-120b"
    groq_rpd_limit: int = 1000

    # gemini-2.5-flash was the documented default and is now refused for new
    # accounts. Pinned rather than using the floating `gemini-flash-latest`
    # alias: a model that changes underneath a run would break the guarantee
    # that identical inputs produce identical specs (NFR-02). Check staleness
    # with `papersynth models --provider gemini`.
    gemini_model: str = "gemini-3.6-flash"
    gemini_rpd_limit: int = 1500

    openrouter_free_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    openrouter_rpd_limit: int = 50

    vllm_url: str = "http://localhost:11434/v1"
    vllm_model: str = "qwen2.5:14b"

    #: Self-throttle at this fraction of a provider's known daily ceiling, so
    #: the router steps aside before burning a call to discover it is capped.
    rpd_safety_margin: float = Field(default=0.9, gt=0.0, le=1.0)

    # -- ingestion ---------------------------------------------------------
    grobid_url: str = "http://localhost:8070"
    grobid_timeout_s: float = 120.0
    prefer_latex: bool = True
    arxiv_api_url: str = "https://export.arxiv.org/api/query"

    # -- call volume (section 6.4.5) --------------------------------------
    self_consistency_n: int = Field(default=1, ge=1, le=9)
    verify_batch_size: int = Field(default=10, ge=1, le=100)
    cache_by_prompt_hash: bool = True

    # -- pipeline ----------------------------------------------------------
    workspace: Path = Path("./runs")
    cache_dir: Path = Path("./.papersynth_cache")
    #: Papers extracted concurrently. One by default, and that is not
    #: timidity: the binding constraint on a hosted free tier is tokens per
    #: minute, not latency. Groq allows 8,000 and one extraction prompt is
    #: roughly 3,000, so three concurrent papers put 9,000 in flight and
    #: guarantee the 429 that concurrency was supposed to avoid waiting for.
    #: Raise it on the local vLLM path, which has no per-minute cap.
    max_parallel_papers: int = Field(default=1, ge=1, le=32)
    policy: Path = Path("config/reconcile_policy.yaml")
    range_rules: Path = Path("config/range_rules.yaml")
    checklist: Path = Path("config/implementability_checklist.yaml")

    #: Below this, a claim stays 'extracted' rather than 'verified', which
    #: excludes it from auto-resolution (section 8.3.4).
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    @field_validator("provider_chain", mode="before")
    @classmethod
    def _split_chain(cls, v: object) -> object:
        """Accept 'groq,gemini' as well as a real list."""
        if isinstance(v, str):
            return [part.strip() for part in v.split(",") if part.strip()]
        return v

    def api_key(self, provider_id: str) -> str | None:
        """The configured key, or None when this leg is simply not set up.

        A missing key is not an error - section 6.4.4 is explicit that an unset
        OPENROUTER_API_KEY narrows the chain rather than crashing the run.
        """
        key = {
            "groq": self.groq_api_key,
            "gemini": self.google_api_key,
            "openrouter": self.openrouter_api_key,
        }.get(provider_id)
        return key.strip() or None if key else None

    def rpd_limit(self, provider_id: str) -> int | None:
        return {
            "groq": self.groq_rpd_limit,
            "gemini": self.gemini_rpd_limit,
            "openrouter": self.openrouter_rpd_limit,
        }.get(provider_id)

    def model_for(self, provider_id: str) -> str:
        return {
            "groq": self.groq_model,
            "gemini": self.gemini_model,
            "openrouter": self.openrouter_free_model,
            "vllm": self.vllm_model,
            "stub": "stub",
        }.get(provider_id, "unknown")

    def reproducibility_fingerprint(self) -> dict[str, object]:
        """The subset of settings that can change model output.

        Recorded in the run manifest so a spec that differs from a prior run
        can be attributed to a config change rather than model drift (R-06).
        """
        return {
            "provider_chain": list(self.provider_chain),
            "models": {p: self.model_for(p) for p in self.provider_chain},
            "temperature": self.temperature,
            "self_consistency_n": self.self_consistency_n,
            "confidence_threshold": self.confidence_threshold,
            "prefer_latex": self.prefer_latex,
        }


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    """Override the singleton. Tests and the CLI use this; nothing else should."""
    global _settings
    _settings = settings


def reset_settings() -> None:
    global _settings
    _settings = None
