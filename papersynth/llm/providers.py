"""Concrete providers and chain construction.

Chain order is fixed and deliberate (section 6.4.2): Groq first because it is
fastest and has the largest usable free allowance, Gemini second because its
quota is genuinely independent of Groq's, OpenRouter last because its unfunded
tier is the smallest of the three - putting it earlier would exhaust the whole
chain sooner, not extend it.

No paid model is configured anywhere here.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from papersynth.core.config import Settings, get_settings
from papersynth.core.errors import (
    CapacityError,
    ContentPolicyError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)
from papersynth.llm.base import (
    Completion,
    LLMProvider,
    Usage,
    parse_json_response,
    schema_instruction,
)
from papersynth.llm.openai_compat import OpenAICompatibleProvider

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def groq_provider(settings: Settings, api_key: str) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        provider_id="groq",
        base_url=GROQ_BASE_URL,
        model=settings.groq_model,
        api_key=api_key,
    )


def openrouter_provider(settings: Settings, api_key: str) -> OpenAICompatibleProvider:
    model = settings.openrouter_free_model
    if not model.endswith(":free"):
        # Structural guard, not a style preference: this leg must not be able
        # to reach a paid model even by misconfiguration (section 6.4.3).
        raise ProviderError(
            f"PAPERSYNTH_OPENROUTER_FREE_MODEL is {model!r}, which is not a ':free' "
            "model. This provider is configured for free-tier use only."
        )
    return OpenAICompatibleProvider(
        provider_id="openrouter",
        base_url=OPENROUTER_BASE_URL,
        model=model,
        api_key=api_key,
        extra_headers={
            "HTTP-Referer": "https://github.com/Daksha1611/papersynth",
            "X-Title": "PaperSynth",
        },
    )


def vllm_provider(settings: Settings) -> OpenAICompatibleProvider:
    """Local Ollama or vLLM. No quota, no key, fully offline."""
    return OpenAICompatibleProvider(
        provider_id="vllm",
        base_url=settings.vllm_url,
        model=settings.vllm_model,
        api_key=os.getenv("PAPERSYNTH_VLLM_API_KEY"),
        timeout_s=600.0,
    )


class GeminiProvider:
    """Google Generative Language REST API.

    Not OpenAI-compatible in request shape, so it gets its own client. Uses
    native structured output, which Gemini honours more reliably than a
    prompt-only instruction.
    """

    def __init__(self, model: str, api_key: str, timeout_s: float = 120.0) -> None:
        self.provider_id = "gemini"
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    def complete(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> Completion:
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if schema is not None:
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["contents"][0]["parts"][0]["text"] = f"{prompt}\n\n{schema_instruction(schema)}"

        url = f"{GEMINI_BASE_URL}/models/{self.model}:generateContent"
        started = time.perf_counter()
        try:
            response = httpx.post(
                url,
                json=body,
                headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
                timeout=self.timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise CapacityError(f"gemini timed out after {self.timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise CapacityError(f"gemini unreachable: {exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code == 429:
            raise RateLimitError("gemini", retry_after=None)
        if response.status_code == 404:
            raise ModelNotFoundError(
                f"gemini does not serve model {self.model!r}: {response.text[:200]}"
            )
        if response.status_code in (401, 403):
            raise ProviderError(
                f"gemini rejected the API key ({response.status_code}). Check GOOGLE_API_KEY."
            )
        if response.status_code >= 500:
            raise CapacityError(f"gemini returned {response.status_code}: {response.text[:200]}")
        if response.status_code != 200:
            raise ProviderError(f"gemini returned {response.status_code}: {response.text[:300]}")

        payload = response.json()

        feedback = payload.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise ContentPolicyError(f"gemini blocked the prompt: {feedback['blockReason']}")

        candidates = payload.get("candidates") or []
        if not candidates:
            raise ProviderError(f"gemini returned no candidates: {response.text[:200]}")

        candidate = candidates[0]
        if candidate.get("finishReason") == "SAFETY":
            raise ContentPolicyError("gemini stopped generation for safety reasons")

        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)

        usage_payload = payload.get("usageMetadata") or {}
        completion = Completion(
            text=text,
            provider_id="gemini",
            model=self.model,
            usage=Usage(
                input_tokens=int(usage_payload.get("promptTokenCount", 0)),
                output_tokens=int(usage_payload.get("candidatesTokenCount", 0)),
            ),
            cost_usd=0.0,
            latency_ms=latency_ms,
            finish_reason=candidate.get("finishReason"),
        )
        if schema is not None:
            completion.parsed = parse_json_response(text, provider_id="gemini")
        return completion


def build_provider(provider_id: str, settings: Settings | None = None) -> LLMProvider | None:
    """Instantiate one provider, or None when its key is absent.

    A missing key is not an error. Section 6.4.4 is explicit that an unset
    OPENROUTER_API_KEY simply narrows the chain rather than crashing the run.
    """
    settings = settings or get_settings()

    key = settings.api_key(provider_id)

    if provider_id == "groq":
        return groq_provider(settings, key) if key else None

    if provider_id == "gemini":
        # GEMINI_API_KEY is the older name and still in circulation.
        key = key or os.getenv("GEMINI_API_KEY")
        return GeminiProvider(settings.gemini_model, key) if key else None

    if provider_id == "openrouter":
        return openrouter_provider(settings, key) if key else None

    if provider_id == "vllm":
        # Local endpoints need no key, so this leg is always available.
        return vllm_provider(settings)

    return None


def build_chain(settings: Settings | None = None) -> list[LLMProvider]:
    """Build the configured chain, skipping legs whose key is unset."""
    settings = settings or get_settings()
    chain: list[LLMProvider] = []
    for provider_id in settings.provider_chain:
        provider = build_provider(provider_id, settings)
        if provider is not None:
            chain.append(provider)
    return chain
