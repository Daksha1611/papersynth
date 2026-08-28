"""Shared client for OpenAI-compatible chat endpoints.

Groq, OpenRouter, and Ollama/vLLM all speak this dialect, so they share one
implementation rather than three near-identical SDK wrappers. Talking plain
REST also keeps the provider SDKs out of the default dependency set - one
fewer thing to break when a vendor ships a major version.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from papersynth.core.errors import (
    CapacityError,
    ContentPolicyError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    SchemaValidationError,
)
from papersynth.llm.base import (
    Completion,
    Usage,
    as_object_schema,
    parse_json_response,
    schema_instruction,
)


class OpenAICompatibleProvider:
    """POST /chat/completions against any OpenAI-compatible server."""

    def __init__(
        self,
        *,
        provider_id: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        supports_json_mode: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.supports_json_mode = supports_json_mode
        self.extra_headers = extra_headers or {}

    def complete(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> Completion:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})

        user_content = prompt
        wire_schema = as_object_schema(schema) if schema is not None else None
        if wire_schema is not None:
            user_content = f"{prompt}\n\n{schema_instruction(wire_schema)}"
        messages.append({"role": "user", "content": user_content})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if wire_schema is not None and self.supports_json_mode:
            body["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers=headers,
                timeout=self.timeout_s,
            )
        except httpx.TimeoutException as exc:
            # A timeout is a capacity symptom, so it falls through to the next
            # provider rather than failing the run.
            raise CapacityError(f"{self.provider_id} timed out after {self.timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise CapacityError(f"{self.provider_id} unreachable: {exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000
        self._raise_for_status(response)

        try:
            payload = response.json()
            choice = payload["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(
                f"{self.provider_id} returned an unexpected response shape: {response.text[:200]}"
            ) from exc

        usage_payload = payload.get("usage") or {}
        completion = Completion(
            text=text,
            provider_id=self.provider_id,
            model=payload.get("model", self.model),
            usage=Usage(
                input_tokens=int(usage_payload.get("prompt_tokens", 0)),
                output_tokens=int(usage_payload.get("completion_tokens", 0)),
            ),
            cost_usd=0.0,
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason"),
        )

        if schema is not None:
            completion.parsed = parse_json_response(text, provider_id=self.provider_id)
        return completion

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code == 200:
            return

        detail = response.text[:300]

        if response.status_code in (429, 413):
            # 413 reads as "payload too large" but Groq returns it for a
            # tokens-per-minute rejection, which is a rate limit that clears on
            # its own. Treating it as terminal made a run give up on a paper
            # that would have succeeded a minute later.
            raise RateLimitError(self.provider_id, retry_after=_retry_after(response))

        if response.status_code == 404:
            # Free model lineups rotate; a configured ID that 404s must say so
            # loudly rather than being skipped silently (section 6.4.4).
            raise ModelNotFoundError(
                f"{self.provider_id} does not serve model {self.model!r}. "
                "Free-tier model IDs are delisted without notice - check the "
                f"provider's current catalogue. Response: {detail}"
            )

        if response.status_code in (400, 422) and "json_validate_failed" in detail:
            raise SchemaValidationError(
                f"{self.provider_id} JSON mode",
                [
                    "the provider rejected the model's output as not a JSON object. "
                    "Array schemas must be wrapped before sending; see as_object_schema."
                ],
            )

        if response.status_code in (400, 422) and _mentions_content_policy(detail):
            raise ContentPolicyError(f"{self.provider_id} refused the request: {detail}")

        if response.status_code in (401, 403):
            raise ProviderError(
                f"{self.provider_id} rejected the API key ({response.status_code}). "
                "Check the corresponding key in .env."
            )

        if response.status_code >= 500 or response.status_code == 503:
            raise CapacityError(f"{self.provider_id} returned {response.status_code}: {detail}")

        raise ProviderError(f"{self.provider_id} returned {response.status_code}: {detail}")


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after") or response.headers.get("x-ratelimit-reset-requests")
    if not raw:
        return None
    try:
        return float(str(raw).rstrip("s"))
    except ValueError:
        return None


def _mentions_content_policy(detail: str) -> bool:
    lowered = detail.lower()
    return any(
        token in lowered
        for token in ("content_policy", "content policy", "safety", "blocked", "moderation")
    )
