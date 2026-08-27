"""LLM provider protocol and shared response types.

No pipeline stage imports a provider directly (constraint in section 3.3).
Stages take a ``LLMProvider`` and are indifferent to which free tier actually
served the call - that choice belongs to the router.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from papersynth.core.errors import SchemaValidationError


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Completion:
    """One model response, plus everything the ledger needs to account for it."""

    text: str
    provider_id: str
    model: str
    usage: Usage = field(default_factory=Usage)
    #: Always 0.0 on the default chain - every leg is free tier (DD-07).
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str | None = None
    #: Populated when the call requested structured output.
    parsed: Any = None
    #: True when served from the prompt-hash cache rather than the network.
    cached: bool = False


@runtime_checkable
class LLMProvider(Protocol):
    """The one interface stages see."""

    provider_id: str
    model: str

    def complete(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> Completion:
        """Run a completion. Raises RateLimitError to invite fall-through."""
        ...


# ---------------------------------------------------------------------------
# JSON recovery
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def parse_json_response(text: str, *, provider_id: str) -> Any:
    """Parse a model's JSON output, repairing only deterministic packaging noise.

    The repairs here are strictly syntactic: unwrapping a markdown fence,
    trimming prose around the object, dropping a trailing comma. Every one is a
    pure function of the text with no model involved.

    What this deliberately does not do is retry the model or ask another
    provider. A response that is still not JSON after unwrapping is a prompt
    bug, and section 6.4.3 is explicit that hiding it behind a second provider
    "coincidentally" succeeding is how such bugs never get found.
    """
    candidates = [text]

    fence = _FENCE.search(text)
    if fence:
        candidates.append(fence.group("body"))

    # Outermost object or array, for a model that prefixed an explanation.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped:
            continue
        for attempt in (stripped, _TRAILING_COMMA.sub(r"\1", stripped)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue

    raise SchemaValidationError(
        f"{provider_id} response",
        [f"not valid JSON after unwrapping: {text[:200]!r}"],
    )


def schema_instruction(schema: dict[str, Any]) -> str:
    """Prompt fragment pinning the output shape.

    Sent alongside the provider's native JSON mode rather than instead of it:
    free-tier models vary in how well they honour a response_format, and the
    two together are meaningfully more reliable than either alone.
    """
    return (
        "Respond with a single JSON value conforming exactly to this JSON Schema.\n"
        "Emit no prose, no markdown fences, and no fields the schema does not define.\n\n"
        f"{json.dumps(schema, indent=2)}"
    )
