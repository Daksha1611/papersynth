"""Scripted provider for tests and cassette replay (section 14.4).

Non-determinism is handled by recording rather than by mocking behaviour: CI
replays request-hash -> response pairs, so the suite is deterministic, offline,
and free. A live nightly job re-records and diffs the resulting specs, which is
what turns prompt or model drift into a visible signal (R-06).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from papersynth.core import ids
from papersynth.core.errors import ProviderError
from papersynth.llm.base import Completion, Usage


class StubProvider:
    """Returns canned responses. Raises rather than inventing one."""

    def __init__(
        self,
        responses: Sequence[Any] | Callable[[str], Any] | None = None,
        *,
        provider_id: str = "stub",
        model: str = "stub-model",
        by_hash: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.model = model
        self._responses = list(responses) if isinstance(responses, Sequence) else responses
        self._by_hash = by_hash or {}
        self._error = error
        self._index = 0
        #: Every prompt this provider was asked to serve, for test assertions.
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    def complete(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> Completion:
        self.prompts.append(prompt)
        self.systems.append(system)

        if self._error is not None:
            raise self._error

        payload = self._resolve(prompt)
        text = payload if isinstance(payload, str) else json.dumps(payload)

        return Completion(
            text=text,
            provider_id=self.provider_id,
            model=self.model,
            usage=Usage(input_tokens=len(prompt) // 4, output_tokens=len(text) // 4),
            cost_usd=0.0,
            latency_ms=0.1,
            finish_reason="stop",
            parsed=None if schema is None else (payload if not isinstance(payload, str) else None),
        )

    def _resolve(self, prompt: str) -> Any:
        key = ids.content_hash(prompt)[:16]
        if key in self._by_hash:
            return self._by_hash[key]

        if callable(self._responses):
            return self._responses(prompt)

        if isinstance(self._responses, list):
            if self._index >= len(self._responses):
                raise ProviderError(
                    f"StubProvider ran out of scripted responses at call "
                    f"{self._index + 1}; the test scripted {len(self._responses)}."
                )
            response = self._responses[self._index]
            self._index += 1
            return response

        raise ProviderError(
            "StubProvider has no response for this prompt and no fallback was "
            "configured. Scripting the response explicitly keeps the test honest."
        )

    # -- cassette helpers --------------------------------------------------

    @classmethod
    def from_cassette(cls, path: Path, **kwargs: Any) -> StubProvider:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(by_hash=payload, **kwargs)

    @property
    def call_count(self) -> int:
        return len(self.prompts)

    def assert_no_prompt_contains_multiple_paper_ids(self, paper_ids: Sequence[str]) -> None:
        """ER-09: an extractor must never see two papers at once.

        If extraction were corpus-aware, a model holding Paper A's learning rate
        in context while extracting Paper B's would drift toward agreement, and
        genuine contradictions would vanish before the detector ever ran.
        """
        for prompt in self.prompts:
            present = [pid for pid in paper_ids if pid in prompt]
            if len(present) > 1:
                raise AssertionError(
                    f"extractor isolation violated (ER-09): a single prompt referenced {present}"
                )
