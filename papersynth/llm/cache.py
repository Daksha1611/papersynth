"""Prompt-hash response cache.

Unchanged papers plus unchanged prompts never re-call a provider. This matters
more than the fallback chain does (section 6.4.5): the cheapest call is the one
never made, and during development the same extraction runs many times over an
identical document.

The key includes the prompt template, the rendered prompt, and the model, so a
prompt edit or a model swap invalidates the affected entries automatically
(ER-10).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from papersynth.llm.base import Completion, Usage


class PromptCache:
    """Content-addressed cache of completions, one JSON file per key."""

    def __init__(self, root: Path | None, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled and root is not None
        if self.enabled and root is not None:
            root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Completion | None:
        if not self.enabled or self.root is None:
            return None
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            usage = payload.pop("usage", {})
            completion = Completion(**payload)
            completion.usage = Usage(**usage)
        except (json.JSONDecodeError, OSError, TypeError):
            return None
        # A cache hit is not a provider call. Marking it keeps the ledger's
        # cost and quota accounting honest.
        completion.cached = True
        return completion

    def put(self, key: str, completion: Completion) -> None:
        if not self.enabled or self.root is None:
            return
        payload = asdict(completion)
        payload["cached"] = False
        try:
            tmp = self.root / f"{key}.tmp"
            tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            tmp.replace(self.root / f"{key}.json")
        except (OSError, TypeError):
            pass

    def clear(self) -> int:
        if self.root is None or not self.root.exists():
            return 0
        removed = 0
        for path in self.root.glob("*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed
