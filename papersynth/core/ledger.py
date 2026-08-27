"""Append-only accounting of every LLM call (NFR-06).

One JSONL line per call, written as it happens rather than buffered, so a run
that crashes still leaves a complete record up to the failure. The ledger is
the first thing to read when a spec looks wrong: it says which provider served
each call, under which prompt hash, and whether a fallback fired.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from papersynth.core.models import utcnow


@dataclass
class LedgerEntry:
    ts: str
    stage: str
    call_id: str
    provider_id: str
    model: str
    prompt_hash: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    cached: bool = False
    paper_id: str | None = None
    extractor: str | None = None
    #: Set when this call happened only because an earlier provider fell through.
    fallback_from: str | None = None
    fallback_reason: str | None = None
    error: str | None = None


@dataclass
class LedgerSummary:
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    errors: int = 0
    fallbacks: int = 0
    by_provider: dict[str, int] = field(default_factory=dict)
    by_stage: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Ledger:
    """Writes ledger.jsonl for one run."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._entries: list[LedgerEntry] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, **kwargs: Any) -> LedgerEntry:
        entry = LedgerEntry(ts=utcnow(), **kwargs)
        with self._lock:
            self._entries.append(entry)
            if self.path is not None:
                try:
                    with open(self.path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(asdict(entry), separators=(",", ":")) + "\n")
                except OSError:
                    # Losing an audit line must not abort a run mid-stage; the
                    # in-memory copy still backs the end-of-run summary.
                    pass
        return entry

    @property
    def entries(self) -> list[LedgerEntry]:
        with self._lock:
            return list(self._entries)

    def summary(self) -> LedgerSummary:
        summary = LedgerSummary()
        by_provider: dict[str, int] = defaultdict(int)
        by_stage: dict[str, int] = defaultdict(int)

        for entry in self.entries:
            summary.calls += 1
            summary.input_tokens += entry.input_tokens
            summary.output_tokens += entry.output_tokens
            summary.cost_usd += entry.cost_usd
            summary.latency_ms += entry.latency_ms
            by_provider[entry.provider_id] += 1
            by_stage[entry.stage] += 1
            if entry.cached:
                summary.cached_calls += 1
            if entry.error:
                summary.errors += 1
            if entry.fallback_from:
                summary.fallbacks += 1

        summary.by_provider = dict(by_provider)
        summary.by_stage = dict(by_stage)
        return summary

    @classmethod
    def load(cls, path: Path) -> Ledger:
        """Re-read a ledger from disk, for `papersynth cost` on a finished run."""
        ledger = cls(path=None)
        if not path.exists():
            return ledger
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ledger._entries.append(LedgerEntry(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
        ledger.path = path
        return ledger
