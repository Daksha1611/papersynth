"""Per-provider quota accounting against known free-tier ceilings.

This exists because free tiers cap requests per *day*, not just per minute, and
a daily counter does not reset because you restarted the process. State is
persisted to disk so that six debugging runs in an afternoon see one shared
count rather than six independent ones that each believe the quota is untouched
(section 6.4.1).

The tracker predicts exhaustion rather than only reacting to it: the router
skips a provider it already knows is tapped out instead of spending a call to
rediscover that fact.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from papersynth.llm.base import Usage


@dataclass
class ProviderUsage:
    """One provider's counters for one UTC day."""

    day: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    spend_usd: float = 0.0
    #: Unix timestamp before which the provider should not be retried.
    blocked_until: float = 0.0
    rate_limit_hits: int = 0


@dataclass
class UsageSnapshot:
    """What `papersynth cost --by-provider` renders."""

    provider_id: str
    requests: int
    limit: int | None
    input_tokens: int
    output_tokens: int
    spend_usd: float
    exhausted: bool
    #: The ceiling the router actually enforces: limit * safety_margin.
    safe_limit: int | None = None
    blocked_for_s: float = 0.0
    rate_limit_hits: int = 0

    @property
    def headroom(self) -> int | None:
        """Calls still usable, against the ceiling the router enforces.

        Reporting against the raw limit would advertise headroom the router
        will refuse to spend, which is exactly the kind of number that sends
        you debugging a phantom problem.
        """
        if self.safe_limit is None:
            return None
        return max(0, self.safe_limit - self.requests)


class UsageTracker:
    """Tracks requests per provider per day and answers 'is this one tapped out?'."""

    def __init__(
        self,
        state_path: Path | None = None,
        *,
        limits: dict[str, int | None] | None = None,
        safety_margin: float = 0.9,
        persist: bool = True,
    ) -> None:
        self.state_path = state_path
        self.limits = limits or {}
        self.safety_margin = safety_margin
        self.persist = persist and state_path is not None
        self._lock = threading.Lock()
        self._state: dict[str, ProviderUsage] = {}
        self._load()

    # -- queries -----------------------------------------------------------

    def likely_exhausted(self, provider_id: str) -> bool:
        """True when the provider is rate-limit-blocked or at its safe ceiling.

        Self-throttling below the real ceiling leaves room for the calls already
        in flight and for a limit that turns out lower than documented - free
        tiers are adjusted without notice (R-13).
        """
        with self._lock:
            usage = self._today(provider_id)
            if usage.blocked_until > time.time():
                return True
            limit = self.limits.get(provider_id)
            if limit is None:
                return False
            return usage.requests >= int(limit * self.safety_margin)

    def headroom(self, provider_id: str) -> int | None:
        with self._lock:
            limit = self.limits.get(provider_id)
            if limit is None:
                return None
            return max(0, int(limit * self.safety_margin) - self._today(provider_id).requests)

    def snapshot(self, provider_ids: list[str] | None = None) -> list[UsageSnapshot]:
        with self._lock:
            ids = provider_ids or sorted(self._state.keys())
            now = time.time()
            out = []
            for provider_id in ids:
                usage = self._today(provider_id)
                limit = self.limits.get(provider_id)
                safe_limit = None if limit is None else int(limit * self.safety_margin)
                blocked = max(0.0, usage.blocked_until - now)
                at_ceiling = safe_limit is not None and usage.requests >= safe_limit
                out.append(
                    UsageSnapshot(
                        provider_id=provider_id,
                        requests=usage.requests,
                        limit=limit,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        spend_usd=usage.spend_usd,
                        exhausted=bool(blocked > 0 or at_ceiling),
                        safe_limit=safe_limit,
                        blocked_for_s=blocked,
                        rate_limit_hits=usage.rate_limit_hits,
                    )
                )
            return out

    # -- mutations ---------------------------------------------------------

    def record(self, provider_id: str, usage: Usage) -> None:
        with self._lock:
            entry = self._today(provider_id)
            entry.requests += 1
            entry.input_tokens += usage.input_tokens
            entry.output_tokens += usage.output_tokens
            self._save()

    def record_spend(self, provider_id: str, cost_usd: float) -> None:
        """Only ever called with a non-zero cost if a paid provider was added
        outside the default chain - every leg of that chain is free tier."""
        if cost_usd <= 0:
            return
        with self._lock:
            self._today(provider_id).spend_usd += cost_usd
            self._save()

    def mark_exhausted(self, provider_id: str, retry_after: float | None = None) -> None:
        """Block a provider after a 429. Default backoff is until the next UTC day."""
        with self._lock:
            entry = self._today(provider_id)
            entry.rate_limit_hits += 1
            if retry_after is not None:
                entry.blocked_until = max(entry.blocked_until, time.time() + retry_after)
            else:
                # No Retry-After header means we cannot tell a per-minute limit
                # from a per-day one. Assume the per-day case and also push the
                # request count to the ceiling, so headroom reporting is honest.
                entry.blocked_until = max(entry.blocked_until, _seconds_until_utc_midnight())
                limit = self.limits.get(provider_id)
                if limit is not None:
                    entry.requests = max(entry.requests, int(limit * self.safety_margin))
            self._save()

    def clear_block(self, provider_id: str) -> None:
        with self._lock:
            self._today(provider_id).blocked_until = 0.0
            self._save()

    def reset(self) -> None:
        with self._lock:
            self._state = {}
            self._save()

    # -- persistence -------------------------------------------------------

    def _today(self, provider_id: str) -> ProviderUsage:
        today = date.today().isoformat()
        entry = self._state.get(provider_id)
        if entry is None or entry.day != today:
            entry = ProviderUsage(day=today)
            self._state[provider_id] = entry
        return entry

    def _load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt quota file must not stop a run. Undercounting risks one
            # wasted call; refusing to start costs the whole run.
            return
        for provider_id, payload in raw.items():
            try:
                self._state[provider_id] = ProviderUsage(**payload)
            except TypeError:
                continue

    def _save(self) -> None:
        if not self.persist or not self.state_path:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({k: asdict(v) for k, v in self._state.items()}, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self.state_path)
        except OSError:
            pass


def _seconds_until_utc_midnight() -> float:
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return time.time() + (tomorrow - now).total_seconds()
