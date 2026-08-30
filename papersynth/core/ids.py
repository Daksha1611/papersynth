"""Content-derived identifiers.

Every ID in the system is a pure function of the content it names. This is what
makes provenance survive across runs: re-ingesting the same PDF produces the
same span IDs, which produces the same claim IDs, which means a spec emitted
today can still be diffed against one emitted last month (FR-17).

Never introduce a random or time-based ID for anything that is referenced by
another artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _canonical(obj: Any) -> str:
    """Stable JSON: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def content_hash(*parts: Any) -> str:
    """Full sha256 hex of the canonical rendering of ``parts``."""
    payload = "\x1f".join(_canonical(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def quote_hash(text: str) -> str:
    """Hash of source text, stored instead of the text itself (R-12).

    Whitespace is normalized first so that a re-ingest whose line wrapping
    differs still verifies. Anything more aggressive would let genuinely
    different text hash equal, which would defeat citation_trace.
    """
    normalized = " ".join(text.split())
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def slugify(text: str, max_len: int = 40) -> str:
    """Lowercase, underscore-separated slug suitable for an ID component."""
    slug = _SLUG_RE.sub("_", text.strip().lower()).strip("_")
    return slug[:max_len].rstrip("_") or "unnamed"


def claim_id(paper_id: str, claim_type: str, span_id: str, payload: Any) -> str:
    """Stable claim ID: clm_ + 6 hex of (paper, type, span, payload)."""
    return "clm_" + content_hash(paper_id, claim_type, span_id, payload)[:6]


def span_id(paper_id: str, section_index: int, paragraph_index: int, char_offset: int) -> str:
    """``{paper_id}#s{sec}.p{para}.{offset}`` - the one addressing scheme."""
    return f"{paper_id}#s{section_index}.p{paragraph_index}.{char_offset}"


def scope_id(span: str) -> str:
    """The section a span belongs to: ``{paper_id}#s{sec}``.

    Claims that share a scope came from the same passage. That is the signal
    section 10.1 needs: several count-valued facts from one section of one
    paper - a study's sample sizes, say - are stages of one described
    procedure, not independent settings that happen to disagree.
    """
    head, _, tail = span.partition("#s")
    if not tail:
        return span
    return f"{head}#s{tail.split('.', 1)[0]}"


def cluster_id(concept_type: str, canonical_name: str) -> str:
    return f"cnc_{slugify(concept_type)[:4]}_{slugify(canonical_name)}"


def contradiction_id(cluster: str, conflict_type: str, positions: list[str]) -> str:
    return "ctr_" + content_hash(cluster, conflict_type, sorted(positions))[:8]


def resolution_id(contradiction: str) -> str:
    return "res_" + contradiction.removeprefix("ctr_")


def gap_id(component_id: str | None, field: str) -> str:
    return "gap_" + content_hash(component_id or "global", field)[:8]


def component_id(name: str) -> str:
    return f"cmp_{slugify(name)}"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_hash(template: str, rendered: str, model: str) -> str:
    """Cache key for an LLM call (PAPERSYNTH_CACHE_BY_PROMPT_HASH)."""
    return content_hash(template, rendered, model)[:32]
