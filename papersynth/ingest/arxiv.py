"""arXiv fetching.

Prefers e-print source over the PDF whenever it exists, because LaTeX gives
``latex_native`` math fidelity and the PDF path cannot. Some submissions are
PDF-only (the author uploaded a PDF directly), so the PDF path is always the
fallback rather than an error.
"""

from __future__ import annotations

import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import httpx

from papersynth.core.config import Settings, get_settings
from papersynth.core.errors import InvalidPaperRef

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

#: Modern (2404.01234) and legacy (math.GT/0309136) identifier forms.
_MODERN = re.compile(r"(?P<id>\d{4}\.\d{4,5})(?:v(?P<version>\d+))?")
_LEGACY = re.compile(r"(?P<id>[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v(?P<version>\d+))?")

_USER_AGENT = "papersynth/0.1 (https://github.com/Daksha1611/papersynth)"

#: arXiv asks for roughly three seconds between API calls. Ignoring that earns
#: a 429, and a 429 during ingestion silently costs a whole paper - the run
#: continues with fewer sources than were asked for, which is a far worse
#: outcome than waiting.
_MIN_INTERVAL_S = 3.0

#: Attempts before giving up on a paper. arXiv's throttling clears quickly, so
#: a short wait usually recovers what would otherwise be a missing source.
_MAX_ATTEMPTS = 3

_last_request_at = 0.0
_throttle = threading.Lock()


def _polite_wait() -> None:
    """Space requests out, however many callers there are."""
    global _last_request_at
    with _throttle:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < _MIN_INTERVAL_S:
            time.sleep(_MIN_INTERVAL_S - elapsed)
        _last_request_at = time.monotonic()


def _get(url: str, **kwargs: object) -> httpx.Response:
    """GET with arXiv's rate limit respected and short throttles waited out."""
    last: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        _polite_wait()
        try:
            response = httpx.get(url, **kwargs)  # type: ignore[arg-type]
        except httpx.HTTPError as exc:
            last = exc
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            time.sleep(_MIN_INTERVAL_S * (attempt + 2))
            continue

        if response.status_code == 429 and attempt < _MAX_ATTEMPTS - 1:
            retry_after = response.headers.get("retry-after")
            try:
                wait = float(retry_after) if retry_after else _MIN_INTERVAL_S * (attempt + 2)
            except ValueError:
                wait = _MIN_INTERVAL_S * (attempt + 2)
            time.sleep(min(wait, 30.0))
            continue

        return response

    raise last if last else httpx.HTTPError("arXiv request failed")


@dataclass(frozen=True)
class ArxivMetadata:
    arxiv_id: str
    title: str
    year: int | None
    venue: str | None
    abstract: str


def normalize_arxiv_id(ref: str) -> str:
    """Accept an ID, an abs/pdf URL, or an ``arXiv:`` prefix. Strips version.

    The version is dropped so that a run citing 2504.17192v2 and one citing
    2504.17192 address the same paper. Pinning a version would be more precise
    but would fragment provenance across runs for no practical gain.
    """
    ref = ref.strip()
    for pattern in (_LEGACY, _MODERN):
        match = pattern.search(ref)
        if match:
            return match.group("id")
    raise InvalidPaperRef(
        f"{ref!r} is not a recognizable arXiv reference. "
        "Expected e.g. 1706.03762, arXiv:1706.03762, or an arxiv.org URL."
    )


def looks_like_arxiv_ref(ref: str) -> bool:
    if Path(ref).exists():
        return False
    return bool(_MODERN.search(ref) or _LEGACY.search(ref)) or "arxiv.org" in ref


class ArxivFetcher:
    """Downloads metadata and source for an arXiv paper."""

    def __init__(self, settings: Settings | None = None, cache_dir: Path | None = None) -> None:
        self.settings = settings or get_settings()
        self.cache_dir = cache_dir or (self.settings.cache_dir / "arxiv")

    def fetch_metadata(self, arxiv_id: str) -> ArxivMetadata:
        try:
            response = _get(
                self.settings.arxiv_api_url,
                params={"id_list": arxiv_id, "max_results": 1},
                timeout=60.0,
                headers={"User-Agent": _USER_AGENT},
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise InvalidPaperRef(f"Cannot reach the arXiv API for {arxiv_id}: {exc}") from exc

        root = ET.fromstring(response.text)
        entry = root.find("atom:entry", ATOM_NS)
        if entry is None or entry.find("atom:id", ATOM_NS) is None:
            raise InvalidPaperRef(f"arXiv has no record of {arxiv_id}")

        published = _text(entry.find("atom:published", ATOM_NS))
        year_match = re.match(r"(\d{4})", published or "")
        journal = _text(entry.find("arxiv:journal_ref", ATOM_NS))
        comment = _text(entry.find("arxiv:comment", ATOM_NS))

        return ArxivMetadata(
            arxiv_id=arxiv_id,
            title=re.sub(r"\s+", " ", _text(entry.find("atom:title", ATOM_NS)) or "").strip(),
            year=int(year_match.group(1)) if year_match else None,
            venue=journal or _venue_from_comment(comment),
            abstract=re.sub(r"\s+", " ", _text(entry.find("atom:summary", ATOM_NS)) or "").strip(),
        )

    def fetch_source(self, arxiv_id: str) -> Path | None:
        """Download the e-print tarball. Returns None when source is unavailable."""
        path = self._cached(arxiv_id, ".tar.gz")
        if path.exists():
            return path

        url = f"https://arxiv.org/e-print/{arxiv_id}"
        try:
            response = _get(
                url, timeout=180.0, follow_redirects=True, headers={"User-Agent": _USER_AGENT}
            )
        except httpx.HTTPError:
            return None
        if response.status_code != 200 or not response.content:
            return None

        # A PDF-only submission serves a PDF from the e-print endpoint.
        if response.content[:5] == b"%PDF-":
            return None

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return path

    def fetch_pdf(self, arxiv_id: str) -> Path:
        path = self._cached(arxiv_id, ".pdf")
        if path.exists():
            return path

        url = f"https://arxiv.org/pdf/{arxiv_id}"
        try:
            response = _get(
                url, timeout=180.0, follow_redirects=True, headers={"User-Agent": _USER_AGENT}
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise InvalidPaperRef(f"Cannot download the PDF for {arxiv_id}: {exc}") from exc

        if response.content[:5] != b"%PDF-":
            raise InvalidPaperRef(f"arXiv did not return a PDF for {arxiv_id}")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return path

    def _cached(self, arxiv_id: str, suffix: str) -> Path:
        return self.cache_dir / f"{arxiv_id.replace('/', '_')}{suffix}"


def _text(element: ET.Element | None) -> str | None:
    return element.text if element is not None and element.text else None


def _venue_from_comment(comment: str | None) -> str | None:
    """arXiv comments routinely carry the venue: 'Accepted at NeurIPS 2017'."""
    if not comment:
        return None
    match = re.search(
        r"\b(NeurIPS|NIPS|ICML|ICLR|AAAI|ACL|EMNLP|NAACL|CVPR|ICCV|ECCV|COLM|TMLR|JMLR)\b"
        r"\s*(\d{4})?",
        comment,
        re.IGNORECASE,
    )
    if not match:
        return None
    return " ".join(part for part in (match.group(1), match.group(2)) if part)
