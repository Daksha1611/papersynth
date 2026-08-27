"""Ingestion: any input form to one canonical StructuredDocument (section 8.1).

Dispatch order encodes the fidelity preference. An arXiv reference resolves to
e-print source when it exists and falls back to the PDF when it does not; a
local PDF goes through GROBID and falls back to pdftotext only with a recorded
warning. Degrading is always visible, never silent (R-09).
"""

from __future__ import annotations

from pathlib import Path

from papersynth.core.config import Settings, get_settings
from papersynth.core.document import StructuredDocument
from papersynth.core.errors import IngestError, InvalidPaperRef
from papersynth.ingest.arxiv import ArxivFetcher, looks_like_arxiv_ref, normalize_arxiv_id
from papersynth.ingest.base import DocumentBuilder, Ingestor, normalize_text
from papersynth.ingest.latex import LatexIngestor, parse_latex
from papersynth.ingest.pdf import GrobidIngestor, PdftotextIngestor, parse_tei

__all__ = [
    "ArxivFetcher",
    "DocumentBuilder",
    "GrobidIngestor",
    "Ingestor",
    "LatexIngestor",
    "PdftotextIngestor",
    "ingest",
    "ingest_arxiv",
    "ingest_pdf",
    "normalize_arxiv_id",
    "normalize_text",
    "parse_latex",
    "parse_tei",
]


def ingest(
    ref: str,
    *,
    settings: Settings | None = None,
    prefer_latex: bool | None = None,
    allow_no_grobid: bool = False,
) -> StructuredDocument:
    """Ingest any supported reference: arXiv ID/URL, PDF path, or LaTeX source."""
    settings = settings or get_settings()

    if looks_like_arxiv_ref(ref):
        return ingest_arxiv(
            ref,
            settings=settings,
            prefer_latex=prefer_latex,
            allow_no_grobid=allow_no_grobid,
        )

    path = Path(ref)
    if not path.exists():
        raise InvalidPaperRef(f"{ref!r} is neither an arXiv reference nor an existing path.")

    if path.suffix.lower() == ".pdf":
        return ingest_pdf(path, settings=settings, allow_no_grobid=allow_no_grobid)

    latex = LatexIngestor()
    if latex.can_handle(str(path)):
        return latex.ingest(str(path))

    raise InvalidPaperRef(
        f"Cannot ingest {ref!r}: expected a .pdf, a .tex, a directory of LaTeX "
        "sources, or an e-print tarball."
    )


def ingest_arxiv(
    ref: str,
    *,
    settings: Settings | None = None,
    prefer_latex: bool | None = None,
    allow_no_grobid: bool = False,
) -> StructuredDocument:
    """Fetch and ingest an arXiv paper, preferring e-print source."""
    settings = settings or get_settings()
    want_latex = settings.prefer_latex if prefer_latex is None else prefer_latex

    arxiv_id = normalize_arxiv_id(ref)
    fetcher = ArxivFetcher(settings)
    metadata = fetcher.fetch_metadata(arxiv_id)

    if want_latex:
        source = fetcher.fetch_source(arxiv_id)
        if source is not None:
            try:
                doc = LatexIngestor().ingest(str(source), paper_id=arxiv_id)
                return _apply_metadata(doc, metadata)
            except IngestError as exc:
                # A malformed tarball is worth reporting, but it must not cost
                # us the paper when the PDF is right there.
                doc = _ingest_arxiv_pdf(fetcher, arxiv_id, settings, allow_no_grobid)
                doc.warnings.append(f"LaTeX source unusable, fell back to PDF: {exc}")
                return _apply_metadata(doc, metadata)

    doc = _ingest_arxiv_pdf(fetcher, arxiv_id, settings, allow_no_grobid)
    if want_latex:
        doc.warnings.append(
            "no e-print source on arXiv; math fidelity is text_layer, not latex_native"
        )
    return _apply_metadata(doc, metadata)


def ingest_pdf(
    path: Path | str,
    *,
    settings: Settings | None = None,
    paper_id: str | None = None,
    allow_no_grobid: bool = False,
) -> StructuredDocument:
    """Ingest a local PDF through GROBID, or pdftotext if explicitly allowed."""
    settings = settings or get_settings()
    grobid = GrobidIngestor(settings)

    if grobid.is_alive():
        return grobid.ingest(str(path), paper_id=paper_id)

    if not allow_no_grobid:
        raise IngestError(
            f"GROBID is not reachable at {settings.grobid_url}. Start it with:\n"
            "  docker run --rm -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.1\n"
            "Or pass --no-grobid to ingest with pdftotext only, which degrades "
            "math fidelity and records a warning on the document."
        )

    return PdftotextIngestor().ingest(str(path), paper_id=paper_id)


def _ingest_arxiv_pdf(
    fetcher: ArxivFetcher,
    arxiv_id: str,
    settings: Settings,
    allow_no_grobid: bool,
) -> StructuredDocument:
    pdf = fetcher.fetch_pdf(arxiv_id)
    return ingest_pdf(pdf, settings=settings, paper_id=arxiv_id, allow_no_grobid=allow_no_grobid)


def _apply_metadata(doc: StructuredDocument, metadata: object) -> StructuredDocument:
    """Overlay authoritative arXiv metadata onto whatever the parser guessed.

    The arXiv API knows the real title, year, and venue; a LaTeX \\title
    command or a PDF header is a guess at them.
    """
    from papersynth.ingest.arxiv import ArxivMetadata

    if not isinstance(metadata, ArxivMetadata):
        return doc
    if metadata.title:
        doc.title = metadata.title
    if metadata.year:
        doc.year = metadata.year
    if metadata.venue:
        doc.venue = metadata.venue
    return doc
