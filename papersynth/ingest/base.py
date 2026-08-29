"""Ingestor protocol and the shared document builder.

Every ingestor funnels through ``DocumentBuilder`` rather than constructing a
``StructuredDocument`` directly. That is deliberate: the builder is the single
place that guarantees section and paragraph indices match their positions, and
that guarantee is what makes span IDs stable across re-ingestion (section 8.1).
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from papersynth.core.document import (
    IngestMethod,
    Paragraph,
    RawAlgorithm,
    RawEquation,
    RawTable,
    Reference,
    Section,
    SourceFidelity,
    StructuredDocument,
)

#: Paragraphs shorter than this are almost always artifacts of PDF extraction -
#: stray page numbers, orphaned figure labels, running heads. Keeping them adds
#: noise to every downstream prompt without adding extractable content.
MIN_PARAGRAPH_CHARS = 20


@runtime_checkable
class Ingestor(Protocol):
    """Turns one input reference into one canonical document."""

    method: IngestMethod

    def can_handle(self, ref: str) -> bool:
        """True if this ingestor recognizes the reference form."""
        ...

    def ingest(self, ref: str) -> StructuredDocument:
        """Produce a StructuredDocument. Raises IngestError on failure."""
        ...


class DocumentBuilder:
    """Accumulates document parts, assigning positional indices as it goes."""

    def __init__(
        self,
        paper_id: str,
        *,
        title: str,
        ingest_method: IngestMethod,
        sha256: str,
        venue: str | None = None,
        year: int | None = None,
        math_fidelity: SourceFidelity = "text_layer",
    ) -> None:
        self.paper_id = paper_id
        self.title = title
        self.ingest_method: IngestMethod = ingest_method
        self.sha256 = sha256
        self.venue = venue
        self.year = year
        self.math_fidelity: SourceFidelity = math_fidelity

        self._sections: list[Section] = []
        self._equations: list[RawEquation] = []
        self._algorithms: list[RawAlgorithm] = []
        self._tables: list[RawTable] = []
        self._references: list[Reference] = []
        self._warnings: list[str] = []

    # -- construction ------------------------------------------------------

    def add_section(self, title: str) -> int:
        """Open a new section; returns its index."""
        index = len(self._sections)
        self._sections.append(Section(index=index, title=title.strip() or f"Section {index}"))
        return index

    def add_paragraph(self, section_index: int, text: str, page: int | None = None) -> int | None:
        """Append a paragraph to a section. Returns its index, or None if dropped.

        Text is normalized here and nowhere else, so the characters a span
        addresses are exactly the characters stored.
        """
        cleaned = normalize_text(text)
        if len(cleaned) < MIN_PARAGRAPH_CHARS:
            return None
        section = self._sections[section_index]
        index = len(section.paragraphs)
        section.paragraphs.append(Paragraph(index=index, text=cleaned, page=page))
        return index

    def add_equation(self, equation: RawEquation) -> None:
        self._equations.append(equation)
        # Either damaged value degrades the whole document's math fidelity;
        # downstream confidence penalties key off this.
        if equation.source_fidelity in ("ocr_recovered", "text_layer_suspect"):
            self.math_fidelity = equation.source_fidelity

    def add_algorithm(self, algorithm: RawAlgorithm) -> None:
        self._algorithms.append(algorithm)

    def add_table(self, table: RawTable) -> None:
        self._tables.append(table)

    def add_reference(self, reference: Reference) -> None:
        self._references.append(reference)

    def warn(self, message: str) -> None:
        """Record a non-fatal problem. Surfaced in the run manifest (NFR-09)."""
        if message not in self._warnings:
            self._warnings.append(message)

    @property
    def equation_count(self) -> int:
        return len(self._equations)

    # -- finalization ------------------------------------------------------

    def build(self) -> StructuredDocument:
        """Drop empty sections, renumber, and validate."""
        kept = [s for s in self._sections if s.paragraphs]
        remap: dict[int, int] = {}
        renumbered: list[Section] = []
        for new_index, section in enumerate(kept):
            remap[section.index] = new_index
            renumbered.append(
                Section(index=new_index, title=section.title, paragraphs=section.paragraphs)
            )

        if not renumbered:
            self.warn("no sections survived ingestion; document has no addressable text")

        return StructuredDocument(
            paper_id=self.paper_id,
            title=self.title,
            venue=self.venue,
            year=self.year,
            ingest_method=self.ingest_method,
            sha256=self.sha256,
            math_fidelity=self.math_fidelity,
            sections=renumbered,
            equations=[self._remap_equation(e, remap) for e in self._equations],
            algorithms_raw=[self._remap_algorithm(a, remap) for a in self._algorithms],
            tables=self._tables,
            references=self._references,
            warnings=self._warnings,
        )

    @staticmethod
    def _remap_equation(eq: RawEquation, remap: dict[int, int]) -> RawEquation:
        return eq.model_copy(update={"section_index": remap.get(eq.section_index, 0)})

    @staticmethod
    def _remap_algorithm(alg: RawAlgorithm, remap: dict[int, int]) -> RawAlgorithm:
        return alg.model_copy(update={"section_index": remap.get(alg.section_index, 0)})


_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    " ": " ",
}

_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_WHITESPACE = re.compile(r"[ \t]+")
_NEWLINES = re.compile(r"\n{2,}")


def normalize_text(text: str) -> str:
    """Canonicalize extracted text once, at ingest.

    PDF text layers routinely contain ligature codepoints and words split
    across a line break with a hyphen. Left alone, both defeat verbatim
    matching in ``citation_trace`` - the model quotes "efficient" while the
    document holds "eﬃ-\\ncient" - so a claim that was correctly extracted
    gets rejected. Normalizing at the single point of entry keeps the stored
    characters and the addressable characters identical.
    """
    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = _NEWLINES.sub("\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()
