"""The canonical document representation every ingestor must produce.

One rule governs this module: span IDs are content-derived and stable
(section 8.1). Re-ingesting the same source must produce the same span IDs, or
provenance breaks across runs and the whole traceability guarantee (NFR-01)
evaporates.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from papersynth.core import ids

IngestMethod = Literal["latex", "pdf", "pdf_no_grobid", "prepared_json"]
#: How an equation's LaTeX was obtained. text_layer_suspect is text the math
#: layer flagged as damaged and could not recover; it earns the same confidence
#: penalty as ocr_recovered but records honestly that no OCR ran.
SourceFidelity = Literal["latex_native", "text_layer", "text_layer_suspect", "ocr_recovered"]


class Span(BaseModel):
    """A resolved character range within a paragraph."""

    span_id: str
    paper_id: str
    section_index: int
    paragraph_index: int
    char_start: int
    char_end: int
    text: str
    section_title: str
    page: int | None = None

    @property
    def quote_hash(self) -> str:
        return ids.quote_hash(self.text)


class Paragraph(BaseModel):
    index: int
    text: str
    page: int | None = None

    def span_for(self, paper_id: str, section_index: int, start: int, end: int) -> tuple[int, int]:
        """Clamp a requested range to this paragraph's bounds."""
        start = max(0, min(start, len(self.text)))
        end = max(start, min(end, len(self.text)))
        return start, end


class Section(BaseModel):
    index: int
    title: str
    paragraphs: list[Paragraph] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)


class RawEquation(BaseModel):
    """An equation as found during ingestion, before extraction interprets it."""

    label: str | None = None
    latex: str
    section_index: int
    paragraph_index: int
    char_start: int = 0
    char_end: int = 0
    source_fidelity: SourceFidelity = "text_layer"
    page: int | None = None


class RawAlgorithm(BaseModel):
    """An algorithm block as found during ingestion (float/environment), unparsed."""

    label: str | None = None
    caption: str | None = None
    body: str
    section_index: int
    paragraph_index: int
    page: int | None = None


class TableCell(BaseModel):
    row: int
    col: int
    text: str


class RawTable(BaseModel):
    label: str | None = None
    caption: str | None = None
    cells: list[TableCell] = Field(default_factory=list)
    section_index: int = 0
    page: int | None = None

    def as_text(self) -> str:
        rows: dict[int, list[str]] = {}
        for cell in sorted(self.cells, key=lambda c: (c.row, c.col)):
            rows.setdefault(cell.row, []).append(cell.text)
        return "\n".join(" | ".join(r) for r in rows.values())


class Reference(BaseModel):
    """A bibliography entry, used by reference_trace (section 10.2)."""

    key: str
    raw: str
    title: str | None = None
    year: int | None = None
    arxiv_id: str | None = None


class StructuredDocument(BaseModel):
    """The one representation stages 1+ are allowed to see."""

    paper_id: str
    title: str
    venue: str | None = None
    year: int | None = None
    ingest_method: IngestMethod
    sha256: str
    math_fidelity: SourceFidelity = "text_layer"

    sections: list[Section] = Field(default_factory=list)
    equations: list[RawEquation] = Field(default_factory=list)
    algorithms_raw: list[RawAlgorithm] = Field(default_factory=list)
    tables: list[RawTable] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sections_are_indexed_consistently(self) -> StructuredDocument:
        for i, section in enumerate(self.sections):
            if section.index != i:
                raise ValueError(
                    f"section {i} declares index {section.index}; "
                    "section.index must equal its position for span IDs to be stable"
                )
            for j, para in enumerate(section.paragraphs):
                if para.index != j:
                    raise ValueError(
                        f"section {i} paragraph {j} declares index {para.index}; "
                        "paragraph.index must equal its position"
                    )
        return self

    # -- span addressing ---------------------------------------------------

    def make_span(
        self, section_index: int, paragraph_index: int, char_start: int, char_end: int
    ) -> Span:
        """Build a Span, clamping the range to the paragraph's actual bounds."""
        section = self._section(section_index)
        para = self._paragraph(section, paragraph_index)
        start, end = para.span_for(self.paper_id, section_index, char_start, char_end)
        return Span(
            span_id=ids.span_id(self.paper_id, section_index, paragraph_index, start),
            paper_id=self.paper_id,
            section_index=section_index,
            paragraph_index=paragraph_index,
            char_start=start,
            char_end=end,
            text=para.text[start:end],
            section_title=section.title,
            page=para.page,
        )

    def resolve_span(self, span_id: str, char_end: int | None = None) -> Span | None:
        """Resolve a span ID back to its text. Returns None when unresolvable.

        ``citation_trace`` treats None as a hard rejection (ER-01), so this must
        never raise for merely malformed input.
        """
        parsed = parse_span_id(span_id)
        if parsed is None:
            return None
        paper_id, section_index, paragraph_index, offset = parsed
        if paper_id != self.paper_id:
            return None
        try:
            section = self._section(section_index)
            para = self._paragraph(section, paragraph_index)
        except (IndexError, KeyError):
            return None
        if offset > len(para.text):
            return None
        end = len(para.text) if char_end is None else min(char_end, len(para.text))
        return Span(
            span_id=span_id,
            paper_id=self.paper_id,
            section_index=section_index,
            paragraph_index=paragraph_index,
            char_start=offset,
            char_end=end,
            text=para.text[offset:end],
            section_title=section.title,
            page=para.page,
        )

    def find_span(self, needle: str, *, section_filter: list[int] | None = None) -> Span | None:
        """Locate ``needle`` verbatim and return its span. Used to anchor a
        claim the model quoted but did not correctly address."""
        target = " ".join(needle.split())
        if not target:
            return None
        for section in self.sections:
            if section_filter is not None and section.index not in section_filter:
                continue
            for para in section.paragraphs:
                idx = _normalized_find(para.text, target)
                if idx is not None:
                    start, length = idx
                    return self.make_span(section.index, para.index, start, start + length)
        return None

    def sections_matching(self, pattern: str) -> list[Section]:
        """Sections whose title matches a regex, case-insensitively."""
        import re

        rx = re.compile(pattern, re.IGNORECASE)
        return [s for s in self.sections if rx.search(s.title)]

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"## {s.title}\n{s.text}" for s in self.sections)

    def _section(self, index: int) -> Section:
        if index < 0 or index >= len(self.sections):
            raise IndexError(f"no section {index} in {self.paper_id}")
        return self.sections[index]

    @staticmethod
    def _paragraph(section: Section, index: int) -> Paragraph:
        if index < 0 or index >= len(section.paragraphs):
            raise IndexError(f"no paragraph {index} in section {section.index}")
        return section.paragraphs[index]


def parse_span_id(span_id: str) -> tuple[str, int, int, int] | None:
    """``paper#s1.p2.30`` -> ``("paper", 1, 2, 30)``; None if malformed."""
    import re

    match = re.fullmatch(r"(?P<paper>[^#]+)#s(?P<sec>\d+)\.p(?P<para>\d+)\.(?P<off>\d+)", span_id)
    if not match:
        return None
    return (
        match.group("paper"),
        int(match.group("sec")),
        int(match.group("para")),
        int(match.group("off")),
    )


def _normalized_find(haystack: str, normalized_needle: str) -> tuple[int, int] | None:
    """Find ``normalized_needle`` in ``haystack`` tolerating whitespace runs.

    Returns (start_offset, length) in *original* haystack coordinates, so the
    resulting span still addresses real characters in the document.
    """
    # Map each non-space char in haystack to its original index.
    condensed_chars: list[str] = []
    original_index: list[int] = []
    prev_space = False
    for i, ch in enumerate(haystack):
        if ch.isspace():
            if not prev_space and condensed_chars:
                condensed_chars.append(" ")
                original_index.append(i)
            prev_space = True
        else:
            condensed_chars.append(ch)
            original_index.append(i)
            prev_space = False
    condensed = "".join(condensed_chars).strip()
    if not condensed:
        return None
    # Recompute index offset lost to the leading strip.
    lead = len("".join(condensed_chars)) - len("".join(condensed_chars).lstrip())
    pos = condensed.find(normalized_needle)
    if pos < 0:
        return None
    start_c = pos + lead
    end_c = start_c + len(normalized_needle) - 1
    if end_c >= len(original_index):
        return None
    start = original_index[start_c]
    end = original_index[end_c] + 1
    return start, end - start
