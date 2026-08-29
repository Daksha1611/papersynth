"""PDF ingestion via GROBID, with a pdftotext-only degraded path.

This is PaperCoder's exact chain (GROBID / s2orc-doc2json), adopted so that
JSON prepared for PaperCoder runs through PaperSynth unmodified and vice versa
(DD-04). Interoperability with the prior art is free; giving it up for a newer
parser would make benchmark comparison dishonest.

GROBID's math handling is the known weakness. Equations arrive as text-layer
approximations, which is why ``math_fidelity`` is recorded per document and why
the LaTeX path is preferred whenever e-print source exists.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from papersynth.core import ids
from papersynth.core.config import Settings, get_settings
from papersynth.core.document import (
    RawEquation,
    RawTable,
    Reference,
    StructuredDocument,
    TableCell,
)
from papersynth.core.errors import IngestError
from papersynth.ingest.base import DocumentBuilder
from papersynth.ingest.math_layer import MathRecoverer, review

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

#: A run of these characters in a text-layer line is a strong signal the line is
#: a mangled equation rather than prose.
_MATH_GLYPHS = set("∑∏∫√≤≥≠≈∈∀∃∂∇⊗⊕±×÷αβγδεθλμνπρστφχψωΓΔΘΛΞΠΣΦΨΩ^_{}")


class GrobidIngestor:
    """Sends a PDF to GROBID and parses the returned TEI XML."""

    method = "pdf"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def can_handle(self, ref: str) -> bool:
        return Path(ref).suffix.lower() == ".pdf" and Path(ref).exists()

    def is_alive(self) -> bool:
        try:
            response = httpx.get(f"{self.settings.grobid_url}/api/isalive", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def ingest(self, ref: str, paper_id: str | None = None) -> StructuredDocument:
        path = Path(ref)
        if not path.exists():
            raise IngestError(f"No such PDF: {ref}")

        sha = ids.file_sha256(str(path))
        pid = paper_id or f"sha256:{sha[:16]}"
        tei = self._call_grobid(path)
        return parse_tei(tei, paper_id=pid, sha256=sha, pdf_path=str(path))

    def _call_grobid(self, path: Path) -> str:
        url = f"{self.settings.grobid_url}/api/processFulltextDocument"
        try:
            with open(path, "rb") as handle:
                response = httpx.post(
                    url,
                    files={"input": (path.name, handle, "application/pdf")},
                    data={
                        "consolidateHeader": "1",
                        "segmentSentences": "0",
                        "includeRawCitations": "1",
                    },
                    timeout=self.settings.grobid_timeout_s,
                )
        except httpx.HTTPError as exc:
            raise IngestError(
                f"GROBID unreachable at {self.settings.grobid_url}: {exc}. "
                "Start it with: docker run --rm -d -p 8070:8070 lfoppiano/grobid:0.8.1"
            ) from exc

        if response.status_code != 200:
            raise IngestError(
                f"GROBID returned {response.status_code} for {path.name}: {response.text[:200]}"
            )
        return response.text


class PdftotextIngestor:
    """Degraded fallback used when GROBID is unavailable.

    Produces no section structure worth the name and no reliable math, so it
    records a warning and marks the document accordingly. It exists so a run
    can proceed with a recorded caveat rather than failing outright (NFR-09) -
    never as a silent substitute for the real path.
    """

    method = "pdf_no_grobid"

    def can_handle(self, ref: str) -> bool:
        return Path(ref).suffix.lower() == ".pdf" and shutil.which("pdftotext") is not None

    def ingest(self, ref: str, paper_id: str | None = None) -> StructuredDocument:
        path = Path(ref)
        if shutil.which("pdftotext") is None:
            raise IngestError("pdftotext not found; install poppler-utils")

        try:
            result = subprocess.run(
                ["pdftotext", "-layout", str(path), "-"],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise IngestError(f"pdftotext failed on {path.name}: {exc}") from exc

        sha = ids.file_sha256(str(path))
        pid = paper_id or f"sha256:{sha[:16]}"

        builder = DocumentBuilder(
            paper_id=pid,
            title=path.stem,
            ingest_method="pdf_no_grobid",
            sha256=sha,
            math_fidelity="text_layer",
        )
        builder.warn(
            "ingested without GROBID: no section structure, degraded math fidelity. "
            "Equation-derived claims from this document should not be trusted."
        )
        # No <formula> tagging exists on this path, so any math present is
        # inline in prose and unverifiable. Saying so is the point of R-09.
        builder.math_fidelity = "text_layer_suspect"

        section_index = builder.add_section("Body")
        for page_number, page in enumerate(result.stdout.split("\f"), start=1):
            for para in re.split(r"\n\s*\n", page):
                builder.add_paragraph(section_index, para, page=page_number)

        doc = builder.build()
        if not doc.sections:
            raise IngestError(f"{path.name} has no extractable text layer; it may be a scan")
        return doc


def parse_tei(
    tei_xml: str,
    *,
    paper_id: str,
    sha256: str,
    pdf_path: str | None = None,
    recoverer: MathRecoverer | None = None,
) -> StructuredDocument:
    """Parse GROBID TEI into a StructuredDocument."""
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError as exc:
        raise IngestError(f"GROBID returned unparseable TEI for {paper_id}: {exc}") from exc

    builder = DocumentBuilder(
        paper_id=paper_id,
        title=_tei_title(root) or paper_id,
        ingest_method="pdf",
        sha256=sha256,
        venue=_tei_venue(root),
        year=_tei_year(root),
        math_fidelity="text_layer",
    )

    abstract = _tei_abstract(root)
    if abstract:
        index = builder.add_section("Abstract")
        builder.add_paragraph(index, abstract)

    body = root.find(".//tei:text/tei:body", TEI_NS)
    if body is not None:
        for div in body.findall("tei:div", TEI_NS):
            head = div.find("tei:head", TEI_NS)
            title = _tei_text(head) if head is not None else "Untitled section"
            number = head.get("n") if head is not None else None
            section_index = builder.add_section(f"{number} {title}".strip() if number else title)

            for paragraph in div.findall("tei:p", TEI_NS):
                text = _tei_text(paragraph)
                if not text:
                    continue
                builder.add_paragraph(section_index, text)
                for latex in _suspect_math_lines(text):
                    builder.add_equation(
                        RawEquation(
                            label=None,
                            latex=latex,
                            section_index=section_index,
                            paragraph_index=0,
                            source_fidelity="text_layer",
                        )
                    )

            for formula in div.findall(".//tei:formula", TEI_NS):
                latex = _tei_text(formula)
                if latex:
                    builder.add_equation(
                        RawEquation(
                            label=formula.get("{http://www.w3.org/XML/1998/namespace}id"),
                            latex=latex,
                            section_index=section_index,
                            paragraph_index=0,
                            source_fidelity="text_layer",
                        )
                    )

    for table in _tei_tables(root):
        builder.add_table(table)
    for reference in _tei_references(root):
        builder.add_reference(reference)

    if builder.math_fidelity == "text_layer" and builder.equation_count:
        builder.warn(
            "equations came from a PDF text layer, not LaTeX source. "
            "Re-ingest with --prefer-latex if e-print source exists (R-01)."
        )

    doc = builder.build()

    # The R-01 mitigation. Every text-layer equation is checked for damage, and
    # anything unreliable is re-recognized where a backend exists or flagged
    # where one does not. Running after build() means it sees the equations
    # with their final section indices.
    reviewed, math_warnings = review(doc.equations, recoverer=recoverer, pdf_path=pdf_path)
    doc.equations = reviewed
    doc.warnings.extend(math_warnings)
    degraded = {e.source_fidelity for e in reviewed} & {"ocr_recovered", "text_layer_suspect"}
    if degraded:
        doc.math_fidelity = sorted(degraded)[0]
    if not doc.sections:
        raise IngestError(f"GROBID produced no body text for {paper_id}")
    return doc


def _tei_text(element: ET.Element | None) -> str:
    """All descendant text, joined. GROBID nests <ref> inside <p> freely."""
    if element is None:
        return ""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def _tei_title(root: ET.Element) -> str | None:
    node = root.find(".//tei:titleStmt/tei:title", TEI_NS)
    return _tei_text(node) or None


def _tei_abstract(root: ET.Element) -> str | None:
    node = root.find(".//tei:profileDesc/tei:abstract", TEI_NS)
    return _tei_text(node) or None


def _tei_venue(root: ET.Element) -> str | None:
    node = root.find(".//tei:monogr/tei:title", TEI_NS)
    return _tei_text(node) or None


def _tei_year(root: ET.Element) -> int | None:
    for node in root.findall(".//tei:date", TEI_NS):
        when = node.get("when") or _tei_text(node)
        match = re.search(r"(19|20)\d{2}", when or "")
        if match:
            return int(match.group(0))
    return None


def _tei_tables(root: ET.Element) -> list[RawTable]:
    tables: list[RawTable] = []
    for figure in root.findall(".//tei:figure", TEI_NS):
        table_node = figure.find("tei:table", TEI_NS)
        if table_node is None:
            continue
        cells = [
            TableCell(row=r, col=c, text=_tei_text(cell))
            for r, row in enumerate(table_node.findall("tei:row", TEI_NS))
            for c, cell in enumerate(row.findall("tei:cell", TEI_NS))
        ]
        tables.append(
            RawTable(
                label=figure.get("{http://www.w3.org/XML/1998/namespace}id"),
                caption=_tei_text(figure.find("tei:figDesc", TEI_NS)),
                cells=cells,
            )
        )
    return tables


def _tei_references(root: ET.Element) -> list[Reference]:
    out: list[Reference] = []
    for i, bibl in enumerate(root.findall(".//tei:listBibl/tei:biblStruct", TEI_NS)):
        key = bibl.get("{http://www.w3.org/XML/1998/namespace}id") or f"b{i}"
        raw = _tei_text(bibl)[:500]
        title = _tei_text(bibl.find(".//tei:title", TEI_NS)) or None
        year_match = re.search(r"(19|20)\d{2}", raw)
        arxiv = re.search(r"(\d{4}\.\d{4,5})", raw)
        out.append(
            Reference(
                key=key,
                raw=raw,
                title=title,
                year=int(year_match.group(0)) if year_match else None,
                arxiv_id=arxiv.group(1) if arxiv else None,
            )
        )
    return out


def _suspect_math_lines(text: str, threshold: float = 0.12) -> list[str]:
    """Lines whose symbol density suggests a flattened equation.

    GROBID sometimes leaves display math inline in a <p> instead of tagging it
    as <formula>. Flagging those lines here means symbol_check gets a chance to
    catch the corruption downstream, rather than the fragment silently becoming
    prose that an extractor quotes as a sentence.
    """
    suspects = []
    for line in text.split("\n"):
        stripped = line.strip()
        if len(stripped) < 8:
            continue
        density = sum(ch in _MATH_GLYPHS for ch in stripped) / len(stripped)
        if density >= threshold:
            suspects.append(stripped)
    return suspects
