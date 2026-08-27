"""LaTeX source ingestion - the preferred path.

When e-print source is available it beats every PDF parser, because equations
arrive as the author wrote them rather than as a text layer guessed at them.
This is the only path that yields ``latex_native`` math fidelity, which is why
``prefer_latex`` defaults to true (DD-04 trade-off mitigation).

The parser is deliberately regex-based rather than a full LaTeX engine. It only
needs structure - sections, display math, algorithm blocks, bibliography keys -
and a real engine would drag in a TeX distribution for no gain in extractable
content.
"""

from __future__ import annotations

import re
import tarfile
from pathlib import Path

from papersynth.core import ids
from papersynth.core.document import (
    RawAlgorithm,
    RawEquation,
    Reference,
    StructuredDocument,
)
from papersynth.core.errors import IngestError
from papersynth.ingest.base import DocumentBuilder

# Comments, but not an escaped \%
_COMMENT = re.compile(r"(?<!\\)%.*?$", re.MULTILINE)

_SECTION = re.compile(
    r"\\(?P<level>section|subsection|subsubsection)\*?\s*\{(?P<title>(?:[^{}]|\{[^{}]*\})*)\}"
)

_MATH_ENVS = ("equation", "align", "gather", "multline", "eqnarray", "flalign", "displaymath")
_EQUATION = re.compile(
    r"\\begin\{(?P<env>" + "|".join(_MATH_ENVS) + r")\*?\}(?P<body>.*?)\\end\{(?P=env)\*?\}",
    re.DOTALL,
)
_DISPLAY_MATH = re.compile(r"(?<!\\)\\\[(?P<body>.*?)(?<!\\)\\\]", re.DOTALL)

_ALGORITHM = re.compile(
    r"\\begin\{(?P<env>algorithm|algorithmic|algorithm2e)\*?\}(?P<body>.*?)"
    r"\\end\{(?P=env)\*?\}",
    re.DOTALL,
)

_LABEL = re.compile(r"\\label\{(?P<label>[^}]*)\}")
_CAPTION = re.compile(r"\\caption\{(?P<caption>(?:[^{}]|\{[^{}]*\})*)\}")
_TITLE = re.compile(r"\\title\s*\{(?P<title>(?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
_BIBITEM = re.compile(
    r"\\bibitem(?:\[[^\]]*\])?\s*\{(?P<key>[^}]*)\}(?P<body>.*?)(?=\\bibitem|\Z)", re.DOTALL
)
_ARXIV_IN_TEXT = re.compile(r"ar[Xx]iv[:\s]*(?P<id>\d{4}\.\d{4,5})")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")

_INPUT = re.compile(r"\\(?:input|include)\s*\{(?P<path>[^}]*)\}")

# Inline commands whose argument is the visible text; unwrapped so prose reads
# as prose rather than as markup that no extractor prompt should have to parse.
_UNWRAP = re.compile(
    r"\\(?:emph|textit|textbf|texttt|textsc|text|mbox|mathrm|textrm|underline)"
    r"\s*\{(?P<inner>[^{}]*)\}"
)
_CITE = re.compile(
    r"\\(?:cite|citep|citet|citealp|autocite)\*?(?:\[[^\]]*\])*\s*\{(?P<keys>[^}]*)\}"
)
_REF = re.compile(r"\\(?:ref|autoref|eqref|cref|Cref)\s*\{(?P<key>[^}]*)\}")
_INLINE_MATH = re.compile(r"(?<!\\)\$(?P<body>[^$]+?)(?<!\\)\$")
_DROP_ENVS = re.compile(
    r"\\begin\{(figure|table|tabular|thebibliography|abstract)\*?\}.*?"
    r"\\end\{\1\*?\}",
    re.DOTALL,
)
_STRAY_COMMAND = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?")


class LatexIngestor:
    """Ingests a .tex file, a directory of sources, or an e-print tarball."""

    method = "latex"

    def can_handle(self, ref: str) -> bool:
        path = Path(ref)
        if not path.exists():
            return False
        if path.is_dir():
            return any(path.rglob("*.tex"))
        return path.suffix in {".tex", ".gz", ".tgz", ".tar"}

    def ingest(self, ref: str, paper_id: str | None = None) -> StructuredDocument:
        path = Path(ref)
        if not path.exists():
            raise IngestError(f"No such LaTeX source: {ref}")

        source, sha = self._read_source(path)
        pid = paper_id or f"sha256:{sha[:16]}"
        return parse_latex(source, paper_id=pid, sha256=sha)

    def _read_source(self, path: Path) -> tuple[str, str]:
        if path.is_dir():
            main = self._pick_main_tex(list(path.rglob("*.tex")))
            return self._resolve_inputs(main, path), ids.file_sha256(str(main))
        if path.suffix == ".tex":
            return self._resolve_inputs(path, path.parent), ids.file_sha256(str(path))
        return self._read_tarball(path), ids.file_sha256(str(path))

    def _read_tarball(self, path: Path) -> str:
        try:
            with tarfile.open(path) as tar:
                members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".tex")]
                if not members:
                    raise IngestError(f"No .tex file inside {path.name}")
                # Concatenate every .tex, main file first, so \input'd sections
                # are present without having to resolve paths inside the archive.
                ordered = sorted(members, key=lambda m: (not self._looks_main(m.name), m.name))
                parts = []
                for member in ordered:
                    handle = tar.extractfile(member)
                    if handle is None:
                        continue
                    parts.append(handle.read().decode("utf-8", errors="replace"))
                return "\n".join(parts)
        except tarfile.TarError as exc:
            raise IngestError(f"Cannot read e-print archive {path.name}: {exc}") from exc

    @staticmethod
    def _looks_main(name: str) -> bool:
        stem = Path(name).stem.lower()
        return stem in {"main", "paper", "ms", "article", "root", "arxiv"}

    def _pick_main_tex(self, candidates: list[Path]) -> Path:
        if not candidates:
            raise IngestError("No .tex files found")
        for path in candidates:
            if "\\begin{document}" in path.read_text(encoding="utf-8", errors="replace"):
                return path
        return candidates[0]

    def _resolve_inputs(self, main: Path, root: Path, depth: int = 0) -> str:
        """Inline \\input and \\include so section order is preserved."""
        text = main.read_text(encoding="utf-8", errors="replace")
        if depth >= 3:
            return text

        def replace(match: re.Match[str]) -> str:
            target = match.group("path").strip()
            for candidate in (root / target, root / f"{target}.tex"):
                if candidate.is_file():
                    return self._resolve_inputs(candidate, root, depth + 1)
            return ""

        return _INPUT.sub(replace, text)


def parse_latex(source: str, *, paper_id: str, sha256: str) -> StructuredDocument:
    """Parse LaTeX source into a StructuredDocument."""
    source = _COMMENT.sub("", source)
    body = _extract_body(source)

    builder = DocumentBuilder(
        paper_id=paper_id,
        title=_extract_title(source) or paper_id,
        ingest_method="latex",
        sha256=sha256,
        math_fidelity="latex_native",
        year=_extract_year(source),
    )

    for reference in _extract_references(source):
        builder.add_reference(reference)

    # The abstract is dropped from prose (it is a float environment) but states
    # the objective more directly than any body section, so it is kept as its
    # own addressable section - matching what the GROBID path produces.
    abstract = _extract_abstract(source)
    if abstract:
        builder.add_paragraph(builder.add_section("Abstract"), abstract)

    segments = _split_sections(body)
    for title, chunk in segments:
        section_index = builder.add_section(title)
        _collect_floats(builder, chunk, section_index)
        prose = _strip_environments(chunk)
        for para_text in _split_paragraphs(prose):
            builder.add_paragraph(section_index, para_text)

    doc = builder.build()
    if not doc.sections:
        raise IngestError(
            f"LaTeX source for {paper_id} yielded no addressable text; "
            "the file may be a stub or use an unsupported document structure"
        )
    return doc


def _extract_body(source: str) -> str:
    match = re.search(
        r"\\begin\{document\}(?P<body>.*?)(?:\\end\{document\}|\Z)", source, re.DOTALL
    )
    return match.group("body") if match else source


def _extract_title(source: str) -> str | None:
    match = _TITLE.search(source)
    if not match:
        return None
    return _clean_inline(match.group("title")) or None


def _extract_year(source: str) -> int | None:
    """Year from \\date, or from the preamble - never from the bibliography.

    Scanning the whole source would happily return a cited work's year, and a
    wrong year is worse than no year: the `prefer_recent_peer_reviewed` policy
    rule keys off it, so a bibliography year could silently drive a real
    auto-resolution the wrong way. Absent beats guessed (ER-02).
    """
    date = re.search(r"\\date\s*\{(?P<date>[^{}]*)\}", source)
    if date:
        match = _YEAR.search(date.group("date"))
        if match:
            return int(match.group(0))

    preamble = source.split("\\begin{document}", 1)[0]
    match = _YEAR.search(preamble)
    return int(match.group(0)) if match else None


def _extract_abstract(source: str) -> str | None:
    match = re.search(r"\\begin\{abstract\}(?P<body>.*?)\\end\{abstract\}", source, re.DOTALL)
    if not match:
        return None
    return _clean_inline(match.group("body")) or None


def _extract_references(source: str) -> list[Reference]:
    out: list[Reference] = []
    for match in _BIBITEM.finditer(source):
        raw = _clean_inline(match.group("body"))[:500]
        arxiv = _ARXIV_IN_TEXT.search(raw)
        year = _YEAR.search(raw)
        out.append(
            Reference(
                key=match.group("key").strip(),
                raw=raw,
                year=int(year.group(0)) if year else None,
                arxiv_id=arxiv.group("id") if arxiv else None,
            )
        )
    return out


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split into (title, chunk). Text before the first heading becomes Preamble."""
    matches = list(_SECTION.finditer(body))
    if not matches:
        return [("Body", body)]

    segments: list[tuple[str, str]] = []
    lead = body[: matches[0].start()].strip()
    if lead:
        segments.append(("Preamble", lead))

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        title = _clean_inline(match.group("title"))
        segments.append((title or f"Section {i + 1}", body[match.end() : end]))
    return segments


def _collect_floats(builder: DocumentBuilder, chunk: str, section_index: int) -> None:
    """Pull equations and algorithm blocks out before prose normalization."""
    for match in _EQUATION.finditer(chunk):
        latex = match.group("body").strip()
        builder.add_equation(
            RawEquation(
                label=_first_label(latex),
                latex=_strip_labels(latex),
                section_index=section_index,
                paragraph_index=0,
                source_fidelity="latex_native",
            )
        )

    for match in _DISPLAY_MATH.finditer(chunk):
        latex = match.group("body").strip()
        if latex:
            builder.add_equation(
                RawEquation(
                    label=_first_label(latex),
                    latex=_strip_labels(latex),
                    section_index=section_index,
                    paragraph_index=0,
                    source_fidelity="latex_native",
                )
            )

    for match in _ALGORITHM.finditer(chunk):
        body = match.group("body")
        caption = _CAPTION.search(body)
        builder.add_algorithm(
            RawAlgorithm(
                label=_first_label(body),
                caption=_clean_inline(caption.group("caption")) if caption else None,
                body=body.strip(),
                section_index=section_index,
                paragraph_index=0,
            )
        )


def _strip_environments(chunk: str) -> str:
    """Remove float environments and display math from prose.

    Equations are captured separately as first-class objects; leaving their
    LaTeX inline would let an extractor quote raw markup as if it were a
    sentence, which then fails citation_trace against normalized prose.
    """
    chunk = _EQUATION.sub(" ", chunk)
    chunk = _DISPLAY_MATH.sub(" ", chunk)
    chunk = _ALGORITHM.sub(" ", chunk)
    chunk = _DROP_ENVS.sub(" ", chunk)
    return chunk


def _split_paragraphs(text: str) -> list[str]:
    cleaned = _clean_inline(text)
    return [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]


def _clean_inline(text: str) -> str:
    """Turn LaTeX prose into plain prose, preserving readable content."""
    for _ in range(3):  # nested \textbf{\emph{...}}
        text = _UNWRAP.sub(r"\g<inner>", text)
    text = _CITE.sub(lambda m: f"[{m.group('keys').split(',')[0].strip()}]", text)
    text = _REF.sub("", text)
    text = _INLINE_MATH.sub(lambda m: m.group("body").strip(), text)
    text = _LABEL.sub("", text)
    text = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", " ", text)
    text = _STRAY_COMMAND.sub(" ", text)
    text = text.replace("~", " ").replace("\\&", "&").replace("\\%", "%")
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _first_label(text: str) -> str | None:
    match = _LABEL.search(text)
    return match.group("label").strip() if match else None


def _strip_labels(latex: str) -> str:
    return re.sub(r"\s+", " ", _LABEL.sub("", latex)).strip()
