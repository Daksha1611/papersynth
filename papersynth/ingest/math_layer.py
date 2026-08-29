"""Detecting and recovering unreliable math from a PDF text layer (section 8.1).

This is the R-01 mitigation. Corrupted equations producing silently wrong specs
is the highest likelihood-and-impact risk in the register, and the three
behaviours that respond to it - the confidence penalty, the symbol_check
corruption heuristic, and the document fidelity flag - all key off a fidelity
value that nothing produced until this module existed. A mitigation nothing
triggers is worse than none, because a reader trusts it.

Detection is deliberately separate from recovery. Recognising that a text layer
mangled an equation needs no model, no OCR and no dependencies, so it always
runs; re-recognising the equation needs a vision model that pulls in torch, so
it is optional. Splitting them means the risk is flagged on every install
rather than only where a two-gigabyte dependency happens to be present.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from papersynth.core.document import RawEquation

#: Share of characters that must be unusual before the text is treated as
#: mangled. Real display math is symbol-dense, so this counts characters that
#: do not belong in LaTeX at all rather than mathematical symbols.
SUSPECT_GLYPH_RATIO = 0.08

#: Below this length a display equation is almost certainly a fragment of one.
MIN_DISPLAY_LENGTH = 6

_DELIMITERS = (("(", ")"), ("[", "]"), ("{", "}"))
_ESCAPED = re.compile(r"\\[(){}\[\]]")
_LEFT_RIGHT = re.compile(r"\\(left|right)\b")

#: Characters a PDF text layer emits when it could not map a glyph.
_REPLACEMENT = {"\ufffd", "\ufeff"}


@dataclass
class MathQuality:
    """Whether an equation's text can be trusted, and why not."""

    reliable: bool = True
    reasons: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> MathQuality:
        self.reliable = False
        self.reasons.append(reason)
        return self


def assess(latex: str) -> MathQuality:
    """Judge a text-layer equation, using only the string itself.

    Three signals, all from section 8.1: unbalanced delimiters, glyphs the
    extractor could not map, and a density of unusable characters. Each is
    something a correct LaTeX equation does not have, so a failure means the
    text is damaged rather than that the mathematics is unusual.
    """
    quality = MathQuality()
    stripped = latex.strip()

    if not stripped:
        return quality.fail("equation text is empty")

    if len(stripped) < MIN_DISPLAY_LENGTH:
        quality.fail(f"only {len(stripped)} characters; likely a fragment")

    unmatched = _unbalanced(stripped)
    if unmatched:
        quality.fail(f"unbalanced {unmatched}")

    if _LEFT_RIGHT.findall(stripped):
        lefts = len(re.findall(r"\\left\b", stripped))
        rights = len(re.findall(r"\\right\b", stripped))
        if lefts != rights:
            quality.fail(f"{lefts} \\left against {rights} \\right")

    bad = _unmappable(stripped)
    if bad:
        quality.fail(f"unmappable glyphs: {''.join(sorted(bad)[:6])!r}")

    ratio = _unusable_ratio(stripped)
    if ratio > SUSPECT_GLYPH_RATIO:
        quality.fail(f"{ratio:.0%} of characters are unusable in LaTeX")

    return quality


def _unbalanced(text: str) -> str | None:
    """Which delimiter pair does not balance, ignoring escaped literals."""
    cleaned = _ESCAPED.sub("", text)
    for opener, closer in _DELIMITERS:
        if cleaned.count(opener) != cleaned.count(closer):
            return f"{opener}{closer}"
    return None


def _unmappable(text: str) -> set[str]:
    """Characters a text layer emits when it could not identify a glyph."""
    found = set()
    for char in text:
        if char in _REPLACEMENT:
            found.add(char)
            continue
        category = unicodedata.category(char)
        # Co is the private use area, which is where a subsetted math font
        # lands when its encoding is missing. Cn is unassigned.
        if category in ("Co", "Cn") or (category == "Cc" and char not in "\t\n\r"):
            found.add(char)
    return found


def _unusable_ratio(text: str) -> float:
    """Share of characters that cannot appear in well-formed LaTeX.

    Mathematical symbols are explicitly not counted. An equation full of
    integrals and Greek is normal; an equation full of control characters and
    unassigned codepoints is damaged.
    """
    if not text:
        return 0.0
    unusable = sum(
        1
        for char in text
        if char in _REPLACEMENT or unicodedata.category(char) in ("Co", "Cn", "Cs")
    )
    return unusable / len(text)


@runtime_checkable
class MathRecoverer(Protocol):
    """Re-recognizes an equation from the page image."""

    available: bool

    def recover(self, equation: RawEquation, pdf_path: str | None) -> str | None:
        """Return recovered LaTeX, or None when recovery was not possible."""
        ...


class NullRecoverer:
    """The default. Detects, never recovers.

    Present so the detection path runs on every install. Recovery needs a
    vision model and a couple of gigabytes of dependencies, and gating the
    whole R-01 mitigation behind that would leave it inert on most machines -
    which is the state this module was written to fix.
    """

    available = False

    def recover(self, equation: RawEquation, pdf_path: str | None) -> str | None:
        return None


class Pix2TexRecoverer:
    """Optional backend using pix2tex. Requires the `math` extra."""

    def __init__(self) -> None:
        self.available = False
        self._model = None
        try:
            from pix2tex.cli import LatexOCR

            self._model = LatexOCR()
            self.available = True
        except Exception:
            self._model = None

    def recover(self, equation: RawEquation, pdf_path: str | None) -> str | None:
        if not self.available or pdf_path is None or equation.page is None:
            return None
        image = _rasterize(pdf_path, equation.page)
        if image is None:
            return None
        try:
            return str(self._model(image)) or None  # type: ignore[misc]
        except Exception:
            return None


def _rasterize(pdf_path: str, page: int, dpi: int = 300):  # type: ignore[no-untyped-def]
    """Render one page via poppler's pdftoppm, which is already required."""
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    if shutil.which("pdftoppm") is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        try:
            subprocess.run(
                [
                    "pdftoppm",
                    "-png",
                    "-r",
                    str(dpi),
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    pdf_path,
                    str(prefix),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        rendered = sorted(Path(tmp).glob("page*.png"))
        return Image.open(rendered[0]).copy() if rendered else None


def review(
    equations: list[RawEquation],
    *,
    recoverer: MathRecoverer | None = None,
    pdf_path: str | None = None,
) -> tuple[list[RawEquation], list[str]]:
    """Assess every text-layer equation, recovering where possible.

    Returns the equations with fidelity updated, and warnings naming what was
    found. LaTeX-native equations are skipped: they came from the author's own
    source, so there is no text layer to have mangled them.
    """
    reviewed: list[RawEquation] = []
    warnings: list[str] = []
    recoverer = recoverer or NullRecoverer()

    for equation in equations:
        if equation.source_fidelity != "text_layer":
            reviewed.append(equation)
            continue

        quality = assess(equation.latex)
        if quality.reliable:
            reviewed.append(equation)
            continue

        label = equation.label or f"p{equation.page or '?'}"
        recovered = recoverer.recover(equation, pdf_path)

        if recovered:
            reviewed.append(
                equation.model_copy(update={"latex": recovered, "source_fidelity": "ocr_recovered"})
            )
            warnings.append(
                f"equation {label}: {'; '.join(quality.reasons)} - re-recognized by OCR"
            )
        else:
            # Detected as damaged and not recovered. Distinct from
            # ocr_recovered on purpose: claiming OCR ran when it did not would
            # misdescribe how the text was obtained, and the two situations
            # need different follow-up - one wants the OCR checked, the other
            # wants the extra installed or LaTeX source found.
            reviewed.append(equation.model_copy(update={"source_fidelity": "text_layer_suspect"}))
            warnings.append(
                f"equation {label}: {'; '.join(quality.reasons)} - not recovered "
                f"({'OCR failed' if recoverer.available else 'no OCR backend installed'})"
            )

    return reviewed, warnings


def build_recoverer(enabled: bool = False) -> MathRecoverer:
    """The configured backend, falling back to detection-only."""
    if not enabled:
        return NullRecoverer()
    recoverer = Pix2TexRecoverer()
    return recoverer if recoverer.available else NullRecoverer()
