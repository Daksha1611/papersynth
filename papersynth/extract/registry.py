"""Extractor registry (FR-14, NFR-07).

Adding a claim type is a schema file, a prompt, and a ~30-line class. No
pipeline code changes, and third parties register through the
``papersynth.extractors`` entry point without vendoring anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from papersynth.core.document import StructuredDocument
from papersynth.core.errors import PaperSynthError
from papersynth.extract.base import ExtractionResult, LLMExtractor

if TYPE_CHECKING:
    from papersynth.llm.base import LLMProvider

EXTRACTORS: dict[str, type[LLMExtractor]] = {}
_ENTRY_POINTS_LOADED = False


def register(cls: type[LLMExtractor]) -> type[LLMExtractor]:
    """Class decorator. Self-registers on import."""
    if not cls.claim_type:
        raise PaperSynthError(f"{cls.__name__} does not declare a claim_type")
    EXTRACTORS[cls.claim_type] = cls
    return cls


def load_entry_points() -> None:
    """Discover third-party extractors declared under `papersynth.extractors`."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True

    from importlib.metadata import entry_points

    for entry in entry_points(group="papersynth.extractors"):
        try:
            candidate = entry.load()
        except Exception:
            continue
        if isinstance(candidate, type) and issubclass(candidate, LLMExtractor):
            register(candidate)


def available() -> list[str]:
    _ensure_builtins()
    return sorted(EXTRACTORS)


def build(
    names: list[str],
    provider: LLMProvider,
    *,
    temperature: float = 0.0,
) -> list[LLMExtractor]:
    """Instantiate the named extractors, rejecting unknown names loudly."""
    _ensure_builtins()
    load_entry_points()

    unknown = [n for n in names if n not in EXTRACTORS]
    if unknown:
        raise PaperSynthError(
            f"Unknown extractor(s): {', '.join(unknown)}. Available: {', '.join(available())}"
        )
    return [EXTRACTORS[name](provider, temperature=temperature) for name in names]


def run_all(
    doc: StructuredDocument,
    extractors: list[LLMExtractor],
) -> ExtractionResult:
    """Run every extractor over one document.

    One extractor failing does not abort the paper (NFR-09); the failure is
    recorded as a warning and the remaining extractors still run. A partial
    claim set with a visible gap beats no claim set at all.
    """
    result = ExtractionResult()
    for extractor in extractors:
        try:
            result.extend(extractor.extract(doc))
        except PaperSynthError as exc:
            result.warnings.append(f"{extractor.extractor_version} failed on {doc.paper_id}: {exc}")
    return result


def _ensure_builtins() -> None:
    """Import the bundled extractors so their decorators run."""
    from papersynth.extract import extractors as _builtins  # noqa: F401


def describe() -> dict[str, dict[str, Any]]:
    _ensure_builtins()
    return {
        name: {"version": cls.version, "sections": cls.section_pattern or "(all)"}
        for name, cls in sorted(EXTRACTORS.items())
    }
