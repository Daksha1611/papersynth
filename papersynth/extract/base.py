"""Extractor protocol and the LLM-backed base class (section 8.2).

The central mechanism here is how provenance gets anchored. Models are poor at
emitting a correct structured span ID and good at quoting text verbatim, so the
contract asks for the quote and resolves the span deterministically on our
side. A claim whose quote cannot be located in the document is rejected rather
than downgraded (ER-01), which means every surviving claim has a span that
provably contains its evidence - no model self-report involved.

Extractors see exactly one document. Cross-paper reasoning starts at stage 3
(ER-09); if extraction could see a sibling paper, an LLM would harmonize claims
toward agreement and genuine contradictions would vanish before the detector
ever ran.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from papersynth.core import ids
from papersynth.core.document import Section, StructuredDocument
from papersynth.core.models import Claim, ClaimType, Provenance
from papersynth.llm.base import LLMProvider
from papersynth.schemas import validate

#: Cap on how much text goes into one extraction prompt. Beyond this, recall
#: drops and cost climbs; `applicable_sections` should have narrowed it first.
MAX_PROMPT_CHARS = 24_000


@dataclass
class RejectedClaim:
    """A candidate that failed the extraction contract, kept for the report.

    Rejections are as informative as claims: a run rejecting most of what it
    extracted signals a prompt or ingestion problem, not a quiet success.
    """

    reason: str
    payload: dict[str, Any]
    quote: str | None = None


@dataclass
class ExtractionResult:
    claims: list[Claim] = field(default_factory=list)
    rejected: list[RejectedClaim] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def extend(self, other: ExtractionResult) -> None:
        self.claims.extend(other.claims)
        self.rejected.extend(other.rejected)
        self.warnings.extend(other.warnings)


@runtime_checkable
class Extractor(Protocol):
    """Turns one document into claims of one type."""

    claim_type: ClassVar[str]
    version: ClassVar[str]
    output_schema: ClassVar[dict[str, Any]]

    def applicable_sections(self, doc: StructuredDocument) -> list[Section]:
        """Narrow the search space before spending tokens."""
        ...

    def extract(self, doc: StructuredDocument, sections: list[Section]) -> ExtractionResult:
        """Emit claims. Must populate provenance for every claim."""
        ...


class LLMExtractor(ABC):
    """Base for LLM-backed extractors.

    Subclasses supply a prompt, a payload schema, and a section filter. This
    class owns everything that must not vary between extractors: schema
    validation, quote anchoring, provenance construction, and rejection
    accounting.
    """

    claim_type: ClassVar[str] = ""
    version: ClassVar[str] = "0.0.0"
    output_schema: ClassVar[dict[str, Any]] = {}
    payload_schema_name: ClassVar[str] = ""
    section_pattern: ClassVar[str] = ""
    system_prompt: ClassVar[str] = ""

    def __init__(self, provider: LLMProvider, *, temperature: float = 0.0) -> None:
        self.provider = provider
        self.temperature = temperature

    # -- subclass hooks ----------------------------------------------------

    @abstractmethod
    def build_prompt(self, doc: StructuredDocument, sections: list[Section]) -> str:
        """Render the extraction prompt for these sections."""

    def normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Last chance to canonicalize before validation. Must not invent values."""
        return payload

    # -- protocol ----------------------------------------------------------

    @property
    def extractor_version(self) -> str:
        return f"{self.claim_type}@{self.version}"

    def applicable_sections(self, doc: StructuredDocument) -> list[Section]:
        """Sections worth reading. Falls back to the whole paper if none match.

        Falling back rather than returning nothing is deliberate: a paper whose
        headings do not match our regex still has hyperparameters in it, and
        silently extracting nothing from it would look like a clean run.
        """
        if not self.section_pattern:
            return list(doc.sections)
        matched = doc.sections_matching(self.section_pattern)
        return matched or list(doc.sections)

    def extract(
        self, doc: StructuredDocument, sections: list[Section] | None = None
    ) -> ExtractionResult:
        sections = sections if sections is not None else self.applicable_sections(doc)
        result = ExtractionResult()
        if not sections:
            result.warnings.append(f"{self.claim_type}: no sections to read in {doc.paper_id}")
            return result

        prompt = self.build_prompt(doc, sections)
        if len(prompt) > MAX_PROMPT_CHARS:
            prompt = prompt[:MAX_PROMPT_CHARS]
            result.warnings.append(
                f"{self.claim_type}: prompt truncated to {MAX_PROMPT_CHARS} chars for "
                f"{doc.paper_id}; recall on later sections may be reduced"
            )

        completion = self._call(prompt, doc)
        raw_items = _as_items(completion.parsed)

        section_indices = [s.index for s in sections]
        for item in raw_items:
            self._admit(item, doc, section_indices, result)
        return result

    # -- internals ---------------------------------------------------------

    def _call(self, prompt: str, doc: StructuredDocument) -> Any:
        kwargs: dict[str, Any] = {
            "schema": self.output_schema,
            "temperature": self.temperature,
            "system": self.system_prompt or None,
        }
        # The router accepts ledger attribution kwargs; a bare provider does not.
        if hasattr(self.provider, "chain"):
            kwargs |= {
                "stage": "extract",
                "paper_id": doc.paper_id,
                "extractor": self.extractor_version,
                "template_id": self.extractor_version,
            }
        return self.provider.complete(prompt, **kwargs)

    def _admit(
        self,
        item: Any,
        doc: StructuredDocument,
        section_indices: list[int],
        result: ExtractionResult,
    ) -> None:
        """Apply the extraction contract to one candidate."""
        if not isinstance(item, dict):
            result.rejected.append(RejectedClaim("not a JSON object", {"raw": item}))
            return

        # Copy before touching it. Mutating the caller's object is invisible in
        # production, where each parsed response is fresh, and corrupting for
        # anything that reuses a candidate - a retry, a self-consistency
        # re-scan, or a fixture.
        data = dict(item)
        quote = data.pop("quote", None)
        payload = self.normalize_payload(
            {k: v for k, v in data.items() if v is not None or k in _NULLABLE}
        )

        if not isinstance(quote, str) or not quote.strip():
            result.rejected.append(
                RejectedClaim("no supporting quote supplied", payload, quote=None)
            )
            return

        span = doc.find_span(quote, section_filter=section_indices) or doc.find_span(quote)
        if span is None:
            # The model quoted text that is not in the paper. That is either a
            # fabrication or a paraphrase; either way the claim is unsupported.
            result.rejected.append(
                RejectedClaim("quote does not appear in the document", payload, quote=quote)
            )
            return

        if self.payload_schema_name:
            errors = validate(payload, self.payload_schema_name)
            if errors:
                result.rejected.append(
                    RejectedClaim(f"payload failed schema: {errors[0]}", payload, quote=quote)
                )
                return

        provenance = Provenance(
            paper_id=doc.paper_id,
            span_id=span.span_id,
            section=span.section_title,
            page=span.page,
            char_start=span.char_start,
            char_end=span.char_end,
            quote_hash=ids.quote_hash(span.text),
            extraction_method="llm",
            extractor_version=self.extractor_version,
            confidence=_confidence(doc),
        )
        claim = Claim.build(
            paper_id=doc.paper_id,
            claim_type=self.claim_type,  # type: ignore[arg-type]
            provenance=provenance,
            payload=payload,
            confidence=provenance.confidence,
        )
        if any(c.claim_id == claim.claim_id for c in result.claims):
            return  # identical content at the same span; not a second claim
        result.claims.append(claim)


#: Payload keys whose null is meaningful rather than absent.
_NULLABLE = {"condition", "unit", "paper_symbol", "label"}


def _as_items(parsed: Any) -> list[Any]:
    """Accept a bare list, or an object wrapping one under a plausible key."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("hyperparameters", "equations", "algorithms", "items", "claims", "results"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        # A single object is a single claim.
        return [parsed] if parsed else []
    return []


def _confidence(doc: StructuredDocument) -> float:
    """Base confidence, penalized for degraded source fidelity.

    An OCR-recovered document earns less trust before any check runs, because
    garbled math and lost decimal points originate there (R-01).
    """
    return {"latex_native": 0.95, "text_layer": 0.85, "ocr_recovered": 0.6}.get(
        doc.math_fidelity, 0.8
    )


def render_sections(doc: StructuredDocument, sections: list[Section]) -> str:
    """Render sections with stable markers the model can quote from."""
    blocks = []
    for section in sections:
        body = "\n\n".join(p.text for p in section.paragraphs)
        blocks.append(f"### {section.title}\n{body}")
    return "\n\n".join(blocks)


def claim_type_of(extractor: Extractor | LLMExtractor) -> ClaimType:
    return extractor.claim_type  # type: ignore[return-value]
