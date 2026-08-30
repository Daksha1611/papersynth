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

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from papersynth.core import ids
from papersynth.core.document import Section, Span, StructuredDocument
from papersynth.core.models import Claim, ClaimType, Provenance, SecondaryProvenance
from papersynth.llm.base import LLMProvider
from papersynth.schemas import validate

#: Character budget for one extraction prompt.
#:
#: Sized against a free tier's per-minute token allowance rather than a model's
#: context window. Groq's free tier permits 8,000 tokens per minute, and real
#: paper text runs close to 3 characters per token, so a single 24,000-character
#: prompt was rejected outright with a 413. Sections are batched under this
#: budget across several calls instead, which also removes the truncation that
#: silently cost recall on long papers.
MAX_PROMPT_CHARS = 9_000


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

    def __init__(
        self,
        provider: LLMProvider,
        *,
        temperature: float = 0.0,
        self_consistency_n: int = 1,
    ) -> None:
        self.provider = provider
        self.temperature = temperature
        #: Re-extractions used to score agreement (section 8.3.4). One means
        #: the check is off, which is the default: it multiplies extraction
        #: cost by n, and section 6.4.5 reserves it for final reviewed runs.
        self.self_consistency_n = max(1, self_consistency_n)

    # -- subclass hooks ----------------------------------------------------

    @abstractmethod
    def build_prompt(self, doc: StructuredDocument, sections: list[Section]) -> str:
        """Render the extraction prompt for these sections."""

    def normalize_payload(self, payload: dict[str, Any], doc: StructuredDocument) -> dict[str, Any]:
        """Canonicalize before validation. Must not invent values.

        Receives the document because some normalization is only decidable
        against it - resolving a symbol's claimed definition to a real span,
        for instance, where a definition the model cannot point to does not
        count as one.
        """
        return payload

    def anchor(
        self,
        item: dict[str, Any],
        quote: str | None,
        doc: StructuredDocument,
        section_indices: list[int],
    ) -> Span | None:
        """Resolve this candidate to a span in the document.

        The default asks the model for a verbatim quote and locates it, which
        is right whenever the claim comes out of prose. Extractors whose source
        object already has a known position in the document - an equation, an
        algorithm block - override this and anchor deterministically instead,
        so no model quote is involved in their provenance at all.
        """
        if not isinstance(quote, str) or not quote.strip():
            return None
        return doc.find_span(quote, section_filter=section_indices) or doc.find_span(quote)

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

        if self.self_consistency_n <= 1:
            return self._single_pass(doc, sections)

        passes = [
            self._single_pass(doc, sections, rotation=i) for i in range(self.self_consistency_n)
        ]
        return _merge_passes(passes, self.self_consistency_n)

    def _single_pass(
        self, doc: StructuredDocument, sections: list[Section], rotation: int = 0
    ) -> ExtractionResult:
        """One extraction sweep over the sections, optionally reordered.

        Rotation is what makes a second opinion an opinion at all. Extraction
        runs at temperature 0 and responses are cached by prompt hash, so
        re-issuing an identical prompt returns the identical answer - for free,
        and worth nothing. Section 8.3.4 asks for different prompt orderings
        precisely because the same facts presented in a different order is a
        genuinely different question to ask.
        """
        result = ExtractionResult()
        for batch in self.batch_sections(doc, sections):
            ordered = _rotate(batch, rotation)
            completion = self._call(self.build_prompt(doc, ordered), doc)
            section_indices = [s.index for s in ordered]
            for item in _as_items(completion.parsed):
                self._admit(item, doc, section_indices, result)
        return result

    def batch_sections(
        self, doc: StructuredDocument, sections: list[Section]
    ) -> list[list[Section]]:
        """Split sections into groups that each fit one prompt.

        Batching rather than truncating matters for recall: a paper whose
        training details sit in an appendix would have had them silently cut
        off, and the run would have reported a clean extraction that simply
        never read the relevant page.

        A single section larger than the budget still goes out alone. Splitting
        mid-section would hand the model a fragment with no context, and a
        claim extracted from half a sentence is worse than one not extracted.
        """
        overhead = len(self.build_prompt(doc, []))
        budget = max(1_000, MAX_PROMPT_CHARS - overhead)

        batches: list[list[Section]] = []
        current: list[Section] = []
        size = 0

        for section in sections:
            section_size = len(section.text) + len(section.title) + 8
            if current and size + section_size > budget:
                batches.append(current)
                current, size = [], 0
            current.append(section)
            size += section_size

        if current:
            batches.append(current)
        return batches

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
            {k: v for k, v in data.items() if v is not None or k in _NULLABLE}, doc
        )

        span = self.anchor(data, quote if isinstance(quote, str) else None, doc, section_indices)
        if span is None:
            # Either the model supplied no quote, or it quoted text that is not
            # in the paper - a fabrication or a paraphrase. Either way the
            # claim is unsupported and is rejected rather than downgraded.
            reason = (
                "no supporting quote supplied"
                if not isinstance(quote, str) or not quote.strip()
                else "quote does not appear in the document"
            )
            result.rejected.append(
                RejectedClaim(reason, payload, quote=quote if isinstance(quote, str) else None)
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
            secondary=reference_trace(span.text, doc),
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


#: Citations as the LaTeX ingestor renders them: \cite{key} becomes [key].
_CITATION = re.compile(r"\[([A-Za-z][A-Za-z0-9_:.\-]{2,})\]")


def reference_trace(span_text: str, doc: StructuredDocument) -> SecondaryProvenance | None:
    """Record the work a claim's span cites (section 10.2).

    "Following [12], we use cosine decay" credits the citing paper with a
    method it borrowed unless the reference is carried alongside the claim.

    What this establishes is narrow and worth stating: the span cites this
    work. It does not establish that the claim was borrowed - "unlike [12], we
    use X" cites without borrowing - so this is evidence for a reader and for
    later stages to weigh, never a verdict on its own.
    """
    known = {r.key: r for r in doc.references}

    for match in _CITATION.finditer(span_text):
        key = match.group(1)
        resolved = known.get(key)
        if resolved is None and not _looks_like_bibkey(key):
            continue
        return SecondaryProvenance(
            cited_ref=key,
            resolved_paper_id=resolved.arxiv_id if resolved and resolved.arxiv_id else None,
        )
    return None


def _looks_like_bibkey(key: str) -> bool:
    """Whether a bracketed token is plausibly a citation key.

    Needed because papers bracket things that are not citations. BERT writes
    [MASK], [CLS] and [SEP] constantly, and reading those as references
    attributed BERT's own masking decisions to a cited work. A bibliography key
    almost always carries a year; a special token is short, all-caps and has no
    digits.
    """
    return any(ch.isdigit() for ch in key) and not key.isupper()


def _confidence(doc: StructuredDocument) -> float:
    """Base confidence, penalized for degraded source fidelity.

    An OCR-recovered document earns less trust before any check runs, because
    garbled math and lost decimal points originate there (R-01).
    """
    return {
        "latex_native": 0.95,
        "text_layer": 0.85,
        # Detected as damaged, whether or not OCR managed to recover it. Both
        # mean the characters may not be what the author wrote.
        "text_layer_suspect": 0.6,
        "ocr_recovered": 0.6,
    }.get(doc.math_fidelity, 0.8)


def render_sections(doc: StructuredDocument, sections: list[Section]) -> str:
    """Render sections with stable markers the model can quote from."""
    blocks = []
    for section in sections:
        body = "\n\n".join(p.text for p in section.paragraphs)
        blocks.append(f"### {section.title}\n{body}")
    return "\n\n".join(blocks)


def claim_type_of(extractor: Extractor | LLMExtractor) -> ClaimType:
    return extractor.claim_type  # type: ignore[return-value]


def _rotate(sections: list[Section], rotation: int) -> list[Section]:
    """Present the same sections starting from a different one."""
    if not sections or rotation % len(sections) == 0:
        return sections
    offset = rotation % len(sections)
    return sections[offset:] + sections[:offset]


def agreement_key(claim: Claim) -> tuple[str, str, str]:
    """What counts as "the same fact" across re-extractions.

    Deliberately not the claim_id, which folds in the span. A model quoting a
    different sentence for the same value on a second pass still reported the
    same fact, and scoring that as disagreement would penalise exactly the
    claims that were found twice.
    """
    payload = claim.payload
    for field_name in ("canonical_name", "sub_problem", "label", "name", "metric"):
        value = payload.get(field_name)
        if isinstance(value, str) and value.strip():
            name = value.strip().lower()
            break
    else:
        name = claim.claim_id

    raw = payload.get("value", payload.get("approach", ""))
    if isinstance(raw, bool):
        rendered = str(raw)
    elif isinstance(raw, int | float):
        rendered = f"{float(raw):.12g}"
    else:
        rendered = str(raw).strip().lower()

    return (claim.type, name, rendered)


def _merge_passes(passes: list[ExtractionResult], n: int) -> ExtractionResult:
    """Combine re-extractions, scoring each fact by how often it appeared.

    A fact every pass reported keeps full confidence. One that appeared once in
    three is reported with its agreement recorded and its confidence scaled, so
    the verifier can decline to promote it. Nothing is dropped here: a claim
    seen once may still be correct, and discarding it silently would trade a
    visible low-confidence entry for an invisible absence.
    """
    merged = ExtractionResult()
    seen: dict[tuple[str, str, str], Claim] = {}
    counts: dict[tuple[str, str, str], int] = {}

    for result in passes:
        merged.warnings.extend(result.warnings)
        merged.rejected.extend(result.rejected)
        for claim in result.claims:
            key = agreement_key(claim)
            counts[key] = counts.get(key, 0) + 1
            seen.setdefault(key, claim)

    for key, claim in sorted(seen.items(), key=lambda kv: kv[1].claim_id):
        agreed = counts[key]
        claim.verification.self_consistency = f"{agreed}/{n}"
        claim.confidence = round(claim.confidence * (agreed / n), 4)
        if agreed < n:
            claim.verification.notes.append(f"reported in {agreed} of {n} extractions")
        merged.claims.append(claim)

    return merged
