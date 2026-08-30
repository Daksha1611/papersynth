"""Citation tracing (section 8.3.1, 10.1).

Four checks, cheapest first, so an obviously-broken claim never reaches an LLM:

  1. the span resolves at all
  2. the stored quote hash still matches the document's text
  3. a numeric value literally appears in the cited span
  4. the span actually entails the payload (adversarial LLM judge)

Checks 1 and 2 nearly always pass for a freshly extracted claim, because the
span was derived from the model's own quote. Their real work is on
re-verification: a claim loaded from a previous run against a re-ingested
document, where a changed parser or a different PDF would otherwise let stale
provenance pass unnoticed.

Check 3 is where deterministic value is concentrated. A model that quotes
"learning rate of 0.0001" and reports 0.001 is caught here with no call spent,
and that transcription slip is the single most common extraction error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from papersynth.core import ids
from papersynth.core.document import StructuredDocument
from papersynth.core.models import Claim
from papersynth.llm.base import LLMProvider
from papersynth.verify.range_check import CheckOutcome

TRACER_SYSTEM = (
    "You are an adversarial fact-checker. Your job is to find reasons a quoted "
    "passage does NOT support a claim. You are not trying to be agreeable. If "
    "the passage only partially supports the claim, that is not support."
)

#: Numbers in prose: 0.0001, 1e-4, 1.2 x 10^-4, 100,000, 64.
_NUMBER = re.compile(
    r"(?<![\w.])"
    r"(?P<num>\d[\d,]*\.?\d*(?:\s*[eE]\s*[-+]?\d+)?"
    r"|\.\d+(?:\s*[eE]\s*[-+]?\d+)?)"
    r"(?![\w])"
)
# The caret is optional and must not swallow the exponent's sign - treating
# "^-" as one marker turns 1.2 x 10^-4 into 1.2 x 10^4.
_SCIENTIFIC = re.compile(r"(?P<mant>\d+\.?\d*)\s*(?:x|×|\*)\s*10\s*\^?\s*(?P<exp>[-−+]?\d+)")


@dataclass
class TraceResult:
    span_resolves: CheckOutcome
    quote_hash: CheckOutcome
    numeric_literal: CheckOutcome
    entailment: CheckOutcome

    @property
    def outcome(self) -> CheckOutcome:
        """Collapse to a single verdict. Any hard failure fails the trace."""
        for check in (self.span_resolves, self.quote_hash, self.numeric_literal, self.entailment):
            if check.result == "fail":
                return check
        if any(
            c.result == "warn" for c in (self.numeric_literal, self.entailment, self.quote_hash)
        ):
            reasons = "; ".join(
                c.reason
                for c in (self.quote_hash, self.numeric_literal, self.entailment)
                if c.result == "warn"
            )
            return CheckOutcome("warn", reasons)
        return CheckOutcome("pass")


def trace(
    claim: Claim,
    doc: StructuredDocument,
    *,
    provider: LLMProvider | None = None,
    entailment: CheckOutcome | None = None,
) -> TraceResult:
    """Run the trace.

    An `entailment` verdict may be supplied by a caller that judged claims in
    batches; otherwise a provider is asked for one, and with neither the check
    is skipped.
    """
    span = doc.resolve_span(claim.provenance.span_id, char_end=claim.provenance.char_end)

    if span is None:
        unresolvable = CheckOutcome(
            "fail", f"span {claim.provenance.span_id} does not resolve in {doc.paper_id}"
        )
        return TraceResult(
            unresolvable, CheckOutcome("n/a"), CheckOutcome("n/a"), CheckOutcome("n/a")
        )

    hash_check = (
        CheckOutcome("pass")
        if ids.quote_hash(span.text) == claim.provenance.quote_hash
        else CheckOutcome(
            "fail",
            "source text at this span no longer matches the recorded hash; "
            "the document changed or the span misaligned",
        )
    )
    if hash_check.failed:
        return TraceResult(
            CheckOutcome("pass"), hash_check, CheckOutcome("n/a"), CheckOutcome("n/a")
        )

    numeric = check_numeric_literal(claim, span.text)

    if entailment is None:
        entailment = (
            check_entailment(claim, span.text, provider)
            if provider is not None
            else CheckOutcome("n/a", "no provider supplied")
        )

    return TraceResult(CheckOutcome("pass"), hash_check, numeric, entailment)


def check_numeric_literal(claim: Claim, span_text: str) -> CheckOutcome:
    """Does the claimed number actually appear in the cited text?

    Deterministic and free. Comparison is on numeric value rather than on
    string form, so 1e-4, 0.0001, and 1 x 10^-4 all match a claimed 0.0001 -
    papers write the same quantity many ways, and a string comparison would
    reject correct claims constantly.
    """
    value = claim.payload.get("value")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return CheckOutcome("n/a")

    for candidate in _numbers_in(span_text):
        if _close(candidate, float(value)):
            return CheckOutcome("pass")

    return CheckOutcome(
        "warn",
        f"value {value} does not appear literally in the cited span; "
        "it may have been inferred, converted, or mis-transcribed",
    )


def check_entailment(claim: Claim, span_text: str, provider: LLMProvider) -> CheckOutcome:
    """Adversarial judge: does this passage support this claim?

    Prompted to look for reasons the passage does NOT support the claim, and
    ties resolve as failure. A judge asked "does this support it?" agrees far
    too readily, which would make the check decorative.
    """
    prompt = (
        "PASSAGE FROM THE PAPER:\n"
        f"{span_text}\n\n"
        "CLAIM EXTRACTED FROM THAT PASSAGE:\n"
        f"{_render_claim(claim)}\n\n"
        "Does the passage state or directly support this claim?\n"
        "Look for reasons it does not: a different value, a different scope or "
        "condition, a different quantity with a similar name, or a claim that "
        "goes beyond what the passage says.\n"
        'Answer JSON: {"entailed": true|false, "reason": "<one sentence>"}'
    )
    schema = {
        "type": "object",
        "properties": {"entailed": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["entailed", "reason"],
    }

    kwargs: dict[str, Any] = {"schema": schema, "temperature": 0.0, "system": TRACER_SYSTEM}
    if hasattr(provider, "chain"):
        kwargs |= {
            "stage": "verify",
            "paper_id": claim.paper_id,
            "extractor": "tracer@1.0.0",
            "template_id": "tracer@1.0.0",
        }

    completion = provider.complete(prompt, **kwargs)
    verdict = completion.parsed if isinstance(completion.parsed, dict) else {}

    # An unreadable verdict must not silently pass. Ties resolve as FAIL.
    if "entailed" not in verdict:
        return CheckOutcome("fail", "tracer returned no verdict; treated as unsupported")

    if verdict.get("entailed") is True:
        return CheckOutcome("pass")

    reason = str(verdict.get("reason", "")).strip() or "tracer found the span unsupportive"
    return CheckOutcome("fail", reason)


def _render_claim(claim: Claim) -> str:
    payload = claim.payload
    if claim.type == "hyperparameter":
        parts = [f"{payload.get('canonical_name')} = {payload.get('value')!r}"]
        if payload.get("unit"):
            parts.append(f"unit: {payload['unit']}")
        if payload.get("condition"):
            parts.append(f"condition: {payload['condition']}")
        if payload.get("applies_to") and payload["applies_to"] != "global":
            parts.append(f"applies to: {payload['applies_to']}")
        return "\n".join(parts)
    return "\n".join(f"{k}: {v}" for k, v in payload.items() if v is not None)


def _numbers_in(text: str) -> list[float]:
    """Every number in the text, including 1.2 x 10^-4 forms."""
    found: list[float] = []

    for match in _SCIENTIFIC.finditer(text):
        try:
            exponent = match.group("exp").replace("−", "-")
            found.append(float(match.group("mant")) * (10 ** int(exponent)))
        except (ValueError, OverflowError):
            continue

    for match in _NUMBER.finditer(text):
        raw = match.group("num").replace(",", "").replace(" ", "")
        try:
            found.append(float(raw))
        except (ValueError, OverflowError):
            continue

    return found


def _close(a: float, b: float) -> bool:
    """Equality with tolerance for how the value was written down."""
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    if scale == 0:
        return True
    return abs(a - b) / scale < 1e-6


def check_entailment_batch(
    pairs: list[tuple[Claim, str]], provider: LLMProvider
) -> dict[str, CheckOutcome]:
    """Judge several claims in one call, keyed by claim id.

    One call per claim is the largest single source of calls in a run - the
    entailment check runs on every extracted claim, where extraction itself
    runs once per section batch. Section 6.4.5 puts batching ahead of the
    fallback chain for exactly this reason: the cheapest call is the one never
    made, and on a free tier capped per minute the difference decides whether a
    run finishes.

    Batching does cost something real: the judge sees several claims at once
    and could let a confident verdict on one colour another. That is why each
    passage is given with its own claim and asked about separately within the
    call, and why an answer that omits a claim fails closed rather than
    defaulting to entailed.
    """
    if not pairs:
        return {}
    if len(pairs) == 1:
        claim, span_text = pairs[0]
        return {claim.claim_id: check_entailment(claim, span_text, provider)}

    rendered = "\n\n".join(
        f"--- CLAIM {claim.claim_id} ---\nPASSAGE:\n{span_text}\n\nCLAIM:\n{_render_claim(claim)}"
        for claim, span_text in pairs
    )
    prompt = (
        "Judge each claim below against its own passage, independently. A "
        "verdict on one claim tells you nothing about another.\n\n"
        f"{rendered}\n\n"
        "For each, look for reasons the passage does NOT support the claim: a "
        "different value, a different scope or condition, a different quantity "
        "with a similar name, or a claim going beyond what the passage says.\n"
        'Answer JSON: {"verdicts": [{"claim_id": "...", "entailed": true|false, '
        '"reason": "<one sentence>"}]}'
    )
    schema = {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "entailed": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["claim_id", "entailed"],
                },
            }
        },
        "required": ["verdicts"],
    }

    kwargs: dict[str, Any] = {"schema": schema, "temperature": 0.0, "system": TRACER_SYSTEM}
    if hasattr(provider, "chain"):
        kwargs |= {
            "stage": "verify",
            "paper_id": pairs[0][0].paper_id,
            "extractor": "tracer@1.0.0",
            "template_id": "tracer_batch@1.0.0",
        }

    completion = provider.complete(prompt, **kwargs)
    payload = completion.parsed if isinstance(completion.parsed, dict) else {}

    verdicts = {
        str(v.get("claim_id")): v
        for v in payload.get("verdicts") or []
        if isinstance(v, dict) and v.get("claim_id")
    }

    out: dict[str, CheckOutcome] = {}
    for claim, _ in pairs:
        verdict = verdicts.get(claim.claim_id)
        if verdict is None or "entailed" not in verdict:
            # A claim the judge did not answer for is not a claim it approved.
            # Defaulting to entailed would let a truncated response silently
            # verify everything it ran out of room to consider.
            out[claim.claim_id] = CheckOutcome(
                "fail", "tracer returned no verdict for this claim; treated as unsupported"
            )
        elif verdict.get("entailed") is True:
            out[claim.claim_id] = CheckOutcome("pass")
        else:
            out[claim.claim_id] = CheckOutcome(
                "fail",
                str(verdict.get("reason", "")).strip() or "tracer found the span unsupportive",
            )
    return out
