"""Core entities (section 7).

The JSON Schemas in ``papersynth/schemas/`` are authoritative - these Pydantic
models are the in-process mirror. ``payload`` stays an untyped dict here on
purpose: it is validated against the owning extractor's ``output_schema``, so
adding a claim type never requires touching this file (FR-14, NFR-07).
Typed payload helpers live alongside their extractors.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from papersynth.core import ids

ClaimType = Literal[
    "algorithm",
    "equation",
    "hyperparameter",
    "dataset",
    "component",
    "result",
    "assumption",
    #: A design decision: which approach a paper took to a sub-problem, and
    #: whether it adopted or explicitly rejected it. Distinct from a
    #: hyperparameter because it carries no value to compare - two papers can
    #: disagree completely without a number between them.
    "method",
]
ClaimStatus = Literal["extracted", "verified", "rejected", "superseded"]
CheckResult = Literal["pass", "fail", "warn", "n/a"]
Criticality = Literal["BLOCKING", "MATERIAL", "COSMETIC"]
ConflictType = Literal[
    "VALUE_CONFLICT",
    "METHOD_CONFLICT",
    "RESULT_CONFLICT",
    "DEFINITION_CONFLICT",
    "SCOPE_CONFLICT",
]
Outcome = Literal["SELECTED", "SCOPED", "MERGED", "ESCALATED", "DEFERRED"]
Agreement = Literal["unanimous", "majority", "conflicting", "singleton"]
ExtractionMethod = Literal["llm", "regex", "latex_parser", "ocr_math"]


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ---------------------------------------------------------------------------
# Provenance (7.2)
# ---------------------------------------------------------------------------


class SecondaryProvenance(_Model):
    """Set when a claim is attributed to a work this paper cites.

    Without this, "following [12], we use cosine decay" would credit the citing
    paper with a method it borrowed (section 10.2, reference_trace).
    """

    cited_ref: str
    resolved_paper_id: str | None = None


class Provenance(_Model):
    paper_id: str
    span_id: str
    section: str
    page: int | None = None
    char_start: int
    char_end: int
    quote_hash: str
    extraction_method: ExtractionMethod
    extractor_version: str
    confidence: float = Field(ge=0.0, le=1.0)
    secondary: SecondaryProvenance | None = None
    #: The section this claim came from. Claims sharing a scope were extracted
    #: from the same passage, which is how section 10.1 tells "stages of one
    #: procedure" apart from "independent facts that disagree".
    scope_id: str = ""

    @model_validator(mode="after")
    def _derive_scope(self) -> Provenance:
        if not self.scope_id and self.span_id:
            object.__setattr__(self, "scope_id", ids.scope_id(self.span_id))
        return self

    @field_validator("quote_hash")
    @classmethod
    def _hash_shape(cls, v: str) -> str:
        if not v.startswith("sha256:") or len(v) != 71:
            raise ValueError("quote_hash must be 'sha256:' + 64 hex chars")
        return v


# ---------------------------------------------------------------------------
# Claim (7.3)
# ---------------------------------------------------------------------------


class Verification(_Model):
    citation_trace: CheckResult = "n/a"
    symbol_check: CheckResult = "n/a"
    range_check: CheckResult = "n/a"
    self_consistency: str = "n/a"
    notes: list[str] = Field(default_factory=list)

    @property
    def failed_checks(self) -> list[str]:
        return [
            name
            for name in ("citation_trace", "symbol_check", "range_check")
            if getattr(self, name) == "fail"
        ]

    @property
    def passes(self) -> bool:
        """True when no check failed. A 'warn' does not block verification."""
        return not self.failed_checks


class Claim(_Model):
    claim_id: str
    paper_id: str
    type: ClaimType
    status: ClaimStatus = "extracted"
    provenance: Provenance
    verification: Verification = Field(default_factory=Verification)
    payload: dict[str, Any]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @classmethod
    def build(
        cls,
        *,
        paper_id: str,
        claim_type: ClaimType,
        provenance: Provenance,
        payload: dict[str, Any],
        confidence: float = 1.0,
    ) -> Claim:
        """Construct with a content-derived claim_id."""
        return cls(
            claim_id=ids.claim_id(paper_id, claim_type, provenance.span_id, payload),
            paper_id=paper_id,
            type=claim_type,
            provenance=provenance,
            payload=payload,
            confidence=confidence,
        )

    @property
    def is_usable(self) -> bool:
        """Only verified claims may drive auto-resolution (section 8.3.4)."""
        return self.status == "verified"

    @property
    def scope_id(self) -> str:
        return self.provenance.scope_id


class ClaimSet(_Model):
    """All claims from one paper. The stage 1/2 artifact."""

    paper_id: str
    claims: list[Claim] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def by_type(self, claim_type: ClaimType) -> list[Claim]:
        return [c for c in self.claims if c.type == claim_type]

    @property
    def verified(self) -> list[Claim]:
        return [c for c in self.claims if c.status == "verified"]

    @property
    def rejected(self) -> list[Claim]:
        return [c for c in self.claims if c.status == "rejected"]


# ---------------------------------------------------------------------------
# ConceptCluster (7.5)
# ---------------------------------------------------------------------------


class ConceptCluster(_Model):
    cluster_id: str
    canonical_name: str
    concept_type: ClaimType
    member_claims: list[str] = Field(min_length=1)
    symbol_aliases: list[str] = Field(default_factory=list)
    papers: list[str] = Field(min_length=1)
    agreement: Agreement = "singleton"
    split_check: CheckResult = "n/a"

    @property
    def is_multi_paper(self) -> bool:
        """Only multi-paper clusters can host a cross-paper contradiction."""
        return len(set(self.papers)) > 1


class ConceptGraph(_Model):
    """The stage 3 artifact. Serialized as JSON, not YAML - it is machine-facing."""

    clusters: list[ConceptCluster] = Field(default_factory=list)
    claims: dict[str, Claim] = Field(default_factory=dict)
    symbol_map: dict[str, str] = Field(default_factory=dict)

    def cluster(self, cluster_id: str) -> ConceptCluster | None:
        return next((c for c in self.clusters if c.cluster_id == cluster_id), None)

    def claims_in(self, cluster: ConceptCluster) -> list[Claim]:
        return [self.claims[cid] for cid in cluster.member_claims if cid in self.claims]


# ---------------------------------------------------------------------------
# Contradiction (7.6)
# ---------------------------------------------------------------------------


class Support(_Model):
    """Evidence weight for one position. Drives the policy rules in section 8.5."""

    venue: str | None = None
    year: int | None = None
    is_primary: bool = False
    specificity: float = Field(default=0.5, ge=0.0, le=1.0)
    peer_reviewed: bool = False
    stated_explicitly: bool = True
    #: The claim named an explicit scope ("base model", "for WMT14"). Kept as a
    #: fact rather than folded into specificity, because prefer_scoped_over_global
    #: needs to know whether a condition exists, not how much one is worth.
    has_condition: bool = False


class Position(_Model):
    claim_id: str
    paper_id: str
    position: str
    support: Support = Field(default_factory=Support)


class Contradiction(_Model):
    contradiction_id: str
    cluster_id: str
    type: ConflictType
    severity: Criticality
    description: str
    positions: list[Position] = Field(min_length=2)
    detected_by: str

    @property
    def claim_ids(self) -> list[str]:
        return [p.claim_id for p in self.positions]


# ---------------------------------------------------------------------------
# Resolution (7.7)
# ---------------------------------------------------------------------------


class Resolution(_Model):
    resolution_id: str
    contradiction_id: str
    outcome: Outcome
    selected_claim_id: str | None = None
    rule_fired: str | None = None
    rationale: str = ""
    resolved_by: Literal["policy", "human"] = "policy"
    resolved_at: str = Field(default_factory=utcnow)
    human_note: str | None = None

    @property
    def is_open(self) -> bool:
        """ESCALATED and DEFERRED are unresolved; they still need a human."""
        return self.outcome in ("ESCALATED", "DEFERRED")


class ReconciliationResult(_Model):
    """Stage 5 artifact."""

    policy_version: str
    resolutions: list[Resolution] = Field(default_factory=list)

    def for_contradiction(self, contradiction_id: str) -> Resolution | None:
        return next(
            (r for r in self.resolutions if r.contradiction_id == contradiction_id),
            None,
        )

    @property
    def open(self) -> list[Resolution]:
        return [r for r in self.resolutions if r.is_open]

    @property
    def auto_resolved(self) -> list[Resolution]:
        return [r for r in self.resolutions if not r.is_open]


# ---------------------------------------------------------------------------
# Gap (7.8)
# ---------------------------------------------------------------------------


class Gap(_Model):
    gap_id: str
    component_id: str | None = None
    field: str
    question: str
    criticality: Criticality
    searched_papers: list[str] = Field(default_factory=list)
    suggested_sources: list[str] = Field(default_factory=list)
    resolution_status: Literal["open", "resolved_by_human", "resolved_by_reference"] = "open"
