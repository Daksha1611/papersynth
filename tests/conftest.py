from __future__ import annotations

import pytest

from papersynth.core import ids
from papersynth.core.document import Paragraph, Section, StructuredDocument
from papersynth.core.models import Claim, Provenance


def make_doc(paper_id: str = "1706.03762") -> StructuredDocument:
    """A small but realistic document, used across the unit suite."""
    return StructuredDocument(
        paper_id=paper_id,
        title="Attention Is All You Need",
        venue="NeurIPS",
        year=2017,
        ingest_method="latex",
        sha256="a" * 64,
        math_fidelity="latex_native",
        sections=[
            Section(
                index=0,
                title="1 Introduction",
                paragraphs=[
                    Paragraph(
                        index=0, text="Recurrent models typically factor computation.", page=1
                    ),
                    Paragraph(
                        index=1, text="We propose the Transformer, a model architecture.", page=1
                    ),
                ],
            ),
            Section(
                index=1,
                title="3.2 Attention",
                paragraphs=[
                    Paragraph(
                        index=0,
                        text=(
                            "We train using the Adam optimizer with a learning rate of 0.0001 "
                            "and a dropout rate of 0.1 on the base model."
                        ),
                        page=4,
                    ),
                    Paragraph(
                        index=1,
                        text="The key dimension d_k is set to 64 for each attention head.",
                        page=4,
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def doc() -> StructuredDocument:
    return make_doc()


@pytest.fixture
def second_doc() -> StructuredDocument:
    return StructuredDocument(
        paper_id="2504.17192",
        title="Paper2Code",
        venue="ICLR",
        year=2026,
        ingest_method="pdf",
        sha256="b" * 64,
        sections=[
            Section(
                index=0,
                title="4 Experiments",
                paragraphs=[
                    Paragraph(
                        index=0,
                        text=(
                            "We use a learning rate of 0.0003 with dropout 0.1 for the base model."
                        ),
                        page=6,
                    )
                ],
            )
        ],
    )


def make_claim(
    doc: StructuredDocument,
    *,
    section: int = 1,
    paragraph: int = 0,
    start: int = 0,
    end: int = 40,
    canonical_name: str = "learning_rate",
    value: float | str | bool = 0.0001,
    condition: str | None = "base model",
    stated_explicitly: bool = True,
    status: str = "verified",
) -> Claim:
    span = doc.make_span(section, paragraph, start, end)
    payload = {
        "canonical_name": canonical_name,
        "paper_symbol": None,
        "value": value,
        "value_type": "float" if isinstance(value, float) else "categorical",
        "unit": None,
        "applies_to": "global",
        "condition": condition,
        "stated_explicitly": stated_explicitly,
    }
    claim = Claim.build(
        paper_id=doc.paper_id,
        claim_type="hyperparameter",
        provenance=Provenance(
            paper_id=doc.paper_id,
            span_id=span.span_id,
            section=span.section_title,
            page=span.page,
            char_start=span.char_start,
            char_end=span.char_end,
            quote_hash=ids.quote_hash(span.text),
            extraction_method="llm",
            extractor_version="hyperparameter@1.0.0",
            confidence=0.95,
        ),
        payload=payload,
    )
    claim.status = status  # type: ignore[assignment]
    return claim


@pytest.fixture
def claim_factory():
    return make_claim
