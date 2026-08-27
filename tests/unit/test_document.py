"""Span addressing (section 8.1). The round-trip test is the load-bearing one:
if it fails, provenance does not survive re-ingestion and NFR-01 is unenforceable."""

from __future__ import annotations

import pytest

from papersynth.core.document import Paragraph, Section, StructuredDocument, parse_span_id
from tests.conftest import make_doc


def test_span_roundtrip_is_lossless(doc):
    """ingest -> span_id -> resolve -> identical text."""
    for section in doc.sections:
        for para in section.paragraphs:
            span = doc.make_span(section.index, para.index, 0, len(para.text))
            resolved = doc.resolve_span(span.span_id, char_end=span.char_end)
            assert resolved is not None
            assert resolved.text == span.text == para.text


def test_span_ids_are_stable_across_reingestion():
    """Same source in, same span IDs out - otherwise provenance breaks between runs."""
    a, b = make_doc(), make_doc()
    ids_a = [a.make_span(s.index, p.index, 0, 10).span_id for s in a.sections for p in s.paragraphs]
    ids_b = [b.make_span(s.index, p.index, 0, 10).span_id for s in b.sections for p in s.paragraphs]
    assert ids_a == ids_b


def test_partial_span_resolves_to_the_requested_slice(doc):
    span = doc.make_span(1, 0, 10, 30)
    assert span.text == doc.sections[1].paragraphs[0].text[10:30]
    assert span.char_start == 10


def test_resolve_span_returns_none_for_malformed_id(doc):
    """ER-01 rejects; it must not crash on a model hallucinating a span format."""
    assert doc.resolve_span("garbage") is None
    assert doc.resolve_span("1706.03762#section-three") is None
    assert doc.resolve_span("1706.03762#s99.p0.0") is None
    assert doc.resolve_span("1706.03762#s0.p99.0") is None


def test_resolve_span_rejects_a_foreign_paper_id(doc):
    """A claim citing another paper's span must not resolve here (ER-09 backstop)."""
    assert doc.resolve_span("2504.17192#s0.p0.0") is None


def test_resolve_span_rejects_offset_past_end(doc):
    para_len = len(doc.sections[0].paragraphs[0].text)
    assert doc.resolve_span(f"1706.03762#s0.p0.{para_len + 5}") is None


def test_make_span_clamps_out_of_range_requests(doc):
    span = doc.make_span(0, 0, 5, 10_000)
    assert span.char_end == len(doc.sections[0].paragraphs[0].text)


def test_parse_span_id():
    assert parse_span_id("1706.03762#s3.p2.114") == ("1706.03762", 3, 2, 114)
    assert parse_span_id("sha256:abc#s0.p0.0") == ("sha256:abc", 0, 0, 0)
    assert parse_span_id("no-hash-here") is None


def test_find_span_locates_verbatim_text(doc):
    span = doc.find_span("learning rate of 0.0001")
    assert span is not None
    assert "learning rate of 0.0001" in span.text
    assert span.section_index == 1


def test_find_span_tolerates_whitespace_differences(doc):
    """A model re-wrapping a quote must still anchor to the right span."""
    span = doc.find_span("Adam   optimizer\n  with a learning rate")
    assert span is not None
    assert "Adam optimizer with a learning rate" in " ".join(span.text.split())


def test_find_span_returns_none_when_absent(doc):
    assert doc.find_span("a sentence that is nowhere in this paper") is None


def test_quote_hash_is_whitespace_insensitive(doc):
    from papersynth.core.ids import quote_hash

    assert quote_hash("a  b\nc") == quote_hash("a b c")
    assert quote_hash("a b c") != quote_hash("a b d")


def test_sections_matching(doc):
    assert [s.title for s in doc.sections_matching(r"attention")] == ["3.2 Attention"]
    assert doc.sections_matching(r"ablation") == []


def test_misindexed_section_is_rejected():
    """Positional indices are what make span IDs stable; drift must fail loudly."""
    with pytest.raises(ValueError, match="index"):
        StructuredDocument(
            paper_id="x",
            title="t",
            ingest_method="pdf",
            sha256="c" * 64,
            sections=[Section(index=7, title="s", paragraphs=[])],
        )


def test_misindexed_paragraph_is_rejected():
    with pytest.raises(ValueError, match="index"):
        StructuredDocument(
            paper_id="x",
            title="t",
            ingest_method="pdf",
            sha256="c" * 64,
            sections=[Section(index=0, title="s", paragraphs=[Paragraph(index=3, text="hi")])],
        )
