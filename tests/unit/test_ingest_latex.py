"""LaTeX ingestion. Offline and deterministic - no network, no GROBID."""

from __future__ import annotations

from pathlib import Path

import pytest

from papersynth.core.errors import IngestError
from papersynth.ingest.base import normalize_text
from papersynth.ingest.latex import LatexIngestor, parse_latex

FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_paper.tex"


@pytest.fixture(scope="module")
def parsed():
    return LatexIngestor().ingest(str(FIXTURE), paper_id="test.0001")


def test_title_comes_from_the_title_command(parsed):
    assert parsed.title == "A Minimal Attention Model"


def test_latex_path_reports_native_math_fidelity(parsed):
    assert parsed.math_fidelity == "latex_native"
    assert parsed.ingest_method == "latex"


def test_abstract_is_kept_as_an_addressable_section(parsed):
    """It states the objective more directly than any body section."""
    assert parsed.sections[0].title == "Abstract"
    assert "ingestion fixture" in parsed.sections[0].paragraphs[0].text


def test_sections_are_found_in_order(parsed):
    titles = [s.title for s in parsed.sections]
    assert titles == ["Abstract", "Introduction", "Method", "Experiments"]


def test_comments_are_stripped(parsed):
    assert "must never appear" not in parsed.full_text


def test_equation_is_captured_with_its_label(parsed):
    assert len(parsed.equations) == 1
    equation = parsed.equations[0]
    assert equation.label == "eq:attention"
    assert "softmax" in equation.latex
    assert equation.source_fidelity == "latex_native"


def test_equation_latex_has_its_label_command_removed(parsed):
    assert "\\label" not in parsed.equations[0].latex


def test_display_math_is_not_left_inline_in_prose(parsed):
    """An extractor quoting raw markup as a sentence fails citation_trace."""
    assert "\\frac{QK" not in parsed.full_text


def test_algorithm_block_is_captured(parsed):
    assert len(parsed.algorithms_raw) == 1
    algorithm = parsed.algorithms_raw[0]
    assert algorithm.label == "alg:train"
    assert algorithm.caption == "Training loop"
    assert "Initialize parameters" in algorithm.body


def test_figure_body_does_not_become_prose(parsed):
    assert "includegraphics" not in parsed.full_text


def test_inline_formatting_is_unwrapped(parsed):
    assert "The model is simple and fast to train." in parsed.full_text
    assert "\\emph" not in parsed.full_text


def test_citations_render_as_bracketed_keys(parsed):
    assert "[bahdanau2014]" in parsed.full_text


def test_bibliography_is_parsed(parsed):
    assert len(parsed.references) == 1
    reference = parsed.references[0]
    assert reference.key == "bahdanau2014"
    assert reference.arxiv_id == "1409.0473"
    assert reference.year == 2014


def test_year_is_not_borrowed_from_the_bibliography(parsed):
    """A cited work's year driving `prefer_recent_peer_reviewed` would silently
    resolve a real conflict the wrong way. Absent beats guessed (ER-02)."""
    assert parsed.year is None


def test_year_is_read_from_the_date_command():
    source = r"\date{March 2019}\begin{document}\section{S}" + "\n" + "x" * 40 + r"\end{document}"
    doc = parse_latex(source, paper_id="p", sha256="a" * 64)
    assert doc.year == 2019


def test_spans_resolve_after_ingestion(parsed):
    """The invariant every extracted claim depends on."""
    for section in parsed.sections:
        for para in section.paragraphs:
            span = parsed.make_span(section.index, para.index, 0, len(para.text))
            resolved = parsed.resolve_span(span.span_id, char_end=span.char_end)
            assert resolved is not None
            assert resolved.text == para.text


def test_find_span_anchors_a_hyperparameter_quote(parsed):
    span = parsed.find_span("learning rate of 0.0001")
    assert span is not None
    assert span.text == "learning rate of 0.0001"
    assert parsed.sections[span.section_index].title == "Experiments"


def test_ingesting_a_directory_of_sources(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    (root / "main.tex").write_text(r"\title{Split Paper}\begin{document}\input{body}\end{document}")
    (root / "body.tex").write_text(
        "\\section{Method}\nWe set the learning rate to 0.0003 for all experiments."
    )
    doc = LatexIngestor().ingest(str(root), paper_id="split.001")
    assert doc.title == "Split Paper"
    assert [s.title for s in doc.sections] == ["Method"]
    assert "0.0003" in doc.full_text


def test_input_resolution_does_not_recurse_forever(tmp_path):
    root = tmp_path / "loop"
    root.mkdir()
    (root / "main.tex").write_text(
        r"\begin{document}\section{A}" + "\n" + "x" * 40 + r"\input{main}\end{document}"
    )
    doc = LatexIngestor().ingest(str(root), paper_id="loop.001")
    assert doc.sections


def test_source_with_no_text_is_rejected_not_silently_empty():
    with pytest.raises(IngestError, match="no addressable text"):
        parse_latex(r"\begin{document}\end{document}", paper_id="empty", sha256="a" * 64)


def test_missing_file_raises():
    with pytest.raises(IngestError, match="No such LaTeX source"):
        LatexIngestor().ingest("/nonexistent/paper.tex")


def test_can_handle_rejects_nonexistent_paths():
    assert not LatexIngestor().can_handle("/nonexistent/paper.tex")
    assert LatexIngestor().can_handle(str(FIXTURE))


class TestNormalizeText:
    def test_ligatures_are_expanded(self):
        """Left alone, these defeat verbatim matching in citation_trace."""
        assert normalize_text("eﬃcient inference") == "efficient inference"
        assert normalize_text("ﬁne-tuning the ﬂow") == "fine-tuning the flow"

    def test_hyphenated_line_breaks_are_rejoined(self):
        assert normalize_text("effi-\ncient") == "efficient"

    def test_smart_quotes_are_normalized(self):
        assert normalize_text("\u201cattention\u201d") == '"attention"'

    def test_whitespace_runs_collapse(self):
        assert normalize_text("a    b\t\tc") == "a b c"

    def test_nonbreaking_space_becomes_a_space(self):
        assert normalize_text("Figure\u00a01") == "Figure 1"
