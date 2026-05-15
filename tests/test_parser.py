"""Tests for src/2_parser.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eei_parser as parser  # registered by conftest


# ---------------------------------------------------------------------------
# Q&A section detection
# ---------------------------------------------------------------------------
def test_qa_section_found(sample_transcript_text):
    qa = parser.split_qa_section(sample_transcript_text)
    assert "Toni Sacconaghi" in qa
    assert "prepared remarks omitted" not in qa


def test_qa_section_alternate_opener():
    txt = "Some text. begin the question-and-answer session.\nJohn Smith -- Goldman Sachs -- Analyst\nQ"
    qa = parser.split_qa_section(txt)
    assert "John Smith" in qa
    assert "Some text" not in qa


def test_qa_section_falls_back_when_missing():
    txt = "Random text with no opener phrase."
    qa = parser.split_qa_section(txt)
    assert qa == txt  # falls back to whole text


# ---------------------------------------------------------------------------
# Speaker classification
# ---------------------------------------------------------------------------
def test_classify_analyst():
    affil, title, role = parser._classify("Goldman Sachs", "Analyst")
    assert role == "analyst"
    assert affil == "Goldman Sachs"


def test_classify_executive_by_officer_keyword():
    affil, title, role = parser._classify("Chief Financial Officer", "")
    assert role == "executive"


def test_classify_executive_by_ceo_keyword():
    _, _, role = parser._classify("CEO", "")
    assert role == "executive"


def test_classify_unknown():
    _, _, role = parser._classify("Some Other", "")
    assert role == "unknown"


# ---------------------------------------------------------------------------
# Turn splitting + pairing
# ---------------------------------------------------------------------------
def test_split_turns_and_pair(sample_transcript_text):
    qa_text = parser.split_qa_section(sample_transcript_text)
    turns = parser._split_turns(qa_text)
    pairs = parser._pair_turns(turns)
    assert len(pairs) >= 3
    # First pair should be Sacconaghi -> Maestri.
    p0 = pairs[0]
    assert "Sacconaghi" in p0.analyst_name
    assert "Bernstein" in p0.analyst_firm
    assert "Maestri" in p0.executive_name


def test_pair_extracts_topics(sample_transcript_text):
    qa_text = parser.split_qa_section(sample_transcript_text)
    pairs = parser._pair_turns(parser._split_turns(qa_text))
    assert any("margins" in p.question_topics for p in pairs)
    assert any("litigation" in p.question_topics for p in pairs)


def test_pair_handles_empty_qa():
    pairs = parser._pair_turns(parser._split_turns(""))
    assert pairs == []


def test_single_exchange():
    txt = (
        "Now we'll open up the call for questions.\n\n"
        "Alice Doe -- Bernstein -- Analyst\nWhat's your margin?\n\n"
        "Bob Roe -- Chief Financial Officer\nIt's 45%.\n\n"
    )
    pairs = parser._pair_turns(parser._split_turns(parser.split_qa_section(txt)))
    assert len(pairs) == 1
    assert pairs[0].analyst_name == "Alice Doe"
    assert pairs[0].executive_name == "Bob Roe"


# ---------------------------------------------------------------------------
# Linguistic features
# ---------------------------------------------------------------------------
def test_count_phrases_basic():
    from config import HEDGE_WORDS
    text = "We expect approximately 5% growth, and we believe we may see further gains."
    n = parser.count_phrases(text, HEDGE_WORDS)
    assert n >= 3  # "we expect", "approximately", "we believe", "may"


def test_count_phrases_case_insensitive():
    n = parser.count_phrases("WE BELIEVE strongly", ["we believe"])
    assert n == 1


def test_topic_extraction_margins():
    topics = parser.extract_topics("What about gross margin trajectory?")
    assert "margins" in topics


def test_topic_extraction_capex_and_debt():
    topics = parser.extract_topics("Capex was elevated and your debt level grew.")
    assert "capex" in topics
    assert "debt" in topics


def test_topic_extraction_no_match():
    topics = parser.extract_topics("Hello world.")
    assert topics == []


def test_jaccard_overlap_identical():
    assert parser.jaccard_overlap("margin growth strong", "margin growth strong") == pytest.approx(1.0)


def test_jaccard_overlap_disjoint():
    assert parser.jaccard_overlap("apple banana cherry", "xenon yttrium zinc") == 0.0


def test_jaccard_overlap_partial():
    o = parser.jaccard_overlap("margin guidance growth", "margin commentary outlook")
    assert 0.0 < o < 1.0


def test_jaccard_overlap_empty():
    assert parser.jaccard_overlap("", "anything here") == 0.0


# ---------------------------------------------------------------------------
# Full file parse smoke test
# ---------------------------------------------------------------------------
def test_parse_transcript_file(tmp_path, sample_transcript_text):
    from src.utils import write_json
    f = tmp_path / "AAPL_2024-04-25.json"
    write_json(f, {
        "company": "Apple Inc.", "ticker": "AAPL", "date": "2024-04-25",
        "quarter": "Q2 2024", "raw_text": sample_transcript_text, "source": "test",
    })
    call = parser.parse_transcript_file(f)
    assert call.ticker == "AAPL"
    assert len(call.qa_pairs) >= 3
    assert all(p.answer_word_count > 0 for p in call.qa_pairs)
