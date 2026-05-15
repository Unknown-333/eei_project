"""Tests for src/1_scraper.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eei_scraper as scraper  # registered by conftest


# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------
def test_synthetic_calls_default_count():
    out = scraper.synthetic_calls(tickers=["AAPL"], per_ticker=4, start_year=2023, end_year=2023)
    assert len(out) == 4


def test_synthetic_calls_have_qa_section():
    out = scraper.synthetic_calls(tickers=["MSFT"], per_ticker=2, start_year=2024, end_year=2024)
    for t in out:
        assert "open up the call for questions" in t.raw_text.lower()
        assert "Analyst" in t.raw_text


def test_synthetic_deterministic_with_seed():
    a = scraper.synthetic_calls(tickers=["AAPL"], per_ticker=2, start_year=2024, end_year=2024)
    b = scraper.synthetic_calls(tickers=["AAPL"], per_ticker=2, start_year=2024, end_year=2024)
    assert a[0].raw_text == b[0].raw_text


def test_synthetic_evasion_profile_varies_by_ticker():
    p1 = scraper._evasion_profile("AAPL", "Q1 2024")
    p2 = scraper._evasion_profile("XOM", "Q1 2024")
    # XOM has higher base evasion; should generally have more "evasive" labels.
    assert p1 != p2 or set(p1) != set(p2)


def test_make_qa_returns_question_and_answer_text():
    rng = scraper._seeded_rng("AAPL", "Q1 2024")
    q, a = scraper._make_qa(rng, "AAPL", "direct")
    assert "Analyst" in q
    assert len(a) > 50


# ---------------------------------------------------------------------------
# Live scraping (HTTP mocked)
# ---------------------------------------------------------------------------
def _mock_response(text: str, status: int = 200):
    r = MagicMock()
    r.status_code = status
    r.text = text
    return r


def test_request_returns_none_on_short_body():
    with patch.object(scraper.requests, "get", return_value=_mock_response("short", 200)):
        assert scraper._request("http://x") is None


def test_request_returns_none_on_non_200():
    with patch.object(scraper.requests, "get", return_value=_mock_response("a" * 5000, 404)):
        assert scraper._request("http://x") is None


def test_request_returns_response_on_success():
    with patch.object(scraper.requests, "get", return_value=_mock_response("a" * 5000, 200)):
        r = scraper._request("http://x")
        assert r is not None and r.status_code == 200


def test_motley_fool_index_handles_request_failure():
    with patch.object(scraper, "_request", return_value=None):
        urls = scraper.scrape_motley_fool_index(pages=1)
        assert urls == []


def test_parse_motley_fool_article_returns_none_when_body_missing():
    html = "<html><body><div class='other'>nothing here</div></body></html>"
    with patch.object(scraper, "_request", return_value=_mock_response(html * 50, 200)):
        out = scraper.parse_motley_fool_article("http://x")
        assert out is None


def test_save_all_writes_files(tmp_path, monkeypatch):
    monkeypatch.setattr(scraper, "TRANSCRIPTS_DIR", tmp_path)
    transcripts = scraper.synthetic_calls(tickers=["AAPL"], per_ticker=2,
                                          start_year=2024, end_year=2024)
    n = scraper.save_all(transcripts)
    assert n == 2
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_quarter_dates_count():
    out = scraper._quarter_dates(2023, 2023)
    assert len(out) == 4
    assert all(label.startswith("Q") for _, label in out)
