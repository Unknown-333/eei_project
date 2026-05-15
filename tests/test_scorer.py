"""Tests for src/3_evasion_scorer.py."""

from __future__ import annotations

import asyncio
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eei_scorer as scorer  # registered by conftest


# ---------------------------------------------------------------------------
# Heuristic scorer invariants
# ---------------------------------------------------------------------------
def test_heuristic_score_in_range(sample_qa_pairs):
    for p in sample_qa_pairs:
        s = scorer.heuristic_score(p, "AAPL", "2024-04-25")
        assert 0.0 <= s["evasion_score"] <= 1.0


def test_heuristic_levels_match_score_buckets(sample_qa_pairs):
    for p in sample_qa_pairs:
        s = scorer.heuristic_score(p, "AAPL", "2024-04-25")
        if s["evasion_score"] < 0.35:
            assert s["evasion_level"] == "direct"
        elif s["evasion_score"] < 0.65:
            assert s["evasion_level"] == "intermediate"
        else:
            assert s["evasion_level"] == "fully_evasive"


def test_heuristic_evasive_pair_higher_than_direct_pair(sample_qa_pairs):
    direct = scorer.heuristic_score(sample_qa_pairs[0], "AAPL", "d")["evasion_score"]
    evasive = scorer.heuristic_score(sample_qa_pairs[1], "AAPL", "d")["evasion_score"]
    assert evasive > direct


def test_heuristic_legal_shield_detected(sample_qa_pairs):
    s = scorer.heuristic_score(sample_qa_pairs[2], "AAPL", "d")
    assert s["evasion_tactics"]["legal_shield"] is True


def test_heuristic_red_flag_field_present(sample_qa_pairs):
    for p in sample_qa_pairs:
        s = scorer.heuristic_score(p, "AAPL", "d")
        assert isinstance(s["red_flag"], (bool, np.bool_))


# ---------------------------------------------------------------------------
# JSON parsing of LLM responses
# ---------------------------------------------------------------------------
def test_parse_json_clean(mock_anthropic_response):
    raw = json.dumps(mock_anthropic_response)
    parsed = scorer._parse_json(raw)
    assert parsed["evasion_score"] == 0.82


def test_parse_json_with_preamble(mock_anthropic_response):
    raw = "Here is the JSON:\n" + json.dumps(mock_anthropic_response) + "\nThanks."
    parsed = scorer._parse_json(raw)
    assert parsed is not None
    assert parsed["evasion_level"] == "fully_evasive"


def test_parse_json_garbage():
    assert scorer._parse_json("not json at all") is None


def test_parse_json_truncated():
    assert scorer._parse_json('{"evasion_score": 0.5,') is None


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def test_cache_roundtrip(tmp_cache_dir, sample_qa_pairs, mock_anthropic_response):
    pair = sample_qa_pairs[0]
    assert scorer._load_cached(pair, "llm") is None
    scorer._save_cached(pair, "llm", mock_anthropic_response)
    cached = scorer._load_cached(pair, "llm")
    assert cached is not None
    assert cached["evasion_score"] == 0.82


def test_cache_keyed_by_text(tmp_cache_dir, sample_qa_pairs, mock_anthropic_response):
    a, b = sample_qa_pairs[0], sample_qa_pairs[1]
    scorer._save_cached(a, "llm", mock_anthropic_response)
    assert scorer._load_cached(b, "llm") is None  # different question/answer => different key


# ---------------------------------------------------------------------------
# Async LLM scoring (mocked)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_llm_score_one_returns_parsed_json(mock_anthropic_client, sample_qa_pairs):
    sem = asyncio.Semaphore(1)
    meter = scorer.CostMeter()
    out = await scorer.llm_score_one(
        mock_anthropic_client, sample_qa_pairs[0], "AAPL", "2024-04-25", sem, meter,
    )
    assert out is not None
    assert out["evasion_score"] == 0.82
    assert meter.n_calls == 1
    assert meter.in_tokens == 350
    assert meter.out_tokens == 180


@pytest.mark.asyncio
async def test_llm_score_one_retries_on_exception(sample_qa_pairs, mock_anthropic_response):
    bad_msg = MagicMock()
    bad_msg.content = [MagicMock(text="not-json")]
    bad_msg.usage = MagicMock(input_tokens=10, output_tokens=5)
    good_msg = MagicMock()
    good_msg.content = [MagicMock(text=json.dumps(mock_anthropic_response))]
    good_msg.usage = MagicMock(input_tokens=10, output_tokens=5)

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=[bad_msg, good_msg])

    sem = asyncio.Semaphore(1)
    out = await scorer.llm_score_one(
        client, sample_qa_pairs[0], "AAPL", "2024-04-25", sem, scorer.CostMeter(),
        max_retries=3,
    )
    assert out is not None
    assert out["evasion_score"] == 0.82


@pytest.mark.asyncio
async def test_llm_score_one_gives_up_after_max_retries(sample_qa_pairs):
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("boom"))
    out = await scorer.llm_score_one(
        client, sample_qa_pairs[0], "AAPL", "2024-04-25",
        asyncio.Semaphore(1), scorer.CostMeter(), max_retries=2,
    )
    assert out is None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _scored_call(sample_qa_pairs):
    call = {
        "ticker": "AAPL", "company": "Apple", "date": "2024-04-25",
        "quarter": "Q2 2024", "qa_pairs": sample_qa_pairs,
    }
    scored = [scorer.heuristic_score(p, "AAPL", "d") for p in sample_qa_pairs]
    return call, scored


def test_aggregate_call_basic_fields(sample_qa_pairs):
    call, scored = _scored_call(sample_qa_pairs)
    row = scorer.aggregate_call(call, scored)
    assert row["ticker"] == "AAPL"
    assert row["n_pairs"] == len(sample_qa_pairs)
    assert 0.0 <= row["EEI_raw"] <= 1.0
    assert 0.0 <= row["EEI_weighted"] <= 1.0
    assert 0.0 <= row["fully_evasive_pct"] <= 1.0


def test_aggregate_call_weighted_differs_for_tier1():
    qa = [
        {"analyst_name": "x", "analyst_firm": "Goldman Sachs", "question_text": "q", "question_topics": [], "answer_text": "a", "answer_word_count": 1, "executive_name": "x", "executive_title": "x",
         "hedge_count": 0, "deflection_keywords": 0, "question_marks_in_answer": 0, "answer_question_word_overlap": 0.5, "answer_to_question_length_ratio": 1.0},
        {"analyst_name": "x", "analyst_firm": "Small Bank LLC", "question_text": "q", "question_topics": [], "answer_text": "a", "answer_word_count": 1, "executive_name": "x", "executive_title": "x",
         "hedge_count": 0, "deflection_keywords": 0, "question_marks_in_answer": 0, "answer_question_word_overlap": 0.5, "answer_to_question_length_ratio": 1.0},
    ]
    call = {"ticker": "T", "company": "T", "date": "d", "quarter": "q", "qa_pairs": qa}
    scored = [{"evasion_score": 0.9, "evasion_level": "fully_evasive", "evasion_tactics": {}, "red_flag": False},
              {"evasion_score": 0.1, "evasion_level": "direct", "evasion_tactics": {}, "red_flag": False}]
    row = scorer.aggregate_call(call, scored)
    # Tier-1 (Goldman) is the high-score one and gets weight 1.5 → weighted > raw.
    assert row["EEI_weighted"] > row["EEI_raw"]


def test_add_cross_call_features_computes_delta(sample_eei_df):
    df = scorer.add_cross_call_features(sample_eei_df.drop(columns=["EEI_delta"]))
    assert "EEI_delta" in df.columns
    # First chronological row of each ticker must have NaN delta.
    df_sorted = df.sort_values(["ticker", "date"])
    firsts = df_sorted.groupby("ticker").nth(0)
    assert firsts["EEI_delta"].isna().all()


def test_add_cross_call_features_trend_column_exists(sample_eei_df):
    df = scorer.add_cross_call_features(sample_eei_df.drop(columns=["EEI_delta"]))
    assert "EEI_trend" in df.columns


def test_cost_meter_accumulates():
    m = scorer.CostMeter()
    m.add(100, 50)
    m.add(200, 75)
    assert m.in_tokens == 300
    assert m.out_tokens == 125
    assert m.n_calls == 2
    assert m.usd > 0
