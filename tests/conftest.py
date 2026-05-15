"""Shared pytest fixtures for the EEI test suite."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_numbered_module(filename: str, alias: str):
    """Import a `src/N_*.py` module under a clean alias.

    Modules whose names begin with a digit cannot be imported with `import`,
    so we use importlib here. We must register in `sys.modules` BEFORE running
    the module, otherwise dataclass decorators (Python 3.13) blow up trying
    to resolve `sys.modules[cls.__module__].__dict__`.
    """
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, ROOT / "src" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load all four numbered modules so test files can `from conftest import ...`
# or just call the loader.
SCRAPER = _load_numbered_module("1_scraper.py", "eei_scraper")
PARSER = _load_numbered_module("2_parser.py", "eei_parser")
SCORER = _load_numbered_module("3_evasion_scorer.py", "eei_scorer")
BACKTESTER = _load_numbered_module("4_backtester.py", "eei_backtester")


@pytest.fixture
def scraper_module():
    return SCRAPER


@pytest.fixture
def parser_module():
    return PARSER


@pytest.fixture
def scorer_module():
    return SCORER


@pytest.fixture
def backtester_module():
    return BACKTESTER


# ---------------------------------------------------------------------------
# Transcript / Q&A fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_transcript_text() -> str:
    """A realistic multi-turn earnings call transcript (~500 words)."""
    return """Operator: Good afternoon and welcome to the Apple Inc. Q2 2024 earnings conference call.
At this time all participants are in a listen-only mode. After the speakers' presentation
there will be a question-and-answer session.

IR: Thank you operator. With me on the call today are Tim Cook and Luca Maestri. Before we begin
I'd like to remind you that today's discussion will include forward-looking statements.

[Prepared remarks omitted for brevity]

Operator: Now we'll open up the call for questions.

Toni Sacconaghi -- Bernstein -- Analyst
On gross margin -- you guided 45% last quarter and you printed 46.6%. What were the puts and
takes and how should we think about the trajectory next quarter?

Luca Maestri -- Chief Financial Officer
Sure Toni. Gross margin came in at 46.6% which was ahead of our internal plan by about 80 basis
points. The puts were favorable FX and lower freight; the takes were promotional intensity in
the wearables segment. Going into next quarter we expect a roughly 45-46% range, and for the
full year we're tightening to 45-47%. The biggest line-item delta is component costs.

Katy Huberty -- Morgan Stanley -- Analyst
Capex was 22% above the Street. What's the duration of this elevated investment cycle?

Tim Cook -- Chief Executive Officer
Look, as we've discussed previously, we don't break out the components of capex at that level
of granularity. What I would say is we feel really good about the long-term trajectory and we'll
continue to invest behind the things that matter for our customers. Over time you should expect
us to deliver, and we're confident in our ability to execute against our plan. As I mentioned in
the prepared remarks there are a lot of moving pieces in any given quarter.

Eric Sheridan -- Goldman Sachs -- Analyst
Can you give us an update on the DOJ antitrust matter? Any view on timing or potential exposure?

Tim Cook -- Chief Executive Officer
On the legal matter -- as you'd expect, I'm not in a position to comment on ongoing litigation
beyond what's in our public disclosures. What I would say more broadly is we have strong defenses,
we believe the positions we've taken are correct, and we'll defend them vigorously. I'd refer you
to the 10-Q for the specific language.

Wamsi Mohan -- Bank of America -- Analyst
Headcount was down 4% sequentially. Is this a structural reset or a pause?

Luca Maestri -- Chief Financial Officer
Thanks Wamsi. The headcount reduction reflects normal attrition plus targeted reorganizations in
two business units. We continue to invest in AI and silicon engineering. The operating-margin
glide path is unchanged from what we communicated at the analyst day.

Operator: This concludes today's question-and-answer session. Thank you.
"""


@pytest.fixture
def sample_qa_pairs() -> list[dict]:
    """Five realistic Q&A pairs with mixed evasion levels."""
    return [
        {
            "analyst_name": "Toni Sacconaghi", "analyst_firm": "Bernstein",
            "question_text": "On gross margin you guided 45% and you printed 46.6%. What are the puts and takes?",
            "question_topics": ["margins"],
            "answer_text": "Gross margin came in at 46.6%, 80 basis points ahead of plan. The puts were favorable FX and lower freight. We expect 45-46% next quarter.",
            "answer_word_count": 25, "executive_name": "Luca Maestri",
            "executive_title": "Chief Financial Officer",
            "hedge_count": 0, "deflection_keywords": 0, "question_marks_in_answer": 0,
            "answer_question_word_overlap": 0.35, "answer_to_question_length_ratio": 1.2,
        },
        {
            "analyst_name": "Tim Smith", "analyst_firm": "Goldman Sachs",
            "question_text": "What's your full-year guidance?",
            "question_topics": ["guidance"],
            "answer_text": "Look, as we've discussed previously, we don't break out guidance at that level. We feel good about the long-term trajectory and we'll continue to invest. Over time you should expect us to deliver. As I mentioned, there are a lot of moving pieces.",
            "answer_word_count": 45, "executive_name": "Tim Cook",
            "executive_title": "Chief Executive Officer",
            "hedge_count": 5, "deflection_keywords": 4, "question_marks_in_answer": 0,
            "answer_question_word_overlap": 0.05, "answer_to_question_length_ratio": 9.0,
        },
        {
            "analyst_name": "Eric Sheridan", "analyst_firm": "Goldman Sachs",
            "question_text": "Update on the DOJ antitrust case?",
            "question_topics": ["litigation"],
            "answer_text": "I'm not in a position to comment on ongoing litigation beyond our public disclosures. We'll defend vigorously. I'd refer you to the 10-Q.",
            "answer_word_count": 22, "executive_name": "Tim Cook",
            "executive_title": "Chief Executive Officer",
            "hedge_count": 1, "deflection_keywords": 1, "question_marks_in_answer": 0,
            "answer_question_word_overlap": 0.08, "answer_to_question_length_ratio": 4.4,
        },
        {
            "analyst_name": "Brian Nowak", "analyst_firm": "Morgan Stanley",
            "question_text": "Capex came in 20% higher than expectations. What's the duration?",
            "question_topics": ["capex"],
            "answer_text": "Capex was elevated and reflects peak build-out. Run-rate normalizes back-half next year, and FCF conversion gets back above 90% by mid-2026. The driver is data center buildout.",
            "answer_word_count": 30, "executive_name": "Luca Maestri",
            "executive_title": "Chief Financial Officer",
            "hedge_count": 0, "deflection_keywords": 0, "question_marks_in_answer": 0,
            "answer_question_word_overlap": 0.25, "answer_to_question_length_ratio": 2.0,
        },
        {
            "analyst_name": "Mike Mayo", "analyst_firm": "Wells Fargo",
            "question_text": "Headcount was down 4%. Structural reset or pause?",
            "question_topics": ["hiring"],
            "answer_text": "The reduction reflects attrition and targeted reorganizations. We continue to invest in AI and engineering. The margin path is unchanged from analyst day.",
            "answer_word_count": 24, "executive_name": "Luca Maestri",
            "executive_title": "Chief Financial Officer",
            "hedge_count": 0, "deflection_keywords": 0, "question_marks_in_answer": 0,
            "answer_question_word_overlap": 0.20, "answer_to_question_length_ratio": 2.4,
        },
    ]


@pytest.fixture
def sample_eei_df() -> pd.DataFrame:
    """A small but realistic EEI scores DataFrame, 3 tickers x 4 quarters."""
    rng = np.random.default_rng(42)
    rows = []
    for ticker in ("AAPL", "MSFT", "GOOGL"):
        for q, dt in enumerate(("2024-01-25", "2024-04-25", "2024-07-25", "2024-10-25")):
            base = {"AAPL": 0.25, "MSFT": 0.20, "GOOGL": 0.45}[ticker]
            score = float(np.clip(base + rng.normal(0, 0.05), 0, 1))
            rows.append({
                "ticker": ticker, "company": ticker, "date": dt,
                "quarter": f"Q{q+1} 2024", "n_pairs": 12,
                "EEI_raw": score, "EEI_weighted": score * 1.02,
                "evasion_concentration": 0.1, "fully_evasive_pct": 0.15,
                "red_flag_count": q,
                "EEI_topic_guidance": score, "EEI_topic_margins": score - 0.05,
                "tactic_topic_pivot": 0.3, "tactic_verbosity_shield": 0.2,
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["EEI_delta"] = df.groupby("ticker")["EEI_raw"].diff()
    return df


@pytest.fixture
def sample_prices_df() -> pd.DataFrame:
    """100 trading days of synthetic prices for 3 tickers."""
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(7)
    df = pd.DataFrame(index=idx)
    for t in ("AAPL", "MSFT", "GOOGL", "SPY"):
        rets = rng.normal(0.0005, 0.012, len(idx))
        df[t] = 100 * np.exp(np.cumsum(rets))
    return df


# ---------------------------------------------------------------------------
# Mocked Anthropic API response
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_anthropic_response() -> dict:
    """Realistic JSON payload an LLM scorer would emit."""
    return {
        "evasion_level": "fully_evasive",
        "evasion_score": 0.82,
        "question_was_answered": False,
        "answer_addresses_question": 0.20,
        "evasion_tactics": {
            "topic_pivot": True, "false_precision": False, "time_deflection": True,
            "legal_shield": False, "verbosity_shield": True, "question_reframing": True,
            "competitive_shield": False, "macro_deflection": False,
        },
        "topic_evaded": "guidance",
        "severity_rationale": "Multiple hedging phrases and a clear topic pivot away from numerical guidance.",
        "red_flag": True,
        "red_flag_reason": "verbose response with no numerical content",
    }


@pytest.fixture
def mock_anthropic_client(mock_anthropic_response):
    """An AsyncMock client whose .messages.create returns the fixture JSON."""
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(mock_anthropic_response))]
    msg.usage = MagicMock(input_tokens=350, output_tokens=180)
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=msg)
    return client


@pytest.fixture
def tmp_cache_dir(tmp_path, monkeypatch):
    """Redirect CACHE_DIR to a tmp folder so tests don't pollute the real cache."""
    cache = tmp_path / "cache"
    cache.mkdir()
    import config
    monkeypatch.setattr(config, "CACHE_DIR", cache)
    # The scorer module captured CACHE_DIR at import time — patch it there too.
    monkeypatch.setattr(SCORER, "CACHE_DIR", cache)
    return cache
