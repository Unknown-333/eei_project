"""
Configuration module for the Executive Evasion Index (EEI) research system.

All tunable parameters, API credentials, and lexical resources are centralized
here so the rest of the codebase can stay deterministic and reproducible.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
TRANSCRIPTS_DIR: Final[Path] = DATA_DIR / "transcripts"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
PRICES_DIR: Final[Path] = DATA_DIR / "prices"
CACHE_DIR: Final[Path] = DATA_DIR / "cache"
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"

for _d in (TRANSCRIPTS_DIR, PROCESSED_DIR, PRICES_DIR, CACHE_DIR, OUTPUT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Anthropic / LLM
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: Final[str | None] = os.getenv("ANTHROPIC_API_KEY")
MODEL: Final[str] = os.getenv("EEI_MODEL", "claude-opus-4-20250514")
SCORER_MODEL: Final[str] = os.getenv("EEI_SCORER_MODEL", MODEL)
SYNTHETIC_MODEL: Final[str] = os.getenv("EEI_SYNTHETIC_MODEL", "claude-sonnet-4-20250514")
MAX_TOKENS: Final[int] = 1000
SYNTHETIC_MAX_TOKENS: Final[int] = 4096
LLM_CONCURRENCY: Final[int] = int(os.getenv("EEI_CONCURRENCY", "8"))

# Pricing (USD per million tokens) — used purely for cost telemetry.
COST_PER_MTOK_INPUT: Final[float] = 15.0
COST_PER_MTOK_OUTPUT: Final[float] = 75.0

# ---------------------------------------------------------------------------
# Universe and time window
# ---------------------------------------------------------------------------
TICKERS: Final[list[str]] = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA",
    "JPM", "GS", "BAC", "C", "MS",
    "XOM", "CVX",
    "WMT", "HD", "COST",
    "PFE", "JNJ", "UNH",
    "TSLA",
]

START_DATE: Final[str] = "2021-01-01"
END_DATE: Final[str] = "2024-12-31"

# Number of synthetic transcripts per ticker when scraping fails.
SYNTHETIC_PER_TICKER: Final[int] = 12

# ---------------------------------------------------------------------------
# Backtest parameters
# ---------------------------------------------------------------------------
RISK_FREE_RATE: Final[float] = 0.045
TRADING_DAYS: Final[int] = 252
QUINTILES: Final[int] = 5
HORIZONS_DAYS: Final[tuple[int, ...]] = (1, 5, 20, 60)
BENCHMARK_TICKER: Final[str] = "SPY"

# Tier-1 sell-side firms get higher weight in EEI_weighted.
TIER1_FIRMS: Final[set[str]] = {
    "goldman sachs", "goldman", "morgan stanley", "jpmorgan", "j.p. morgan",
    "citi", "citigroup", "bank of america", "bofa", "merrill lynch",
}
TIER1_WEIGHT: Final[float] = 1.5
TIER2_WEIGHT: Final[float] = 1.0

# ---------------------------------------------------------------------------
# Lexicons (deterministic, fast linguistic features)
# ---------------------------------------------------------------------------
HEDGE_WORDS: Final[list[str]] = [
    "approximately", "roughly", "around", "we believe", "we expect", "we think",
    "going forward", "over time", "broadly", "generally", "potentially", "may",
    "might", "could", "should", "would", "in the near term", "in due course",
    "in general", "to some extent", "for the most part",
]

DEFLECTION_PHRASES: Final[list[str]] = [
    "as i mentioned", "as we've discussed", "as we discussed",
    "in general", "broadly speaking", "over time",
    "we don't comment", "we don't disclose", "we don't break out",
    "as you know", "look,", "again,", "i'd refer you to",
    "it's too early", "we'll talk about that",
]

TOPIC_KEYWORDS: Final[dict[str, list[str]]] = {
    "margins": ["margin", "gross margin", "operating margin", "profitability"],
    "guidance": ["guidance", "outlook", "forecast", "next quarter", "full year", "fy"],
    "competition": ["competitor", "competition", "market share", "competitive"],
    "capex": ["capex", "capital expenditure", "investment", "spend"],
    "debt": ["debt", "leverage", "balance sheet", "refinanc"],
    "litigation": ["litigation", "lawsuit", "regulatory", "investigation", "doj", "ftc"],
    "macro": ["macro", "recession", "fed", "rates", "inflation", "tariff"],
    "hiring": ["hiring", "headcount", "layoff", "workforce", "employees"],
    "product": ["product", "launch", "roadmap", "ai", "model", "platform"],
    "buybacks": ["buyback", "repurchase", "dividend", "capital return"],
}

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
SEED: Final[int] = 42
LOG_FILE: Final[Path] = LOG_DIR / "eei.log"


def cost_estimate(input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for an Anthropic call."""
    return (
        input_tokens / 1_000_000 * COST_PER_MTOK_INPUT
        + output_tokens / 1_000_000 * COST_PER_MTOK_OUTPUT
    )
