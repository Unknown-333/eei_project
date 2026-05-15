"""End-to-end integration test: scrape (synthetic) → parse → score → backtest."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import write_json  # noqa: E402
import eei_scraper as scraper  # noqa: E402  registered by conftest
import eei_parser as parser  # noqa: E402
import eei_scorer as scorer  # noqa: E402
import eei_backtester as bt  # noqa: E402


REQUIRED_EEI_COLS = {
    "ticker", "company", "date", "quarter", "n_pairs",
    "EEI_raw", "EEI_weighted", "evasion_concentration", "fully_evasive_pct",
    "red_flag_count", "EEI_delta",
}


@pytest.mark.integration
def test_full_pipeline_synthetic(tmp_path, monkeypatch, sample_prices_df):
    # Redirect all data dirs to tmp.
    import config
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    for d in (config.TRANSCRIPTS_DIR, config.PROCESSED_DIR,
              config.CACHE_DIR, config.OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    # Re-bind the constants that the modules captured at import time.
    monkeypatch.setattr(scraper, "TRANSCRIPTS_DIR", config.TRANSCRIPTS_DIR)
    monkeypatch.setattr(parser, "TRANSCRIPTS_DIR", config.TRANSCRIPTS_DIR)
    monkeypatch.setattr(parser, "PROCESSED_DIR", config.PROCESSED_DIR)
    monkeypatch.setattr(scorer, "PROCESSED_DIR", config.PROCESSED_DIR)
    monkeypatch.setattr(scorer, "OUTPUT_DIR", config.OUTPUT_DIR)
    monkeypatch.setattr(scorer, "CACHE_DIR", config.CACHE_DIR)
    monkeypatch.setattr(bt, "OUTPUT_DIR", config.OUTPUT_DIR)

    # 1. Scrape (synthetic) — 3 tickers x 4 calls.
    calls = scraper.synthetic_calls(
        tickers=["AAPL", "MSFT", "GOOGL"], per_ticker=4,
        start_year=2024, end_year=2024,
    )
    n = scraper.save_all(calls)
    assert n == 12

    # 2. Parse.
    files = sorted(config.TRANSCRIPTS_DIR.glob("*.json"))
    parsed_count = 0
    for f in files:
        call = parser.parse_transcript_file(f)
        if not call.qa_pairs:
            continue
        out_name = f"{call.ticker}_{call.date}_qa_pairs.json"
        from dataclasses import asdict
        write_json(config.PROCESSED_DIR / out_name, {
            "company": call.company, "ticker": call.ticker,
            "date": call.date, "quarter": call.quarter,
            "source": call.source,
            "qa_pairs": [asdict(p) for p in call.qa_pairs],
        })
        parsed_count += 1
    assert parsed_count == 12

    # 3. Score (heuristic — never hits real API).
    rows = []
    for f in sorted(config.PROCESSED_DIR.glob("*_qa_pairs.json")):
        from src.utils import read_json
        c = read_json(f)
        scored = scorer.score_call_heuristic(c)
        row = scorer.aggregate_call(c, scored)
        if row:
            rows.append(row)
    df = pd.DataFrame(rows)
    df = scorer.add_cross_call_features(df)
    out_csv = config.OUTPUT_DIR / "eei_scores.csv"
    df.to_csv(out_csv, index=False)

    assert out_csv.exists()
    assert REQUIRED_EEI_COLS.issubset(set(df.columns)), \
        f"Missing columns: {REQUIRED_EEI_COLS - set(df.columns)}"

    # 4. Backtester reads the CSV and produces stats.
    eei = pd.read_csv(out_csv, parse_dates=["date"])
    cfg = bt.SignalConfig("EEI Level", "EEI_raw", direction=-1)

    # Build prices that include all tickers we generated, plus SPY benchmark.
    prices = sample_prices_df.copy()
    panel = bt.build_signal_panel(eei, prices, cfg, horizon=5)
    if not panel.empty:
        ls = bt.long_short_returns(panel, direction=-1)
        if not ls.empty:
            stats = bt.stats_dict(ls["long_short"], ppy=bt.annualization_factor(5), label="EEI Level")
            for k in ("ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_rate"):
                assert k in stats
