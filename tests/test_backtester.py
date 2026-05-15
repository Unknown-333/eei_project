"""Tests for src/4_backtester.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eei_backtester as bt  # registered by conftest


# ---------------------------------------------------------------------------
# Sharpe / drawdown
# ---------------------------------------------------------------------------
def test_sharpe_positive_returns():
    r = pd.Series([0.01, 0.02, -0.005, 0.015, 0.008])
    s = bt.sharpe(r, periods_per_year=4, rf=0.0)
    assert s > 0


def test_sharpe_zero_std_returns_nan():
    r = pd.Series([0.01, 0.01, 0.01])
    s = bt.sharpe(r, periods_per_year=4, rf=0.0)
    assert np.isnan(s)


def test_sharpe_empty_series():
    s = bt.sharpe(pd.Series([], dtype=float), periods_per_year=4)
    assert np.isnan(s)


def test_max_drawdown_known_input():
    cum = pd.Series([1.0, 1.10, 1.20, 0.90, 1.05, 1.30])
    # Peak at 1.20, trough at 0.90 → DD = -0.25
    dd = bt.max_drawdown(cum)
    assert dd == pytest.approx(-0.25, abs=1e-6)


def test_max_drawdown_monotonic_up():
    cum = pd.Series([1.0, 1.1, 1.2, 1.3])
    assert bt.max_drawdown(cum) == pytest.approx(0.0)


def test_max_drawdown_empty():
    assert np.isnan(bt.max_drawdown(pd.Series([], dtype=float)))


# ---------------------------------------------------------------------------
# Annualization helper
# ---------------------------------------------------------------------------
def test_annualization_factor():
    assert bt.annualization_factor(1) == bt.TRADING_DAYS
    assert bt.annualization_factor(20) == pytest.approx(bt.TRADING_DAYS / 20)
    assert bt.annualization_factor(60) == pytest.approx(bt.TRADING_DAYS / 60)


# ---------------------------------------------------------------------------
# Forward-return alignment + signal panel
# ---------------------------------------------------------------------------
def test_forward_returns_horizon(sample_prices_df):
    fr = bt.forward_returns(sample_prices_df, horizon=5)
    # First 5 rows should be NaN at the *end*; explicit check on a known offset.
    assert fr["AAPL"].iloc[0] == pytest.approx(
        sample_prices_df["AAPL"].iloc[5] / sample_prices_df["AAPL"].iloc[0] - 1
    )


def test_build_signal_panel_quintile_assignment(sample_eei_df, sample_prices_df):
    cfg = bt.SignalConfig("EEI Level", "EEI_raw", direction=-1)
    panel = bt.build_signal_panel(sample_eei_df, sample_prices_df, cfg, horizon=5)
    assert not panel.empty
    assert {"score", "fwd_ret", "quintile"}.issubset(panel.columns)
    assert panel["fwd_ret"].notna().all()


def test_build_signal_panel_skips_missing_tickers(sample_eei_df, sample_prices_df):
    df = sample_eei_df.copy()
    df.loc[df["ticker"] == "GOOGL", "ticker"] = "ZZZZ"  # not in prices
    cfg = bt.SignalConfig("EEI Level", "EEI_raw")
    panel = bt.build_signal_panel(df, sample_prices_df, cfg, horizon=5)
    assert "ZZZZ" not in set(panel["ticker"])


# ---------------------------------------------------------------------------
# Long/short returns
# ---------------------------------------------------------------------------
def test_long_short_returns_columns(sample_eei_df, sample_prices_df):
    cfg = bt.SignalConfig("EEI Level", "EEI_raw", direction=-1)
    panel = bt.build_signal_panel(sample_eei_df, sample_prices_df, cfg, horizon=5)
    if panel["quintile"].notna().sum() == 0:
        pytest.skip("not enough cross-sectional spread for quintile in tiny fixture")
    ls = bt.long_short_returns(panel, direction=-1)
    if not ls.empty:
        assert {"long", "short", "long_short"}.issubset(ls.columns)


def test_long_short_returns_empty_panel():
    out = bt.long_short_returns(pd.DataFrame(), direction=-1)
    assert out.empty


# ---------------------------------------------------------------------------
# IC
# ---------------------------------------------------------------------------
def test_information_coefficient_matches_scipy():
    rng = np.random.default_rng(0)
    rows = []
    for q in pd.period_range("2024Q1", periods=2, freq="Q"):
        scores = rng.normal(0, 1, 30)
        rets = -0.5 * scores + rng.normal(0, 0.5, 30)  # negative correlation by construction
        for s, r in zip(scores, rets):
            rows.append({"quarter": q, "score": float(s), "fwd_ret": float(r)})
    panel = pd.DataFrame(rows)
    ic_mean, ic_series = bt.information_coefficient(panel, direction=-1)
    # direction=-1 flips the sign so IC should be POSITIVE.
    assert ic_mean > 0
    rho_q1, _ = spearmanr(-panel[panel["quarter"] == panel["quarter"].iloc[0]]["score"],
                          panel[panel["quarter"] == panel["quarter"].iloc[0]]["fwd_ret"])
    assert ic_series.iloc[0] == pytest.approx(rho_q1, rel=1e-6)


def test_information_coefficient_empty_panel():
    ic_mean, ic_series = bt.information_coefficient(pd.DataFrame(), direction=-1)
    assert np.isnan(ic_mean)
    assert ic_series.empty


# ---------------------------------------------------------------------------
# Stats dict
# ---------------------------------------------------------------------------
def test_stats_dict_keys():
    r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015])
    s = bt.stats_dict(r, ppy=4, label="X")
    for key in ("ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_rate"):
        assert key in s


def test_stats_dict_empty_series():
    s = bt.stats_dict(pd.Series([], dtype=float), ppy=4, label="X")
    assert s == {"label": "X"}


# ---------------------------------------------------------------------------
# Alpha vs benchmark
# ---------------------------------------------------------------------------
def test_alpha_beta_vs_benchmark_recovers_known_beta():
    rng = np.random.default_rng(11)
    bench = pd.Series(rng.normal(0.005, 0.02, 100))
    strat = 0.001 + 0.7 * bench + rng.normal(0, 0.005, 100)
    alpha, beta, r2 = bt.alpha_beta_vs_benchmark(strat, bench, ppy=4)
    assert beta == pytest.approx(0.7, abs=0.1)
    assert r2 > 0.5


def test_alpha_beta_too_few_points_returns_nan():
    a, b, r = bt.alpha_beta_vs_benchmark(pd.Series([0.01]), pd.Series([0.01]), ppy=4)
    assert np.isnan(a) and np.isnan(b) and np.isnan(r)


# ---------------------------------------------------------------------------
# Trading-day alignment + benchmark return helper
# ---------------------------------------------------------------------------
def test_next_trading_day_finds_following_session():
    idx = pd.bdate_range("2024-01-02", periods=10)
    out = bt._next_trading_day(idx, pd.Timestamp("2024-01-06"))  # Saturday
    assert out is not None and out.weekday() < 5


def test_next_trading_day_returns_none_when_past_end():
    idx = pd.bdate_range("2024-01-02", periods=10)
    out = bt._next_trading_day(idx, pd.Timestamp("2030-01-01"))
    assert out is None


def test_benchmark_quarterly_returns_returns_series(sample_prices_df):
    quarters = pd.PeriodIndex(["2024Q1", "2024Q2"], freq="Q")
    out = bt.benchmark_quarterly_returns(sample_prices_df["SPY"], quarters, horizon=20)
    assert isinstance(out, pd.Series)


def test_stats_dict_known_returns_sharpe_sign():
    r = pd.Series([0.05, 0.04, 0.03, 0.06, 0.05])  # all positive, low std
    s = bt.stats_dict(r, ppy=4, label="X")
    assert s["sharpe"] > 0
    assert s["hit_rate"] == 1.0
