"""
Module 4 — Backtester.

Tests three signals built from the per-call EEI table emitted by module 3:

    Signal 1 — EEI Level
        Each rebalance, rank the cross-section by EEI_raw.
        SHORT the top quintile (most evasive), LONG the bottom quintile.

    Signal 2 — EEI Delta (typically the strongest signal)
        Rank by EEI_delta = EEI_raw - prior-quarter EEI_raw.
        SHORT companies whose evasion increased the most.
        LONG companies whose evasion dropped the most.

    Signal 3 — Topic-Specific (guidance evasion)
        Same long/short construction, but ranks on EEI_topic_guidance.

For each signal we report the standard quant tear-sheet stats:
    annualized return, Sharpe, max drawdown, hit rate, IC (Spearman),
    ICIR, alpha vs SPY (OLS), and a multi-panel tear sheet PNG.

Usage::
    python src/4_backtester.py
    python src/4_backtester.py --quintiles 5 --no-plot
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    BENCHMARK_TICKER,
    END_DATE,
    HORIZONS_DAYS,
    OUTPUT_DIR,
    PRICES_DIR,
    QUINTILES,
    RISK_FREE_RATE,
    START_DATE,
    TICKERS,
    TRADING_DAYS,
)
from src.utils import get_logger  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
LOG = get_logger("backtester")
sns.set_theme(style="whitegrid", context="paper")


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------
def download_prices(tickers: list[str], force: bool = False) -> pd.DataFrame:
    """Download adjusted-close prices and cache to ``data/prices/prices.csv``."""
    cache = PRICES_DIR / "prices.csv"
    if cache.exists() and not force:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        missing = [t for t in tickers if t not in df.columns]
        if not missing:
            return df
        LOG.info("price cache missing tickers %s; refetching", missing)
    LOG.info("downloading prices for %d tickers", len(tickers))
    raw = yf.download(
        tickers, start=START_DATE, end=END_DATE,
        auto_adjust=True, progress=False, group_by="ticker", threads=True,
    )
    # yfinance returns either a single-level frame (1 ticker) or multi-level.
    if isinstance(raw.columns, pd.MultiIndex):
        close = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw.columns.levels[0]})
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.dropna(how="all").sort_index()
    close.to_csv(cache)
    return close


def forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return percentage forward returns over `horizon` trading days."""
    return prices.pct_change(horizon).shift(-horizon)


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def _next_trading_day(idx: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    pos = idx.searchsorted(day)
    if pos >= len(idx):
        return None
    return idx[pos]


@dataclass
class SignalConfig:
    name: str
    score_col: str
    direction: int = -1   # -1 → high score => short (negative alpha)


def build_signal_panel(
    eei: pd.DataFrame, prices: pd.DataFrame, cfg: SignalConfig, horizon: int,
) -> pd.DataFrame:
    """Return a per-call dataframe with score, T+h forward return, and quintile."""
    rows = []
    px_idx = prices.index
    for _, r in eei.iterrows():
        ticker = r["ticker"]
        if ticker not in prices.columns:
            continue
        score = r.get(cfg.score_col)
        if pd.isna(score):
            continue
        d_event = pd.Timestamp(r["date"])
        d_entry = _next_trading_day(px_idx, d_event)
        if d_entry is None:
            continue
        try:
            entry_pos = px_idx.get_loc(d_entry)
        except KeyError:
            continue
        exit_pos = entry_pos + horizon
        if exit_pos >= len(px_idx):
            continue
        p_in = prices[ticker].iloc[entry_pos]
        p_out = prices[ticker].iloc[exit_pos]
        if not (np.isfinite(p_in) and np.isfinite(p_out)) or p_in <= 0:
            continue
        rows.append({
            "ticker": ticker,
            "event_date": d_event,
            "entry_date": d_entry,
            "score": float(score),
            "fwd_ret": float(p_out / p_in - 1.0),
            "horizon": horizon,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Quintile rank within each event-quarter cross-section.
    df["quarter"] = df["event_date"].dt.to_period("Q")
    df["quintile"] = df.groupby("quarter")["score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), q=QUINTILES, labels=False, duplicates="drop") + 1
        if s.notna().sum() >= QUINTILES else np.nan
    )
    return df


def long_short_returns(panel: pd.DataFrame, direction: int) -> pd.DataFrame:
    """Aggregate to per-quarter long, short, and L/S returns."""
    if panel.empty:
        return pd.DataFrame()
    high = panel["quintile"] == QUINTILES
    low = panel["quintile"] == 1
    if direction == -1:
        # high score = short, low score = long
        short_leg = panel[high].groupby("quarter")["fwd_ret"].mean()
        long_leg = panel[low].groupby("quarter")["fwd_ret"].mean()
    else:
        long_leg = panel[high].groupby("quarter")["fwd_ret"].mean()
        short_leg = panel[low].groupby("quarter")["fwd_ret"].mean()
    out = pd.DataFrame({"long": long_leg, "short": short_leg})
    out["long_short"] = out["long"] - out["short"]
    return out.dropna(how="all").sort_index()


# ---------------------------------------------------------------------------
# Performance stats
# ---------------------------------------------------------------------------
def annualization_factor(horizon_days: int) -> float:
    """Approximate periods-per-year for given holding-period in trading days."""
    return TRADING_DAYS / horizon_days


def sharpe(returns: pd.Series, periods_per_year: float, rf: float = RISK_FREE_RATE) -> float:
    if returns.empty or returns.std(ddof=0) == 0:
        return float("nan")
    excess = returns - rf / periods_per_year
    return float(excess.mean() / returns.std(ddof=0) * np.sqrt(periods_per_year))


def max_drawdown(cum: pd.Series) -> float:
    if cum.empty:
        return float("nan")
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def alpha_beta_vs_benchmark(strategy_ret: pd.Series, bench_ret: pd.Series, ppy: float) -> tuple[float, float, float]:
    df = pd.concat([strategy_ret.rename("s"), bench_ret.rename("b")], axis=1).dropna()
    if len(df) < 4:
        return float("nan"), float("nan"), float("nan")
    X = sm.add_constant(df["b"])
    res = sm.OLS(df["s"], X).fit()
    alpha_per = res.params["const"]
    beta = res.params["b"]
    return float(alpha_per * ppy), float(beta), float(res.rsquared)


def information_coefficient(panel: pd.DataFrame, direction: int) -> tuple[float, pd.Series]:
    """Spearman IC between (direction * score) and forward returns, by quarter."""
    if panel.empty:
        return float("nan"), pd.Series(dtype=float)
    by_q = []
    for q, g in panel.groupby("quarter"):
        if g["fwd_ret"].nunique() < 3:
            continue
        rho, _ = spearmanr(direction * g["score"], g["fwd_ret"])
        if np.isnan(rho):
            continue
        by_q.append((q, rho))
    if not by_q:
        return float("nan"), pd.Series(dtype=float)
    s = pd.Series({q: r for q, r in by_q}).sort_index()
    return float(s.mean()), s


def benchmark_quarterly_returns(prices: pd.Series, ls_index: pd.PeriodIndex, horizon: int) -> pd.Series:
    """Compute the benchmark's H-day forward return measured at each L/S rebalance."""
    out: dict[pd.Period, float] = {}
    px_idx = prices.index
    for q in ls_index:
        # Use end of quarter as proxy event date; align to next trading day.
        d = pd.Timestamp(q.end_time.date())
        entry = _next_trading_day(px_idx, d)
        if entry is None:
            continue
        try:
            i0 = px_idx.get_loc(entry)
        except KeyError:
            continue
        i1 = i0 + horizon
        if i1 >= len(px_idx):
            continue
        out[q] = float(prices.iloc[i1] / prices.iloc[i0] - 1.0)
    return pd.Series(out).sort_index()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def stats_dict(returns: pd.Series, ppy: float, label: str) -> dict[str, float]:
    if returns.empty:
        return {"label": label}
    cum = (1 + returns.fillna(0)).cumprod()
    return {
        "label": label,
        "n_periods": int(returns.notna().sum()),
        "ann_return": float((cum.iloc[-1]) ** (ppy / max(1, len(returns))) - 1),
        "ann_vol": float(returns.std(ddof=0) * np.sqrt(ppy)),
        "sharpe": sharpe(returns, ppy),
        "max_drawdown": max_drawdown(cum),
        "hit_rate": float((returns > 0).mean()),
    }


def make_tearsheet(
    results: dict[str, dict],
    bench_ret: pd.Series,
    ic_series: pd.Series,
    out_path: Path,
) -> None:
    """Render a multi-panel performance tear sheet."""
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.3)

    # Panel 1: cumulative returns of every signal's L/S vs benchmark.
    ax1 = fig.add_subplot(gs[0, :])
    for name, payload in results.items():
        ls = payload["ls_returns"]["long_short"]
        cum = (1 + ls.fillna(0)).cumprod()
        ax1.plot(cum.index.to_timestamp(), cum.values, label=f"{name} L/S", linewidth=2)
    if not bench_ret.empty:
        cum_b = (1 + bench_ret.fillna(0)).cumprod()
        ax1.plot(cum_b.index.to_timestamp(), cum_b.values, label="SPY (T+20)", linestyle="--", color="grey")
    ax1.set_title("Cumulative return — long/short EEI signals vs benchmark", fontsize=13, weight="bold")
    ax1.set_ylabel("Growth of $1")
    ax1.legend(loc="best")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Panel 2: IC over time.
    ax2 = fig.add_subplot(gs[1, 0])
    if not ic_series.empty:
        ax2.bar(range(len(ic_series)), ic_series.values, color=["#2c7" if v > 0 else "#c33" for v in ic_series.values])
        ax2.axhline(0, color="k", linewidth=0.8)
        ax2.set_xticks(range(len(ic_series)))
        ax2.set_xticklabels([str(p) for p in ic_series.index], rotation=45, fontsize=8)
        ax2.set_title("EEI Δ signal IC by quarter (T+20)", fontsize=11, weight="bold")
        ax2.set_ylabel("Spearman ρ")

    # Panel 3: long vs short leg cumulative.
    ax3 = fig.add_subplot(gs[1, 1])
    payload = next(iter(results.values()))
    ls = payload["ls_returns"]
    for col, color in [("long", "#2c7"), ("short", "#c33"), ("long_short", "#36c")]:
        if col in ls.columns:
            cum = (1 + ls[col].fillna(0)).cumprod()
            ax3.plot(cum.index.to_timestamp(), cum.values, label=col, color=color, linewidth=2)
    ax3.set_title(f"{next(iter(results.keys()))} — long vs short legs", fontsize=11, weight="bold")
    ax3.legend()

    # Panel 4: stats table.
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    table_data = []
    for name, payload in results.items():
        s = payload["stats"]
        table_data.append([
            name,
            f"{s.get('ann_return', np.nan):.2%}",
            f"{s.get('sharpe', np.nan):.2f}",
            f"{s.get('max_drawdown', np.nan):.2%}",
            f"{s.get('hit_rate', np.nan):.2%}",
            f"{payload.get('ic_mean', np.nan):.3f}",
            f"{payload.get('icir', np.nan):.2f}",
            f"{payload.get('alpha', np.nan):.2%}",
            f"{payload.get('beta', np.nan):.2f}",
        ])
    headers = ["Signal", "Ann. Ret", "Sharpe", "MaxDD", "HitRate", "IC", "ICIR", "α vs SPY", "β"]
    tbl = ax4.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for k in range(len(headers)):
        tbl[(0, k)].set_facecolor("#222"); tbl[(0, k)].set_text_props(color="white", weight="bold")
    ax4.set_title("Performance summary (T+20 forward returns, quarterly rebalance)", fontsize=12, weight="bold", pad=20)

    # Panel 5: monthly distribution.
    ax5 = fig.add_subplot(gs[3, 0])
    primary = next(iter(results.values()))["ls_returns"]["long_short"].dropna()
    if not primary.empty:
        ax5.hist(primary.values, bins=15, color="#36c", edgecolor="white")
        ax5.axvline(0, color="k", linewidth=0.8)
        ax5.set_title("Distribution of L/S quarterly returns", fontsize=11, weight="bold")
        ax5.set_xlabel("return")

    # Panel 6: rolling Sharpe of primary signal.
    ax6 = fig.add_subplot(gs[3, 1])
    if len(primary) >= 4:
        roll = primary.rolling(4).apply(lambda s: s.mean() / s.std(ddof=0) * np.sqrt(4) if s.std(ddof=0) > 0 else np.nan)
        ax6.plot(range(len(roll)), roll.values, color="#36c", linewidth=2)
        ax6.axhline(0, color="k", linewidth=0.8)
        ax6.set_title("Rolling 4Q Sharpe (annualized)", fontsize=11, weight="bold")

    fig.suptitle(
        "Executive Evasion Index — Performance Tear Sheet",
        fontsize=16, weight="bold", y=0.995,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote tear sheet to %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--quintiles", type=int, default=QUINTILES)
    ap.add_argument("--horizon", type=int, default=20,
                    help="primary horizon (days) used for tear-sheet narrative")
    args = ap.parse_args()

    eei_path = OUTPUT_DIR / "eei_scores.csv"
    if not eei_path.exists():
        LOG.error("missing %s — run src/3_evasion_scorer.py first", eei_path)
        sys.exit(2)
    eei = pd.read_csv(eei_path, parse_dates=["date"])
    LOG.info("loaded EEI table: %d rows, %d tickers", len(eei), eei["ticker"].nunique())

    universe = sorted(set(eei["ticker"]).union({BENCHMARK_TICKER}, set(TICKERS)))
    prices = download_prices(universe)
    if BENCHMARK_TICKER not in prices.columns:
        LOG.error("benchmark %s missing from prices; abort", BENCHMARK_TICKER)
        sys.exit(2)

    signals = [
        SignalConfig("EEI Level", "EEI_raw", direction=-1),
        SignalConfig("EEI Delta", "EEI_delta", direction=-1),
        SignalConfig("Topic-Guidance", "EEI_topic_guidance", direction=-1),
    ]

    results: dict[str, dict] = {}
    primary_horizon = args.horizon
    ppy = annualization_factor(primary_horizon)

    # Pre-compute multi-horizon stats per signal.
    multi_horizon_table: list[dict] = []
    primary_panel = None
    primary_ic_series: pd.Series = pd.Series(dtype=float)

    for cfg in signals:
        if cfg.score_col not in eei.columns:
            LOG.warning("score column %s missing; skipping %s", cfg.score_col, cfg.name)
            continue
        for h in HORIZONS_DAYS:
            panel = build_signal_panel(eei, prices, cfg, h)
            ls = long_short_returns(panel, cfg.direction)
            if ls.empty:
                continue
            ic_mean, ic_series = information_coefficient(panel, cfg.direction)
            icir = float(ic_series.mean() / ic_series.std(ddof=0)) if ic_series.std(ddof=0) > 0 else float("nan")
            ppy_h = annualization_factor(h)
            stats = stats_dict(ls["long_short"], ppy_h, cfg.name)
            bench_q = benchmark_quarterly_returns(prices[BENCHMARK_TICKER], ls.index, h)
            alpha_a, beta, r2 = alpha_beta_vs_benchmark(ls["long_short"], bench_q, ppy_h)
            multi_horizon_table.append({
                "signal": cfg.name, "horizon_days": h,
                **{k: v for k, v in stats.items() if k != "label"},
                "ic_mean": ic_mean, "icir": icir,
                "alpha_ann": alpha_a, "beta": beta, "r2": r2,
            })
            if h == primary_horizon:
                results[cfg.name] = {
                    "stats": stats, "ls_returns": ls, "ic_mean": ic_mean,
                    "icir": icir, "alpha": alpha_a, "beta": beta,
                }
                if cfg.name == "EEI Delta":
                    primary_panel = panel
                    primary_ic_series = ic_series

    if not results:
        LOG.error("no signal results computed; abort")
        sys.exit(2)

    # Save numerical outputs.
    summary_df = pd.DataFrame(multi_horizon_table)
    summary_df.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    summary_json = {row["signal"] + f"_T+{int(row['horizon_days'])}": {k: (v if pd.notna(v) else None) for k, v in row.items()} for _, row in summary_df.iterrows()}
    (OUTPUT_DIR / "performance_summary.json").write_text(json.dumps(summary_json, indent=2, default=str))

    LOG.info("\n%s", summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if not args.no_plot:
        # Use EEI Δ for IC chart if available.
        bench_q_primary = benchmark_quarterly_returns(prices[BENCHMARK_TICKER], next(iter(results.values()))["ls_returns"].index, primary_horizon)
        make_tearsheet(results, bench_q_primary, primary_ic_series, OUTPUT_DIR / "performance_tearsheet.png")


if __name__ == "__main__":
    main()
