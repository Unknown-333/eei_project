"""
Module 6 — Extended factor library.

Builds 4 new signals on top of the EEI scoring + a multi-factor combiner:

    1. confidence_score        prosody proxy from punctuation + sentence-length variance
    2. analyst_skepticism      density of skeptical / pressure markers in QUESTIONS
    3. evasion_under_pressure  EEI conditioned on high-skepticism quarters
    4. ceo_cfo_evasion_gap     CEO-vs-CFO answer evasion delta within a call
    5. evasion_momentum_8q     8-quarter linear-trend slope of EEI per ticker
    6. composite_signal        scipy-optimized linear combo maximizing IS Sharpe

Outputs:
    outputs/signals_panel.csv          one row per (ticker, date) with all factors
    outputs/signals_ic.csv             Spearman IC of each signal at T+1/5/20/60
    outputs/composite_weights.json     fitted multi-factor combiner weights
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import HORIZONS_DAYS, OUTPUT_DIR, PROCESSED_DIR  # noqa: E402
from src.utils import get_logger  # noqa: E402

LOG = get_logger("signals")


# ---------------------------------------------------------------------------
# 1. Confidence (audio-prosody proxy from text only)
# ---------------------------------------------------------------------------
_LOW_CONF_PUNCT = re.compile(r"[—–\-]{1,2}|\.{2,}|,\s*(?:um+|uh+|er+|well|i mean|you know)\b", re.I)
_FILLER_RE = re.compile(r"\b(um+|uh+|er+|kind of|sort of|i think|i mean|you know|maybe|perhaps)\b", re.I)
_SENT_SPLIT = re.compile(r"[.!?]+\s+")


def confidence_score_for_answer(text: str) -> float:
    """0 = nervous / fragmented, 1 = crisp & confident.

    Combines (a) low filler density, (b) low low-confidence-punctuation density,
    (c) low coefficient-of-variation of sentence length.
    """
    if not text or len(text.split()) < 4:
        return 0.5
    words = text.split()
    n_words = len(words)
    fillers = len(_FILLER_RE.findall(text))
    lcp = len(_LOW_CONF_PUNCT.findall(text))
    sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    sent_lens = [len(s.split()) for s in sents] or [n_words]
    if len(sent_lens) > 1 and statistics.mean(sent_lens) > 0:
        cv = statistics.pstdev(sent_lens) / statistics.mean(sent_lens)
    else:
        cv = 0.0
    raw = (
        1.0
        - 1.5 * (fillers / n_words)
        - 1.5 * (lcp / n_words)
        - 0.4 * min(cv, 1.0)
    )
    return float(max(0.0, min(1.0, raw)))


# ---------------------------------------------------------------------------
# 2. Analyst skepticism (on the QUESTION side)
# ---------------------------------------------------------------------------
_SKEPTIC_RE = re.compile(
    r"\b(but|however|really|though|actually|isn'?t|doesn'?t|aren'?t|wasn'?t|"
    r"can you|could you|why is|why are|how come|to be clear|to clarify|"
    r"a follow[- ]up|just to push back|color on|specifically|concretely|"
    r"walk (us|me) through|how confident|what gives you confidence)\b",
    re.I,
)


def skepticism_score_for_question(text: str) -> float:
    """0 = soft lob, 1 = hostile pressure question."""
    if not text:
        return 0.0
    n_words = max(len(text.split()), 1)
    hits = len(_SKEPTIC_RE.findall(text))
    qmarks = text.count("?")
    long_q = min(n_words / 80.0, 1.0)  # longer questions usually = more challenging
    raw = 0.6 * (hits / max(n_words / 30.0, 1.0)) + 0.2 * min(qmarks / 2.0, 1.0) + 0.2 * long_q
    return float(max(0.0, min(1.0, raw)))


# ---------------------------------------------------------------------------
# 3 & 4. Per-call aggregation from raw QA pairs
# ---------------------------------------------------------------------------
def _ceo_cfo_role(title: str) -> str:
    t = (title or "").lower()
    if "chief executive" in t or t.endswith(" ceo") or t == "ceo":
        return "CEO"
    if "chief financial" in t or t.endswith(" cfo") or t == "cfo":
        return "CFO"
    return "OTHER"


def _heuristic_evasion(answer: str) -> float:
    """Lightweight rule-based evasion proxy — duplicates the heuristic scorer
    so this module stays self-contained without circular imports."""
    if not answer:
        return 0.5
    n = max(len(answer.split()), 1)
    hedge_words = ["may", "might", "could", "perhaps", "approximately", "roughly", "broadly", "generally"]
    deflect = ["macro", "competitive", "legal", "regulatory", "market conditions", "going forward", "in due course"]
    hedge = sum(answer.lower().count(w) for w in hedge_words)
    defl = sum(answer.lower().count(w) for w in deflect)
    raw = 0.4 * (hedge / max(n / 30.0, 1.0)) + 0.4 * (defl / max(n / 30.0, 1.0))
    return float(min(raw + 0.1, 1.0))


def features_for_call(call: dict[str, Any]) -> dict[str, float]:
    pairs = call.get("qa_pairs", [])
    if not pairs:
        return {}
    confs, skeps, evasions, ceo_ev, cfo_ev = [], [], [], [], []
    for p in pairs:
        a = p.get("answer_text", "")
        q = p.get("question_text", "")
        conf = confidence_score_for_answer(a)
        skep = skepticism_score_for_question(q)
        ev = _heuristic_evasion(a)
        confs.append(conf); skeps.append(skep); evasions.append(ev)
        role = _ceo_cfo_role(p.get("executive_title", ""))
        if role == "CEO":
            ceo_ev.append(ev)
        elif role == "CFO":
            cfo_ev.append(ev)

    skep_arr = np.array(skeps); ev_arr = np.array(evasions)
    high_skep_mask = skep_arr >= np.median(skep_arr) if len(skep_arr) else np.array([])
    eup = float(ev_arr[high_skep_mask].mean()) if high_skep_mask.any() else float(ev_arr.mean())

    return {
        "confidence_score": float(np.mean(confs)),
        "analyst_skepticism": float(np.mean(skeps)),
        "evasion_under_pressure": eup,
        "ceo_cfo_evasion_gap": (float(np.mean(ceo_ev)) - float(np.mean(cfo_ev))) if ceo_ev and cfo_ev else 0.0,
    }


# ---------------------------------------------------------------------------
# 5. Evasion momentum (8-quarter linear slope)
# ---------------------------------------------------------------------------
def add_evasion_momentum(df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()

    def _slope(series: pd.Series) -> float:
        s = series.dropna()
        if len(s) < 3:
            return np.nan
        x = np.arange(len(s), dtype=float)
        return float(np.polyfit(x, s.to_numpy(dtype=float), 1)[0])

    df["evasion_momentum_8q"] = (
        df.groupby("ticker")["EEI_raw"]
        .rolling(window, min_periods=3).apply(_slope, raw=False)
        .reset_index(level=0, drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# Build the master signals panel
# ---------------------------------------------------------------------------
def build_signals_panel() -> pd.DataFrame:
    eei = pd.read_csv(OUTPUT_DIR / "eei_scores.csv", parse_dates=["date"])

    rows: list[dict[str, Any]] = []
    for f in sorted(PROCESSED_DIR.glob("*_qa_pairs.json")):
        call = json.loads(f.read_text(encoding="utf-8"))
        feats = features_for_call(call)
        if not feats:
            continue
        rows.append({
            "ticker": call["ticker"], "date": pd.to_datetime(call["date"]),
            **feats,
        })
    new_feats = pd.DataFrame(rows)

    panel = eei.merge(new_feats, on=["ticker", "date"], how="left")
    panel = add_evasion_momentum(panel, window=8)
    return panel


# ---------------------------------------------------------------------------
# IC computation + composite optimization
# ---------------------------------------------------------------------------
def _forward_returns(prices: pd.DataFrame, dates: pd.Series, tickers: pd.Series, h: int) -> pd.Series:
    out = []
    for tk, dt in zip(tickers, dates):
        if tk not in prices.columns:
            out.append(np.nan); continue
        idx = prices.index.searchsorted(dt)
        if idx + h >= len(prices):
            out.append(np.nan); continue
        p0 = prices[tk].iloc[idx]; p1 = prices[tk].iloc[idx + h]
        out.append(np.nan if pd.isna(p0) or pd.isna(p1) or p0 == 0 else float(p1 / p0 - 1))
    return pd.Series(out, index=dates.index)


def compute_ics(panel: pd.DataFrame, prices: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    rows = []
    for h in HORIZONS_DAYS:
        rets = _forward_returns(prices, panel["date"], panel["ticker"], h)
        for col in signal_cols:
            sub = pd.DataFrame({"sig": panel[col], "ret": rets}).dropna()
            if len(sub) < 30:
                ic = np.nan
            else:
                ic = float(spearmanr(sub["sig"], sub["ret"])[0])
            rows.append({"signal": col, "horizon_days": h, "ic": ic, "n": len(sub)})
    return pd.DataFrame(rows)


def fit_composite(panel: pd.DataFrame, prices: pd.DataFrame, signal_cols: list[str], horizon: int = 20) -> dict[str, float]:
    """Find linear weights that maximize cross-sectional Sharpe of long-short
    (top vs. bottom quintile by composite signal) at the given horizon.
    Negative IC signals are pre-flipped so all factors are 'higher = bullish'.
    """
    rets = _forward_returns(prices, panel["date"], panel["ticker"], horizon)
    sub = panel[signal_cols].copy()
    sub["__ret"] = rets
    sub = sub.dropna()
    if len(sub) < 50:
        LOG.warning("fit_composite: only %d obs, returning equal weights", len(sub))
        return {c: 1.0 / len(signal_cols) for c in signal_cols}

    # Normalise each factor + flip if negatively correlated with returns.
    signs = {}
    X = sub[signal_cols].copy()
    for c in signal_cols:
        rho = spearmanr(X[c], sub["__ret"])[0]
        signs[c] = -1.0 if (rho is not None and rho < 0) else 1.0
        X[c] = signs[c] * (X[c] - X[c].mean()) / (X[c].std() + 1e-9)
    y = sub["__ret"].to_numpy()

    def neg_sharpe(w: np.ndarray) -> float:
        sig = X.to_numpy() @ w
        order = np.argsort(sig)
        n = len(sig)
        bottom = y[order[: max(n // 5, 1)]]
        top = y[order[-max(n // 5, 1):]]
        ls = top.mean() - bottom.mean()
        sd = np.std(np.concatenate([top, -bottom])) + 1e-9
        return -(ls / sd) * np.sqrt(252 / horizon)

    w0 = np.ones(len(signal_cols)) / len(signal_cols)
    cons = ({"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0},)
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=[(-1, 1)] * len(signal_cols),
                   constraints=cons, options={"maxiter": 200, "ftol": 1e-6})
    w = res.x
    LOG.info("composite fit: IS Sharpe ≈ %.2f at H=%d", -res.fun, horizon)
    return {c: float(signs[c] * w[i]) for i, c in enumerate(signal_cols)}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    panel = build_signals_panel()
    panel.to_csv(OUTPUT_DIR / "signals_panel.csv", index=False)
    LOG.info("wrote signals_panel.csv (%d rows × %d cols)", *panel.shape)

    # Load price cache produced by the backtester.
    prices_path = ROOT / "data" / "prices" / "prices.parquet"
    if not prices_path.exists():
        LOG.warning("price cache not found; run backtester first to enable IC. skipping.")
        return
    prices = pd.read_parquet(prices_path)

    signal_cols = [
        "EEI_raw", "EEI_weighted", "EEI_delta", "EEI_trend",
        "confidence_score", "analyst_skepticism", "evasion_under_pressure",
        "ceo_cfo_evasion_gap", "evasion_momentum_8q",
    ]
    signal_cols = [c for c in signal_cols if c in panel.columns]

    ic_df = compute_ics(panel, prices, signal_cols)
    ic_df.to_csv(OUTPUT_DIR / "signals_ic.csv", index=False)
    LOG.info("IC table:\n%s", ic_df.to_string(index=False))

    weights = fit_composite(panel, prices, signal_cols, horizon=20)
    composite = pd.Series(0.0, index=panel.index)
    for c, w in weights.items():
        col = (panel[c] - panel[c].mean()) / (panel[c].std() + 1e-9)
        composite = composite + w * col.fillna(0.0)
    panel["composite_signal"] = composite
    panel.to_csv(OUTPUT_DIR / "signals_panel.csv", index=False)

    (OUTPUT_DIR / "composite_weights.json").write_text(
        json.dumps(weights, indent=2), encoding="utf-8"
    )
    LOG.info("composite weights: %s", weights)


if __name__ == "__main__":
    main()
