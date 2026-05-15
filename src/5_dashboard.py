"""
Module 5 — Streamlit research cockpit for the Executive Evasion Index.

Run::
    streamlit run src/5_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import OUTPUT_DIR, PROCESSED_DIR  # noqa: E402
from src.utils import read_json  # noqa: E402

st.set_page_config(
    page_title="Executive Evasion Index",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_eei() -> pd.DataFrame:
    path = OUTPUT_DIR / "eei_scores.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    return df


@st.cache_data(show_spinner=False)
def load_processed_pairs() -> pd.DataFrame:
    rows: list[dict] = []
    for f in sorted(PROCESSED_DIR.glob("*_qa_pairs.json")):
        try:
            payload = read_json(f)
        except Exception:
            continue
        for p in payload.get("qa_pairs", []):
            rows.append({
                "ticker": payload["ticker"],
                "company": payload["company"],
                "date": payload["date"],
                "quarter": payload["quarter"],
                **p,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_perf_summary() -> pd.DataFrame:
    path = OUTPUT_DIR / "performance_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_scored_pairs(eei: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Join LLM/heuristic scores from cache onto Q&A pairs (best-effort)."""
    from src.utils import stable_hash
    from config import CACHE_DIR, SCORER_MODEL
    if pairs.empty:
        return pairs
    rows = []
    for _, p in pairs.iterrows():
        for mode_label, model_tag in (("llm", SCORER_MODEL), ("heuristic", "heuristic-v1")):
            h = stable_hash({"q": p["question_text"], "a": p["answer_text"], "model": model_tag})
            cp = CACHE_DIR / f"score_{h}.json"
            if cp.exists():
                try:
                    data = read_json(cp)
                except Exception:
                    continue
                rows.append({**p.to_dict(), **{
                    "evasion_score": data.get("evasion_score"),
                    "evasion_level": data.get("evasion_level"),
                    "topic_evaded": data.get("topic_evaded"),
                    "red_flag": data.get("red_flag"),
                    "red_flag_reason": data.get("red_flag_reason"),
                    "scorer": data.get("_scorer", mode_label),
                }})
                break
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🕵️ EEI Cockpit")
page = st.sidebar.radio(
    "Navigation",
    [
        "📈 Leaderboard",
        "🔬 Company Deep-Dive",
        "🚩 Red-Flag Alerts",
        "💰 Alpha Dashboard",
        "📊 Raw Q&A Explorer",
    ],
)

eei = load_eei()
pairs = load_processed_pairs()
perf = load_perf_summary()
scored_pairs = load_scored_pairs(eei, pairs)

if eei.empty:
    st.error("`outputs/eei_scores.csv` not found — run modules 1 → 3 first.")
    st.stop()


# ---------------------------------------------------------------------------
# Page: Leaderboard
# ---------------------------------------------------------------------------
def _cmap_red_green(v: float) -> str:
    if pd.isna(v):
        return ""
    norm = max(0.0, min(1.0, v))
    r = int(255 * norm)
    g = int(180 * (1 - norm))
    return f"background-color: rgba({r},{g},80,0.55); color: white"


if page == "📈 Leaderboard":
    st.title("Executive Evasion Leaderboard")
    st.caption("Cross-section of all companies ranked by their most recent EEI. Red = evasive, green = transparent.")

    cutoff = st.sidebar.date_input(
        "Date range",
        value=(eei["date"].min().date(), eei["date"].max().date()),
    )
    if isinstance(cutoff, tuple) and len(cutoff) == 2:
        cutoff_low, cutoff_high = cutoff
    else:
        cutoff_low, cutoff_high = eei["date"].min().date(), eei["date"].max().date()

    sub = eei[(eei["date"].dt.date >= cutoff_low) & (eei["date"].dt.date <= cutoff_high)]
    latest = sub.sort_values("date").groupby("ticker").tail(1).sort_values("EEI_raw", ascending=False)

    cols = ["ticker", "company", "date", "quarter", "EEI_raw", "EEI_weighted", "EEI_delta", "fully_evasive_pct", "red_flag_count", "n_pairs"]
    cols = [c for c in cols if c in latest.columns]
    show = latest[cols].copy()
    show["date"] = show["date"].dt.date

    styled = show.style.applymap(_cmap_red_green, subset=[c for c in ("EEI_raw", "EEI_weighted") if c in show.columns]).format({
        "EEI_raw": "{:.3f}", "EEI_weighted": "{:.3f}", "EEI_delta": "{:+.3f}",
        "fully_evasive_pct": "{:.1%}",
    })
    st.dataframe(styled, use_container_width=True, height=600)

    c1, c2, c3 = st.columns(3)
    c1.metric("Companies tracked", show["ticker"].nunique())
    c2.metric("Median EEI", f"{show['EEI_raw'].median():.3f}")
    c3.metric("Mean Δ", f"{show['EEI_delta'].mean():+.3f}")


# ---------------------------------------------------------------------------
# Page: Company Deep-Dive
# ---------------------------------------------------------------------------
elif page == "🔬 Company Deep-Dive":
    st.title("Company Deep-Dive")
    ticker = st.selectbox("Select company", sorted(eei["ticker"].unique()))
    co = eei[eei["ticker"] == ticker].sort_values("date")
    if co.empty:
        st.warning("No data for ticker.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest EEI", f"{co['EEI_raw'].iloc[-1]:.3f}")
    c2.metric("Quarter Δ", f"{co['EEI_delta'].iloc[-1]:+.3f}" if pd.notna(co['EEI_delta'].iloc[-1]) else "—")
    c3.metric("Trend (4Q slope)", f"{co['EEI_trend'].iloc[-1]:+.3f}" if 'EEI_trend' in co.columns and pd.notna(co['EEI_trend'].iloc[-1]) else "—")
    c4.metric("Total red flags", int(co["red_flag_count"].sum()))

    # EEI trend
    fig = px.line(co, x="date", y=["EEI_raw", "EEI_weighted"], markers=True,
                  title=f"{ticker} — EEI trajectory")
    fig.update_yaxes(title="evasion score (0–1)")
    st.plotly_chart(fig, use_container_width=True)

    # Topic radar
    topic_cols = [c for c in co.columns if c.startswith("EEI_topic_") and not c.endswith("_delta")]
    if topic_cols:
        latest_topics = co[topic_cols].iloc[-1].dropna()
        if not latest_topics.empty:
            radar = go.Figure(go.Scatterpolar(
                r=latest_topics.values,
                theta=[c.replace("EEI_topic_", "").title() for c in latest_topics.index],
                fill="toself",
                line=dict(color="#c33"),
            ))
            radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False, title=f"{ticker} — topic-level evasion (latest call)",
                height=420,
            )
            st.plotly_chart(radar, use_container_width=True)

    # Tactic stack
    tactic_cols = [c for c in co.columns if c.startswith("tactic_")]
    if tactic_cols:
        st.subheader("Evasion tactic frequency over time")
        tdf = co.melt(id_vars=["date"], value_vars=tactic_cols, var_name="tactic", value_name="freq")
        tdf["tactic"] = tdf["tactic"].str.replace("tactic_", "")
        fig = px.area(tdf, x="date", y="freq", color="tactic", groupnorm=None)
        st.plotly_chart(fig, use_container_width=True)

    # Most evasive Q&A pairs
    st.subheader("Most evasive Q&A pairs")
    if not scored_pairs.empty:
        top = scored_pairs[scored_pairs["ticker"] == ticker].sort_values("evasion_score", ascending=False).head(5)
        for _, r in top.iterrows():
            with st.expander(f"📅 {r['date'].date()} | {r['executive_name']} ({r['executive_title']}) | EEI={r['evasion_score']:.2f}"):
                st.markdown(f"**Analyst:** {r['analyst_name']} ({r['analyst_firm']})")
                st.markdown(f"**Q:** {r['question_text']}")
                st.markdown(f"**A:** {r['answer_text']}")
                if r.get("red_flag"):
                    st.error(f"🚩 Red flag — {r.get('red_flag_reason') or 'high evasion'}")

    # Executive split
    if not scored_pairs.empty:
        co_pairs = scored_pairs[scored_pairs["ticker"] == ticker]
        if not co_pairs.empty:
            st.subheader("Executive evasion profile")
            ex = co_pairs.groupby("executive_name")["evasion_score"].agg(["mean", "count"]).reset_index()
            ex = ex[ex["count"] >= 3].sort_values("mean", ascending=False)
            fig = px.bar(ex, x="executive_name", y="mean", text="count",
                         labels={"mean": "avg EEI", "count": "n pairs"})
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Red-Flag Alerts
# ---------------------------------------------------------------------------
elif page == "🚩 Red-Flag Alerts":
    st.title("Red-Flag Alerts")
    st.caption("Q&A exchanges the scorer flagged as suspicious / noteworthy.")
    if scored_pairs.empty:
        st.info("No scored pairs cached. Run module 3 first.")
        st.stop()
    flagged = scored_pairs[scored_pairs["red_flag"] == True].sort_values("evasion_score", ascending=False)
    if flagged.empty:
        st.success("No red flags in current dataset.")
        st.stop()

    topic_filter = st.multiselect("Filter by evaded topic",
                                  sorted(flagged["topic_evaded"].dropna().unique()),
                                  default=None)
    if topic_filter:
        flagged = flagged[flagged["topic_evaded"].isin(topic_filter)]

    color_map = {"margins": "#c33", "guidance": "#e67", "litigation": "#fa3", "macro": "#39c"}
    for _, r in flagged.head(50).iterrows():
        bg = color_map.get(r.get("topic_evaded", ""), "#888")
        st.markdown(
            f"<div style='border-left:6px solid {bg};padding:8px;margin-bottom:8px;background:#1a1a1a;color:#eee;border-radius:4px'>"
            f"<b>{r['ticker']}</b> | {r['date'].date()} | EEI={r['evasion_score']:.2f} | topic: <i>{r.get('topic_evaded','?')}</i><br/>"
            f"<small>{r.get('red_flag_reason','')}</small>"
            f"</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Show full Q&A"):
            st.markdown(f"**Analyst:** {r['analyst_name']} ({r['analyst_firm']})")
            st.markdown(f"**Q:** {r['question_text']}")
            st.markdown(f"**Executive:** {r['executive_name']} ({r['executive_title']})")
            st.markdown(f"**A:** {r['answer_text']}")


# ---------------------------------------------------------------------------
# Page: Alpha Dashboard
# ---------------------------------------------------------------------------
elif page == "💰 Alpha Dashboard":
    st.title("Alpha Dashboard")
    if perf.empty:
        st.warning("No performance summary found — run `python src/4_backtester.py`.")
        st.stop()

    st.subheader("Performance summary across signals & horizons")
    st.dataframe(perf.style.format({
        "ann_return": "{:+.2%}", "ann_vol": "{:.2%}", "sharpe": "{:.2f}",
        "max_drawdown": "{:.2%}", "hit_rate": "{:.2%}",
        "ic_mean": "{:+.3f}", "icir": "{:+.2f}",
        "alpha_ann": "{:+.2%}", "beta": "{:.2f}", "r2": "{:.2f}",
    }), use_container_width=True)

    ts = OUTPUT_DIR / "performance_tearsheet.png"
    if ts.exists():
        st.image(str(ts), caption="Performance tear sheet (T+20)", use_container_width=True)

    # Current portfolio (latest cross-section by EEI Δ).
    st.subheader("Current model portfolio (EEI Δ signal, latest cross-section)")
    latest_q = eei["date"].dt.to_period("Q").max()
    cs = eei[eei["date"].dt.to_period("Q") == latest_q].dropna(subset=["EEI_delta"])
    if not cs.empty:
        cs = cs.sort_values("EEI_delta")
        n_leg = max(1, len(cs) // 5)
        longs = cs.head(n_leg)[["ticker", "company", "EEI_delta"]]
        shorts = cs.tail(n_leg)[["ticker", "company", "EEI_delta"]]
        c1, c2 = st.columns(2)
        c1.markdown("### LONG  (evasion ↓)")
        c1.dataframe(longs.style.format({"EEI_delta": "{:+.3f}"}), use_container_width=True)
        c2.markdown("### SHORT (evasion ↑)")
        c2.dataframe(shorts.style.format({"EEI_delta": "{:+.3f}"}), use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Raw Q&A Explorer
# ---------------------------------------------------------------------------
elif page == "📊 Raw Q&A Explorer":
    st.title("Raw Q&A Explorer")
    if scored_pairs.empty:
        st.info("No scored pairs found.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    tickers = c1.multiselect("Ticker", sorted(scored_pairs["ticker"].unique()))
    levels = c2.multiselect("Evasion level",
                            sorted(scored_pairs["evasion_level"].dropna().unique()))
    text = c3.text_input("Text search (regex)")

    df = scored_pairs.copy()
    if tickers:
        df = df[df["ticker"].isin(tickers)]
    if levels:
        df = df[df["evasion_level"].isin(levels)]
    if text:
        try:
            mask = df["question_text"].str.contains(text, case=False, regex=True, na=False) | \
                   df["answer_text"].str.contains(text, case=False, regex=True, na=False)
            df = df[mask]
        except Exception as exc:  # noqa: BLE001
            st.warning(f"regex error: {exc}")

    show_cols = ["ticker", "date", "executive_name", "analyst_firm", "evasion_score",
                 "evasion_level", "topic_evaded", "answer_word_count",
                 "hedge_count", "deflection_keywords", "answer_question_word_overlap"]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[show_cols].sort_values("evasion_score", ascending=False),
                 use_container_width=True, height=600)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered CSV", data=csv,
                       file_name="eei_qa_pairs_filtered.csv", mime="text/csv")
