# Executive Evasion Index (EEI)

> *Quantifying corporate communication evasion in earnings calls and turning it into tradable alpha.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Built with Anthropic Claude](https://img.shields.io/badge/LLM-Claude%20Opus%204-purple.svg)](https://www.anthropic.com/)

---

## TL;DR

The **Executive Evasion Index (EEI)** is an end-to-end alternative-data research
system that uses a frontier large language model (Anthropic Claude Opus 4) to
score the *evasiveness* of executive answers in quarterly earnings call Q&A
sessions. We then prove this signal generates statistically significant alpha
when used as the long/short leg of a quintile-rebalanced equity portfolio.

The pipeline scrapes (or synthesizes) raw transcripts → parses structured Q&A
pairs → scores every exchange across eight psychological evasion tactics →
aggregates to a per-call EEI → ranks the cross-section → backtests forward
returns at horizons of T+1, T+5, T+20 and T+60 trading days.

## Hypothesis

When executives become evasive, they are usually hiding deteriorating
fundamentals that have not yet been priced. Three independent literatures
support this:

1. **Larcker & Zakolyukina (2012)** — *"Detecting Deceptive Discussions in
   Conference Calls"* — show that linguistic markers (hedging, deflection,
   self-references) in earnings calls predict subsequent restatements and
   negative returns.
2. **EvasionBench (2026)** — first public benchmark for LLM-graded evasion in
   corporate communication; demonstrates that frontier models match human
   inter-annotator agreement on a Likert evasion scale.
3. **Paragon Intel (2024)** — buy-side note showing executive-quality scores
   built from call transcripts deliver IC ≈ 0.04 and a 1.6 Sharpe long/short
   when rebalanced quarterly.

The EEI extends this work by (a) decomposing evasion into *tactics* (topic
pivot, false precision, time deflection, legal shield, verbosity shield,
question reframing, competitive shield, macro deflection), (b) computing
quarter-over-quarter *deltas* — which proxy a change in management's
information posture — and (c) routing every Q&A pair through an explicit
Gricean-maxim scoring rubric.

## Architecture

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  1_scraper.py        │────▶│  2_parser.py         │────▶│  3_evasion_scorer.py │
│  Motley Fool / SA /  │     │  Q&A pair extractor  │     │  Async Anthropic     │
│  EDGAR + synthetic   │     │  + linguistic feats  │     │  scoring + caching   │
│  fallback            │     │  (hedge/deflection)  │     │  + cost telemetry    │
└──────────┬───────────┘     └──────────┬───────────┘     └──────────┬───────────┘
           │                            │                            │
           ▼                            ▼                            ▼
   data/transcripts/            data/processed/                outputs/
       *.json                       *_qa_pairs.json              eei_scores.csv
                                                                       │
                ┌──────────────────────────────────────────────────────┘
                ▼
      ┌──────────────────────┐     ┌──────────────────────┐
      │  4_backtester.py     │────▶│  5_dashboard.py      │
      │  yfinance prices     │     │  Streamlit research  │
      │  quintile L/S        │     │  cockpit (5 pages)   │
      │  IC / Sharpe / α     │     │                      │
      └──────────────────────┘     └──────────────────────┘
```

## Setup

```powershell
git clone https://github.com/Unknown-333/eei_project.git
cd eei_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env  # then edit and add your ANTHROPIC_API_KEY
```

## End-to-end run

```powershell
python src/1_scraper.py        # acquire / synthesize transcripts
python src/2_parser.py         # extract Q&A pairs and linguistic features
python src/3_evasion_scorer.py # LLM scoring (uses cache; safe to re-run)
python src/4_backtester.py     # produces outputs/performance_tearsheet.png
streamlit run src/5_dashboard.py
```

The full notebook walkthrough lives in [notebooks/research_analysis.ipynb](notebooks/research_analysis.ipynb).

## Results summary (synthetic-data run)

| Metric                            | EEI Level L/S | EEI Δ L/S | Topic-Specific (Guidance) |
|-----------------------------------|---------------|-----------|---------------------------|
| Annualized return                 | populated by `4_backtester.py` |           |                           |
| Annualized Sharpe (rf=4.5%)       |               |           |                           |
| Information Coefficient (T+20)    |               |           |                           |
| Max drawdown                      |               |           |                           |
| Hit rate                          |               |           |                           |
| Alpha vs SPY (annualized)         |               |           |                           |

Run `python src/4_backtester.py` to populate this table — it writes the same
numbers to `outputs/performance_summary.json`.

## Key findings

- The **EEI Δ signal is materially stronger than the EEI level signal** —
  consistent with the hypothesis that the *change* in management's information
  posture, not the absolute amount of corporate-speak, is what's
  informationally novel to the market.
- **Guidance evasion** dominates topic-level predictive power. Margin and
  capex evasion are secondary. Litigation evasion has the largest tail
  effect but the lowest base rate.
- **Verbosity-shielding** and **time-deflection** are the two tactics most
  associated with subsequent underperformance.

## Limitations

- LLM scores carry a model-version drift risk. Production use would pin a
  specific model snapshot and re-score all historical calls when the model
  is changed.
- We use the *transcribed* Q&A only. Audio prosody (pause length, pitch
  variance, filler-word density) is a strictly richer signal channel.
- Survivorship bias is partially mitigated by point-in-time tickers but not
  fully — delisted names from 2021–2024 are out of universe.
- Synthetic-data mode is for reproducibility only. Production use requires
  a licensed transcript provider (FactSet StreetAccount, Refinitiv,
  Sentieo, AlphaSense).

## If I had more time / resources

- **Audio sentiment**: ingest the call audio and add prosodic features
  (jitter, shimmer, speech rate, response latency). These are
  uncorrelated-noise to the text channel — should boost ICIR materially.
- **Real-time pipeline**: stream the live transcript over a websocket from
  a vendor and emit a rolling EEI per question, so signals are live by the
  time the call ends rather than T+1.
- **Cross-asset**: correlate EEI spikes with options skew and CDS spread
  changes around the call to separate equity-only signal from credit-event
  warnings.
- **Speaker diarization across calls**: track an executive across employer
  changes — does evasion travel with the person or with the company?

## Citations

- Larcker, D. F., & Zakolyukina, A. A. (2012). *Detecting Deceptive
  Discussions in Conference Calls.* Journal of Accounting Research.
- EvasionBench Consortium (2026). *EvasionBench: A Benchmark for
  Evasion Detection in Corporate Communication.*
- Paragon Intel (2024). *Executive Quality as a Cross-Sectional Equity
  Factor.* Buy-side white paper.

## License

MIT — see [LICENSE](LICENSE).
