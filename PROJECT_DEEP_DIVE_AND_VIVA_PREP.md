# PROJECT DEEP DIVE AND VIVA PREP
## Executive Evasion Index (EEI)

---

# 1. High-Level Architecture & Data Flow

## The Elevator Pitch

The Executive Evasion Index (EEI) is a quantitative NLP pipeline that ingests corporate earnings call transcripts, scores executive evasion in Q&A sections using both heuristic rules and LLM evaluation, and backtests the resulting signals as cross-sectional equity factors to predict short-term stock underperformance. It turns "how much is the CEO dodging questions" into a tradeable alpha signal.

## System Architecture

```
┌─────────────────┐      ┌────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  1_scraper.py   │─────>│  2_parser.py   │─────>│ 3_evasion_scorer │─────>│  4_backtester   │
│                 │      │                │      │     .py          │      │     .py         │
│ • Motley Fool   │      │ • Q&A split    │      │ • Heuristic mode │      │ • yfinance px   │
│ • SEC EDGAR     │      │ • Speaker turn │      │ • LLM mode      │      │ • Quintile L/S  │
│ • Synthetic gen │      │   parsing      │      │ • Caching layer  │      │ • Multi-horizon │
│                 │      │ • Linguistic   │      │ • Aggregation    │      │ • Tear sheet    │
│                 │      │   features     │      │                  │      │                 │
└────────┬────────┘      └───────┬────────┘      └────────┬─────────┘      └────────┬────────┘
         │                       │                        │                          │
         ▼                       ▼                        ▼                          ▼
  data/transcripts/      data/processed/          data/cache/              outputs/
  {TICKER}_{DATE}.json   {T}_{D}_qa_pairs.json    score_{hash}.json       eei_scores.csv
                                                                           performance_summary.json
                                                                           performance_tearsheet.png
                                                                                    │
                                                                                    ▼
                                                                           ┌─────────────────┐
                                                                           │  5_dashboard.py  │
                                                                           │   (Streamlit)    │
                                                                           └─────────────────┘
                                                                                    │
                                                                                    ▼
                                                                           ┌─────────────────┐
                                                                           │   signals.py     │
                                                                           │ Extended factors │
                                                                           │ + composite opt  │
                                                                           └─────────────────┘
```

### Data Flow in Detail

1. **Acquisition (`1_scraper.py`):** Attempts to scrape earnings call transcripts from Motley Fool (HTML parsing) and SEC EDGAR (full-text 8-K search). If live scraping yields <50 transcripts, the system falls back to a **deterministic synthetic generator** that produces realistic, variably-evasive transcripts. Each transcript is serialized as JSON into `data/transcripts/`.

2. **Parsing (`2_parser.py`):** Reads raw transcripts, locates the Q&A section boundary via regex ("now we'll open up the call for questions…"), parses speaker turns using a multi-pattern header regex, classifies speakers as analyst/executive/operator, pairs consecutive analyst→executive turns into structured Q&A pairs, and computes cheap linguistic features (hedge count, deflection density, Jaccard word overlap, length ratio, topic tags). Output: one `*_qa_pairs.json` per call in `data/processed/`.

3. **Scoring (`3_evasion_scorer.py`):** Each Q&A pair is scored on a [0, 1] evasion scale via either:
   - **Heuristic mode (default):** A deterministic weighted formula of 5 components (hedge density, deflection density, topic overlap, length ratio, tactic detection) → cost $0.
   - **LLM mode:** An Anthropic Claude API call per pair using a detailed rubric prompt → returns structured JSON with evasion_score, tactics, rationale.
   
   All results are cached to disk keyed by SHA-1(question + answer + model). Per-call aggregation produces `outputs/eei_scores.csv` with: EEI_raw, EEI_weighted, EEI_delta, EEI_trend, per-topic breakdowns, and tactic frequencies.

4. **Backtesting (`4_backtester.py`):** Downloads adjusted-close prices via yfinance (cached to Parquet), constructs a signal panel by linking each earnings call's EEI score to the stock's forward return over T+1/5/20/60 days, assigns quintile ranks within each quarterly cross-section, computes long-short portfolio returns (long Q1, short Q5), and reports: annualized return, Sharpe, max drawdown, hit rate, Spearman IC, ICIR, alpha/beta vs. SPY.

5. **Dashboard (`5_dashboard.py`):** A Streamlit app with 8 pages — leaderboard, company deep-dive, red-flag alerts, alpha dashboard, Q&A explorer, comparison view, live scoring playground, and export.

6. **Extended Signals (`signals.py`):** Builds 4 additional factors (confidence score, analyst skepticism, evasion-under-pressure, CEO-CFO gap) + an 8-quarter momentum slope, then optimizes a linear composite via scipy SLSQP maximizing in-sample Sharpe.

## Tech Stack Rationale

| Technology | Purpose | Why This Over Alternatives |
|---|---|---|
| **Python 3.11** | Pipeline language | NumPy/pandas/scipy ecosystem dominates quant finance; fastest CPython release at time of build |
| **Anthropic Claude (Opus/Sonnet)** | LLM scorer | Best-in-class at structured JSON extraction and nuanced textual analysis; native system-prompt adherence superior to GPT-4 for rubric-following; no hallucinated keys |
| **pandas + NumPy** | Data wrangling & numerics | Industry standard for tabular quant data; vectorized operations for IC/Sharpe computation |
| **yfinance** | Price data | Free, no-API-key equity price source; acceptable for research-stage; the Parquet cache avoids repeated downloads |
| **BeautifulSoup4** | HTML parsing | Lightweight, sufficient for structured article pages; lxml parser adds C-speed when available |
| **Streamlit** | Dashboard | Rapid prototyping of data apps with zero frontend code; native df/chart support; good for research demos |
| **scipy.optimize** | Composite signal fitting | SLSQP handles constrained optimization (weights sum to 1, bounded [-1,1]) in <200 iterations |
| **statsmodels** | OLS regression | Alpha/beta decomposition vs. benchmark; standard econometric engine |
| **Plotly + matplotlib + seaborn** | Visualization | Plotly for interactive Streamlit charts; matplotlib for static tear sheets |
| **asyncio + Semaphore** | LLM concurrency | Bounded concurrent API calls (default 8) to respect rate limits without blocking |
| **SHA-1 content hashing** | Cache keys | Deterministic, collision-resistant for ~10K pairs; 16-char hex prefix is compact and unique enough |
| **Parquet** | Price storage | 10x smaller than CSV, columnar access, preserves dtypes — critical for 4-year daily price matrices |
| **Docker + docker-compose** | Deployment | Single-command deployment of dashboard; volume mounts for data persistence |
| **pytest + pytest-asyncio** | Testing | Async test support for LLM scorer; parametrized fixtures for edge cases |

### Why NOT:
- **PostgreSQL/SQLite:** Not needed. The dataset is small (~240 rows of scores, ~2500 Q&A pairs). File-based JSON + CSV + Parquet is simpler, portable, and version-controllable.
- **Flask/FastAPI:** No REST API needed. This is a research pipeline, not a production service. Streamlit serves both UI and interactivity.
- **Spark/Dask:** The data fits in memory many times over. Distributed computing adds complexity with zero benefit at this scale.
- **Hugging Face Transformers:** Local models would require GPU infra and produce inferior structured output compared to Claude for this nuanced linguistic task.

---

# 2. Codebase Anatomy (Nook & Cranny Breakdown)

## `config.py` — Central Configuration

**Responsibility:** Single source of truth for all tunable parameters, paths, API keys, lexicons.

**Key elements:**
- `TICKERS` — 20-stock universe (mega-cap diversified: tech, banks, energy, retail, pharma, auto)
- `START_DATE / END_DATE` — 2021-01-01 to 2024-12-31 (3-year backtest window)
- `HORIZONS_DAYS = (1, 5, 20, 60)` — Forward-return horizons tested
- `QUINTILES = 5` — Cross-sectional sort buckets
- `RISK_FREE_RATE = 0.045` — Used in excess-return Sharpe computation
- `HEDGE_WORDS / DEFLECTION_PHRASES` — Handcrafted lexicons for heuristic scoring
- `TOPIC_KEYWORDS` — 10-topic taxonomy for per-topic evasion tracking
- `TIER1_FIRMS / TIER1_WEIGHT` — Sell-side analyst weighting (Tier-1 gets 1.5x weight)
- `cost_estimate()` — Token-to-USD conversion for telemetry

**Interaction:** Imported by every module. Acts as the "settings.py" equivalent.

---

## `src/1_scraper.py` — Transcript Acquisition

**Responsibility:** Acquire raw earnings call transcripts from multiple sources.

**Key functions/classes:**
- `Transcript` dataclass — Standardized schema (company, ticker, date, quarter, raw_text, source)
- `scrape_motley_fool()` — Paginated index scrape → article parse → structured transcript
- `search_edgar_8k()` — SEC EDGAR full-text search for 8-K exhibits carrying call transcripts
- `synthetic_calls()` — **The most critical function.** Generates deterministic fake transcripts with controlled evasion profiles per ticker and per quarter.
- `_evasion_profile()` — Assigns each ticker a base evasion probability (e.g., XOM=0.50, NVDA=0.15) with random drift, then samples Q&A modes (direct/intermediate/evasive) for each pair.
- `_make_qa()` — Template-based question+answer generation using curated templates per evasion mode.
- `_seeded_rng()` — Deterministic per-ticker-per-quarter RNG for reproducibility.

**Interaction:** Writes to `data/transcripts/`. The synthetic path ensures reviewers can run the full pipeline with zero API keys or data feeds.

---

## `src/2_parser.py` — Q&A Extraction + Linguistic Features

**Responsibility:** Transform raw text into structured Q&A pairs with cheap NLP features.

**Key functions/classes:**
- `ParsedCall` / `QAPair` dataclasses — Schema for structured output
- `split_qa_section()` — Regex-based detection of where the Q&A starts in the transcript
- `_HEADER_RE` — Multi-pattern regex matching speaker headers ("Name -- Firm -- Title")
- `_split_turns()` — Walk text, chunk by speaker headers, classify each turn
- `_classify()` — Heuristic role classification (analyst/executive/operator) from title keywords
- `_pair_turns()` — Pair consecutive analyst→executive turns into Q&A objects
- `_make_pair()` — Compute all linguistic features for a single pair:
  - `count_phrases()` — Case-insensitive lexicon hit counting
  - `jaccard_overlap()` — Content-word Jaccard between question and answer (stop-word filtered)
  - `extract_topics()` — Topic-keyword matching against 10-topic taxonomy
  - `answer_to_question_length_ratio` — Verbosity proxy

**Interaction:** Reads from `data/transcripts/`, writes to `data/processed/`. Features here flow downstream to the heuristic scorer as direct inputs.

---

## `src/3_evasion_scorer.py` — Evasion Scoring Engine

**Responsibility:** Score each Q&A pair on evasion [0, 1] and aggregate per call.

**Key functions:**
- `heuristic_score()` — **The core formula:**
  ```
  score = 0.18 * hedge_density_norm
        + 0.22 * deflection_density_norm
        + 0.18 * (1 - overlap_norm)
        + 0.10 * length_ratio_norm
        + 0.32 * (tactics_triggered / 8)
  ```
  Plus 8 regex-based tactic detectors (topic_pivot, false_precision, time_deflection, legal_shield, verbosity_shield, question_reframing, competitive_shield, macro_deflection).

- `llm_score_one()` — Async Anthropic API call with structured prompt → JSON parse → retry logic with exponential backoff (max 5 retries, capped at 30s delay).
- `_cache_key() / _load_cached() / _save_cached()` — Content-addressed caching layer.
- `aggregate_call()` — Per-call aggregation: mean EEI, weighted EEI (analyst-tier weighting), per-topic breakdown, tactic frequencies, concentration metric, red-flag counting.
- `add_cross_call_features()` — Computes EEI_delta (quarter-over-quarter diff), EEI_trend (4-quarter rolling slope), and per-topic deltas.
- `score_call_llm()` — Orchestrates async LLM scoring with fallback to heuristic on failure.

**Interaction:** Reads `data/processed/`, writes `data/cache/score_*.json` and `outputs/eei_scores.csv`.

---

## `src/4_backtester.py` — Signal Construction & Performance Evaluation

**Responsibility:** Test whether evasion scores predict equity returns.

**Key functions:**
- `download_prices()` — Batched yfinance download with Parquet caching, retry logic, staleness check.
- `forward_returns()` — Shift-aligned H-day forward percentage returns.
- `build_signal_panel()` — For each earnings event: find next trading day ≥ event date, compute entry/exit prices at horizon, assign quintile rank within the quarterly cross-section.
- `long_short_returns()` — Aggregate per-quarter: average return of Q1 (long) minus Q5 (short).
- `sharpe()` — Annualized Sharpe with configurable risk-free rate.
- `max_drawdown()` — Peak-to-trough on cumulative return series.
- `alpha_beta_vs_benchmark()` — OLS regression of strategy returns on SPY.
- `information_coefficient()` — Per-quarter Spearman rank correlation between score and forward return.
- `make_tearsheet()` — 6-panel matplotlib figure (cumulative returns, IC bars, long/short legs, stats table, return distribution, rolling Sharpe).

**Critical design: No look-ahead bias.** Entry is on the *next trading day on or after* the event date, NOT the event date itself. Quintile ranks are computed within the same quarter's cross-section only (no future information leaks into ranking).

**Interaction:** Reads `outputs/eei_scores.csv` + prices; writes `outputs/performance_summary.{csv,json}` + PNG tear sheet.

---

## `src/5_dashboard.py` — Streamlit Research Cockpit

**Responsibility:** Interactive exploration of EEI data for research/demo purposes.

**Key features:**
- 8-page navigation (Leaderboard, Deep-Dive, Red-Flags, Alpha, Q&A Explorer, Compare, Live Scoring, Export)
- Password-gated access via env var (`EEI_DASHBOARD_PASSWORD`)
- `@st.cache_data` for all data loaders — avoids re-reading on each interaction
- Live scoring page: paste a Q&A pair, runs heuristic scorer in real-time
- Export: generates PDF/Excel reports via reportlab/openpyxl

**Interaction:** Reads all `outputs/` files + `data/processed/` + `data/cache/`.

---

## `src/signals.py` — Extended Factor Library

**Responsibility:** Build orthogonal factors and an optimized multi-factor composite.

**Key functions:**
- `confidence_score_for_answer()` — Text-only prosody proxy: penalizes fillers, fragmented punctuation, high sentence-length variance.
- `skepticism_score_for_question()` — Measures analyst pressure (pushback markers, multi-question density).
- `features_for_call()` — Per-call aggregation of confidence, skepticism, evasion-under-pressure, CEO-CFO gap.
- `add_evasion_momentum()` — 8-quarter rolling slope of EEI_raw.
- `compute_ics()` — Spearman IC of each signal at all horizons.
- `fit_composite()` — **scipy SLSQP optimizer** maximizing in-sample Sharpe of a quintile L/S on a linear combination of all factors. Constraint: |weights| sum to 1, each ∈ [-1, 1]. Negative-IC factors are pre-flipped.

**Interaction:** Reads `outputs/eei_scores.csv` + `data/processed/` + `data/prices/`; writes `outputs/signals_panel.csv`, `outputs/signals_ic.csv`, `outputs/composite_weights.json`.

---

## `src/perf.py` — Performance Utilities

**Responsibility:** Profiling, memory tracking, and cache telemetry.

- `measure()` — Context manager tracking wall time, CPU time, peak RSS.
- `profile_to_file()` — cProfile wrapper dumping .prof + .txt.
- `CacheStats` — Hit/miss accounting with estimated dollar savings.

---

## `src/utils.py` — Shared Utilities

- `get_logger()` — Console + rotating file handler (2MB max, 3 backups).
- `stable_hash()` — `json.dumps(sort_keys=True)` → SHA-1[:16]. Deterministic across runs.
- `read_json() / write_json()` — Standard JSON I/O with mkdir-p.

---

## `tests/` — Test Suite

- `conftest.py` — `importlib`-based loader for numbered modules (Python doesn't allow `import 1_scraper`). Provides fixtures for sample transcripts, Q&A pairs, mock Anthropic responses, and price DataFrames.
- `test_scorer.py` — Heuristic invariants, level-bucket consistency, tactic detection, JSON parsing edge cases, cache round-trips, async LLM mock tests.
- `test_backtester.py` — Sharpe/drawdown math verification, annualization, forward-return alignment, quintile assignment, IC correctness vs. scipy.
- `test_parser.py` — Speaker-turn splitting, role classification, Q&A pairing, linguistic-feature computation.
- `test_scraper.py` — Synthetic generation determinism, transcript schema validation.
- `test_integration.py` — End-to-end pipeline (scrape synthetic → parse → score → aggregate).

---

# 3. Critical Design Decisions & Trade-offs

## Decision 1: Synthetic Data as the Default Path

**The Choice:** Instead of requiring users to have paid transcript feeds (FactSet, Bloomberg), the system generates realistic synthetic transcripts with controlled evasion profiles.

**What We Gained:**
- Any reviewer can run the full pipeline in <60 seconds with zero setup cost
- Deterministic seeding (`SEED=42` + ticker+quarter salt) ensures reproducibility
- Known ground truth: each pair's evasion mode (direct/intermediate/evasive) is baked into generation, allowing calibration of the scorer

**What We Sacrificed:**
- The backtest is testing "can the scorer recover the synthetic evasion labels" rather than "does real-world evasion predict real returns"
- Price data is real (yfinance) but disconnected from the synthetic event dates — there's no causal link between the fake transcript and the real price move
- This makes the backtest results a **methodology proof-of-concept**, not an alpha validation

---

## Decision 2: Heuristic Scorer with LLM as Upgrade Path

**The Choice:** Ship a deterministic rule-based scorer as default, with LLM scoring as an opt-in `--mode llm` flag.

**What We Gained:**
- Zero-cost operation: no API key needed for the primary pipeline
- Determinism: same input always produces same output (critical for reproducible research)
- Speed: ~200 Q&A pairs scored in <1 second vs. ~$20 and 3 minutes for LLM mode
- The heuristic formula is transparent, auditable, and debuggable

**What We Sacrificed:**
- The heuristic can only detect surface-level linguistic markers, not deep semantic evasion
- The "confident non-answer" failure mode: an off-topic but assertive response scores low because it lacks hedge words/deflection phrases
- The heuristic's weight allocation (0.18/0.22/0.18/0.10/0.32) was hand-tuned — no formal calibration against human labels

---

## Decision 3: Content-Addressed Disk Cache for Scores

**The Choice:** SHA-1 hash of (question_text + answer_text + model) → on-disk JSON file as the cache key.

**What We Gained:**
- Idempotent re-runs: re-running the scorer doesn't re-compute or re-call the API
- LLM cost protection: ~$0.05–$0.10 per pair means $50+ for the full universe without caching
- Granular cache: individual pairs can be invalidated without nuking everything
- Git-friendly: cache files are small JSON, diffable, and can be committed for reproducibility

**What We Sacrificed:**
- Cache invalidation is manual (`--clear-cache` flag) — if the prompt changes, stale results persist
- Disk I/O overhead: ~2500 small files. On HDD this would be slow; on SSD it's negligible.
- SHA-1 is technically breakable (though collision resistance for 10K cache entries is a non-issue)

---

## Decision 4: Quintile Long/Short with Quarterly Rebalance

**The Choice:** Sort the cross-section into 5 buckets each quarter, long the least evasive quintile, short the most evasive.

**What We Gained:**
- Standard quant-research methodology — directly comparable to academic factor papers
- No continuous position sizing or leverage decisions — pure relative ranking
- Robust to score-level drift — only cross-sectional *ranking* matters

**What We Sacrificed:**
- With 20 tickers / 5 quintiles = 4 names per bucket — extremely thin statistical power
- No transaction costs modeled — real-world slippage would materially erode short-horizon signals
- Quarterly rebalance means we don't exploit intra-quarter signal decay
- Equal-weighted quintiles ignore magnitude of evasion differences

---

## Decision 5: Bounded Async Concurrency for LLM Calls

**The Choice:** `asyncio.Semaphore(8)` limiting concurrent API calls, with exponential backoff (1s → 2s → 4s → … → 30s cap) over 5 retries, plus fallback to heuristic on total failure.

**What We Gained:**
- Respect for API rate limits without explicit rate-limit parsing
- Graceful degradation: if the LLM fails on one pair, we get a heuristic score instead of a crash
- ~8x throughput vs. sequential calls (240 calls × 8 parallel ≈ 30 batches vs. 240 sequential)

**What We Sacrificed:**
- No adaptive rate limiting (if Anthropic returns 429, we still wait fixed backoff rather than respecting `Retry-After` header)
- The fallback-to-heuristic means mixed scoring provenance in the same call — could confound analysis
- `asyncio` complexity vs. simpler threading — but justified by I/O-bound nature of API calls

---

# 4. Edge Cases, Limitations & Technical Debt

## What Happens When the System Breaks

| Failure Point | Current Behavior | Risk |
|---|---|---|
| Motley Fool / EDGAR unreachable | Falls back to synthetic generation (if <50 scraped) | Silent degradation — user may not realize they're running on synthetic data |
| Anthropic API rate-limited | Exponential backoff × 5, then fallback to heuristic | Mixed-provenance scores in same output |
| yfinance returns partial data | Missing tickers are skipped in `build_signal_panel()` | Cross-section shrinks, quintile assignments become noisier |
| Transcript has no Q&A section | `split_qa_section()` returns full text → parser tries to find speaker turns in prepared remarks | May produce garbage pairs from non-Q&A text |
| Empty answer text | Division guards (`max(1, answer_word_count)`) prevent ZeroDivisionError | A 0-word "answer" gets a neutral 0.18 score (not flagged) |
| Cache file corrupted (invalid JSON) | `try/except` returns None → re-scores the pair | Silent re-computation; no alerting |

## Edge Cases Currently Handled

1. **Division by zero:** All density calculations use `max(1, n_words)` as denominator.
2. **Missing price data:** `np.isfinite()` check + `p_in > 0` guard before computing returns.
3. **Quintile ties:** `pd.qcut` with `method="first"` rank and `duplicates="drop"` — deterministic tie-breaking.
4. **NaN propagation:** `pd.isna(score)` check skips entries before panel construction.
5. **Multi-line speaker turns:** Consecutive same-role turns are concatenated (handles follow-up questions / multi-paragraph answers).
6. **Encoded characters:** The data stores raw unicode; `ensure_ascii=False` in JSON serialization preserves non-ASCII names.
7. **Empty cross-sections:** `if s.notna().sum() >= QUINTILES else np.nan` — gracefully handles quarters with too few observations.

## Technical Debt & Rebuild Priorities

### If I Had 3 More Months:

1. **Replace synthetic data with real transcripts.** Partner with a data vendor (e.g., S&P Transcript API, Refinitiv) or build a robust scraping pipeline with anti-bot evasion. The backtest is meaningless on synthetic data.

2. **Fix the "confident non-answer" bug.** The heuristic scores a completely off-topic but assertive response as "direct" because overlap alone (weight 0.18) can't override the absence of hedge/deflection markers. Needs: (a) semantic similarity via embeddings, or (b) hard floor rule when overlap ≈ 0 and length_ratio > 1.5.

3. **Proper cross-validation for composite signal.** The `fit_composite()` function in `signals.py` maximizes in-sample Sharpe — this is textbook overfitting. Needs: walk-forward optimization with rolling 8-quarter train / 4-quarter test windows.

4. **Transaction cost model.** Add configurable slippage (market-impact model or flat bps per side) to the backtester. The T+1 results would collapse immediately.

5. **Statistical significance testing.** Add bootstrap confidence intervals on Sharpe/IC. With 11–12 quarterly observations, none of the current results are statistically significant at p < 0.05.

6. **Adaptive rate limiting.** Parse `Retry-After` and `X-RateLimit-Remaining` headers from Anthropic responses rather than fixed exponential backoff.

7. **Scorer calibration.** Use the known synthetic labels (direct/intermediate/evasive) as ground truth to grid-search optimal heuristic weights via cross-validation on the scorer itself, not just downstream alpha.

8. **Incremental pipeline.** Currently re-processes all files on each run. Add a manifest/checksums to detect new/changed inputs and only process deltas.

---

# 5. The "Grill Me" Section (Viva & Interview Prep)

## Q1: "Your backtest uses synthetic transcripts. How can you claim any alpha exists?"

**Answer:**

"You're absolutely right to challenge this — and I want to be transparent: the backtest as currently configured is a **methodology proof-of-concept**, not an alpha validation. The synthetic transcripts have *controlled* evasion levels injected by design, and the price data is real but causally unconnected to those synthetic events. What the backtest validates is that (a) the scorer correctly recovers the relative ranking of evasion from generated text, and (b) the long-short portfolio construction machinery is mechanically sound — entry timing uses the next trading day on-or-after event date, quintiles are computed within-quarter only, and there are no future-information leaks in the signal panel.

To validate real alpha, I would need 3–5 years of actual earnings call transcripts from a data vendor like S&P or Refinitiv, linked to actual earnings dates. The pipeline is structured to accept that data as a drop-in replacement — the parser handles the standard Motley Fool / Seeking Alpha speaker-turn format, and the scorer is content-agnostic. The synthetic path exists so reviewers can run the full system without a $50K data subscription."

---

## Q2: "Walk me through the exact math of your heuristic evasion score. Why those weights?"

**Answer:**

"The composite score is a 5-component weighted sum on [0, 1]:

```
score = 0.18 × min(1, hedge_density / 6.0)
      + 0.22 × min(1, deflection_density / 4.0)
      + 0.18 × (1 - min(1, overlap × 6.0))
      + 0.10 × min(1, max(0, (length_ratio - 2.0) / 6.0))
      + 0.32 × (tactics_triggered / 8)
```

Each component is linearly normalized to [0, 1] with saturation caps (the `min(1, x/threshold)` pattern). The thresholds (6.0, 4.0, etc.) were calibrated against the known synthetic labels — I ensured that 'direct' template answers score < 0.35, 'intermediate' land in 0.35–0.65, and 'fully_evasive' exceed 0.65.

The weights were set based on intuition from discourse analysis theory — tactics (0.32) carry the highest weight because they represent concrete, detectable behaviors (topic pivots, legal shields, etc.) rather than statistical proxies. Deflection density (0.22) outweighs hedge density (0.18) because deflection phrases ('we don't comment', 'broadly speaking') are more specific signals of evasion than generic hedges ('approximately', 'could').

I'll acknowledge this is hand-tuned, not formally optimized. With real labeled data, I'd cross-validate these weights or replace the whole formula with a logistic regression on the same features. But for a heuristic baseline, the calibration against known synthetic labels provides reasonable discriminant validity."

---

## Q3: "Your EEI Delta signal shows a Sharpe of +1.49 at T+1. Why don't you believe this number?"

**Answer:**

"Three reasons make this Sharpe implausible:

**First, sample size.** It's computed over 11 quarterly cross-sections. With 4 stocks per quintile bucket per quarter, the annualized Sharpe is being estimated from effectively 44 long positions and 44 short positions total. The standard error on a Sharpe estimate from 11 observations is approximately `1/√(11) ≈ 0.30`, so the 95% confidence interval is roughly [0.89, 2.09]. It's not statistically distinguishable from a Sharpe of 1.0 or even 0.5.

**Second, the annualization effect.** The raw per-period return is being annualized by `√252` for a 1-day hold, which inflates the Sharpe by `√252 ≈ 15.9×` relative to the per-period information ratio. The actual per-period signal-to-noise is IC = 0.041 and ICIR = 0.20 — both below the 0.5 threshold that quants typically require for a 'real' signal.

**Third, no transaction costs.** A 1-day hold with quintile rebalance implies 100% quarterly turnover. At 20–50 bps per side, you'd subtract 40–100 bps per trade from a strategy that's making ~2% per quarter. That's a 20–50% drag on returns.

The realistic interpretation is that this is a small-sample accident driven by 2-3 favorable quarters where the short leg happened to underperform by luck."

---

## Q4: "How do you handle look-ahead bias in your signal construction?"

**Answer:**

"Look-ahead bias prevention is baked into three layers:

1. **Entry timing:** The function `_next_trading_day(px_idx, d_event)` returns the first price index date that is ≥ the event date. This means if a call happens after-hours on Tuesday, we enter on Wednesday's close — not Tuesday's. This is conservative; in reality you could enter at the next open, but I use close-to-close for simplicity.

2. **Cross-sectional ranking:** Quintile assignment uses `groupby('quarter')` — each quarter's ranking only sees scores from that same quarter. A stock's EEI from Q3 cannot influence its quintile rank in Q2.

3. **EEI_delta uses `groupby('ticker').diff()`:** This computes the change from the *prior* quarter's score — which would have been available at the time of the current quarter's call. There's no forward-looking information in the diff.

The one subtle risk area is the price cache — if `download_prices()` fetches data through `END_DATE=2024-12-31`, the forward-return computation `prices.iloc[entry_pos + horizon]` could use prices that post-date the backtest window. However, the `exit_pos >= len(px_idx)` guard ensures we never reference beyond available data. And since we're computing point-to-point returns (not rolling), there's no survivor bias either."

---

## Q5: "The standard deviation of EEI_raw is 0.049. Is this signal actually useful?"

**Answer:**

"At the individual Q&A pair level, the scores span the full [0, 1] range — direct templates score ~0.18, evasive templates score ~0.60+. The compression to std=0.049 at the *call-aggregate level* happens because each call averages 8–14 Q&A pairs. By the CLT, averaging N iid random variables reduces std by √N, so: `0.15 (pair-level std) / √10 (avg pairs) ≈ 0.047`. This matches exactly what we observe.

The implication is that the **mean is a blunt instrument** for capturing evasion. A CEO could give 12 direct answers and 1 flagrantly evasive one — the mean barely moves. Better alternatives would be:
- **Max score within a call** — captures the worst-case evasion
- **90th percentile** — robust to outliers but sensitive to tail behavior
- **Evasion concentration** (which I do compute) — max topic score minus mean, capturing whether evasion is concentrated on one sensitive topic

For the *delta* signal (std=0.06), the cross-sectional variation is actually adequate for ranking — it just needs a larger universe (50+ stocks per quarter) to produce statistically meaningful quintile sorts."

---

## Q6: "Explain the async LLM scoring architecture. What happens under failure conditions?"

**Answer:**

"The LLM scoring path uses an `asyncio.Semaphore(8)` to bound concurrency. For each call file, I iterate all Q&A pairs, check the disk cache first, and for cache misses, create a coroutine via `llm_score_one()`. All coroutines for a single call are gathered with `asyncio.gather()`.

Inside `llm_score_one()`, there's a 5-retry loop with exponential backoff (1s → 2s → 4s → 8s → 16s, capped at 30s). Each retry acquires the semaphore before calling `client.messages.create()`. Failure modes:

1. **API exception (network, 5xx, 429):** Caught by the broad `except Exception`, logged, retry after backoff.
2. **Non-JSON response:** The LLM returns text that `_parse_json(regex match for {…})` can't parse. Retry — model occasionally prefixes with prose despite the system prompt.
3. **All 5 retries exhausted:** Returns `None` → the orchestrator falls back to `heuristic_score()` for that pair, tagged as `_scorer: 'llm-failed-fallback-heuristic'`.

This means the output is *always* complete — every pair gets a score. The mixed-provenance risk (some LLM, some heuristic in one call) is logged but not currently flagged in the aggregation. In production, I'd add a `scorer_confidence` field and potentially down-weight fallback scores."

---

## Q7: "Your composite signal optimizer maximizes in-sample Sharpe. Isn't that guaranteed to overfit?"

**Answer:**

"Yes, absolutely — and I want to call that out explicitly. The `fit_composite()` function in `signals.py` is a **demonstration of the methodology**, not a production signal. With 240 observations and 9 features, the optimizer has enough degrees of freedom to find a combination that achieves an in-sample Sharpe of 2+ purely from noise.

In production, I'd implement walk-forward validation:
- Train window: 8 quarters (e.g., 2021Q1–2022Q4)
- Test window: 4 quarters (e.g., 2023Q1–2023Q4)
- Roll forward by 1 quarter, re-fit, record out-of-sample IC

Additionally, the SLSQP constraint (`|weights| sum to 1, each ∈ [-1, 1]`) provides *some* regularization — it prevents extreme leverage — but it's not equivalent to L1/L2 penalization. A ridge regression on factor returns (Bayesian shrinkage) would be more appropriate.

The current output (`composite_weights.json`) should be interpreted as 'which factors have directionally correct IC' rather than 'these are the weights to trade with.'"

---

## Q8: "How does the Jaccard overlap metric actually work, and what are its failure modes?"

**Answer:**

"The `jaccard_overlap()` function:
1. Tokenizes both question and answer using `re.findall(r'[A-Za-z][A-Za-z\-\']{2,}', text.lower())` — only words ≥ 3 chars.
2. Removes a set of 60 stopwords (the, a, and, is, etc.).
3. Computes `|intersection| / |union|` of the remaining content-word sets.

The idea: if the executive's answer shares topic-specific vocabulary with the question, they're probably addressing it. If overlap → 0, they're likely talking about something else.

**Failure modes:**

1. **Semantic synonymy:** 'revenue' in the question, 'top-line' in the answer → zero overlap despite addressing the same concept. An embedding-based similarity (cosine on sentence-transformers) would solve this.

2. **Generic vocabulary inflation:** Long answers naturally share more generic terms with any question, inflating overlap without actually being responsive. The stopword list mitigates this, but domain-specific function words ('quarter', 'year', 'growth') appear in most answers regardless of responsiveness.

3. **Multi-topic questions:** 'What's your capex outlook and how does that relate to your debt covenants?' — an answer addressing only capex would have high overlap despite evading the debt half.

4. **The zero-overlap-but-scored-low bug:** I identified this in testing — when overlap is 0.00, the penalty is only `0.18 × 1.0 = 0.18`. Combined with no hedge words and no deflection phrases, a completely off-topic answer can still score as 'direct'. This is the single biggest flaw in the heuristic."

---

## Q9: "If you were deploying this as a live trading system, what infrastructure would you need?"

**Answer:**

"The pipeline would need to transform from batch-research to event-driven:

1. **Real-time transcript ingestion:** Subscribe to a live transcript feed (e.g., Bloomberg BQuant, S&P Transcripts API). As each call ends, the Q&A section arrives within minutes.

2. **Low-latency scoring:** Pre-warm an LLM connection (or fine-tune a local 7B model like Llama on evasion labels). Target: score all Q&A pairs within 5 minutes of call ending.

3. **Signal generation + risk management:** Compute cross-sectional rank against the current quarter's already-reported calls. Integrate with a portfolio optimizer (e.g., Axioma) that respects sector/factor neutrality constraints.

4. **Execution:** Route orders to a broker API (Interactive Brokers / prime broker) with TWAP/VWAP algos to minimize market impact. For a 20-name quintile L/S, positions are ~5% each side — manageable for a $10–50M AUM book.

5. **Monitoring:** Alerting on (a) model drift (mean EEI shifting over time), (b) IC decay (rolling 4-quarter IC dropping below 0), (c) drawdown limits (automated deleveraging at -15%).

6. **Infrastructure:** Kubernetes cluster with: transcript listener (pub/sub), scorer service (GPU pod for local model or API gateway), signal engine (pandas compute), order management system, and Grafana monitoring.

The key insight is that the *information edge decays fast* — post-earnings drift is strongest in the first 1–5 days. If you're 30 minutes late, you've given up most of the T+1 alpha to faster systems (HFT reading transcript text feeds). The system would need to target T+5 or T+20 to have a realistic edge at human-speed latency."

---

## Q10: "Walk me through a specific failure case. When does the scorer give a wrong answer, and why?"

**Answer:**

"The clearest failure case I found in validation:

**Input:**
- Question: 'Can you give us an update on the consumer class action matter? Any view on timing or potential exposure ranges we should be modeling?'
- Answer: 'Capex was elevated, you're right. We're at the peak of the build-out cycle now; the run-rate normalizes in the back half of next year, and FCF conversion gets back above 90% by early 2027.'

**Score: 0.18 (direct)** — but the answer is about capex/FCF and the question was about litigation. This is a *complete topic pivot*.

**Why it failed:** The heuristic's overlap component registers 0.00 (correct), contributing 0.18 to the score. But the other components are all zero:
- No hedge words (the answer is assertive)
- No deflection phrases (no 'we don't comment', no 'broadly speaking')
- Length ratio is 2.0 (not extreme)
- No tactics triggered — crucially, `topic_pivot` requires `overlap < 0.06 AND n_a > 50`. Here `n_a = 45` words, so the `n_a > 50` condition fails by 5 words.

**Root cause:** The system conflates *linguistic evasion markers* with *evasion itself*. A skilled executive can evade by simply… answering a different question in a confident, direct tone. The heuristic was designed around the 'fumbling executive' archetype (hedging, deflecting, using legal shields), not the 'smooth redirector' archetype.

**Fix:** Either (a) lower the `n_a > 50` threshold for topic_pivot to 30, (b) add a hard-floor rule where overlap=0 forces minimum score 0.45, or (c) add a semantic similarity component using embeddings — if cosine(question_embedding, answer_embedding) < 0.3, force a high evasion floor regardless of surface markers."

---

*End of Document*
