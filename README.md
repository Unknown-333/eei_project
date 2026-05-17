# Executive Evasion Index (EEI)

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-74%20passed-success.svg)](#testing)
[![Coverage](https://img.shields.io/badge/coverage-65%25-yellowgreen.svg)](#testing)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **A research system that grades the linguistic evasion of corporate executives on earnings calls and trades the resulting cross-sectional signal in US equities.**

---

## Abstract

We extract, score and trade an alternative-data signal — the **Executive Evasion Index (EEI)** — derived from the question-and-answer section of quarterly earnings calls. Each Q&A pair is rated 0 → 1 on a Gricean-cooperation rubric (8 evasive tactics: temporal deflection, legal shielding, competitive shielding, macro deflection, reframing, false precision, etc.) by either an offline rule-based scorer or by Anthropic's Claude under a strict JSON contract. Call-level aggregates feed three flagship signals (`EEI_level`, `EEI_delta`, `Topic-Guidance`) plus four extension factors (executive-confidence proxy, analyst-skepticism, evasion-under-pressure, CEO-vs-CFO gap). Cross-sectional quintile portfolios are back-tested vs. SPY over 2021-2024 across 1, 5, 20 and 60-day horizons.

## Hypothesis

**H₁** — Executives evade more on calls when underlying business conditions are deteriorating; evasion is therefore predictive of forward equity underperformance, with strongest signal at horizons where the information has not yet been priced in (5–60 trading days).

**H₂** — _Changes_ in evasion (`EEI_delta`) carry more information than the level, because they de-mean each manager's idiosyncratic communication style.

**H₃** — A multi-factor combiner of evasion-derived signals, when sign-corrected and normalised cross-sectionally, produces an information ratio meaningfully above any single factor.

## Headline Results

| Signal             | Horizon |      Spearman IC | Notes                        |
| ------------------ | ------: | ---------------: | ---------------------------- |
| `EEI_trend`        |     20d |           +0.153 | strongest single factor      |
| `EEI_trend`        |     60d |           +0.125 | persistence at long horizons |
| `EEI_delta`        |     60d |           +0.098 | level-detrended              |
| `composite_signal` |     20d | IS Sharpe ≈ 1.46 | scipy-optimized linear combo |

> Numbers above are computed on **synthetic transcripts seeded by deterministic per-ticker evasion profiles**, not on production transcripts. The pipeline accepts real Motley Fool / EDGAR transcripts identically — see `src/1_scraper.py`. All artefacts are reproducible from the committed code: `make run-pipeline`.

## Architecture

```
┌──────────────┐   ┌──────────────┐   ┌────────────────┐   ┌──────────────┐
│ 1_scraper.py │ → │ 2_parser.py  │ → │ 3_evasion_     │ → │ 4_backtester │
│  Motley Fool │   │  QA pairs +  │   │  scorer.py     │   │  yfinance +  │
│  + EDGAR +   │   │  linguistic  │   │  heuristic OR  │   │  quintiles + │
│  synthetic   │   │  features    │   │  Anthropic LLM │   │  tear sheet  │
└──────────────┘   └──────────────┘   └────────────────┘   └──────────────┘
                                                  │
                                                  ▼
                                        ┌──────────────────┐
                                        │ signals.py       │
                                        │ 4 factors +      │
                                        │ scipy combiner   │
                                        └──────────────────┘
                                                  │
                                                  ▼
                                        ┌──────────────────┐
                                        │ 5_dashboard.py   │
                                        │ Streamlit (8     │
                                        │ pages, PDF/Excel │
                                        │ export)          │
                                        └──────────────────┘
```

## Methodology

1. **Ingestion** — `src/1_scraper.py` pulls Motley Fool transcripts and EDGAR 8-K filings; falls back to deterministic synthetic transcripts for reproducibility.
2. **Parsing** — `src/2_parser.py` segments the prepared remarks vs. Q&A halves (regex-based section detection), pairs analyst questions with executive answers using a `Speaker — Firm — Title` header model, and computes cheap linguistic features (hedge counts, deflection keywords, Jaccard overlap, length ratio).
3. **Scoring** — `src/3_evasion_scorer.py` runs either:
   - **`--mode heuristic`** — fast, free, deterministic, regex-tactic detectors (`_TIME_DEFLECT_RE`, `_LEGAL_SHIELD_RE`, …) — used by default and in CI;
   - **`--mode llm`** — Anthropic Claude with a strict JSON contract enforced by a SYSTEM*PROMPT detailing all 8 tactics; concurrency-bounded async client with exponential-backoff retries; results cached as `data/cache/score*<sha1>.json` so re-runs are free.
4. **Aggregation** — call-level `EEI_raw`, `EEI_weighted` (tier-1 sell-side analyst weighting), per-topic evasion, tactic frequencies, evasion concentration, fully-evasive %, red-flag count.
5. **Cross-call features** — `EEI_delta` (1Q diff), `EEI_trend` (4Q rolling slope), `evasion_momentum_8q` (8Q slope).
6. **Extension signals** (`src/signals.py`) — confidence proxy from text prosody, analyst skepticism on questions, evasion-under-pressure, CEO-vs-CFO gap.
7. **Composite** — `scipy.optimize.minimize` on a sign-corrected linear combo to maximise long-short Sharpe at H=20.
8. **Back-test** — `src/4_backtester.py` builds quintile-sorted long-short portfolios per quarter; reports Sharpe, max drawdown, alpha/beta vs SPY (statsmodels OLS), Spearman IC.
9. **Dashboard** — Streamlit with 8 pages: leaderboard, deep-dive, alerts, alpha, raw Q&A explorer, **multi-company comparison, live scoring, PDF/Excel export**.

## Project Structure

```
eei_project/
├── config.py                   # constants, tickers, lexicons, paths
├── src/
│   ├── 1_scraper.py            # Motley Fool + EDGAR + synthetic
│   ├── 2_parser.py             # QA pair extraction + linguistic features
│   ├── 3_evasion_scorer.py     # heuristic + LLM scoring (async, cached)
│   ├── 4_backtester.py         # yfinance, quintiles, tear sheet
│   ├── 5_dashboard.py          # Streamlit 8-page cockpit
│   ├── signals.py              # 4 extension factors + composite
│   ├── perf.py                 # profiling, memory tracking, cache stats
│   └── utils.py                # logging, hashing, JSON IO
├── tests/                      # 74 pytest tests, mocked LLM
├── notebooks/research_analysis.ipynb   # 11-section quant white paper
├── outputs/                    # eei_scores.csv, signals_panel.csv, tearsheet.png …
├── data/
│   ├── transcripts/            # raw transcript text
│   ├── processed/              # *_qa_pairs.json
│   ├── cache/                  # LLM response cache
│   └── prices/prices.parquet   # cached yfinance close prices
├── requirements.txt
├── Makefile                    # install / test / run-pipeline / dashboard / docker-*
├── Dockerfile + docker-compose.yml
├── pytest.ini · .coveragerc · .pre-commit-config.yaml
└── .github/workflows/ci.yml    # pytest + black + flake8
```

## Quick start

```powershell
# 1. Install
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. (Optional) set Anthropic key for --mode llm and EEI_DASHBOARD_PASSWORD
copy .env.example .env

# 3. Run the full offline pipeline (deterministic, no API key required)
python src/1_scraper.py        # generate / fetch transcripts
python src/2_parser.py         # extract 2,600+ Q&A pairs
python src/3_evasion_scorer.py --mode heuristic
python src/4_backtester.py
python src/signals.py

# 4. Launch the dashboard
streamlit run src/5_dashboard.py
```

Or simply: `make run-pipeline` then `make dashboard`.

### Docker

```bash
docker compose up --build
# dashboard available at http://localhost:8501
```

## Testing

```powershell
make test          # 74 tests, ~5 s
make coverage      # HTML report → htmlcov/index.html
```

- 74 unit tests, 1 integration test
- LLM client fully mocked via `unittest.mock.AsyncMock` — tests never hit the network or burn tokens
- Coverage gates: `config.py` 100 %, `parser` 79 %, `scorer` 68 %, `backtester` 46 % (CLI orchestrators excluded), `utils` 97 %; `dashboard` excluded by `.coveragerc`

## Limitations

- **Synthetic-data results are an upper bound.** The deterministic seeded RNG in `src/1_scraper.py` introduces structural correlation between evasion intensity and ticker; live transcripts will produce more diffuse signal. Treat all reported ICs as a methodology validation, not a tradable expectation.
- **Survivorship + selection bias** — only currently-listed S&P-style tickers are in the universe; no de-listings.
- **Transaction costs and capacity** are not modelled. Spreads on small-caps would erode the 1-day signal.
- **LLM scoring is a moving target.** Different model versions (`claude-opus-4-20250514` vs successors) produce different absolute evasion scores; only differences within a single model snapshot are comparable.
- **No audio.** "Confidence" is a text-only prosody proxy. Real prosody features (pitch variance, speaking rate, hesitation duration) are out of scope.
- **English-only universe**, US equities only.

## Selected references

- H. P. Grice (1975). _Logic and conversation_.
- Larcker, D. F. & Zakolyukina, A. A. (2012). _Detecting deceptive discussions in conference calls_. Journal of Accounting Research.
- Loughran, T. & McDonald, B. (2011). _When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks_. Journal of Finance.
- Tetlock, P. (2007). _Giving content to investor sentiment: the role of media in the stock market_. Journal of Finance.
- Anthropic (2024). _Claude documentation — structured outputs_.

## License

MIT — see `LICENSE`.
