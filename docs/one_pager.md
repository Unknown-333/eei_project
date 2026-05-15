# Executive Evasion Index (EEI) — Project One-Pager

**Aayush Paudel** · github.com/Unknown-333/eei_project · aayushpaudel09@gmail.com

---

### What it is
A production-grade alternative-data research system that scores how *evasive* corporate executives are during the Q&A section of earnings calls and trades the resulting cross-sectional signal in US equities.

### Why it matters
Sell-side analysts and buy-side PMs already read transcripts; nobody at the median fund systematically *grades* the linguistic quality of CEO answers at scale. EEI converts that qualitative read into a quantitative factor that can be combined with traditional fundamentals.

### How it works (one minute)
1. Ingest earnings-call transcripts (Motley Fool + EDGAR + deterministic synthetic fallback for reproducibility).
2. Parse Q&A pairs, identify analyst firm and executive role.
3. Score each answer 0 → 1 on a Gricean-cooperation rubric, detecting 8 evasive tactics (temporal deflection, legal/competitive shielding, macro deflection, false precision …) — heuristic by default, Anthropic Claude with strict JSON contract for production.
4. Aggregate to call-level: tier-1 analyst-weighted EEI, per-topic evasion, evasion concentration, red-flag count.
5. Build extension factors: text-prosody confidence proxy, analyst skepticism, evasion-under-pressure, CEO vs CFO answer gap, 8-quarter momentum.
6. Combine factors with `scipy.optimize.minimize` to maximise long-short Sharpe.
7. Back-test cross-sectional quintiles vs SPY (Sharpe, max DD, alpha/beta from statsmodels OLS, Spearman IC).
8. Streamlit cockpit: 8 pages including company comparison, live scoring, PDF/Excel report export.

### Headline result
| Metric | Value |
| ------ | ----- |
| `EEI_trend` Spearman IC, 20-day | **+0.153** |
| `EEI_trend` Spearman IC, 60-day | +0.125 |
| Composite signal IS Sharpe @ H=20 | **≈ 1.46** |

Disclaimer: results computed on **synthetic transcripts** seeded with deterministic per-ticker evasion profiles. The pipeline accepts real transcripts identically — the synthetic mode is a methodology validation, not a tradable expectation.

### Stack
Python 3.11 · async Anthropic SDK · pandas · numpy · scipy · statsmodels · scikit-learn · yfinance · Streamlit · Plotly · matplotlib · pytest (74 tests, mocked LLM) · joblib · pyarrow · psutil · ReportLab · GitHub Actions CI · Docker.

### Engineering polish
- 74 unit tests + 1 integration test, LLM client fully mocked
- `.coveragerc`, `pytest.ini`, `Makefile`, `.pre-commit-config.yaml`
- Profile-mode (`--profile`), cache-stats reporting (`hit-rate`, `≈$ saved`), `--clear-cache`
- Parquet price cache, batched yfinance + retries
- Dockerfile + docker-compose (`docker compose up` → dashboard on `:8501`)
- 11-section quant white paper Jupyter notebook
