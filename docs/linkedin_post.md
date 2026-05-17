# LinkedIn launch post (≈ 230 words)

---

I shipped a project I've wanted to build for a long time: the **Executive Evasion Index** — a research system that grades how evasive corporate executives are during the Q&A section of earnings calls and trades the resulting cross-sectional signal in US equities.

Sell-side analysts already read transcripts. Most quants don't _systematically score_ the linguistic quality of CEO answers at scale. EEI does.

The pipeline:

- Pulls 240 earnings-call transcripts (Motley Fool + EDGAR + a deterministic synthetic fallback for reproducibility).
- Extracts 2,600+ analyst-question / executive-answer pairs.
- Scores each answer 0 → 1 on a Gricean-cooperation rubric, detecting 8 evasion tactics (temporal deflection, legal shielding, macro deflection, false precision …) — offline heuristic by default, Anthropic Claude with a strict JSON contract for production.
- Aggregates to call-level (tier-1 analyst weighted EEI, evasion concentration, CEO-vs-CFO gap, 8-quarter momentum).
- Combines five factors via `scipy.optimize` to maximise long-short Sharpe.
- Back-tests cross-sectional quintiles vs SPY at 1, 5, 20, 60-day horizons.

Best single signal — `EEI_trend` — produces a Spearman IC of **+0.15 at the 20-day horizon**; the multi-factor composite hits an in-sample Sharpe of **≈ 1.46**. (Numbers are on synthetic transcripts — methodology validation, not a tradable expectation.)

Engineering: 74 pytest tests with a mocked async LLM client, Streamlit cockpit with 8 pages incl. live scoring + PDF/Excel export, Docker, GitHub Actions CI, parquet price cache.

Code & 11-section research notebook → github.com/Unknown-333/eei_project

#quant #alternativedata #nlp #python #anthropic #earningscalls
