# Interview talking points — Executive Evasion Index

Five questions an interviewer at Point72 / Citadel / Two Sigma is most likely
to ask, and the answer I would give.

---

## 1. "Walk me through the project end to end."

The system has six modules. **(1)** A scraper pulls real earnings-call
transcripts from Motley Fool and EDGAR, with a deterministic synthetic
fallback so the pipeline is fully reproducible without any API. **(2)** A
parser segments the prepared remarks vs Q&A halves with regex, then pairs
analyst questions with executive answers using a `Speaker — Firm — Title`
header model and computes cheap linguistic features. **(3)** A scorer rates
each answer 0 → 1 on a Gricean-cooperation rubric — eight evasive tactics
including temporal deflection, legal shielding, competitive shielding, macro
deflection and false precision. It runs in two modes: a free, deterministic,
regex-based heuristic that I use for CI and reproducibility, and an async
Anthropic-Claude client with a strict JSON contract, exponential-backoff
retries, content-hashed caching and per-call cost telemetry. **(4)** A
back-tester does cross-sectional quintile sorts at 1, 5, 20 and 60-day
horizons against SPY, reporting Sharpe, max drawdown, alpha and beta from
statsmodels OLS, and Spearman IC. **(5)** An extension-signals module adds
four more factors — a text-prosody confidence proxy, analyst-skepticism on
questions, evasion-under-pressure, the CEO-vs-CFO gap, and 8-quarter momentum
— then runs `scipy.optimize.minimize` over a sign-corrected linear combiner
to maximise long-short Sharpe. **(6)** A Streamlit cockpit with eight pages
exposes everything to a PM: leaderboard, deep dive, alerts, alpha dashboard,
raw Q&A explorer, multi-company comparison, live scoring, and PDF/Excel
report export.

## 2. "How did you avoid look-ahead bias?"

Three places it could leak in and how each is closed:

* **Signal timestamping** — every `EEI` row is keyed to the *transcript date*,
  i.e. the date the call actually happened. Returns are taken strictly
  *forward*: at horizon h I take the next trading day after the call as t₀
  and the close at t₀+h as t₁. The `_forward_returns` helper in
  `src/signals.py` does a `searchsorted` on the price index — never an `iloc`
  on a date that could have been adjusted in-sample.
* **Cross-sectional ranking** — quintile assignment is done *per quarter*,
  using only data available at that quarter. There is no full-sample
  z-scoring of features.
* **Composite weights** — the `scipy.optimize` step is currently in-sample;
  the honest next step is a walk-forward fit (re-fit weights every N quarters
  on rolling data). I flag this in the README's Limitations section.

## 3. "Heuristic vs LLM scoring — why both?"

Three reasons. **Reproducibility:** anyone can `git clone`, `pip install`,
`python src/3_evasion_scorer.py --mode heuristic` and get bit-identical
results. CI does this every push. **Cost & latency:** 2,600 Q&A pairs at
Claude Opus pricing is roughly $40 and ten minutes per full run; the
heuristic is free and finishes in two seconds. I keep the LLM mode for
production-grade scoring on tickers I actually want to trade, and the
heuristic mode for development and ablations. **Validation:** if the
heuristic and LLM agree on rank-order, that's evidence the signal is real
linguistic structure, not LLM-stylistic noise. The architecture is identical
between the two modes — same `aggregate_call`, same per-tactic frequencies —
which makes the comparison clean.

## 4. "Your IC of 0.15 looks suspiciously high. Why isn't this just data
mining?"

Two honest answers. First: that number is from synthetic transcripts whose
evasion intensity is seeded with deterministic per-ticker profiles, so there
is structural correlation built into the data that real-world transcripts
won't have. I'm explicit about this in the README — it's a methodology
validation, not a tradable expectation. Second: even setting that aside, the
guards against pure data mining are (a) the rubric is theory-grounded —
Larcker & Zakolyukina (2012) showed deception markers in Q&A predict
restatements; Loughran-McDonald sentiment predicts returns; Grice's
maxims define what cooperative discourse looks like — so I'm not searching
random feature space, (b) I only test 9 signals and report ICs at four
horizons (36 numbers), not thousands, (c) the multi-factor weights have a
sign constraint and are normalized, which limits how aggressively the
optimizer can over-fit, and (d) the dashboard exposes per-quarter and
per-ticker decomposition so any anomalous result can be inspected by hand.

## 5. "How would you take this to production at a real fund?"

Roughly five things. **(a) Transcript ingestion** — replace Motley Fool
scraping with a paid feed (FactSet StreetEvents or AlphaSense) for SLA and
coverage. **(b) Real-time scoring** — sub-five-minute latency from call end
to scored signal, by streaming the live transcript chunks into the async
scorer instead of batching after the fact. **(c) Universe expansion** —
extend from 20 megacaps to the full Russell 3000, which mostly means more
parallel scraping and a smarter cache shard. **(d) Walk-forward validation
+ regime overlay** — re-fit composite weights on a rolling 24-month window
and overlay a simple regime filter (VIX bucket, NBER recession flag) so the
signal is conditional. **(e) Risk model** — neutralise the long-short by
sector and Barra factors before quintile sorts; the current naked-beta
version is a research artefact, not a portfolio.

The engineering substrate is already production-shaped: async client with
backoff and caching, parquet storage, Docker, CI, mocked tests, profile
telemetry. The next step is more about data and deployment than about
re-architecting the code.
