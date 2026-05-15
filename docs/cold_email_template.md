# Cold-email template

**Subject:** Quant research project — Executive Evasion Index

---

Hi {first_name},

I built a research system over the past few weeks that I think might be of
interest to {fund_name}'s {pod_or_team} team: the **Executive Evasion Index**
— an alternative-data factor that grades how evasive corporate executives
are during the Q&A section of earnings calls and trades the resulting
cross-sectional signal in US equities.

The pipeline ingests 240 earnings-call transcripts, parses 2,600+
analyst-question / executive-answer pairs, scores each on a Gricean
8-tactic rubric (heuristic by default, Anthropic Claude with a strict JSON
contract for production), aggregates to call-level factors, runs a
scipy-optimized multi-factor combiner, and back-tests cross-sectional
quintiles vs SPY at 1, 5, 20 and 60-day horizons.

Best single signal — `EEI_trend` — produces a Spearman IC of +0.15 at the
20-day horizon; the composite hits an in-sample Sharpe of ≈ 1.46. Numbers
are on synthetic transcripts (methodology validation, not a tradable
expectation), but the pipeline accepts real transcripts identically.

Engineering: 74 pytest tests with a mocked async LLM client, parquet price
cache, Streamlit cockpit (8 pages incl. live scoring + PDF export), Docker,
GitHub Actions CI.

Code, 11-section research notebook, and one-pager:
**github.com/Unknown-333/eei_project**

Would love 20 minutes of your time to walk you through the methodology —
I'm specifically interested in feedback on (a) how my walk-forward validation
plan would map to your shop's research workflow, and (b) whether the
CEO-vs-CFO answer-gap factor matches anything you've seen elsewhere.

Best,
Aayush Paudel
aayushpaudel09@gmail.com · linkedin.com/in/{handle}
