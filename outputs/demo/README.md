# Demo artifacts

This folder contains a small reproducible slice of pipeline output, useful for
recruiters who want to inspect the project without running the full pipeline.

| File                     | Description                                                         |
| ------------------------ | ------------------------------------------------------------------- |
| `sample_transcript.json` | One synthetic earnings call (AAPL Q4 2020), parsed Q&A              |
| `sample_eei_scores.csv`  | First 5 rows of `outputs/eei_scores.csv` (one call-quarter per row) |
| `sample_tearsheet.png`   | Long-short quintile back-test tear sheet                            |

## How to record a `demo.gif` of the dashboard

```powershell
# 1. Run the dashboard
streamlit run src/5_dashboard.py

# 2. Use ScreenToGif (https://www.screentogif.com/) or LICEcap to record the
#    browser at 1280x720, 12 fps, ~30 s, looping through:
#       - Leaderboard
#       - Company Deep-Dive (AAPL)
#       - Compare Companies (AAPL, MSFT, NVDA)
#       - Live Scoring (paste a sample question)
#       - Alpha Dashboard
# 3. Save as outputs/demo/demo.gif (max 8 MB for GitHub README rendering).
```

The README in repo root references `outputs/demo/demo.gif` — drop the file
there to make it appear in the project landing page.
