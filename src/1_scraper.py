"""
Module 1 — Transcript Acquisition.

Sources, in priority order:
    1. Motley Fool earnings-call transcript pages
    2. Seeking Alpha transcript pages (best-effort; usually paywalled)
    3. SEC EDGAR full-text search for 8-K filings carrying call transcripts
    4. Synthetic generator (deterministic templates + optional LLM elaboration)

The synthetic path is what makes this repo runnable for reviewers without
paid data feeds. It produces realistic, varied transcripts whose evasion
levels are systematically modulated so the downstream backtest has signal
to find — see ``synthetic_calls`` below.

Run:
    python src/1_scraper.py
    python src/1_scraper.py --synthetic-only
    python src/1_scraper.py --tickers AAPL MSFT --max-per-ticker 4
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    SEED,
    START_DATE,
    SYNTHETIC_PER_TICKER,
    TICKERS,
    TRANSCRIPTS_DIR,
)
from src.utils import get_logger, write_json  # noqa: E402

LOG = get_logger("scraper")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Transcript:
    company: str
    ticker: str
    date: str           # ISO YYYY-MM-DD
    quarter: str        # e.g. "Q1 2024"
    raw_text: str
    source: str

    def filename(self) -> str:
        return f"{self.ticker}_{self.date}.json"


# ---------------------------------------------------------------------------
# Live scrapers (best-effort)
# ---------------------------------------------------------------------------
def _request(url: str, timeout: int = 12) -> requests.Response | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200 and len(r.text) > 1500:
            return r
        LOG.debug("non-200 or short body for %s (status=%s len=%s)", url, r.status_code, len(r.text))
    except requests.RequestException as exc:
        LOG.debug("request failed for %s: %s", url, exc)
    return None


def scrape_motley_fool_index(pages: int = 3) -> list[str]:
    """Return article URLs from the Motley Fool earnings-call index."""
    urls: list[str] = []
    for page in range(1, pages + 1):
        idx = f"https://www.fool.com/earnings-call-transcripts/?page={page}"
        r = _request(idx)
        if not r:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.select("a[href*='/earnings/call-transcripts/']"):
            href = a.get("href", "")
            if href.startswith("/"):
                href = "https://www.fool.com" + href
            if href not in urls:
                urls.append(href)
        time.sleep(0.5)
    LOG.info("motley_fool index returned %d urls", len(urls))
    return urls


def parse_motley_fool_article(url: str) -> Transcript | None:
    r = _request(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    body_node = soup.find("div", class_=re.compile("article-body|tailwind-article-body"))
    if not body_node:
        return None
    text = body_node.get_text("\n", strip=True)
    if len(text) < 2000:
        return None
    title = (soup.find("h1") or soup.title).get_text(strip=True) if soup.find("h1") else url
    ticker_m = re.search(r"\(([A-Z]{1,5})\s*[-–]?\s*([0-9.\-]+)?\s*[%]?\)", title)
    ticker = ticker_m.group(1) if ticker_m else "UNK"
    quarter_m = re.search(r"(Q[1-4])\s*(20\d{2})", title)
    quarter = f"{quarter_m.group(1)} {quarter_m.group(2)}" if quarter_m else "Unknown"
    pub = soup.find("meta", attrs={"property": "article:published_time"})
    iso = pub["content"][:10] if pub and pub.get("content") else date.today().isoformat()
    return Transcript(
        company=title.split("(")[0].strip(),
        ticker=ticker,
        date=iso,
        quarter=quarter,
        raw_text=text,
        source="motley_fool",
    )


def scrape_motley_fool(max_articles: int = 50) -> list[Transcript]:
    out: list[Transcript] = []
    for url in scrape_motley_fool_index(pages=5)[:max_articles]:
        t = parse_motley_fool_article(url)
        if t:
            out.append(t)
            LOG.info("scraped %s %s (%s)", t.ticker, t.quarter, t.date)
        time.sleep(0.7)
    return out


def search_edgar_8k(ticker: str, max_filings: int = 4) -> list[Transcript]:
    """SEC EDGAR full-text search for 8-K exhibits carrying call transcripts."""
    out: list[Transcript] = []
    q = f'"earnings conference call" {ticker}'
    url = (
        "https://efts.sec.gov/LATEST/search-index?"
        f"q={requests.utils.quote(q)}&forms=8-K"
    )
    r = _request(url)
    if not r:
        return out
    try:
        hits = r.json().get("hits", {}).get("hits", [])[:max_filings]
    except ValueError:
        return out
    for h in hits:
        src = h.get("_source", {})
        adsh = src.get("adsh", "").replace("-", "")
        cik = src.get("ciks", ["0"])[0].zfill(10)
        fname = src.get("file_type", "")
        if not adsh or not fname:
            continue
        link = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{adsh}/{src.get('id','')}"
        rr = _request(link)
        if not rr:
            continue
        text = BeautifulSoup(rr.text, "html.parser").get_text("\n", strip=True)
        if len(text) < 2500:
            continue
        out.append(
            Transcript(
                company=ticker,
                ticker=ticker,
                date=src.get("file_date", date.today().isoformat()),
                quarter="Unknown",
                raw_text=text,
                source="sec_edgar",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms, Inc.", "AMZN": "Amazon.com, Inc.", "NVDA": "NVIDIA Corporation",
    "JPM": "JPMorgan Chase & Co.", "GS": "The Goldman Sachs Group, Inc.",
    "BAC": "Bank of America Corporation", "C": "Citigroup Inc.", "MS": "Morgan Stanley",
    "XOM": "Exxon Mobil Corporation", "CVX": "Chevron Corporation",
    "WMT": "Walmart Inc.", "HD": "The Home Depot, Inc.", "COST": "Costco Wholesale Corporation",
    "PFE": "Pfizer Inc.", "JNJ": "Johnson & Johnson", "UNH": "UnitedHealth Group Incorporated",
    "TSLA": "Tesla, Inc.",
}

EXECS: dict[str, list[tuple[str, str]]] = {
    "AAPL": [("Tim Cook", "Chief Executive Officer"), ("Luca Maestri", "Chief Financial Officer")],
    "MSFT": [("Satya Nadella", "Chief Executive Officer"), ("Amy Hood", "Chief Financial Officer")],
    "GOOGL": [("Sundar Pichai", "Chief Executive Officer"), ("Ruth Porat", "President and CIO")],
    "META": [("Mark Zuckerberg", "Chief Executive Officer"), ("Susan Li", "Chief Financial Officer")],
    "AMZN": [("Andy Jassy", "Chief Executive Officer"), ("Brian Olsavsky", "Chief Financial Officer")],
    "NVDA": [("Jensen Huang", "Chief Executive Officer"), ("Colette Kress", "Chief Financial Officer")],
    "JPM": [("Jamie Dimon", "Chief Executive Officer"), ("Jeremy Barnum", "Chief Financial Officer")],
    "GS": [("David Solomon", "Chief Executive Officer"), ("Denis Coleman", "Chief Financial Officer")],
    "BAC": [("Brian Moynihan", "Chief Executive Officer"), ("Alastair Borthwick", "Chief Financial Officer")],
    "C": [("Jane Fraser", "Chief Executive Officer"), ("Mark Mason", "Chief Financial Officer")],
    "MS": [("Ted Pick", "Chief Executive Officer"), ("Sharon Yeshaya", "Chief Financial Officer")],
    "XOM": [("Darren Woods", "Chief Executive Officer"), ("Kathryn Mikells", "Chief Financial Officer")],
    "CVX": [("Mike Wirth", "Chief Executive Officer"), ("Pierre Breber", "Chief Financial Officer")],
    "WMT": [("Doug McMillon", "Chief Executive Officer"), ("John David Rainey", "Chief Financial Officer")],
    "HD": [("Ted Decker", "Chief Executive Officer"), ("Richard McPhail", "Chief Financial Officer")],
    "COST": [("Ron Vachris", "Chief Executive Officer"), ("Richard Galanti", "Chief Financial Officer")],
    "PFE": [("Albert Bourla", "Chief Executive Officer"), ("David Denton", "Chief Financial Officer")],
    "JNJ": [("Joaquin Duato", "Chief Executive Officer"), ("Joseph Wolk", "Chief Financial Officer")],
    "UNH": [("Andrew Witty", "Chief Executive Officer"), ("John Rex", "Chief Financial Officer")],
    "TSLA": [("Elon Musk", "Chief Executive Officer"), ("Vaibhav Taneja", "Chief Financial Officer")],
}

ANALYSTS: list[tuple[str, str]] = [
    ("Toni Sacconaghi", "Bernstein"),
    ("Katy Huberty", "Morgan Stanley"),
    ("Wamsi Mohan", "Bank of America"),
    ("Erik Woodring", "Morgan Stanley"),
    ("Mark Moerdler", "Bernstein"),
    ("Brent Thill", "Jefferies"),
    ("Karl Keirstead", "UBS"),
    ("Brian Nowak", "Morgan Stanley"),
    ("Doug Anmuth", "JPMorgan"),
    ("Eric Sheridan", "Goldman Sachs"),
    ("Mark Mahaney", "Evercore ISI"),
    ("Justin Post", "Bank of America"),
    ("Ross Gerber", "Gerber Kawasaki"),
    ("Stacy Rasgon", "Bernstein"),
    ("Vivek Arya", "Bank of America"),
    ("Timothy Arcuri", "UBS"),
    ("Mike Mayo", "Wells Fargo"),
    ("Glenn Schorr", "Evercore ISI"),
    ("Betsy Graseck", "Morgan Stanley"),
    ("Devin Ryan", "Citizens JMP"),
    ("Neil Mehta", "Goldman Sachs"),
    ("Doug Leggate", "Bank of America"),
    ("Simeon Gutman", "Morgan Stanley"),
    ("Chuck Grom", "Gordon Haskett"),
    ("Lisa Gill", "JPMorgan"),
    ("Geoff Meacham", "Bank of America"),
]

QUESTION_TEMPLATES: list[tuple[str, str]] = [
    ("margins",   "On gross margin — you guided {gm_prev}% last quarter and you printed {gm}. What are the puts and takes for next quarter and how should we think about the trajectory into next year?"),
    ("guidance",  "Your full-year guidance implies a meaningful step-up in the back half. Can you walk us through the bridge — how much is pricing, how much is volume, and how much is mix?"),
    ("competition", "We've seen pretty aggressive moves from {comp}. Are you seeing any share shift, and what's your strategy to defend in the segment?"),
    ("capex",     "Capex was {capex_pct}% above the Street. What's the duration of this elevated investment cycle and when do you expect free cash flow conversion to normalize?"),
    ("debt",      "On the balance sheet — net leverage ticked up to {lev}x. How are you thinking about the optimal capital structure, and is there a refinancing window we should be aware of?"),
    ("litigation", "Can you give us an update on the {litig} matter? Any view on timing or potential exposure ranges we should be modeling?"),
    ("macro",     "Given the rate environment and the consumer slowdown signals, are you seeing any change in customer behavior, and how does that flow through your forward outlook?"),
    ("hiring",    "Headcount was down {hc}% sequentially. Is this a structural reset or a pause, and what does that imply about the operating-margin glide path?"),
    ("product",   "On the {prod} roadmap — competitors have been aggressive on launches. Where are you in the cycle and what's the monetization timeline?"),
    ("buybacks",  "Buybacks slowed materially this quarter. Should we read this as a signal on M&A appetite, or is it pure timing?"),
]

DIRECT_TEMPLATES: list[str] = [
    "Sure, {analyst}. Gross margin came in at {gm}, which was {beat} our internal plan by about {bp} basis points. The puts were {tail1}; the takes were {tail2}. Going into next quarter we expect a roughly {nxt} range, and for the year we're tightening to {fy}. The biggest delta line by line is {delta_line}.",
    "Thanks {analyst}. The bridge is roughly half pricing — about {pct1}% — and the rest is mix and volume in roughly equal proportion. To put a finer point on it, in the back half we've baked in {assump1} and {assump2}, and the sensitivity to a 100bp move in {sens} is approximately ${sens_dollar} of operating income.",
    "Capex was elevated, you're right. We're at the peak of the build-out cycle now; the run-rate normalizes in the back half of next year, and FCF conversion gets back above {fcf}% by {when}. The specific projects driving the variance are {proj1} and {proj2}.",
]

EVASIVE_TEMPLATES: list[str] = [
    "Yeah, look, {analyst}, as we've discussed previously, we don't break out the components of gross margin at that level of granularity. What I would say is that we feel really good about the business, we feel really good about the long-term trajectory, and we'll continue to invest behind the things that matter for our customers. Over time, you should expect us to deliver, and we're confident in our ability to execute against our plan. As I mentioned in the prepared remarks, there are a lot of moving pieces in any given quarter.",
    "It's a great question. Broadly speaking, the framework we use is the same framework we've always used, which is to balance investment with returns over the appropriate time horizon. I don't want to get out ahead of ourselves on guidance — we'll talk more about that on the next call. What I'd point you to is the long-term algorithm we laid out at the analyst day; that hasn't changed.",
    "Yeah, I appreciate the question. We don't comment on competitive dynamics in real time. What I can tell you is that we're maniacally focused on the customer, and when we focus on the customer we tend to do well. There's a lot of noise in the market and we try not to react to every data point. As I've said before, we run our own race.",
    "Look, I think it's premature to draw conclusions from one quarter of data. The macro environment is dynamic — there are a lot of cross-currents — and we want to be appropriately humble about what we can and can't predict. We'll continue to monitor the situation and we'll update you as appropriate. As we've discussed, our model has historically been resilient through cycles.",
    "On the legal matter — as you'd expect, I'm not in a position to comment on ongoing litigation beyond what's in our public disclosures. What I would say more broadly is that we have strong defenses, we believe the positions we've taken are correct, and we'll defend them vigorously. I'd refer you to the 10-Q for the specific language.",
]

INTERMEDIATE_TEMPLATES: list[str] = [
    "Thanks {analyst}. So margin came in at {gm}. Without getting into the line-by-line, the key drivers were a mix shift toward {mix} and some FX tailwinds we'd called out previously. For next quarter we'd expect to be in roughly the same neighborhood, plus or minus. We're not changing the long-term framework.",
    "On capex — yes, the number was elevated, and that reflects the timing of a few large projects. We'd guide you to think of it as a high-water mark; the back half normalizes, but I don't want to give you a specific quarter. The investment is consistent with the plan we laid out.",
    "Look, on guidance — we've tightened the range, which I think is the most useful thing we can do. The puts and takes are roughly what you'd expect: pricing is contributing, volume is mixed, and there's some currency. We feel comfortable with where consensus is sitting.",
]


def _seeded_rng(ticker: str, q: str) -> random.Random:
    return random.Random(f"{ticker}-{q}-{SEED}")


def _quarter_dates(start_year: int, end_year: int) -> list[tuple[str, str]]:
    """Return [(date, quarter_label), ...] for quarter end + 25 days (typical print)."""
    out: list[tuple[str, str]] = []
    for y in range(start_year, end_year + 1):
        for q, m in enumerate([1, 4, 7, 10], start=1):
            # Report ~25 days after quarter end of *previous* quarter.
            d = date(y, m, 25)
            qlabel = f"Q{((q - 1) - 1) % 4 + 1} {y if q != 1 else y - 1}"
            # Simpler: report Q{q-1} {y} for jan, Q1 for apr etc.
            mapping = {1: ("Q4", y - 1), 4: ("Q1", y), 7: ("Q2", y), 10: ("Q3", y)}
            qq, yy = mapping[m]
            out.append((d.isoformat(), f"{qq} {yy}"))
    return out


def _make_qa(rng: random.Random, ticker: str, mode: str) -> tuple[str, str]:
    """Return (question, answer) text for a single Q&A pair.

    ``mode`` is one of ``direct | intermediate | evasive``.
    """
    analyst, firm = rng.choice(ANALYSTS)
    topic, qtmpl = rng.choice(QUESTION_TEMPLATES)
    fills = dict(
        gm_prev=rng.choice([42, 43, 44, 45, 46]),
        gm=f"{rng.choice([41, 42, 43, 44, 45])}.{rng.randint(0,9)}%",
        comp=rng.choice(["TSMC", "Microsoft", "AWS", "Walmart", "Pfizer", "JPMorgan", "Tesla"]),
        capex_pct=rng.randint(8, 35),
        lev=round(rng.uniform(1.1, 3.4), 1),
        litig=rng.choice(["DOJ antitrust", "FTC merger review", "patent infringement", "consumer class action"]),
        hc=rng.randint(2, 9),
        prod=rng.choice(["AI", "cloud", "EV", "GLP-1", "trading platform", "advertising"]),
    )
    question = qtmpl.format(**fills)
    qprefix = f"{analyst} -- {firm} -- Analyst\n"
    if mode == "direct":
        atmpl = rng.choice(DIRECT_TEMPLATES)
        answer = atmpl.format(
            analyst=analyst.split()[0],
            gm=fills["gm"], beat=rng.choice(["beat", "ahead of", "modestly above"]),
            bp=rng.choice([40, 60, 80, 110]),
            tail1=rng.choice(["pricing strength in premium SKUs", "favorable FX", "lower freight"]),
            tail2=rng.choice(["a one-time inventory write-up", "promotional intensity", "mix shift to lower-margin geographies"]),
            nxt=f"{rng.randint(42,46)}-{rng.randint(46,49)}%", fy=f"{rng.randint(43,46)}-{rng.randint(46,48)}%",
            delta_line=rng.choice(["component costs", "logistics", "customer mix"]),
            pct1=rng.randint(40, 60),
            assump1="single-digit volume growth", assump2="modest pricing tailwinds",
            sens=rng.choice(["FX", "freight", "wages"]), sens_dollar=rng.randint(80, 250),
            fcf=rng.randint(85, 105), when=rng.choice(["mid-2026", "late 2026", "early 2027"]),
            proj1=rng.choice(["new fab capacity", "data center buildout", "store remodels"]),
            proj2=rng.choice(["network upgrades", "AI infrastructure", "supply chain modernization"]),
        )
    elif mode == "intermediate":
        atmpl = rng.choice(INTERMEDIATE_TEMPLATES)
        answer = atmpl.format(
            analyst=analyst.split()[0], gm=fills["gm"],
            mix=rng.choice(["services", "premium SKUs", "enterprise"]),
        )
    else:  # evasive
        atmpl = rng.choice(EVASIVE_TEMPLATES)
        answer = atmpl.format(analyst=analyst.split()[0])
    return qprefix + question, answer


def _evasion_profile(ticker: str, qlabel: str) -> list[str]:
    """Pick the evasion-mode mix for one call.

    Some tickers are systematically more evasive; we also drift evasiveness
    over time to give the backtester a real signal to recover.
    """
    rng = _seeded_rng(ticker, qlabel)
    base_evasive = {
        "AAPL": 0.22, "MSFT": 0.18, "GOOGL": 0.40, "META": 0.30, "AMZN": 0.25,
        "NVDA": 0.15, "JPM": 0.20, "GS": 0.30, "BAC": 0.22, "C": 0.45, "MS": 0.28,
        "XOM": 0.50, "CVX": 0.45, "WMT": 0.20, "HD": 0.25, "COST": 0.18,
        "PFE": 0.42, "JNJ": 0.30, "UNH": 0.55, "TSLA": 0.50,
    }.get(ticker, 0.30)
    drift = rng.uniform(-0.15, 0.15)
    p_ev = max(0.05, min(0.85, base_evasive + drift))
    p_dir = max(0.05, 0.7 - p_ev)
    p_int = 1.0 - p_ev - p_dir
    n_pairs = rng.randint(8, 14)
    modes: list[str] = []
    for _ in range(n_pairs):
        r = rng.random()
        if r < p_dir:
            modes.append("direct")
        elif r < p_dir + p_int:
            modes.append("intermediate")
        else:
            modes.append("evasive")
    return modes


def _prepared_remarks(ticker: str, quarter: str) -> str:
    return (
        f"Operator: Good afternoon and welcome to the {COMPANY_NAMES.get(ticker, ticker)} "
        f"{quarter} earnings conference call. At this time all participants are in a "
        f"listen-only mode. After the speakers' presentation there will be a question-and-answer "
        f"session. Today's call is being recorded.\n\n"
        f"I would now like to turn the call over to Investor Relations.\n\n"
        f"IR: Thank you operator and good afternoon everyone. With me on the call today are our "
        f"executives. Before we begin, I'd like to remind you that today's discussion will include "
        f"forward-looking statements that involve risks and uncertainties...\n\n"
        f"[Prepared remarks omitted for brevity]\n\n"
        f"Operator: Now we'll open up the call for questions.\n\n"
    )


def synthetic_calls(
    tickers: Iterable[str] = TICKERS,
    per_ticker: int = SYNTHETIC_PER_TICKER,
    start_year: int = 2021,
    end_year: int = 2024,
) -> list[Transcript]:
    """Generate deterministic synthetic transcripts."""
    out: list[Transcript] = []
    qdates = _quarter_dates(start_year, end_year)[:per_ticker]
    for ticker in tickers:
        execs = EXECS.get(ticker, [("CEO", "Chief Executive Officer"), ("CFO", "Chief Financial Officer")])
        for iso, qlabel in qdates:
            modes = _evasion_profile(ticker, qlabel)
            rng = _seeded_rng(ticker, qlabel + "-pairs")
            body = _prepared_remarks(ticker, qlabel)
            for mode in modes:
                q, a = _make_qa(rng, ticker, mode)
                exec_name, exec_title = rng.choice(execs)
                body += q + "\n\n"
                body += f"{exec_name} -- {exec_title}\n{a}\n\n"
            body += "Operator: This concludes today's question-and-answer session. Thank you.\n"
            out.append(
                Transcript(
                    company=COMPANY_NAMES.get(ticker, ticker),
                    ticker=ticker,
                    date=iso,
                    quarter=qlabel,
                    raw_text=body,
                    source="synthetic",
                )
            )
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def save_all(transcripts: list[Transcript]) -> int:
    n = 0
    for t in transcripts:
        path = TRANSCRIPTS_DIR / t.filename()
        write_json(path, asdict(t))
        n += 1
    LOG.info("wrote %d transcripts to %s", n, TRANSCRIPTS_DIR)
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-only", action="store_true",
                        help="skip live scraping; generate synthetic transcripts only")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--max-per-ticker", type=int, default=SYNTHETIC_PER_TICKER)
    parser.add_argument("--no-edgar", action="store_true")
    args = parser.parse_args()

    tickers = args.tickers or TICKERS
    collected: list[Transcript] = []

    if not args.synthetic_only:
        LOG.info("attempting live scrape from Motley Fool ...")
        try:
            collected.extend(scrape_motley_fool(max_articles=80))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("motley fool scrape failed: %s", exc)
        if not args.no_edgar:
            for tk in tqdm(tickers, desc="EDGAR 8-K search"):
                try:
                    collected.extend(search_edgar_8k(tk))
                except Exception as exc:  # noqa: BLE001
                    LOG.debug("edgar failed for %s: %s", tk, exc)

    if len(collected) < 50:
        LOG.warning(
            "live scrape yielded only %d transcripts (<50). Falling back to synthetic.",
            len(collected),
        )
        collected.extend(
            synthetic_calls(
                tickers=tickers,
                per_ticker=args.max_per_ticker,
                start_year=int(START_DATE[:4]),
                end_year=2024,
            )
        )

    n = save_all(collected)
    LOG.info("DONE — %d transcripts on disk in %s", n, TRANSCRIPTS_DIR)


if __name__ == "__main__":
    main()
