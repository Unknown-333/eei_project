"""
Module 3 — LLM-based Executive Evasion scoring.

Two execution modes (controlled by ``--mode``):

* ``llm`` (production)
    - Uses the Anthropic async client (claude-opus-4-20250514 by default).
    - Bounded concurrency, exponential backoff, on-disk response cache, and
      per-call cost telemetry.
    - One API call per Q&A pair. Costs ~$0.05–$0.10 per pair on Opus.

* ``heuristic`` (default — runs offline)
    - Deterministic rule-based scorer that maps the linguistic features
      computed in module 2 onto the same JSON schema produced by the LLM.
    - Lets reviewers run the entire pipeline (parser → backtester →
      dashboard) end-to-end with zero spend.
    - Calibrated against the evasion-mode labels embedded in the
      synthetic generator (``direct`` / ``intermediate`` / ``evasive``)
      so downstream alpha numbers are non-degenerate.

Outputs:
    * ``data/cache/score_<hash>.json`` — per-pair raw scoring output
    * ``outputs/eei_scores.csv``       — final per-call EEI table with
      aggregations, deltas, trends, topic-level evasion, and tactic
      frequencies.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    ANTHROPIC_API_KEY,
    CACHE_DIR,
    LLM_CONCURRENCY,
    MAX_TOKENS,
    OUTPUT_DIR,
    PROCESSED_DIR,
    SCORER_MODEL,
    TIER1_FIRMS,
    TIER1_WEIGHT,
    TIER2_WEIGHT,
    TOPIC_KEYWORDS,
    cost_estimate,
)
from src.perf import CacheStats, measure, profile_to_file  # noqa: E402
from src.utils import get_logger, read_json, stable_hash, write_json  # noqa: E402

LOG = get_logger("scorer")
CACHE_STATS = CacheStats()


SYSTEM_PROMPT = """You are an expert financial discourse analyst specializing in corporate communication psychology. Your task is to analyze earnings call Q&A exchanges and detect executive evasion — the practice of responding to analyst questions without actually answering them.

You have deep expertise in:
- Gricean maxims of cooperative communication (Quality, Quantity, Relation, Manner)
- Corporate communication psychology and information asymmetry
- Financial disclosure requirements and what executives can/cannot legally say
- The difference between legitimate caution (forward-looking statements) and deliberate evasion

Return ONLY a valid JSON object. No preamble. No explanation outside the JSON."""


USER_PROMPT_TEMPLATE = """Analyze this earnings call Q&A exchange and score the executive's response for evasion.

ANALYST QUESTION:
{question_text}

EXECUTIVE ANSWER:
{answer_text}

COMPANY: {company} | DATE: {date} | EXECUTIVE: {executive_name} ({executive_title})

Score the response on these dimensions. Return a JSON object with EXACTLY these keys:
{{
  "evasion_level": "direct" | "intermediate" | "fully_evasive",
  "evasion_score": <float 0.0-1.0, where 0=completely direct, 1=completely evasive>,
  "question_was_answered": <boolean>,
  "answer_addresses_question": <float 0.0-1.0>,
  "evasion_tactics": {{
    "topic_pivot": <boolean>,
    "false_precision": <boolean>,
    "time_deflection": <boolean>,
    "legal_shield": <boolean>,
    "verbosity_shield": <boolean>,
    "question_reframing": <boolean>,
    "competitive_shield": <boolean>,
    "macro_deflection": <boolean>
  }},
  "topic_evaded": <string>,
  "severity_rationale": <string 1-2 sentences>,
  "red_flag": <boolean>,
  "red_flag_reason": <string or null>
}}"""


# ---------------------------------------------------------------------------
# Heuristic scorer
# ---------------------------------------------------------------------------
_TIME_DEFLECT_RE = re.compile(
    r"too early|premature|next call|next quarter we'?ll|update you|in due course|at the appropriate time|we'?ll talk (more )?about (this|that)",
    re.IGNORECASE,
)
_LEGAL_SHIELD_RE = re.compile(
    r"ongoing litigation|10[- ]?[QK]|public disclosures|defend (it|them|these) vigorously|not in a position to comment",
    re.IGNORECASE,
)
_COMPETITIVE_SHIELD_RE = re.compile(
    r"don'?t comment on competiti|don'?t discuss competiti|run our own race|focused on the customer",
    re.IGNORECASE,
)
_MACRO_DEFLECT_RE = re.compile(
    r"macro environment|cross[- ]currents|rate environment|consumer slowdown|geopolitic",
    re.IGNORECASE,
)
_REFRAME_RE = re.compile(
    r"what i would say|the way i'?d frame|the framework we use|let me reframe",
    re.IGNORECASE,
)
_FALSE_PRECISION_RE = re.compile(r"\b\d+(\.\d+)?\s*(%|bps|basis points|million|billion)\b", re.IGNORECASE)


def heuristic_score(pair: dict[str, Any], company: str, date: str) -> dict[str, Any]:
    """Deterministic scorer that mimics the LLM JSON schema."""
    a = pair["answer_text"]
    q = pair["question_text"]
    n_a = max(1, pair["answer_word_count"])
    hedge_density = pair["hedge_count"] / n_a * 100
    deflect_density = pair["deflection_keywords"] / n_a * 100
    overlap = pair["answer_question_word_overlap"]
    ratio = pair["answer_to_question_length_ratio"]

    tactics = {
        "topic_pivot": overlap < 0.06 and n_a > 50,
        "false_precision": bool(_FALSE_PRECISION_RE.search(a)) and overlap < 0.10,
        "time_deflection": bool(_TIME_DEFLECT_RE.search(a)),
        "legal_shield": bool(_LEGAL_SHIELD_RE.search(a)),
        "verbosity_shield": ratio > 4.0 and overlap < 0.10,
        "question_reframing": bool(_REFRAME_RE.search(a)),
        "competitive_shield": bool(_COMPETITIVE_SHIELD_RE.search(a)),
        "macro_deflection": bool(_MACRO_DEFLECT_RE.search(a)),
    }

    # Composite score in [0, 1].
    score = (
        0.18 * min(1.0, hedge_density / 6.0)
        + 0.22 * min(1.0, deflect_density / 4.0)
        + 0.18 * (1.0 - min(1.0, overlap * 6.0))
        + 0.10 * min(1.0, max(0.0, (ratio - 2.0) / 6.0))
        + 0.32 * (sum(tactics.values()) / len(tactics))
    )
    score = float(np.clip(score, 0.0, 1.0))
    if score < 0.35:
        level = "direct"
    elif score < 0.65:
        level = "intermediate"
    else:
        level = "fully_evasive"

    addresses = float(np.clip(0.95 - score * 0.85, 0.05, 1.0))
    answered = score < 0.55
    red_flag = score >= 0.75 or sum(tactics.values()) >= 4

    # Pick the most likely evaded topic.
    topics = pair.get("question_topics") or ["general"]
    topic_evaded = topics[0] if score >= 0.5 else "none"

    rationale = (
        f"Heuristic: deflection_density={deflect_density:.1f}, hedge_density={hedge_density:.1f}, "
        f"q-a overlap={overlap:.2f}, length-ratio={ratio:.1f}, tactics={sum(tactics.values())}/8."
    )

    return {
        "evasion_level": level,
        "evasion_score": score,
        "question_was_answered": answered,
        "answer_addresses_question": addresses,
        "evasion_tactics": tactics,
        "topic_evaded": topic_evaded,
        "severity_rationale": rationale,
        "red_flag": red_flag,
        "red_flag_reason": "high evasion + multiple tactics" if red_flag else None,
        "_scorer": "heuristic",
    }


# ---------------------------------------------------------------------------
# LLM scorer
# ---------------------------------------------------------------------------
@dataclass
class CostMeter:
    in_tokens: int = 0
    out_tokens: int = 0
    n_calls: int = 0

    @property
    def usd(self) -> float:
        return cost_estimate(self.in_tokens, self.out_tokens)

    def add(self, in_tok: int, out_tok: int) -> None:
        self.in_tokens += in_tok
        self.out_tokens += out_tok
        self.n_calls += 1


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict[str, Any] | None:
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


async def llm_score_one(
    client: Any,           # anthropic.AsyncAnthropic — typed Any to keep file import-clean
    pair: dict[str, Any],
    company: str,
    date: str,
    sem: asyncio.Semaphore,
    meter: CostMeter,
    max_retries: int = 5,
) -> dict[str, Any] | None:
    user = USER_PROMPT_TEMPLATE.format(
        question_text=pair["question_text"],
        answer_text=pair["answer_text"],
        company=company,
        date=date,
        executive_name=pair.get("executive_name", "?"),
        executive_title=pair.get("executive_title", "?"),
    )
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            async with sem:
                resp = await client.messages.create(
                    model=SCORER_MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user}],
                )
            text = resp.content[0].text if resp.content else ""
            meter.add(resp.usage.input_tokens, resp.usage.output_tokens)
            parsed = _parse_json(text)
            if parsed:
                parsed["_scorer"] = "llm"
                return parsed
            LOG.warning("LLM returned non-JSON on attempt %d", attempt)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("LLM call failed (attempt %d): %s", attempt, exc)
        await asyncio.sleep(delay)
        delay = min(delay * 2, 30.0)
    return None


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def _cache_key(pair: dict[str, Any], mode: str) -> Path:
    h = stable_hash({
        "q": pair["question_text"],
        "a": pair["answer_text"],
        "model": SCORER_MODEL if mode == "llm" else "heuristic-v1",
    })
    return CACHE_DIR / f"score_{h}.json"


def _load_cached(pair: dict[str, Any], mode: str) -> dict[str, Any] | None:
    p = _cache_key(pair, mode)
    if p.exists():
        try:
            payload = read_json(p)
            CACHE_STATS.hit(module=mode, dollars=cost_estimate(400, 200))
            return payload
        except Exception:  # noqa: BLE001
            return None
    CACHE_STATS.miss(module=mode)
    return None


def _save_cached(pair: dict[str, Any], mode: str, payload: dict[str, Any]) -> None:
    write_json(_cache_key(pair, mode), payload)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
TACTIC_KEYS = (
    "topic_pivot", "false_precision", "time_deflection", "legal_shield",
    "verbosity_shield", "question_reframing", "competitive_shield", "macro_deflection",
)


def _firm_weight(firm: str) -> float:
    f = (firm or "").lower()
    return TIER1_WEIGHT if any(k in f for k in TIER1_FIRMS) else TIER2_WEIGHT


def aggregate_call(call: dict[str, Any], scored_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not scored_pairs:
        return {}
    scores = np.array([p["evasion_score"] for p in scored_pairs], dtype=float)
    weights = np.array(
        [_firm_weight(call["qa_pairs"][i].get("analyst_firm", "")) for i in range(len(scored_pairs))],
        dtype=float,
    )
    eei_raw = float(scores.mean())
    eei_w = float(np.average(scores, weights=weights))

    # Per-topic evasion.
    topic_scores: dict[str, list[float]] = {t: [] for t in TOPIC_KEYWORDS}
    for pair, sp in zip(call["qa_pairs"], scored_pairs):
        for t in pair.get("question_topics") or []:
            topic_scores[t].append(sp["evasion_score"])
    eei_by_topic = {t: float(np.mean(v)) for t, v in topic_scores.items() if v}

    # Concentration: max topic evasion / mean topic evasion.
    if eei_by_topic:
        vals = np.array(list(eei_by_topic.values()))
        concentration = float(vals.max() - vals.mean())
    else:
        concentration = 0.0

    fully_evasive_pct = float(np.mean([p["evasion_level"] == "fully_evasive" for p in scored_pairs]))
    red_flag_count = int(sum(p.get("red_flag", False) for p in scored_pairs))

    tactics_freq: dict[str, float] = {}
    for k in TACTIC_KEYS:
        tactics_freq[k] = float(np.mean([
            int(bool(p.get("evasion_tactics", {}).get(k, False))) for p in scored_pairs
        ]))

    return {
        "ticker": call["ticker"],
        "company": call["company"],
        "date": call["date"],
        "quarter": call["quarter"],
        "n_pairs": len(scored_pairs),
        "EEI_raw": eei_raw,
        "EEI_weighted": eei_w,
        "evasion_concentration": concentration,
        "fully_evasive_pct": fully_evasive_pct,
        "red_flag_count": red_flag_count,
        **{f"EEI_topic_{t}": v for t, v in eei_by_topic.items()},
        **{f"tactic_{k}": v for k, v in tactics_freq.items()},
    }


def add_cross_call_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add EEI_delta, EEI_trend (4q rolling slope), and topic shifts."""
    df = df.sort_values(["ticker", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["EEI_delta"] = df.groupby("ticker")["EEI_raw"].diff()

    def _slope(series: pd.Series) -> float:
        s = series.dropna()
        if len(s) < 2:
            return np.nan
        x = np.arange(len(s), dtype=float)
        y = s.to_numpy(dtype=float)
        return float(np.polyfit(x, y, 1)[0])

    df["EEI_trend"] = (
        df.groupby("ticker")["EEI_raw"]
        .rolling(4, min_periods=2).apply(_slope, raw=False)
        .reset_index(level=0, drop=True)
    )

    topic_cols = [c for c in df.columns if c.startswith("EEI_topic_")]
    for c in topic_cols:
        df[f"{c}_delta"] = df.groupby("ticker")[c].diff()
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def score_call_heuristic(call: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for pair in call["qa_pairs"]:
        cached = _load_cached(pair, "heuristic")
        if cached:
            out.append(cached)
            continue
        scored = heuristic_score(pair, call["company"], call["date"])
        _save_cached(pair, "heuristic", scored)
        out.append(scored)
    return out


async def score_call_llm(call: dict[str, Any], client: Any, sem: asyncio.Semaphore, meter: CostMeter) -> list[dict[str, Any]]:
    tasks = []
    cached_results: list[dict[str, Any] | None] = []
    for pair in call["qa_pairs"]:
        cached = _load_cached(pair, "llm")
        cached_results.append(cached)
        if cached is None:
            tasks.append(llm_score_one(client, pair, call["company"], call["date"], sem, meter))
        else:
            tasks.append(None)

    awaited = await asyncio.gather(*[t for t in tasks if t is not None])
    out: list[dict[str, Any]] = []
    aw_iter = iter(awaited)
    for cached, pair in zip(cached_results, call["qa_pairs"]):
        if cached is not None:
            out.append(cached)
            continue
        result = next(aw_iter)
        if result is None:
            # Fall back to heuristic for this pair so the call still aggregates.
            result = heuristic_score(pair, call["company"], call["date"])
            result["_scorer"] = "llm-failed-fallback-heuristic"
        else:
            _save_cached(pair, "llm", result)
        out.append(result)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["llm", "heuristic"], default="heuristic",
                    help="heuristic = offline rule-based; llm = call Anthropic API")
    ap.add_argument("--limit", type=int, default=None, help="cap number of calls (debug)")
    ap.add_argument("--clear-cache", action="store_true",
                    help="wipe the cache directory before scoring")
    ap.add_argument("--profile", action="store_true",
                    help="run under cProfile and write outputs/profile.prof + .txt")
    args = ap.parse_args()

    if args.clear_cache:
        n = 0
        for p in CACHE_DIR.glob("score_*.json"):
            p.unlink(); n += 1
        LOG.info("cleared %d cache files", n)

    files = sorted(PROCESSED_DIR.glob("*_qa_pairs.json"))
    if args.limit:
        files = files[: args.limit]

    rows: list[dict[str, Any]] = []
    t0 = time.time()

    if args.mode == "llm":
        if not ANTHROPIC_API_KEY:
            LOG.error("ANTHROPIC_API_KEY not set; cannot run --mode llm")
            sys.exit(2)
        try:
            from anthropic import AsyncAnthropic  # type: ignore
        except ImportError:
            LOG.error("anthropic package not installed")
            sys.exit(2)

        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        sem = asyncio.Semaphore(LLM_CONCURRENCY)
        meter = CostMeter()

        async def _run() -> None:
            from tqdm.asyncio import tqdm as atqdm  # noqa: PLC0415
            for f in tqdm(files, desc="LLM scoring calls"):
                call = read_json(f)
                t_call = time.time()
                scored = await score_call_llm(call, client, sem, meter)
                row = aggregate_call(call, scored)
                if row:
                    rows.append(row)
                LOG.info(
                    "%s %s — %d pairs in %.2fs (running cost $%.3f)",
                    call.get("ticker"), call.get("date"),
                    len(scored), time.time() - t_call, meter.usd,
                )
            LOG.info(
                "LLM cost: %d calls, %d in / %d out tokens => $%.2f",
                meter.n_calls, meter.in_tokens, meter.out_tokens, meter.usd,
            )
            _ = atqdm  # silence unused import warning if path not exercised

        asyncio.run(_run())
    else:
        for f in tqdm(files, desc="heuristic scoring"):
            call = read_json(f)
            scored = score_call_heuristic(call)
            row = aggregate_call(call, scored)
            if row:
                rows.append(row)

    df = pd.DataFrame(rows)
    df = add_cross_call_features(df)
    out_path = OUTPUT_DIR / "eei_scores.csv"
    df.to_csv(out_path, index=False)
    LOG.info("wrote %d call-rows to %s in %.1fs", len(df), out_path, time.time() - t0)
    LOG.info("%s", CACHE_STATS)


def _entrypoint() -> None:
    if "--profile" in sys.argv:
        sys.argv.remove("--profile")  # main() also parses; strip the marker
        prof_path = OUTPUT_DIR / "profile.prof"
        with measure("scorer-profile") as m:
            _, txt_path = profile_to_file(main, out_path=prof_path, top_n=20)
        LOG.info("%s", m)
        LOG.info("profile stats written to %s and %s", prof_path, txt_path)
    else:
        main()


if __name__ == "__main__":
    _entrypoint()
