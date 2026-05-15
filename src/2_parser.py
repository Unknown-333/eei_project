"""
Module 2 — Q&A pair parser + cheap linguistic features.

Reads raw transcripts from ``data/transcripts/`` and emits one structured
``*_qa_pairs.json`` per call into ``data/processed/``.

The parser handles the dominant transcript schemas:

    Motley Fool / synthetic ::
        John Smith -- Goldman Sachs -- Analyst
        <question>

        Tim Cook -- Chief Executive Officer
        <answer>

    Seeking Alpha / older ::
        John Smith
        Goldman Sachs

        Q: <question>

        Tim Cook - Chief Executive Officer
        A: <answer>

It also extracts deterministic linguistic features that are cheap to compute
and serve as a sanity check on the much more expensive LLM scores from
module 3.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    DEFLECTION_PHRASES,
    HEDGE_WORDS,
    PROCESSED_DIR,
    TOPIC_KEYWORDS,
    TRANSCRIPTS_DIR,
)
from src.utils import get_logger, read_json, write_json  # noqa: E402

LOG = get_logger("parser")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class QAPair:
    analyst_name: str
    analyst_firm: str
    question_text: str
    question_topics: list[str]
    answer_text: str
    answer_word_count: int
    executive_name: str
    executive_title: str
    # Deterministic linguistic features
    hedge_count: int = 0
    deflection_keywords: int = 0
    question_marks_in_answer: int = 0
    answer_question_word_overlap: float = 0.0
    answer_to_question_length_ratio: float = 0.0


@dataclass
class ParsedCall:
    company: str
    ticker: str
    date: str
    quarter: str
    source: str
    qa_pairs: list[QAPair] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Q&A section detection
# ---------------------------------------------------------------------------
_QA_OPENERS = [
    r"now we'?ll open up the call for questions",
    r"open(ing)? (the|up the) (call|line|floor) (for|to) questions",
    r"begin the question[- ]and[- ]answer session",
    r"first question (comes from|will come from|is from)",
    r"questions?[- ]and[- ]answers?\b",
]

_QA_OPENER_RE = re.compile("|".join(_QA_OPENERS), re.IGNORECASE)


def split_qa_section(raw_text: str) -> str:
    """Return the substring of ``raw_text`` that follows the Q&A opener."""
    m = _QA_OPENER_RE.search(raw_text)
    if not m:
        return raw_text  # fall back: parse whole transcript
    return raw_text[m.end():]


# ---------------------------------------------------------------------------
# Speaker turn parsing
# ---------------------------------------------------------------------------
# Matches headers like:
#   "John Smith -- Goldman Sachs -- Analyst"
#   "Tim Cook -- Chief Executive Officer"
#   "Tim Cook - Chief Executive Officer"
#   "Operator"
_HEADER_RE = re.compile(
    r"^\s*(?P<name>[A-Z][\w.''\-]+(?:\s+[A-Z][\w.''\-]+){0,3})"
    r"(?:\s+[-–—]{1,2}\s+(?P<role1>[^-\n]+?))?"
    r"(?:\s+[-–—]{1,2}\s+(?P<role2>[^-\n]+?))?\s*$",
    re.MULTILINE,
)


@dataclass
class _Turn:
    speaker: str
    affiliation: str
    title: str
    role: str        # "analyst" | "executive" | "operator" | "unknown"
    text: str


def _classify(role1: str, role2: str) -> tuple[str, str, str]:
    """Return (affiliation, title, role) from two header sub-fields."""
    role1 = (role1 or "").strip()
    role2 = (role2 or "").strip()
    title = role2 or role1
    affil = role1 if role2 else ""
    label = (role2 or role1).lower()
    if "analyst" in label:
        return affil, title, "analyst"
    if any(k in label for k in ("officer", "ceo", "cfo", "president", "chair", "director", "vp", "head of", "founder")):
        return affil, title, "executive"
    if "operator" in (role1.lower(), role2.lower()) or affil.lower() == "operator":
        return "", "Operator", "operator"
    return affil, title, "unknown"


def _split_turns(qa_text: str) -> list[_Turn]:
    """Walk the Q&A text and chunk it into speaker turns."""
    headers = list(_HEADER_RE.finditer(qa_text))
    turns: list[_Turn] = []
    for i, m in enumerate(headers):
        name = m.group("name").strip()
        role1, role2 = m.group("role1") or "", m.group("role2") or ""
        # Filter common false positives from header detection.
        if name.lower() in {"q", "a", "thank you", "thanks", "operator"} and not role1:
            if name.lower() == "operator":
                affil, title, role = "", "Operator", "operator"
            else:
                continue
        else:
            affil, title, role = _classify(role1, role2)
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(qa_text)
        body = qa_text[start:end].strip()
        if not body:
            continue
        turns.append(_Turn(speaker=name, affiliation=affil, title=title, role=role, text=body))
    return turns


def _pair_turns(turns: list[_Turn]) -> list[QAPair]:
    """Pair consecutive analyst → executive turns into Q&A objects."""
    pairs: list[QAPair] = []
    i = 0
    while i < len(turns):
        t = turns[i]
        if t.role == "analyst":
            # Collect contiguous analyst utterances.
            q_text = t.text
            analyst, firm = t.speaker, t.affiliation
            j = i + 1
            while j < len(turns) and turns[j].role == "analyst":
                q_text += "\n" + turns[j].text
                j += 1
            # Collect contiguous executive utterances as the answer.
            if j < len(turns) and turns[j].role == "executive":
                exec_turn = turns[j]
                a_text = exec_turn.text
                k = j + 1
                while k < len(turns) and turns[k].role == "executive":
                    a_text += "\n" + turns[k].text
                    k += 1
                pairs.append(_make_pair(
                    analyst=analyst, firm=firm,
                    question=q_text, answer=a_text,
                    exec_name=exec_turn.speaker, exec_title=exec_turn.title,
                ))
                i = k
                continue
        i += 1
    return pairs


def _make_pair(
    analyst: str, firm: str, question: str, answer: str,
    exec_name: str, exec_title: str,
) -> QAPair:
    return QAPair(
        analyst_name=analyst,
        analyst_firm=firm,
        question_text=question.strip(),
        question_topics=extract_topics(question),
        answer_text=answer.strip(),
        answer_word_count=len(answer.split()),
        executive_name=exec_name,
        executive_title=exec_title,
        hedge_count=count_phrases(answer, HEDGE_WORDS),
        deflection_keywords=count_phrases(answer, DEFLECTION_PHRASES),
        question_marks_in_answer=answer.count("?"),
        answer_question_word_overlap=jaccard_overlap(question, answer),
        answer_to_question_length_ratio=(
            len(answer.split()) / max(1, len(question.split()))
        ),
    )


# ---------------------------------------------------------------------------
# Linguistic features
# ---------------------------------------------------------------------------
def count_phrases(text: str, phrases: list[str]) -> int:
    """Count case-insensitive occurrences of any phrase in ``phrases``."""
    low = text.lower()
    return sum(low.count(p.lower()) for p in phrases)


def extract_topics(text: str) -> list[str]:
    """Return the topic-tags whose keywords appear in ``text``."""
    low = text.lower()
    return [topic for topic, kws in TOPIC_KEYWORDS.items() if any(k in low for k in kws)]


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "in", "on",
    "for", "to", "is", "are", "was", "were", "be", "been", "being", "you",
    "your", "we", "our", "i", "me", "my", "this", "that", "these", "those",
    "it", "its", "as", "at", "by", "with", "about", "from", "have", "has",
    "had", "do", "does", "did", "can", "could", "would", "should", "may",
    "might", "will", "what", "how", "why", "when", "where", "so", "just",
    "any", "some", "all", "more", "less", "very", "really",
}


def _content_words(text: str) -> set[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", text.lower())
    return {t for t in toks if t not in _STOPWORDS}


def jaccard_overlap(a: str, b: str) -> float:
    sa, sb = _content_words(a), _content_words(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def parse_transcript_file(path: Path) -> ParsedCall:
    payload = read_json(path)
    raw = payload["raw_text"]
    qa_text = split_qa_section(raw)
    turns = _split_turns(qa_text)
    pairs = _pair_turns(turns)
    return ParsedCall(
        company=payload["company"],
        ticker=payload["ticker"],
        date=payload["date"],
        quarter=payload["quarter"],
        source=payload.get("source", "unknown"),
        qa_pairs=pairs,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    files = sorted(TRANSCRIPTS_DIR.glob("*.json"))
    if args.limit:
        files = files[: args.limit]

    n_calls, n_pairs = 0, 0
    for f in tqdm(files, desc="parsing transcripts"):
        try:
            call = parse_transcript_file(f)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("parse failed %s: %s", f.name, exc)
            continue
        if not call.qa_pairs:
            LOG.debug("no Q&A pairs extracted from %s", f.name)
            continue
        out_name = f"{call.ticker}_{call.date}_qa_pairs.json"
        write_json(
            PROCESSED_DIR / out_name,
            {
                "company": call.company,
                "ticker": call.ticker,
                "date": call.date,
                "quarter": call.quarter,
                "source": call.source,
                "qa_pairs": [asdict(p) for p in call.qa_pairs],
            },
        )
        n_calls += 1
        n_pairs += len(call.qa_pairs)

    LOG.info("parsed %d calls -> %d Q&A pairs", n_calls, n_pairs)


if __name__ == "__main__":
    main()
