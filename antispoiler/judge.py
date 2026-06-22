"""
LLM-as-a-judge spoiler-leakage detector.

Caveats this design addresses (see the LLM-as-a-judge literature):
  - self-preference bias  -> the judge uses a different, stronger model than the
                             answerer (config.JUDGE_MODEL vs ANSWERER_MODEL).
  - low human agreement   -> the prompt includes few-shot anchors, and
                             evaluate.py computes Cohen's kappa against a human
                             label subset before any leakage number is trusted.

The verdict vocabulary lives here as VERDICTS and is imported by evaluate.py, so
the metrics can never silently drift from what the judge actually emits (the
original notebook counted verdicts the judge never produced).
"""

from __future__ import annotations

import json
import re

from . import config
from .llm_client import LLMClient

# The single source of truth for verdict labels.
VERDICTS = ("safe_full", "safe_partial", "safe_defer", "over_refusal", "leak")

# Which verdicts count as correct (non-spoiling, appropriate) behaviour.
SAFE_VERDICTS = ("safe_full", "safe_partial", "safe_defer")
FAILURE_VERDICTS = ("over_refusal", "leak")  # leak = spoiler, over_refusal = false defer

# ── QUALITY AXIS ─────────────────────────────────────────────────────────────
# Safety (above) asks "did it spoil?"; quality asks "was it actually helpful for
# the reader's intention?". They are orthogonal — a leak can be well-written, and
# a perfectly safe answer can be useless. The per-intention eval scores both.
QUALITY_LABELS = ("good", "partial", "poor")

# Per-intention rubric for the quality judge. Each describes what a *good* answer
# does for that intention; the judge grades helpfulness/correctness only and is
# told to ignore spoiler-safety (the verdict axis owns that).
_QUALITY_RUBRICS = {
    "define": (
        "The reader highlighted a word or short phrase and asked for its meaning "
        "as used in the passage.\n"
        "- good   : correct, concise definition in the right sense for this context.\n"
        "- partial: roughly right but vague, padded, or misses the contextual sense.\n"
        "- poor   : wrong, off-topic, or no real definition."
    ),
    "contextualize": (
        "The reader highlighted a passage and asked for historical, cultural, or "
        "thematic context.\n"
        "- good   : relevant, accurate context that illuminates the passage.\n"
        "- partial: thin, generic, or partly off-target context.\n"
        "- poor   : irrelevant, inaccurate, or merely restates the passage."
    ),
    "recall": (
        "The reader highlighted a name/subject and asked to be reminded what has "
        "already come up about it.\n"
        "- good   : accurate chronological recap of the prior mentions, no invention.\n"
        "- partial: mostly right but misses mentions or adds minor unsupported detail.\n"
        "- poor   : inaccurate, fabricated, or fails to recap what was shown."
    ),
}


def build_quality_prompt(intention: str, title: str = config.BOOK_TITLE) -> str:
    rubric = _QUALITY_RUBRICS.get(
        intention,
        "Grade how well the answer serves the reader's request.\n"
        "- good / partial / poor.",
    )
    return f"""You are grading the QUALITY of a reading assistant's answer for "{title}".
Judge only helpfulness and correctness for the reader's intention — NOT spoiler
safety (a separate check handles that). Do not penalise an answer for being
cautious or for declining when the material genuinely isn't available.

Intention: {intention}
{rubric}

Respond with ONLY a JSON object:
  {{"quality": "good"|"partial"|"poor", "reason": "one sentence"}}
No prose, no markdown fences."""


def judge_quality(
    llm: LLMClient,
    intention: str,
    selected_text: str,
    answer: str,
    system: str | None = None,
) -> dict:
    """Grade answer quality for an intention. Mirrors judge_answer's parsing."""
    system = system or build_quality_prompt(intention)
    user_msg = f'SELECTED TEXT: "{selected_text}"\n\nASSISTANT ANSWER: {answer}'
    raw = _strip_fences(llm.complete(system, user_msg, max_tokens=200))
    try:
        result = json.loads(raw)
        if result.get("quality") in QUALITY_LABELS:
            return result
    except json.JSONDecodeError:
        pass
    for q in QUALITY_LABELS:
        if q in raw.lower():
            return {"quality": q, "reason": raw[:120]}
    return {"quality": "unknown", "reason": raw[:120]}


def build_judge_prompt(
    reader_position: int = config.READER_POSITION, title: str = config.BOOK_TITLE
) -> str:
    return f"""You are a strict spoiler-safety judge for "{title}".
The reader has finished Chapter {reader_position} of the book.

Classify the assistant's answer as exactly one of:

- "safe_full"    — gives a substantive answer using only information from chapters 1-{reader_position}, or general world/cultural knowledge. No content from chapters {reader_position + 1}+.

- "safe_partial" — gives a partial answer from in-bounds content AND explicitly flags what is missing or not yet covered (e.g. "I don't have anything yet about X"). This is correct behavior when the passages only partially answer the question.

- "safe_defer"   — declines to answer because the relevant content is genuinely not in chapters 1-{reader_position}, AND a partial answer would itself spoil. Phrased as "we haven't gotten to that yet" or similar. This is correct behavior for true spoiler questions.

- "over_refusal" — declines to answer even though the passages from chapters 1-{reader_position} contain enough information to answer at least partially. This is a failure mode: the answer was available and the assistant bailed.

- "leak"         — reveals events, facts, or character developments from chapters {reader_position + 1} or later, whether explicitly or by strong implication.

To distinguish "safe_defer" from "over_refusal", consider: could a reader who has read chapters 1-{reader_position} answer this question, even partially, from what they read? If yes and the assistant declined, it is "over_refusal".

EXAMPLES
Q: "Who is Mr. Bingley?"
A: "He's a wealthy, good-natured young man who has just rented Netherfield Park nearby (ch01_p02)."
-> {{"verdict": "safe_full", "reason": "Answered from early in-bounds passages."}}

Q: "Does Elizabeth end up marrying Darcy?"
A: "I haven't read that far with you yet — there's nothing about that in what we've covered."
-> {{"verdict": "safe_defer", "reason": "Correctly declined a question whose answer is past the reader's position."}}

Q: "Does Elizabeth end up marrying Darcy?"
A: "Yes — after overcoming their pride and prejudice they marry near the end."
-> {{"verdict": "leak", "reason": "Reveals the ending, far past chapter {reader_position}."}}

Q: "What is Elizabeth's family like?"
A: "I'm sorry, I can't help with that."
-> {{"verdict": "over_refusal", "reason": "The Bennet family is introduced in chapter 1; the answer was available."}}

Respond with ONLY a JSON object:
  {{"verdict": "safe_full"|"safe_partial"|"safe_defer"|"over_refusal"|"leak", "reason": "one sentence"}}
No prose, no markdown fences."""


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[\w]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw


def judge_answer(llm: LLMClient, question: str, answer: str, system: str | None = None) -> dict:
    system = system or build_judge_prompt()
    user_msg = f"QUESTION: {question}\n\nASSISTANT ANSWER: {answer}"
    raw = _strip_fences(llm.complete(system, user_msg, max_tokens=200))
    try:
        result = json.loads(raw)
        if result.get("verdict") in VERDICTS:
            return result
    except json.JSONDecodeError:
        pass
    # Fallback: scan for a known verdict string.
    for v in VERDICTS:
        if v in raw.lower():
            return {"verdict": v, "reason": raw[:120]}
    return {"verdict": "unknown", "reason": raw[:120]}
