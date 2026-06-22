"""
Generate the evaluation question set with an LLM.

Questions are organised into three spoiler-risk tiers:

  | Tier     | Answer available     | Expected bounded behaviour          |
  |----------|----------------------|-------------------------------------|
  | safe     | Chapters 1..N        | answer correctly                    |
  | boundary | Chapters N-2 .. N+2  | edge cases; useful for debugging    |
  | spoiler  | Chapters N+1 .. end  | defer (unbounded baseline: leaks)   |

There is no off-the-shelf benchmark for *position-bounded* spoiler-free QA:
existing spoiler datasets (Goodreads, TV Tropes) label spoiler-ness globally,
not relative to a reader's position. So we synthesise a tiered set and validate
the judge against human labels (see evaluate.py).

The LLM proposes questions; a human reviews/edits them before the eval runs.
"""

from __future__ import annotations

import json
import re

from . import config
from .llm_client import LLMClient

SYSTEM_QGEN = (
    "You are creating evaluation questions for an anti-spoiler reading assistant test.\n"
    "Output ONLY valid JSON; no prose, no markdown fences, no explanation."
)


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[\w]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw


def generate_questions(
    llm: LLMClient,
    tier: str,
    chapter_range: str,
    n: int,
    reader_position: int = config.READER_POSITION,
    title: str = config.BOOK_TITLE,
    author: str = config.BOOK_AUTHOR,
) -> list[dict]:
    prompt = (
        f'The book is "{title}" by {author}.\n'
        f"Generate {n} questions a reader might ask about this book.\n\n"
        f"Tier: {tier}\n"
        f"The CORRECT ANSWER to each question first appears in {chapter_range}.\n\n"
        "Rules:\n"
        "- Each question must be answerable from the book text (not author biography)\n"
        "- Questions should feel natural, like something a reader would genuinely ask\n"
        f"- For 'spoiler' tier: answer must reveal something that happens AFTER chapter {reader_position}\n"
        f"- For 'safe' tier: answer must be fully covered in chapters 1-{reader_position}\n"
        "- For 'boundary' tier: answer is at the edge of the reader's current position\n\n"
        "Return a JSON array, each object with:\n"
        '  "question": string,\n'
        f'  "tier": "{tier}",\n'
        '  "answer_appears_in": "Chapter N" (your best estimate),\n'
        '  "note": one sentence explaining why this belongs in this tier\n\n'
        "Example (2 items):\n"
        '[{"question": "...", "tier": "' + tier + '", "answer_appears_in": "Chapter 3", "note": "..."}]'
    )
    raw = _strip_fences(llm.complete(SYSTEM_QGEN, prompt))
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error (tier={tier}): {e}\n  Raw: {raw[:300]}")
        return []


def generate_all(
    llm: LLMClient,
    n_per_tier: int = config.N_QUESTIONS_PER_TIER,
    reader_position: int = config.READER_POSITION,
) -> list[dict]:
    safe = generate_questions(llm, "safe", f"chapters 1-{reader_position}", n_per_tier, reader_position)
    boundary = generate_questions(
        llm, "boundary", f"chapters {reader_position-2}-{reader_position+2}", n_per_tier, reader_position
    )
    spoiler = generate_questions(
        llm, "spoiler", f"chapters {reader_position+1}-end", n_per_tier, reader_position
    )
    return safe + boundary + spoiler


# ── PER-INTENTION ITEMS ──────────────────────────────────────────────────────
# The QA set above feeds the generic-companion eval. The *product* is selection +
# intention, so the per-intention eval needs items shaped like the real input: a
# selected span plus an intention. Three intentions are evaluated (paraphrase is
# out of scope — owned elsewhere with its own metrics):
#
#   define        — items are just selection STRINGS (a word/phrase the reader
#                   highlights). No spoiler tier: define's failure mode is
#                   correctness, not position, so it is scored on quality only.
#   contextualize — selected passages, tiered by spoiler risk (does explaining
#                   them tempt the model into later-chapter material?).
#   recall        — selected names/subjects, tiered by whether their prior
#                   mentions sit in-bounds (safe) or mostly later (spoiler).

SYSTEM_IGEN = (
    "You are creating evaluation items for an anti-spoiler reading assistant test.\n"
    "Output ONLY valid JSON; no prose, no markdown fences, no explanation."
)


def _parse_array(raw: str, label: str) -> list[dict]:
    raw = _strip_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error ({label}): {e}\n  Raw: {raw[:300]}")
        return []


def generate_define_items(
    llm: LLMClient,
    n: int,
    reader_position: int = config.READER_POSITION,
    title: str = config.BOOK_TITLE,
    author: str = config.BOOK_AUTHOR,
) -> list[dict]:
    """Words/short phrases a B1-B2 reader might highlight to define (quality-only)."""
    prompt = (
        f'The book is "{title}" by {author}.\n'
        f"A reader at a B1-B2 English level has read chapters 1-{reader_position}.\n"
        f"List {n} words or short phrases from WITHIN chapters 1-{reader_position} "
        "that such a reader might highlight and ask to have defined — archaic, "
        "formal, or otherwise tricky vocabulary as actually used in the text.\n\n"
        "Return a JSON array, each object with:\n"
        '  "intention": "define",\n'
        '  "selected_text": the word or short phrase (verbatim, as a reader would select it),\n'
        '  "note": one sentence on why it is tricky.\n\n'
        'Example: [{"intention": "define", "selected_text": "entailed", "note": "legal term for restricted inheritance."}]'
    )
    return _parse_array(llm.complete(SYSTEM_IGEN, prompt), "define")


def generate_span_items(
    llm: LLMClient,
    intention: str,
    tier: str,
    chapter_range: str,
    n: int,
    reader_position: int = config.READER_POSITION,
    title: str = config.BOOK_TITLE,
    author: str = config.BOOK_AUTHOR,
) -> list[dict]:
    """Tiered selection items for contextualize / recall.

    The selection itself is always something the reader has read (in chapters
    1-{reader_position}); the TIER describes whether responding well tempts the
    model toward later-chapter material — the spoiler pressure the bound resists.
    """
    if intention == "contextualize":
        what = "a short passage (1-2 sentences, quoted verbatim from the text)"
        tier_hint = (
            f"- safe    : a passage whose context is fully self-contained in chapters 1-{reader_position}.\n"
            f"- boundary: a passage whose themes start developing right around chapter {reader_position}.\n"
            f"- spoiler : a passage whose real significance is only revealed after chapter {reader_position} "
            "(contextualising it well would tempt a leak)."
        )
    elif intention == "recall":
        what = "a character name or subject the reader has encountered"
        tier_hint = (
            f"- safe    : a subject with several mentions within chapters 1-{reader_position}.\n"
            f"- boundary: a subject just introduced around chapter {reader_position}.\n"
            f"- spoiler : a subject whose importance lies mostly after chapter {reader_position} "
            "(a faithful recall must stay with the few early mentions, if any)."
        )
    else:
        raise ValueError(f"generate_span_items expects contextualize/recall, got {intention!r}")

    prompt = (
        f'The book is "{title}" by {author}.\n'
        f"The reader has read chapters 1-{reader_position}.\n"
        f"Generate {n} items for the '{intention}' intention, tier '{tier}'.\n"
        f"Each selection is {what}, taken from chapters 1-{reader_position} "
        f"(the reader can only select what they have read).\n"
        f"The correct supporting material first appears in {chapter_range}.\n\n"
        f"Tier meanings:\n{tier_hint}\n\n"
        "Return a JSON array, each object with:\n"
        f'  "intention": "{intention}",\n'
        '  "selected_text": the selection (verbatim),\n'
        f'  "tier": "{tier}",\n'
        '  "note": one sentence on why it belongs in this tier.'
    )
    return _parse_array(llm.complete(SYSTEM_IGEN, prompt), f"{intention}/{tier}")


def generate_all_intentions(
    llm: LLMClient,
    n_define: int = config.N_QUESTIONS_PER_TIER,
    n_per_tier: int = config.N_QUESTIONS_PER_TIER,
    reader_position: int = config.READER_POSITION,
) -> list[dict]:
    """All per-intention eval items: define (flat) + contextualize/recall (tiered)."""
    items: list[dict] = generate_define_items(llm, n_define, reader_position)
    tiers = {
        "safe": f"chapters 1-{reader_position}",
        "boundary": f"chapters {reader_position-2}-{reader_position+2}",
        "spoiler": f"chapters {reader_position+1}-end",
    }
    for intention in ("contextualize", "recall"):
        for tier, rng in tiers.items():
            items += generate_span_items(llm, intention, tier, rng, n_per_tier, reader_position)
    return items
