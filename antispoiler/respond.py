"""
Intention dispatcher for the interactive demo.

The eval harness (evaluate.py) drives a single generic QA prompt over a question
set. The *product*, by contrast, takes a (selected_text, intention,
reader_position) triple — the reader highlights a span and picks what they want
done with it. This module is the one function that maps that triple onto the
existing retrieval + LLM layer:

    respond(llm, index, selected_text, intention, reader_position) -> str

`respond_detailed(...)` is the same dispatch but returns
`{answer, chapters, entity}` — the chapters touched (for the per-intention eval's
spoiler check) and the recalled entity. It also accepts `reader_position=None`,
which disables the position bound (the unbounded baseline the eval contrasts
against).

The four intentions differ only in (a) whether they retrieve and how, and (b)
the prompt framing. The anti-spoiler guarantee is identical across all of them:
every retrieval is bounded by `reader_position` (the same `_in_bounds` filter
the eval validates), and the prompts forbid drawing on out-of-bounds knowledge.

  | Intention    | Retrieval                          | Spoiler risk |
  |--------------|------------------------------------|--------------|
  | paraphrase   | none — operates on the span itself | low          |
  | define       | small bounded embedding context    | low          |
  | contextualize| bounded embedding retrieval        | medium       |
  | recall       | bounded exhaustive mention gather  | high         |
"""

from __future__ import annotations

from . import config
from .index import EmbeddingIndex
from .llm_client import LLMClient
from .retrieval import (
    extract_entity,
    format_context,
    recall_retrieve,
    retrieve_embedding,
)

INTENTIONS = ("define", "paraphrase", "contextualize", "recall")

# Shared framing, mirroring qa.build_system_prompt: tell the model what it IS
# (a bounded companion) rather than handing it prohibitions (Zhou et al. 2023).
_PREAMBLE = (
    'You are a reading companion for "{title}" by {author}. You are reading '
    "alongside the reader and only know what they have read so far — the "
    "passages provided are their knowledge up to this point. Never use outside "
    "knowledge of this book's plot, and never reference or hint at events past "
    "the provided passages. General world knowledge (vocabulary, history, "
    "customs, literary conventions) is fine; book-specific content is not."
)


def _preamble() -> str:
    return _PREAMBLE.format(title=config.BOOK_TITLE, author=config.BOOK_AUTHOR)


def _define(llm: LLMClient, index: EmbeddingIndex, span: str, pos: int | None):
    ctx = retrieve_embedding(index, span, pos, top_k=4)
    system = (
        _preamble() + "\n\n"
        "The reader selected a word or short phrase and wants its meaning AS USED "
        "in what they are reading. Give a brief definition in plain language. For "
        "ordinary vocabulary, answer from general knowledge. If the selection's "
        "sense depends on this book specifically, ground it in the passages below "
        "and nothing later. Two or three sentences at most."
    )
    user = (
        f'SELECTED WORD/PHRASE: "{span}"\n\n'
        f"SURROUNDING CONTEXT (the reader's passages so far):\n{format_context(ctx)}"
        if ctx
        else f'SELECTED WORD/PHRASE: "{span}"\n\n(No surrounding passages retrieved.)'
    )
    return llm.complete(system, user), [c.chapter_index for c in ctx], None


def _paraphrase(llm: LLMClient, span: str):
    system = (
        _preamble() + "\n\n"
        "The reader selected a passage and wants it restated in simpler, clearer "
        "English. Paraphrase ONLY the selected passage — keep the same meaning and "
        "tense, do not add information, do not explain what happens next, do not "
        "foreshadow. Just say the same thing in plainer words."
    )
    user = f'SELECTED PASSAGE:\n"""{span}"""\n\nParaphrase it in plain English.'
    return llm.complete(system, user), [], None


def _contextualize(llm: LLMClient, index: EmbeddingIndex, span: str, pos: int | None):
    ctx = retrieve_embedding(index, span, pos, top_k=config.TOP_K)
    system = (
        _preamble() + "\n\n"
        "The reader selected a passage and wants historical, cultural, or thematic "
        "context for it. Lead with general world knowledge (the period, customs, "
        "references). For anything specific to this book's story or characters, use "
        "ONLY the passages below — do not reference how a theme or situation "
        "develops later. Keep it short and concrete."
    )
    user = (
        f'SELECTED PASSAGE:\n"""{span}"""\n\n'
        f"PASSAGES THE READER HAS READ:\n{format_context(ctx)}"
        if ctx
        else f'SELECTED PASSAGE:\n"""{span}"""\n\n(No in-bounds passages retrieved.)'
    )
    return llm.complete(system, user), [c.chapter_index for c in ctx], None


def _recall(llm: LLMClient, index: EmbeddingIndex, span: str, pos: int | None):
    # The reader highlighted a name/subject; recover its earlier mentions.
    # A short selection IS the entity — use it verbatim. extract_entity is for
    # pulling a name out of a longer phrase, and (being question-oriented) it can
    # return a whole chatty sentence, which then matches nothing. So only invoke
    # it for multi-word spans, and only trust its result if that string actually
    # occurs in the text; otherwise fall back to the raw selection.
    entity = span.strip()
    if len(entity.split()) > 4:
        extracted = extract_entity(llm, span)
        if extracted and any(extracted.lower() in c.text.lower() for c in index.chunks):
            entity = extracted
    hits = recall_retrieve(index.chunks, entity, reader_pos=pos)
    if not hits:
        upto = "the whole book" if pos is None else f"{config.BOOK_TITLE} chapter {pos}"
        msg = f'Nothing about "{entity}" has come up yet in what you\'ve read (up to {upto}).'
        return msg, [], entity
    system = (
        _preamble() + "\n\n"
        'The reader wants to remember what they have already encountered about a '
        'subject earlier in the book ("who was this again?"). Using ONLY the '
        "passages below — every earlier mention within what they have read — "
        "summarise what has been shown about it, in chronological order (earliest "
        "to latest). Cite chunk_ids. Do not add anything beyond these passages and "
        "do not speculate about what comes next."
    )
    user = (
        f'SUBJECT TO RECALL: "{entity}"\n\n'
        f"EARLIER MENTIONS (chronological, within what the reader has read):\n"
        f"{format_context(hits)}\n\n"
        f'Summarise what has been shown about "{entity}" so far.'
    )
    return llm.complete(system, user), [c.chapter_index for c in hits], entity


def _dispatch(
    llm: LLMClient,
    index: EmbeddingIndex,
    span: str,
    intention: str,
    pos: int | None,
):
    """Return (answer, chapters, entity) for a non-empty span. Raises on unknown."""
    if intention == "paraphrase":
        return _paraphrase(llm, span)
    if intention == "define":
        return _define(llm, index, span, pos)
    if intention == "contextualize":
        return _contextualize(llm, index, span, pos)
    if intention == "recall":
        return _recall(llm, index, span, pos)
    raise ValueError(f"Unknown intention {intention!r}; expected one of {INTENTIONS}")


def respond(
    llm: LLMClient,
    index: EmbeddingIndex,
    selected_text: str,
    intention: str,
    reader_position: int | None = config.READER_POSITION,
) -> str:
    """Route a (selection, intention, position) triple to a bounded response."""
    intention = (intention or "").lower().strip()
    span = (selected_text or "").strip()
    if not span:
        return "Select some text first, then choose what you'd like."
    return _dispatch(llm, index, span, intention, reader_position)[0]


def respond_detailed(
    llm: LLMClient,
    index: EmbeddingIndex,
    selected_text: str,
    intention: str,
    reader_position: int | None = config.READER_POSITION,
) -> dict:
    """Same dispatch as respond(), but expose {answer, chapters, entity}.

    `reader_position=None` disables the position bound (the unbounded baseline
    the per-intention eval contrasts against). `chapters` is the set of chapter
    indices the answer was allowed to draw on — the eval checks none exceed the
    bound, and that the unbounded arm reaches past it.
    """
    intention = (intention or "").lower().strip()
    span = (selected_text or "").strip()
    if not span:
        return {"answer": "Select some text first, then choose what you'd like.",
                "chapters": [], "entity": None}
    answer, chapters, entity = _dispatch(llm, index, span, intention, reader_position)
    return {"answer": answer, "chapters": chapters, "entity": entity}
