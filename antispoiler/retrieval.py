"""
Retrieval — bounded vs. unbounded, sharing one strategy.

Design note (de-confounded baseline):
The original notebook compared a *smart* bounded retriever (entity-lexical +
embedding) against a *plain* embedding unbounded retriever. That conflated two
variables — the anti-spoiler filter AND the retrieval strategy — so a measured
difference could not be attributed to the filter alone.

Here a single `retrieve()` implements the smart strategy, and the bounded /
unbounded arms differ ONLY by `reader_pos`:
    bounded   = retrieve(query, reader_pos=N)     # position filter ON
    unbounded = retrieve(query, reader_pos=None)  # position filter OFF

So any difference in spoiler leakage is attributable to the filter — the one
line marked below is the entire anti-spoiler mechanism.
"""

from __future__ import annotations

from .book import Chunk
from .index import EmbeddingIndex
from .llm_client import LLMClient

ENTITY_PROMPT = (
    "If this question is asking about one or more specific named characters, "
    "places, or concepts from a book, return the MOST IMPORTANT name to search "
    "for (just one — pick the most specific or least common one, e.g. 'Wickham' "
    "over 'Darcy' if both appear). If the question is not about any specific "
    "named entity, return exactly 'NONE'."
)


def extract_entity(llm: LLMClient, query: str) -> str | None:
    raw = llm.complete(ENTITY_PROMPT, query, max_tokens=20).strip()
    return None if raw.upper() == "NONE" else raw


def _in_bounds(chunk: Chunk, reader_pos: int | None) -> bool:
    # The anti-spoiler filter. reader_pos=None means "no bound" (baseline).
    return reader_pos is None or chunk.chapter_index <= reader_pos


def retrieve_lexical(
    chunks: list[Chunk],
    entity: str,
    reader_pos: int | None,
    max_results: int = 10,
) -> list[Chunk]:
    """Substring match on the entity, spread across chapters (<=2 per chapter)."""
    matches = [
        c for c in chunks if _in_bounds(c, reader_pos) and entity.lower() in c.text.lower()
    ]
    by_chapter: dict[int, list[Chunk]] = {}
    for c in matches:
        by_chapter.setdefault(c.chapter_index, []).append(c)
    spread: list[Chunk] = []
    for chap in sorted(by_chapter):
        spread.extend(by_chapter[chap][:2])  # prefer earlier intro context
    return spread[:max_results]


def retrieve_embedding(
    index: EmbeddingIndex,
    query: str,
    reader_pos: int | None,
    top_k: int,
) -> list[Chunk]:
    """Embedding kNN, then drop out-of-bounds chunks (oversample to refill)."""
    if reader_pos is None:
        ids = index.search(query, top_k)
        return [index.chunks[i] for i in ids]
    oversample = min(top_k * 10, len(index.chunks))
    ids = index.search(query, oversample)
    filtered = [index.chunks[i] for i in ids if _in_bounds(index.chunks[i], reader_pos)]
    return filtered[:top_k]


def retrieve(
    index: EmbeddingIndex,
    llm: LLMClient,
    query: str,
    reader_pos: int | None,
    top_k: int,
) -> list[Chunk]:
    """
    Smart retrieval, shared by both arms. Entity-focused queries use a lexical
    match (better recall for named characters); everything else uses embeddings.
    `reader_pos=None` disables the position filter (the unbounded baseline).
    """
    entity = extract_entity(llm, query)
    if entity:
        lexical = retrieve_lexical(index.chunks, entity, reader_pos)
        if lexical:
            return lexical
    return retrieve_embedding(index, query, reader_pos, top_k)


def recall_retrieve(
    chunks: list[Chunk],
    entity: str,
    reader_pos: int | None,
    max_results: int = 40,
) -> list[Chunk]:
    """
    Exhaustive (position-bounded) mention gather for the *recall* intention.

    `retrieve_lexical` is tuned for QA: it caps at 2 chunks/chapter and prefers
    early "intro" context, because a QA prompt wants a few representative
    passages. Recall wants the opposite — chronological *completeness* of every
    prior mention of the subject. So this drops the per-chapter cap and orders
    hits by position (earliest -> latest). `max_results` is only a context-window
    guard; if a very common name overflows it, the right fix is map-reduce
    summarisation, not silently dropping the tail (see report / recall notes).

    Caveat shared with all lexical matching: this catches explicit substring
    mentions of `entity`, not pronominal/aliased references. That is a known
    scope limit, not a spoiler risk — the position filter still holds.
    """
    matches = [
        c for c in chunks if _in_bounds(c, reader_pos) and entity.lower() in c.text.lower()
    ]
    matches.sort(key=lambda c: (c.chapter_index, c.paragraph_index))
    return matches[:max_results]


def format_context(retrieved: list[Chunk]) -> str:
    parts = []
    for c in retrieved:
        parts.append(
            f"[{c.chunk_id} | {c.chapter_label}, paragraph {c.paragraph_index} "
            f"(chapter position {c.chapter_index})]\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)
