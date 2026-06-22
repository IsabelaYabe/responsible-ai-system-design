"""
Fetch a public-domain book and chunk it by chapter.

Each chunk carries the metadata the anti-spoiler filter needs:
  - chapter_index   : 1-based monotonic position (the anti-spoiler axis)
  - chapter_label   : human-readable ("Chapter XV")
  - paragraph_index : 1-based index within the chapter
  - text            : the paragraph body

Short paragraphs are greedily merged with the next until MIN_CHARS, to avoid
tiny low-signal chunks. No vendor/ML deps here, so this is unit-testable on its
own (see tests/test_book.py, which uses a synthetic book string).
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass

from . import config

CHAPTER_PATTERN = r"\r?\n(Chapter [\w]+\.?)\r?\n"


@dataclass
class Chunk:
    chunk_id: str  # e.g. "ch15_p07"
    chapter_index: int  # 1-based, the position axis
    chapter_label: str  # "Chapter XV"
    paragraph_index: int  # 1-based within chapter
    text: str


def _split_chapter_body(
    body: str, chapter_index: int, chapter_label: str, min_chars: int
) -> list[Chunk]:
    paragraphs = [p.strip() for p in re.split(r"\r?\n\s*\r?\n", body) if p.strip()]

    # Greedy merge of short paragraphs.
    merged: list[str] = []
    buf = ""
    for p in paragraphs:
        if not buf:
            buf = p
        elif len(buf) < min_chars:
            buf = buf + "\n\n" + p
        else:
            merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)

    return [
        Chunk(
            chunk_id=f"ch{chapter_index:02d}_p{i:02d}",
            chapter_index=chapter_index,
            chapter_label=chapter_label,
            paragraph_index=i,
            text=text,
        )
        for i, text in enumerate(merged, start=1)
    ]


def chunk_text(raw: str, min_chars: int = config.MIN_CHARS) -> list[Chunk]:
    """Chunk an already-fetched book string. Split out for testability."""
    parts = re.split(CHAPTER_PATTERN, raw)

    chunks: list[Chunk] = []
    idx = 0
    i = 1
    while i < len(parts) - 1:
        label = parts[i].strip()
        body = parts[i + 1].strip()
        if body:
            idx += 1
            chunks.extend(_split_chapter_body(body, idx, label, min_chars))
        i += 2
    return chunks


def fetch_book(url: str = config.BOOK_URL) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "research-notebook/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def fetch_and_chunk(url: str = config.BOOK_URL) -> list[Chunk]:
    return chunk_text(fetch_book(url))
