"""
Interactive demo for the anti-spoiler reading companion.

A thin web layer over the `antispoiler` package — the counterpart to the
notebook. Where the notebook is the *evaluation* harness (batch question set,
LLM judge, metrics), this is the *demonstration*: it models the real product
interaction, a (selected_text, intention, reader_position) triple.

The reader sees the book rendered only up to their position (the slider), so
they can only select text they've "read"; selecting a span and clicking an
intention calls `antispoiler.respond.respond`, which keeps every retrieval
bounded by that same position. Moving the slider makes spoilers appear/vanish —
the anti-spoiler mechanism, made visible.

Run (native arm64 env; see antispoiler/README.md):
    conda run -n antispoiler-arm uvicorn app:app --reload --port 8000
then open http://127.0.0.1:8000

First request is slow: it downloads the embedding model and indexes the book
once, at startup.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from antispoiler import config
from antispoiler.book import fetch_and_chunk
from antispoiler.index import build_index
from antispoiler.llm_client import LLMClient, make_validator
from antispoiler.respond import INTENTIONS, respond_with_evidence

from validator import CONF_THRESHOLD, dictionary
from validator.service import VALIDATED_FEATURES, validate_response

app = FastAPI(title="Anti-spoiler reading companion (demo)")

_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Mode banner (prod = Anthropic Haiku+Sonnet; dev = one cheap model via OpenRouter).
print(f"Mode: {config.APP_MODE.upper()}  |  backend={config.BACKEND}  "
      f"generator={config.ANSWERER_MODEL}  validator={config.VALIDATOR_MODEL}")
if config.APP_MODE == "dev":
    print("  ⚠  dev mode: one cheap model via OpenRouter — for iteration only, NOT the "
          "characterized setup (validator==generator breaks D13).")

# Book-independent, built once at import time.
LLM = LLMClient(model=config.ANSWERER_MODEL)          # generator
VALIDATOR = make_validator()                          # validator LLM 3 (config.VALIDATOR_MODEL); validator != generator (D13)

# Per-book chunks+index, built lazily and cached. Embedding a whole novel is slow
# (and the first build downloads the embedding model), so we only build the default
# book at startup; other books are built on first switch. Keyed by book id.
BOOK_STATE: dict[str, dict] = {}


def get_book_state(book_id: str | None = None) -> dict:
    """Cached {meta, chunks, index, max_chapter} for a book; builds + caches on first use."""
    meta = config.get_book(book_id)
    bid = meta["id"]
    state = BOOK_STATE.get(bid)
    if state is None:
        print(f"Loading book {meta['title']!r} and building index "
              f"(first run downloads the embedding model)…")
        chunks = fetch_and_chunk(meta["url"])
        index = build_index(chunks, title=meta["title"], author=meta["author"])
        max_chapter = max(c.chapter_index for c in chunks)
        state = {"meta": meta, "chunks": chunks, "index": index, "max_chapter": max_chapter}
        BOOK_STATE[bid] = state
        print(f"Ready: {meta['title']} — {len(chunks)} chunks across {max_chapter} chapters.")
    return state


# Pre-build the default book so startup cost matches the previous single-book behaviour.
_DEFAULT = get_book_state()
print(f"Validator: model={VALIDATOR.model}  tau={CONF_THRESHOLD}  features={sorted(VALIDATED_FEATURES)}")
_dict_ok, _dict_detail = dictionary.available()  # warms the WordNet corpus; surfaces setup issues now
print(f"Dictionary: {'OK' if _dict_ok else 'UNAVAILABLE'} — {_dict_detail}")


class RespondRequest(BaseModel):
    selected_text: str
    intention: str
    reader_position: int
    book_id: str | None = None


@app.get("/")
def home():
    return FileResponse(os.path.join(_STATIC, "index.html"))


@app.get("/books")
def books():
    """The selectable books (registry order; first is the default)."""
    return [{"id": b["id"], "title": b["title"], "author": b["author"]} for b in config.BOOKS]


@app.get("/config")
def app_config(book: str | None = Query(None)):
    state = get_book_state(book)  # builds + caches on first request for this book
    meta = state["meta"]
    return {
        "book_id": meta["id"],
        "title": meta["title"],
        "author": meta["author"],
        "max_chapter": state["max_chapter"],
        "default_position": meta["default_position"],
        "intentions": INTENTIONS,
        "validated_features": sorted(VALIDATED_FEATURES),
        "conf_threshold": CONF_THRESHOLD,
    }


@app.get("/book")
def book(book: str | None = Query(None), upto: int | None = Query(None)):
    """Chapters 1..upto, grouped — the text the reader is allowed to select from."""
    state = get_book_state(book)
    chunks, max_chapter = state["chunks"], state["max_chapter"]
    if upto is None:
        upto = state["meta"]["default_position"]
    upto = max(1, min(int(upto), max_chapter))
    chapters: list[dict] = []
    cur: dict | None = None
    for c in chunks:
        if c.chapter_index > upto:
            break  # chunks are ordered by (chapter_index, paragraph_index)
        if cur is None or cur["index"] != c.chapter_index:
            cur = {"index": c.chapter_index, "label": c.chapter_label, "paragraphs": []}
            chapters.append(cur)
        cur["paragraphs"].append(c.text)
    return {"upto": upto, "max_chapter": max_chapter, "chapters": chapters}


@app.post("/respond")
def do_respond(req: RespondRequest):
    if req.intention not in INTENTIONS:
        return JSONResponse(
            {"error": f"unknown intention {req.intention!r}; expected {INTENTIONS}"},
            status_code=400,
        )
    if not req.selected_text.strip():
        return JSONResponse({"error": "no text selected"}, status_code=400)
    state = get_book_state(req.book_id)
    index, max_chapter = state["index"], state["max_chapter"]
    pos = max(1, min(int(req.reader_position), max_chapter))

    # Generate (LLM 1/1.2), keeping the retrieved grounding chunks for the validator.
    # A generation failure must not 500 the request — degrade to an honest message
    # (e.g. a cheap dev model returning an empty response).
    try:
        out = respond_with_evidence(LLM, index, req.selected_text, req.intention, pos)
    except Exception as e:
        print(f"[generator] failed: {type(e).__name__}: {e}")
        return {
            "answer": f"The assistant couldn't generate a response this time ({type(e).__name__}).",
            "intention": req.intention,
            "reader_position": pos,
            "validation": {
                "enabled": False,
                "reason": "generation_error",
                "note": f"Generation failed ({type(e).__name__}); there's nothing to validate.",
            },
        }
    answer = out["answer"]

    # Validate (LLM 3) — blocking; the frontend shows a spinner meanwhile.
    # selected_text is the grounding source for paraphrase (D15).
    validation = validate_response(
        VALIDATOR, req.intention, answer, out["chunks"], req.selected_text
    )

    return {
        "answer": answer,
        "intention": req.intention,
        "reader_position": pos,
        "validation": validation,
    }
