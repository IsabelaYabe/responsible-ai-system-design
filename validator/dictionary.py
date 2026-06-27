"""
Dictionary grounding for the definition validator (design doc §3, lexical-semantic).

Looks up a term's senses so the validator can check a definition by *entailment
against retrieved senses* rather than recall from memory — the same "convert
recall into grounding" move web search makes for world-knowledge (§5). The result
is exposed to the reader as the auditable grounding source (§7).

Backend: WordNet (offline, deterministic, no API key) — the TRL-3-friendly choice
(reproducible, characterizable). Coverage is modern English single words; archaic
senses, phrases, and proper nouns are the named validity boundary — a miss returns
no senses, which routes the definition to Unverifiable -> Hedged (honest), never a
crash. The lookup is kept behind one function so the backend can be swapped (API /
curated for the offline gold set) later.
"""

from __future__ import annotations

_POS = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}

_wn = None  # cached wordnet handle


def _ensure_wordnet():
    """Import WordNet, downloading the corpus once if needed. Raises if unavailable."""
    global _wn
    if _wn is not None:
        return _wn
    import nltk
    from nltk.corpus import wordnet as wn

    try:
        wn.synsets("test")  # probe — raises LookupError if the corpus isn't present
    except LookupError:
        for pkg in ("wordnet", "omw-1.4"):
            if not nltk.download(pkg, quiet=True):  # returns False on a failed download
                raise RuntimeError(
                    f"could not download NLTK '{pkg}'. Run once: "
                    f"python -m nltk.downloader wordnet omw-1.4"
                )
        wn.synsets("test")  # re-probe — raises if still missing
    _wn = wn
    return wn


def _clean_gloss(text: str) -> str:
    """WordNet glosses embed example fragments / author attributions after ';'
    (e.g. '...weariness; ; - Mark Twain'). Keep only the real definition clause(s)."""
    parts = [p.strip() for p in text.split(";")]
    parts = [p for p in parts if p and not p.startswith("-") and not p.startswith('"')]
    return "; ".join(parts) if parts else text.strip()


def available() -> tuple[bool, str]:
    """(ok, detail) — whether the dictionary backend is usable, for a startup check.
    Warms the corpus so the first definition request isn't slow."""
    try:
        wn = _ensure_wordnet()
        return True, f"wordnet ({len(wn.synsets('test'))} senses for 'test')"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def lookup(term: str, max_senses: int = 6) -> list[dict]:
    """Term -> up to `max_senses` dictionary senses [{pos, definition, example}].

    Returns [] on any failure (term not found, corpus/lib missing) — the caller
    treats an empty result as "no dictionary coverage" and hedges.
    """
    term = (term or "").strip()
    if not term:
        return []
    try:
        wn = _ensure_wordnet()
    except Exception as e:
        # Backend unavailable (corpus/lib missing) — degrade to a miss, but make it
        # LOUD so it isn't mistaken for "the word isn't in the dictionary".
        print(f"[dictionary] WordNet unavailable: {e}")
        return []

    senses: list[dict] = []
    seen: set[str] = set()
    for form in (term, term.replace(" ", "_")):  # WordNet collocations use underscores
        try:
            synsets = wn.synsets(form)
        except Exception:
            synsets = []
        for syn in synsets:
            if syn.name() in seen:
                continue
            seen.add(syn.name())
            examples = syn.examples()
            senses.append(
                {
                    "pos": _POS.get(syn.pos(), syn.pos()),
                    "definition": _clean_gloss(syn.definition()),
                    "example": examples[0] if examples else None,
                }
            )
            if len(senses) >= max_senses:
                return senses
    return senses


def format_senses(senses: list[dict]) -> str:
    """Senses -> a compact numbered list for the validator prompt and the UI."""
    if not senses:
        return ""
    lines = []
    for i, s in enumerate(senses, 1):
        ex = f'  e.g. "{s["example"]}"' if s.get("example") else ""
        lines.append(f'{i}. ({s["pos"]}) {s["definition"]}{ex}')
    return "\n".join(lines)


def lookup_text(term: str, max_senses: int = 6) -> str:
    """Convenience: lookup + format. Empty string on a miss."""
    return format_senses(lookup(term, max_senses))
