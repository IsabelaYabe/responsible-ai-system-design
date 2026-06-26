"""
Application service — the glue between the anti-spoiler generator and LLM 3.

`core.py` is the standalone validation pipeline: it knows nothing about the book
or retrieval, only a passage string and a validator LLM. This module is the
*bridge* — it routes by feature, grounds the validator on the right source, runs
the pipeline, and shapes a JSON-serializable payload (verdict + §7 validation path
+ evidence) for the UI.

Routing (design doc §3):
  - contextualize / recall : claims checked against the retrieved chunks the
    generator saw (D15).
  - paraphrase             : a separate linguistic-equivalence check against the
    selected span itself — no retrieval (D14/D15, §1 "one module per claim type").

Keeping it here rather than in `app.py` lets the orchestration be reused from a
notebook or a test without FastAPI, and keeps `app.py` a thin web layer. This is
the one validator module allowed to depend on `antispoiler`; `core.py` stays
standalone.
"""

from __future__ import annotations

from typing import Callable

from antispoiler.book import Chunk
from antispoiler.retrieval import format_context

from . import dictionary
from .core import (
    ValidatorLLM,
    to_ui_payload,
    validate,
    validate_definition,
    validate_paraphrase,
)

# All four reader features now route through LLM 3.
VALIDATED_FEATURES = {"contextualize", "recall", "paraphrase", "define"}

# Define is overloaded: a single word / short term is a dictionary lookup (lexical-
# semantic), but a multi-word phrase "definition" is really a contextual explanation.
# Selections at or under this many words take the dictionary check; longer ones are
# routed to the context-grounded check. (Tunable.)
DEFINE_TERM_MAX_WORDS = 2

# Per-feature selective-prediction threshold τ, calibrated offline (D25 / OBSERVATIONS
# Run 12, notebook v3). Two checks underlie the four features: the context-grounded
# check (contextualize, recall, and the multi-word define route) calibrates to 0.80;
# the single-verdict checks (paraphrase, definition) to 0.85. Values are illustrative
# (tiny single-annotator gold sets) — trust the shape, not the exact decimals.
CONF_THRESHOLD_BY_FEATURE = {
    "contextualize": 0.80,
    "recall": 0.80,
    "paraphrase": 0.85,
    "define": 0.85,
}


def _evidence(chunks: list[Chunk]) -> list[dict]:
    """Retrieved grounding chunks -> §7 'evidence selected' + Hedged source audit."""
    return [
        {
            "chunk_id": c.chunk_id,
            "chapter_label": c.chapter_label,
            "paragraph_index": c.paragraph_index,
            "text": c.text,
        }
        for c in chunks
    ]


def _span_evidence(selected_text: str) -> list[dict]:
    """The selected span as a single evidence entry — what a paraphrase is grounded
    against (D15), so the §7 'evidence' disclosure still has something to show."""
    return [
        {
            "chunk_id": "selection",
            "chapter_label": "Selected passage",
            "paragraph_index": 0,
            "text": selected_text,
        }
    ]


def _dictionary_evidence(term: str, senses_text: str) -> list[dict]:
    """The dictionary senses as a single evidence entry — the lexical grounding
    source for a definition (§3/§7)."""
    return [
        {
            "chunk_id": "dictionary",
            "chapter_label": f'Dictionary — "{term}"',
            "paragraph_index": 0,
            "text": senses_text or "No dictionary entry found for this term.",
        }
    ]


def _disabled(reason: str, note: str) -> dict:
    """A payload the UI renders honestly — never a fake 'Valid'."""
    return {"enabled": False, "reason": reason, "note": note}


def _safe_payload(result_fn: Callable[[], dict], evidence: list[dict]) -> dict:
    """Run a validator pipeline, shape the UI payload, attach evidence. A validator
    failure degrades to an honest note rather than sinking the generated answer."""
    try:
        result = result_fn()
    except Exception as e:
        print(f"[validator] failed: {type(e).__name__}: {e}")
        return _disabled(
            "validator_error",
            f"The answer couldn't be validated this time ({type(e).__name__}).",
        )
    payload = to_ui_payload(result)
    payload["evidence"] = evidence
    return payload


def validate_response(
    validator: ValidatorLLM,
    intention: str,
    answer: str,
    chunks: list[Chunk],
    selected_text: str = "",
) -> dict:
    """Validate a generated answer (LLM 3) and return a UI-ready payload.

    Routes by feature: paraphrase is a linguistic-equivalence check against the
    selected span (D15); contextualize/recall check claims against the retrieved
    chunks. Out-of-scope features get a disabled payload the UI renders honestly.
    """
    if intention not in VALIDATED_FEATURES:
        return _disabled("feature_roadmap", f"Validation isn't enabled for '{intention}'.")

    # Paraphrase — bidirectional meaning preservation vs. the selected span (no retrieval).
    if intention == "paraphrase":
        if not selected_text.strip():
            return _disabled("no_grounding", "No selected passage to check the paraphrase against.")
        return _safe_payload(
            lambda: validate_paraphrase(
                validator, selected_text, answer, CONF_THRESHOLD_BY_FEATURE["paraphrase"]
            ),
            _span_evidence(selected_text),
        )

    # Define — routed by selection shape (the button is overloaded):
    #   short term    -> lexical-semantic dictionary check (valid meaning + passage sense)
    #   phrase/clause -> context-grounded faithfulness check (a contextual explanation)
    if intention == "define":
        term = selected_text.strip()
        if not term:
            return _disabled("no_grounding", "No selected text to define.")
        if len(term.split()) <= DEFINE_TERM_MAX_WORDS:
            senses_text = dictionary.lookup_text(term)
            return _safe_payload(
                lambda: validate_definition(
                    validator, term, answer, format_context(chunks), senses_text,
                    CONF_THRESHOLD_BY_FEATURE["define"],
                ),
                _dictionary_evidence(term, senses_text) + _evidence(chunks),
            )
        # Multi-word: validate the explanation for passage-faithfulness (additions allowed
        # if grounded; lexical/world glosses it can't ground surface as Unverifiable -> Hedged).
        # This runs the context-grounded check, so it takes the CONTEXT τ, not the dictionary one.
        passage = f'SELECTED PASSAGE (the text being explained):\n"""{term}"""\n\n{format_context(chunks)}'
        return _safe_payload(
            lambda: validate(validator, answer, passage, CONF_THRESHOLD_BY_FEATURE["contextualize"]),
            _span_evidence(term) + _evidence(chunks),
        )

    # Contextualize and Recall (context-grounded features) - Their claims are checked
    # against the retrieved chunks the generator saw (D15). Both use the context τ.
    if not chunks:
        return _disabled(
            "no_grounding",
            "No in-bounds passages were retrieved, so there's nothing to validate against.",
        )
    return _safe_payload(
        lambda: validate(validator, answer, format_context(chunks), CONF_THRESHOLD_BY_FEATURE[intention]),
        _evidence(chunks),
    )
