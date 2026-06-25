"""
LLM 3 — correctness + uncertainty validator (the live pipeline).

Productionized from `notebooks/validator_judge_poc_v2.ipynb`. This module is the
per-request path only:

    decompose -> route -> grounded per-claim verdict -> worst-case aggregate
    -> selective-prediction UI gate (threshold tuned OFFLINE on the gold set)

What deliberately does NOT live here (it stays in the notebook, §9 of the design
doc): the gold set, AUROC, risk-coverage analysis, threshold tuning, and the
K-run variability study. Those are the offline *evidence* that earns the TRL-3
claim; they produce the single number `CONF_THRESHOLD` that this module consumes.
Keeping them out of the request path is decision D19 — their cost must never reach
the reader.

Design record: `validator/DECISIONS.md`, `validator/OBSERVATIONS.md`.

The validator LLM is any object exposing `complete(system, user, max_tokens=...)
-> str` (antispoiler.llm_client.LLMClient). We call it the *validator* rather than
the *judge* to keep it distinct from the anti-spoiler eval's separate LLM-as-a-
judge. Per D13 the validator should differ from the generator; in this demo the
generator is Haiku and the validator is Sonnet — a size difference, not a *family*
difference. The residual self-preference risk is a named limitation (see
DECISIONS.md D13 / design doc §12), not solved here.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Protocol


class ValidatorLLM(Protocol):
    """Structural type for the validator LLM — antispoiler.llm_client.LLMClient fits."""

    def complete(self, system: str, user: str, max_tokens: int = ...) -> str: ...


# ── Verdict schema (D9) and aggregation severity (D10) ───────────────────────
VERDICT_LABELS = ["Supported", "Partially supported", "Contradicted", "Unverifiable"]
SEVERITY = {"Supported": 0, "Partially supported": 1, "Unverifiable": 2, "Contradicted": 3}

# Commit/abstain threshold for the selective-prediction gate (D19). This is the
# OUTPUT of the offline tuning on the gold set, not a guess: on the 24-item gold
# set the risk-coverage curve gives committed-accuracy 0.92 at full coverage for
# tau=0.85 (and 1.00 at 83% coverage for tau=0.90). We operate at 0.85 for now
# (a project decision — revisit on a larger gold set). See OBSERVATIONS.md Run 11.
CONF_THRESHOLD = 0.85

# Canonical UI copy for each 3-way state (design doc §6 / D11). The frontend
# styles these; this is the source-of-truth wording.
BANNER = {"Valid": "VALID", "Not reliable": "NOT RELIABLE", "Hedged": "HEDGED"}
MESSAGE = {
    "Valid": "Verified against the passages you've read.",
    "Not reliable": "Couldn't give a reliable response this time.",
    "Hedged": "Couldn't fully verify this — read it with care.",
}


# ── helpers ──────────────────────────────────────────────────────────────────
def parse_json_response(raw: str) -> Any:
    """Strip ```fences``` and parse JSON (models often wrap JSON in markdown).

    Falls back to extracting the outermost array/object if the model added stray
    prose around the JSON. Truncated JSON (hit the token cap) cannot be salvaged
    and re-raises with a clear message — the fix for that is a larger max_tokens.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[\w]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)  # tolerate prose around the JSON
        if m:
            return json.loads(m.group(1))
        raise


def parallel_map(fn: Callable, items, max_workers: int = 6) -> list:
    """Run fn over items concurrently (LLM calls are I/O-bound); preserves order."""
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(fn, items))


# ── Stage 4 — decompose & route (D7, D8, D17, P2-1) ──────────────────────────
SYSTEM_DECOMPOSE = """You are the claim-decomposition stage of a validation system for a reading assistant.

Your job: break an assistant's answer into ATOMIC CLAIMS and tag each by its GROUNDING SOURCE.
You do NOT judge whether claims are true. Extract faithfully: include every claim, even ones that
look wrong, and never correct, soften, or rephrase their meaning.

ATOMIC = one checkable fact per claim. Split conjunctions and compound sentences into separate
claims. Resolve pronouns and references so each claim stands on its own (e.g. "He" -> "the lighthouse keeper").

SEPARATE FACTS FROM ATTITUDES/CAUSES. When a sentence ties a feeling, attitude, belief, or reason to
someone ("X is excited because Y", "X is pleased that Y", "X thinks Y"), split it into:
  (a) the underlying factual claim Y, stated plainly, and
  (b) the attitudinal/causal claim linking the person to it.
Each is checked separately - the fact may be in the passage even when the attribution is not.
Example: "The lighthouse keeper was worried because the lamp had gone out" becomes two claims:
  - "The lamp had gone out."
  - "The lighthouse keeper was worried because the lamp had gone out."

GROUNDING SOURCE: tag each claim as exactly one of
- "context"         : a statement about what happens INSIDE the story (characters, events,
                      relationships, motivations, setting as described in the book). Must be checked
                      against the book passage the reader has read.
- "world_knowledge" : a statement about the real world or the book as a real-world artifact
                      (historical facts, publication dates, real people/places, literary background).
                      Would need an external source to verify.

Output ONLY a JSON array, no prose, no markdown fences. Each element:
  {"claim": "<the atomic claim as a standalone sentence>", "grounding": "context" | "world_knowledge"}
"""


def decompose_and_route(validator: ValidatorLLM, answer: str) -> list[dict]:
    """Answer text -> list of {"claim", "grounding"}. Routing only; no verdicts (D8)."""
    # Real generated answers (a contextualize reply is several paragraphs) decompose
    # into many claims, so the JSON array needs generous room — 600 truncated it.
    raw = validator.complete(
        SYSTEM_DECOMPOSE, f"ASSISTANT ANSWER TO DECOMPOSE:\n{answer}", max_tokens=3000
    )
    return parse_json_response(raw)


# ── Stage 5 — grounded per-claim verdict (D8, D9, D16) ───────────────────────
SYSTEM_VERDICT = """You are the per-claim verdict stage of a validation system for a reading assistant.

You are given ONE atomic claim and a SOURCE PASSAGE. Decide how the passage bears on the claim.
Judge ONLY from the passage. Do not use outside knowledge: if the passage does not state or imply the
claim, the verdict is "Unverifiable" - regardless of whether you personally believe the claim is true
or false. Entailment against the provided text - not recall from memory - is the whole point.

Verdict, exactly one of:
- "Supported"           : the passage states or clearly entails the claim.
- "Partially supported" : the passage supports part of the claim but not all of it.
- "Contradicted"        : the passage states or clearly entails the OPPOSITE of the claim.
- "Unverifiable"        : the passage neither supports nor contradicts the claim (it is silent on it).

Tie-break (absence vs. contradiction): use "Contradicted" ONLY when the passage explicitly states or
directly entails the opposite of the claim. If the passage simply does not mention the claim, use
"Unverifiable" - even if other statements in the passage loosely suggest the claim might be false.

Also report your confidence that your verdict is correct, as a number from 0.0 to 1.0.

Output ONLY a JSON object, no prose, no markdown fences:
  {"verdict": "<one label>", "confidence": <0.0-1.0>, "reason": "<one sentence grounded in the passage>"}
"""


def validate_claim_once(validator: ValidatorLLM, claim: str, passage: str) -> dict:
    """One grounded verdict pass for a single claim (temperature 0, single sample).

    Within-run N-sample was retired (D5/Run 8): it was inert — a peaked verdict's
    token distribution gives no diversity under resampling, so agreement was always
    ~1.00. The reliability signal that works (cross-run stability) lives offline.
    """
    user = f"SOURCE PASSAGE:\n{passage}\n\nCLAIM:\n{claim}"
    raw = validator.complete(SYSTEM_VERDICT, user, max_tokens=512)
    return parse_json_response(raw)


def validate_claim(validator: ValidatorLLM, claim: str, grounding: str, passage: str) -> dict:
    """Grounded verdict + verbalized confidence for one claim.

    `world_knowledge` short-circuits to Unverifiable: there is no web-search
    grounding in this PoC (design doc §5/§12), so the validator does not adjudicate
    real-world facts from memory — those route to Hedged downstream.
    """
    if grounding == "world_knowledge":
        return {
            "claim": claim,
            "grounding": grounding,
            "verdict": "Unverifiable",
            "verbalized_conf": 1.0,
            "reason": "World-knowledge claim; no external source available to verify it in this prototype.",
        }
    v = validate_claim_once(validator, claim, passage)
    return {
        "claim": claim,
        "grounding": grounding,
        "verdict": v["verdict"],
        "verbalized_conf": v.get("confidence"),
        "reason": v.get("reason", ""),
    }


# ── Stage 6 — answer-level aggregation (worst-case, D10) ──────────────────────
def aggregate(verdicts: list[dict]) -> dict:
    """Any non-Supported claim flags the whole answer (worst-case, D10)."""
    labels = [v["verdict"] for v in verdicts]
    worst = max(labels, key=lambda l: SEVERITY.get(l, 0)) if labels else "Unverifiable"
    flagged = [v for v in verdicts if v["verdict"] == worst and SEVERITY.get(worst, 0) > 0]
    return {
        "answer_verdict": worst,
        "counts": dict(Counter(labels)),  # plain dict so it is JSON-serializable
        "flagged": flagged,
        "n_claims": len(labels),
    }


# ── Stage 7 — uncertainty gate + 3-way UI mapping (D11, D19) ──────────────────
def map_to_ui(agg: dict, verdicts: list[dict], conf_threshold: float | None = None) -> dict:
    """Selective-prediction gate (D19) then the 3-way UI state (D11 / design doc §6).

    Commit to the verdict (-> Valid / Not reliable) only if the confidence of the
    driving claim(s) clears tau; otherwise abstain (-> Hedged). The high-uncertainty
    abstention is checked first, so an Unverifiable / low-confidence answer hedges
    rather than being forced into a confident state.
    """
    if conf_threshold is None:
        conf_threshold = CONF_THRESHOLD
    verdict = agg["answer_verdict"]
    driving = agg["flagged"] if agg["flagged"] else verdicts
    confs = [v["verbalized_conf"] for v in driving if v.get("verbalized_conf") is not None]
    answer_conf = min(confs) if confs else None
    high_uncertainty = (answer_conf is not None) and (answer_conf < conf_threshold)

    if high_uncertainty:
        state = "Hedged"
    elif verdict == "Supported":
        state = "Valid"
    elif verdict == "Contradicted":
        state = "Not reliable"
    else:  # Unverifiable / Partially supported with adequate confidence
        state = "Hedged"

    return {
        "ui_state": state,
        "answer_verdict": verdict,
        "answer_confidence": answer_conf,
        "high_uncertainty": high_uncertainty,
    }


# ── Capstone — the validator as one call ──────────────────────────────────────
def validate(validator: ValidatorLLM, answer: str, passage: str) -> dict:
    """Run the full per-request validator on one generated answer.

    `passage` is the grounding text — in the live demo this is the SAME retrieved,
    position-bounded chunks the generator saw (D15), formatted as one string. The
    per-claim verdict calls are parallelized (I/O-bound).
    """
    claims = decompose_and_route(validator, answer)
    verdicts = parallel_map(
        lambda c: validate_claim(validator, c["claim"], c["grounding"], passage), claims
    )
    agg = aggregate(verdicts)
    ui = map_to_ui(agg, verdicts)
    return {"claims": claims, "verdicts": verdicts, "aggregate": agg, "ui": ui}


# ── Paraphrase check — linguistic equivalence (design doc §3, D14/D15) ────────
# A SEPARATE check from the context pipeline (§1: one module per claim type).
# Grounding is the selected source span itself — no retrieval (D15) — and the test
# is bidirectional meaning preservation: nothing added, dropped, or distorted.
SYSTEM_PARAPHRASE = """You are the paraphrase-validation stage of a validation system for a reading assistant.

You are given a SOURCE PASSAGE (a short span the reader selected) and a PARAPHRASE of it produced by the
assistant. Judge whether the paraphrase preserves the MEANING of the source, in BOTH directions:

- ADDED     : the paraphrase introduces information, claims, or implications NOT present in the source.
- DROPPED   : the paraphrase omits meaning that is present and important in the source.
- DISTORTED : the paraphrase changes the meaning - negates it, reverses who did what, or shifts strength,
              certainty, tense, or tone in a way that alters what is said.

A good paraphrase restates the SAME meaning in different words. Judge ONLY the source against the
paraphrase: do not use outside knowledge of the book, and do not reward or penalize writing style - only
meaning.

Verdict, exactly one of:
- "Supported"           : meaning preserved both ways; a faithful paraphrase.
- "Partially supported" : core meaning preserved, but with a minor addition or omission.
- "Contradicted"        : the meaning is distorted, negated, or materially changed.
- "Unverifiable"        : the paraphrase is too garbled or off-topic to compare meaningfully.

Also report your confidence that your verdict is correct, as a number from 0.0 to 1.0.

Output ONLY a JSON object, no prose, no markdown fences:
  {"verdict": "<one label>", "confidence": <0.0-1.0>, "reason": "<one or two sentences naming any added / dropped / distorted content, or stating the meaning is fully preserved>"}
"""


def paraphrase_verdict_once(validator: ValidatorLLM, source_text: str, paraphrase_text: str) -> dict:
    """One bidirectional meaning-preservation verdict for a paraphrase (single pass)."""
    user = f"SOURCE PASSAGE:\n{source_text}\n\nPARAPHRASE:\n{paraphrase_text}"
    raw = validator.complete(SYSTEM_PARAPHRASE, user, max_tokens=512)
    return parse_json_response(raw)


def validate_paraphrase(validator: ValidatorLLM, source_text: str, paraphrase_text: str) -> dict:
    """Validate a paraphrase against its source span — same result shape as validate(),
    so the shared aggregate / UI gate / payload machinery is reused unchanged.

    The bidirectional check is a single verdict (the "minimal" paraphrase validator,
    §11) surfaced as one row in the §7 path; the per-direction findings (added /
    dropped / distorted) are named in its `reason`.
    """
    v = paraphrase_verdict_once(validator, source_text, paraphrase_text)
    row = {
        "claim": "The paraphrase preserves the meaning of the selected passage (nothing added, dropped, or distorted).",
        "grounding": "paraphrase",
        "verdict": v["verdict"],
        "verbalized_conf": v.get("confidence"),
        "reason": v.get("reason", ""),
    }
    verdicts = [row]
    claims = [{"claim": row["claim"], "grounding": "paraphrase"}]
    agg = aggregate(verdicts)
    ui = map_to_ui(agg, verdicts)
    return {"claims": claims, "verdicts": verdicts, "aggregate": agg, "ui": ui}


# ── Presentation — internal result -> JSON-serializable UI payload ────────────
def to_ui_payload(result: dict) -> dict:
    """Flatten validate()'s result into the shape the frontend renders.

    Surfaces the §7 validation path: the per-claim verdict + confidence + `reason`
    (the one-sentence grounded justification from the validator), the answer-level
    state, and the counts (for "N of M claims verified"). The caller attaches the
    `evidence` list (the retrieved chunks) since that is app-side, not validator-side.
    """
    ui, agg = result["ui"], result["aggregate"]
    state = ui["ui_state"]
    return {
        "enabled": True,
        "ui_state": state,
        "banner": BANNER[state],
        "message": MESSAGE[state],
        "answer_verdict": ui["answer_verdict"],
        "answer_confidence": ui["answer_confidence"],
        "high_uncertainty": ui["high_uncertainty"],
        "conf_threshold": CONF_THRESHOLD,
        "counts": agg["counts"],
        "n_claims": agg["n_claims"],
        "claims": [
            {
                "claim": v["claim"],
                "grounding": v["grounding"],
                "verdict": v["verdict"],
                "confidence": v.get("verbalized_conf"),
                "reason": v.get("reason", ""),
            }
            for v in result["verdicts"]
        ],
    }
