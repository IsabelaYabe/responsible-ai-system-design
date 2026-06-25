"""
LLM 3 — the validation layer.

`validator/` holds both the design record (DECISIONS.md, OBSERVATIONS.md) and the
live pipeline (core.py). The offline evidence that earns the TRL-3 claim — gold
set, AUROC, risk-coverage, threshold tuning — stays in
`notebooks/validator_judge_poc_v2.ipynb`; this package is the per-request path
the app calls, plus the threshold it produced.
"""

from __future__ import annotations

from .core import (
    BANNER,
    CONF_THRESHOLD,
    MESSAGE,
    VERDICT_LABELS,
    aggregate,
    decompose_and_route,
    map_to_ui,
    to_ui_payload,
    validate,
    validate_claim,
    validate_paraphrase,
)

__all__ = [
    "validate",
    "validate_paraphrase",
    "to_ui_payload",
    "decompose_and_route",
    "validate_claim",
    "aggregate",
    "map_to_ui",
    "CONF_THRESHOLD",
    "VERDICT_LABELS",
    "BANNER",
    "MESSAGE",
]
