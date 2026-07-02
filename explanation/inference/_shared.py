"""Helpers shared by the inference pipelines that call the LLM explainer."""

from __future__ import annotations

from typing import Any


def coerce_span(raw_span: Any) -> tuple[int, int]:
    """Convert a span-like object into a validated ``(start, end)`` tuple."""
    if not isinstance(raw_span, (list, tuple)) or len(raw_span) != 2:
        raise ValueError(f"Invalid span returned by Edit Predictor: {raw_span!r}")

    start = int(raw_span[0])
    end = int(raw_span[1])

    if start < 0 or end <= start:
        raise ValueError(f"Invalid span returned by Edit Predictor: {raw_span!r}")

    return start, end


def explanations_by_word(entries: Any) -> dict[str, str]:
    """Map LLM explanation entries by word, ignoring LLM-provided spans."""
    if not isinstance(entries, list):
        return {}

    explanations: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        word = str(entry.get("word", "")).strip()
        if not word or word in explanations:
            continue

        explanations[word] = str(entry.get("meaning_in_context", "")).strip()

    return explanations
