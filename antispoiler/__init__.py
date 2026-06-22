"""
Anti-spoiler reading assistant — position-bounded retrieval vs. unbounded baseline.

A small package extracted from the original Colab notebook so the logic can be
unit-tested and run locally (VS Code) or in Colab. The notebook is a thin
orchestration layer that imports from here.
"""

from __future__ import annotations

__all__ = [
    "config",
    "llm_client",
    "book",
    "index",
    "retrieval",
    "qa",
    "qgen",
    "judge",
    "evaluate",
]
