"""Explanation pipeline: identify and explain difficult words in English sentences."""

from explanation.inference.pipeline import ExplanationPipeline
from explanation.inference.selection_explainer import SelectionExplainer
from explanation.schemas import DifficultWord, ExplanationResult

__all__ = [
    "ExplanationPipeline",
    "SelectionExplainer",
    "ExplanationResult",
    "DifficultWord",
]
