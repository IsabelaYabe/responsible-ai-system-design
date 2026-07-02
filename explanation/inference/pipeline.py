"""End-to-end explanation pipeline: sentence -> ExplanationResult.

The pipeline has a single responsibility:
1. identify difficult words with the local Edit Predictor;
2. ask the LLM only for contextual explanations.

It does not support LLM-based identification, lexical substitution,
sentence simplification, or synonym generation.
"""

from __future__ import annotations

from pathlib import Path

from explanation.inference._shared import coerce_span as _coerce_span
from explanation.inference._shared import explanations_by_word as _explanations_by_word
from explanation.inference.edit_predictor_identifier import EditPredictorIdentifier
from explanation.llm.openrouter_client import OpenRouterClient
from explanation.schemas import DifficultWord, ExplanationResult


class ExplanationPipeline:
    """Pipeline for explaining difficult words in an English sentence.

    The local Edit Predictor identifies difficult-word spans. The LLM only
    explains those already identified words in context.

    Args:
        edit_predictor_checkpoint: Path to a trained Edit Predictor checkpoint.
        client: Optional OpenRouter client. If omitted, one is created lazily.
        edit_predictor_threshold: Optional probability threshold for class M.
        max_length: Tokenizer max length for local Edit Predictor inference.
        device: Optional torch device string, e.g. ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        edit_predictor_checkpoint: str | Path,
        client: OpenRouterClient | None = None,
        edit_predictor_threshold: float | None = None,
        max_length: int = 128,
        device: str | None = None,
    ) -> None:
        self._client = client
        self._identifier = EditPredictorIdentifier(
            checkpoint_dir=edit_predictor_checkpoint,
            max_length=max_length,
            device=device,
            threshold=edit_predictor_threshold,
        )

    def run(self, sentence: str) -> ExplanationResult:
        """Identify and explain difficult words in *sentence*.

        Args:
            sentence: A single English sentence.

        Returns:
            ExplanationResult containing the original sentence and the
            difficult words with spans and contextual explanations.
        """
        sentence = sentence.strip()
        if not sentence:
            return ExplanationResult(sentence="", difficult_words=[])

        local_items = self._identifier.identify(sentence)
        if not local_items:
            return ExplanationResult(sentence=sentence, difficult_words=[])

        words = [str(item["word"]) for item in local_items]
        raw = self._get_client().explain_difficult_words(sentence, words)
        explanations = _explanations_by_word(raw.get("difficult_words", []))

        difficult_words: list[DifficultWord] = []
        for item in local_items:
            word = str(item["word"])
            start, end = _coerce_span(item["span"])

            difficult_words.append(
                DifficultWord(
                    word=word,
                    span=(start, end),
                    meaning_in_context=explanations.get(word, ""),
                )
            )

        return ExplanationResult(sentence=sentence, difficult_words=difficult_words)

    def _get_client(self) -> OpenRouterClient:
        """Create the OpenRouter client only when an explanation is needed."""
        if self._client is None:
            self._client = OpenRouterClient()
        return self._client