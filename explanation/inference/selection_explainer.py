"""Explain whatever a reader selected, using its paragraph(s) as context.

A reader's selection can be a single word, a full sentence, or an arbitrary
excerpt (spanning several sentences, or cutting into one — and, since text is
made of paragraphs, potentially crossing a paragraph break too). Each shape
is handled differently:

- **Single word:** identification is skipped entirely — the Edit Predictor
  gains nothing from running on one already-known word — and the LLM is
  asked directly for its meaning, grounded on the one paragraph the word
  sits in.
- **Sentence or excerpt:** the Edit Predictor identifies difficult words.
  It always runs on whole sentences (never on a raw, possibly-truncated
  excerpt substring, which is not what it was trained on): every sentence
  the excerpt touches is looked up from its paragraph and run through the
  model in full, even where the excerpt only covers part of it. A selection
  that happens to be exactly one full sentence is the special case of this
  same process where only one sentence (and therefore one paragraph) is
  touched.

In every case, the LLM explanation call is grounded on the paragraph(s) that
contain part of the selection — one paragraph for a word or single-sentence
selection, however many an excerpt actually touches — never the caller's
whole input text and never just the bare selection, since a word's meaning
can depend on nearby context outside the highlighted text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import spacy

from explanation.inference._shared import explanations_by_word
from explanation.inference.edit_predictor_identifier import EditPredictorIdentifier
from explanation.llm.openrouter_client import OpenRouterClient
from explanation.schemas import DifficultWord, ExplanationResult

DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_PARAGRAPH_SEPARATOR = r"\n+"

_NLP_CACHE: dict[str, Any] = {}


def _get_nlp(model_name: str = DEFAULT_SPACY_MODEL) -> Any:
    """Load a spaCy model once and reuse it."""
    if model_name not in _NLP_CACHE:
        _NLP_CACHE[model_name] = spacy.load(model_name)
    return _NLP_CACHE[model_name]


@dataclass(frozen=True)
class Paragraph:
    """One paragraph with its text-relative character offsets."""

    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Sentence:
    """One sentence with its text-relative character offsets."""

    text: str
    start: int
    end: int


def split_into_paragraphs(
    text: str, separator_pattern: str = DEFAULT_PARAGRAPH_SEPARATOR
) -> list[Paragraph]:
    """Split *text* into paragraphs, trimming surrounding whitespace off each.

    Paragraphs are separated by *separator_pattern* — by default, one or more
    newlines. Text with mid-paragraph hard line wraps (e.g. Gutenberg-style
    ``\\r\\n``) should be normalized to spaces by the caller first, so that
    only real paragraph breaks remain as newlines.
    """
    breaks = [match.span() for match in re.finditer(separator_pattern, text)]

    chunk_bounds: list[tuple[int, int]] = []
    cursor = 0
    for break_start, break_end in breaks:
        chunk_bounds.append((cursor, break_start))
        cursor = break_end
    chunk_bounds.append((cursor, len(text)))

    paragraphs: list[Paragraph] = []
    for chunk_start, chunk_end in chunk_bounds:
        raw = text[chunk_start:chunk_end]
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw) - len(raw.rstrip())
        start = chunk_start + left_trim
        end = chunk_end - right_trim
        if start >= end:
            continue
        paragraphs.append(Paragraph(text=text[start:end], start=start, end=end))
    return paragraphs


def split_into_sentences(
    paragraph: str, spacy_model: str = DEFAULT_SPACY_MODEL
) -> list[Sentence]:
    """Split *paragraph* into sentences with paragraph-relative offsets."""
    nlp = _get_nlp(spacy_model)
    doc = nlp(paragraph)
    return [
        Sentence(text=sent.text, start=sent.start_char, end=sent.end_char)
        for sent in doc.sents
        if sent.text.strip()
    ]


def _is_single_word(selection: str) -> bool:
    return len(selection.split()) == 1


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _build_context(
    paragraphs: list[Paragraph],
) -> tuple[str, list[tuple[Paragraph, int]]]:
    """Join *paragraphs* into one context string.

    Returns the joined context plus, for each paragraph, the character
    offset at which its text starts inside that context string.
    """
    parts: list[str] = []
    offsets: list[tuple[Paragraph, int]] = []
    cursor = 0
    for paragraph in paragraphs:
        offsets.append((paragraph, cursor))
        parts.append(paragraph.text)
        cursor += len(paragraph.text) + 2  # "\n\n" joiner
    return "\n\n".join(parts), offsets


def _to_context_span(
    span: tuple[int, int], paragraph_offsets: list[tuple[Paragraph, int]]
) -> tuple[int, int]:
    """Remap a *text*-relative span into the joined-context coordinate space."""
    start, end = span
    for paragraph, context_offset in paragraph_offsets:
        if paragraph.start <= start < paragraph.end:
            local_start = context_offset + (start - paragraph.start)
            return local_start, local_start + (end - start)
    raise ValueError(f"Span {span!r} does not fall inside any touched paragraph.")


class SelectionExplainer:
    """Explain difficult words in a reader's selection, in paragraph context.

    Args:
        edit_predictor_checkpoint: Path to a trained Edit Predictor checkpoint.
        client: Optional OpenRouter client. If omitted, one is created lazily
            the first time an explanation is actually needed.
        edit_predictor_threshold: Optional probability threshold for class M.
        max_length: Tokenizer max length for local Edit Predictor inference.
        device: Optional torch device string, e.g. ``"cuda"`` or ``"cpu"``.
        spacy_model: spaCy model used for sentence segmentation.
        paragraph_separator: Regex used to split the input text into
            paragraphs (see :func:`split_into_paragraphs`).
    """

    def __init__(
        self,
        edit_predictor_checkpoint: str | Path,
        client: OpenRouterClient | None = None,
        edit_predictor_threshold: float | None = None,
        max_length: int = 256,
        device: str | None = None,
        spacy_model: str = DEFAULT_SPACY_MODEL,
        paragraph_separator: str = DEFAULT_PARAGRAPH_SEPARATOR,
    ) -> None:
        self._client = client
        self._identifier = EditPredictorIdentifier(
            checkpoint_dir=edit_predictor_checkpoint,
            max_length=max_length,
            device=device,
            threshold=edit_predictor_threshold,
        )
        self._spacy_model = spacy_model
        self._paragraph_separator = paragraph_separator

    def explain(
        self,
        text: str,
        selection: str,
        selection_span: tuple[int, int] | None = None,
    ) -> ExplanationResult:
        """Explain the difficult word(s) in *selection*, within *text*.

        Args:
            text: The full text the selection was made in. May contain
                several paragraphs; only the ones touched by the selection
                are ever sent to the LLM as context.
            selection: The exact substring the reader selected.
            selection_span: Optional ``(start, end)`` character offsets of
                *selection* within *text*. Pass this when the caller already
                knows the offsets (e.g. from a UI selection event) to avoid
                ambiguity when the same text occurs more than once. If
                omitted, the first occurrence is used.

        Returns:
            ExplanationResult keyed on the joined text of the touched
            paragraph(s), with difficult words restricted to the selected
            range.
        """
        selection = selection.strip()
        if not selection:
            return ExplanationResult(sentence="", difficult_words=[])

        if selection_span is not None:
            selection_start, selection_end = selection_span
        else:
            selection_start = text.find(selection)
            if selection_start == -1:
                raise ValueError(
                    "Selection was not found verbatim inside the text. "
                    "Pass selection_span explicitly if the text repeats or "
                    "whitespace was normalized."
                )
            selection_end = selection_start + len(selection)

        paragraphs = split_into_paragraphs(text, self._paragraph_separator)
        touched_paragraphs = [
            paragraph
            for paragraph in paragraphs
            if _spans_overlap(paragraph.start, paragraph.end, selection_start, selection_end)
        ]
        if not touched_paragraphs:
            return ExplanationResult(sentence="", difficult_words=[])

        context, paragraph_offsets = _build_context(touched_paragraphs)

        if _is_single_word(selection):
            word_span = _to_context_span((selection_start, selection_end), paragraph_offsets)
            return self._explain_single_word(context, selection, word_span)

        sentences: list[Sentence] = []
        for paragraph in touched_paragraphs:
            for local_sentence in split_into_sentences(paragraph.text, self._spacy_model):
                sentences.append(
                    Sentence(
                        text=local_sentence.text,
                        start=paragraph.start + local_sentence.start,
                        end=paragraph.start + local_sentence.end,
                    )
                )

        touched_sentences = [
            sentence
            for sentence in sentences
            if _spans_overlap(sentence.start, sentence.end, selection_start, selection_end)
        ]

        difficult_words = self._identify_across_sentences(touched_sentences)
        difficult_words = [
            item
            for item in difficult_words
            if _spans_overlap(item["span"][0], item["span"][1], selection_start, selection_end)
        ]

        if not difficult_words:
            return ExplanationResult(sentence=context, difficult_words=[])

        for item in difficult_words:
            item["span"] = _to_context_span(item["span"], paragraph_offsets)

        return self._explain_identified_words(context, difficult_words)

    def _identify_across_sentences(self, sentences: list[Sentence]) -> list[dict[str, Any]]:
        """Run the Edit Predictor per full sentence, remapped to *text* offsets."""
        results: list[dict[str, Any]] = []
        for sentence in sentences:
            for item in self._identifier.identify(sentence.text):
                local_start, local_end = item["span"]
                results.append({
                    **item,
                    "span": (sentence.start + local_start, sentence.start + local_end),
                })
        return results

    def _explain_single_word(
        self, context: str, word: str, word_span: tuple[int, int]
    ) -> ExplanationResult:
        raw = self._get_client().explain_difficult_words(context, [word])
        explanations = explanations_by_word(raw.get("difficult_words", []))
        return ExplanationResult(
            sentence=context,
            difficult_words=[
                DifficultWord(
                    word=word,
                    span=word_span,
                    meaning_in_context=explanations.get(word, ""),
                )
            ],
        )

    def _explain_identified_words(
        self, context: str, difficult_words: list[dict[str, Any]]
    ) -> ExplanationResult:
        words = [str(item["word"]) for item in difficult_words]
        raw = self._get_client().explain_difficult_words(context, words)
        explanations = explanations_by_word(raw.get("difficult_words", []))

        return ExplanationResult(
            sentence=context,
            difficult_words=[
                DifficultWord(
                    word=str(item["word"]),
                    span=item["span"],
                    meaning_in_context=explanations.get(str(item["word"]), ""),
                )
                for item in difficult_words
            ],
        )

    def _get_client(self) -> OpenRouterClient:
        """Create the OpenRouter client only when an explanation is needed."""
        if self._client is None:
            self._client = OpenRouterClient()
        return self._client
