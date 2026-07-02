"""Core data structures for the explanation pipeline."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DifficultWord:
    """A single word identified as difficult, with its contextual explanation.

    Attributes:
        word: The surface form of the difficult word as it appears in the sentence.
        span: Character-level (start, end) indices in the original sentence (exclusive end).
        meaning_in_context: Short gloss of what the word means given this sentence.
    """

    word: str
    span: Tuple[int, int]
    meaning_in_context: str


@dataclass
class ExplanationResult:
    """Full output of the explanation pipeline for one input sentence.

    Attributes:
        sentence: The original input sentence.
        difficult_words: List of difficult words found, each with explanations.
    """

    sentence: str
    difficult_words: List[DifficultWord] = field(default_factory=list)
