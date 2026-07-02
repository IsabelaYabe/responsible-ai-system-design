"""Unit tests for explanation.schemas."""

import pytest
from explanation.schemas import DifficultWord, ExplanationResult


def test_difficult_word_construction():
    word = DifficultWord(
        word="laureates",
        span=(32, 41),
        meaning_in_context="people who receive an award",
    )
    assert word.word == "laureates"
    assert word.span == (32, 41)
    assert word.meaning_in_context == "people who receive an award"


def test_explanation_result_defaults_to_empty_list():
    result = ExplanationResult(sentence="The cat sat on the mat.")
    assert result.difficult_words == []


def test_explanation_result_with_words():
    words = [
        DifficultWord(
            word="obfuscate",
            span=(0, 9),
            meaning_in_context="to make confusing",
        )
    ]
    result = ExplanationResult(sentence="Obfuscate the truth.", difficult_words=words)
    assert len(result.difficult_words) == 1
    assert result.difficult_words[0].word == "obfuscate"


def test_span_is_tuple_of_two_ints():
    word = DifficultWord(
        word="test",
        span=(0, 4),
        meaning_in_context="a trial",
    )
    assert isinstance(word.span, tuple)
    assert len(word.span) == 2
    assert all(isinstance(i, int) for i in word.span)
