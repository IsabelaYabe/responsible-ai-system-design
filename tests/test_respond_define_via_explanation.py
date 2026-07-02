"""Tests for antispoiler.respond._define_via_explanation (the Define -> explanation bridge).

No real checkpoint or LLM call happens here: SelectionExplainer is a fake double
whose .explain() is asserted against directly, so these tests only exercise the
bridge's own logic (in-bounds filtering, text normalization, chunk matching,
JSON shaping) -- not the Edit Predictor or OpenRouter.
"""

import json

from antispoiler.book import Chunk
from antispoiler.respond import _define_via_explanation, _normalize_book_text
from explanation.schemas import DifficultWord, ExplanationResult


def _chunk(chunk_id, chapter_index, text, paragraph_index=1, chapter_label=None):
    return Chunk(
        chunk_id=chunk_id,
        chapter_index=chapter_index,
        chapter_label=chapter_label or f"Chapter {chapter_index}",
        paragraph_index=paragraph_index,
        text=text,
    )


class _FakeSelectionExplainer:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def explain(self, text, selection):
        self.calls.append((text, selection))
        return self._result


def test_normalize_book_text_collapses_wraps_but_keeps_paragraph_breaks():
    raw = "It is a truth universally\r\nacknowledged, that a man\r\nneeds a wife.\n\nSo it goes."
    normalized = _normalize_book_text(raw)

    assert normalized == (
        "It is a truth universally acknowledged, that a man needs a wife.\n\nSo it goes."
    )


def test_returns_definitions_from_matching_in_bounds_chunk():
    chunk = _chunk("ch01_p01", chapter_index=1, text="The laureates\r\nsmiled warmly.")
    explainer = _FakeSelectionExplainer(
        ExplanationResult(
            sentence="The laureates smiled warmly.",
            difficult_words=[
                DifficultWord(word="laureates", span=(4, 13), meaning_in_context="award winners")
            ],
        )
    )

    class _Index:
        chunks = [chunk]

    answer, chunks, entity = _define_via_explanation(explainer, _Index(), "laureates", pos=3)

    assert entity is None
    assert chunks == [chunk]
    assert json.loads(answer) == {
        "meaning": "",
        "definitions": [{"word": "laureates", "definition": "award winners"}],
    }
    # SelectionExplainer must see whitespace-normalized chunk text, not the raw \r\n.
    assert explainer.calls == [("The laureates smiled warmly.", "laureates")]


def test_out_of_bounds_only_match_falls_back_to_explaining_without_context():
    in_bounds = _chunk("ch01_p01", chapter_index=1, text="A plain sentence.")
    out_of_bounds = _chunk("ch05_p01", chapter_index=5, text="The laureates smiled.")
    explainer = _FakeSelectionExplainer(
        ExplanationResult(
            sentence="laureates",
            difficult_words=[
                DifficultWord(word="laureates", span=(0, 9), meaning_in_context="award winners")
            ],
        )
    )

    class _Index:
        chunks = [in_bounds, out_of_bounds]

    answer, chunks, entity = _define_via_explanation(explainer, _Index(), "laureates", pos=3)

    # The only chunk containing "laureates" is out of bounds, so no chunk is
    # used as evidence -- but the selection is still explained, on its own.
    assert chunks == []
    assert entity is None
    assert json.loads(answer) == {
        "meaning": "",
        "definitions": [{"word": "laureates", "definition": "award winners"}],
    }
    # No surrounding paragraph: the selection is both the text and the selection.
    assert explainer.calls == [("laureates", "laureates")]


def test_falls_back_to_explaining_without_context_when_selection_not_found_anywhere():
    chunk = _chunk("ch01_p01", chapter_index=1, text="A plain sentence.")
    explainer = _FakeSelectionExplainer(
        ExplanationResult(
            sentence="unprecedented",
            difficult_words=[
                DifficultWord(
                    word="unprecedented", span=(0, 13), meaning_in_context="never done before"
                )
            ],
        )
    )

    class _Index:
        chunks = [chunk]

    answer, chunks, entity = _define_via_explanation(
        explainer, _Index(), "unprecedented", pos=3
    )

    assert chunks == []
    assert entity is None
    assert json.loads(answer) == {
        "meaning": "",
        "definitions": [{"word": "unprecedented", "definition": "never done before"}],
    }
    assert explainer.calls == [("unprecedented", "unprecedented")]


def test_matches_selection_spanning_a_hard_line_wrap():
    chunk = _chunk(
        "ch01_p01", chapter_index=1,
        text="A committee of the institute\r\nappoints the laureates for the prize.",
    )
    explainer = _FakeSelectionExplainer(
        ExplanationResult(sentence="unused", difficult_words=[])
    )

    class _Index:
        chunks = [chunk]

    # Selection crosses where the raw text was hard-wrapped with \r\n.
    _define_via_explanation(explainer, _Index(), "institute\r\nappoints", pos=3)

    assert explainer.calls == [
        ("A committee of the institute appoints the laureates for the prize.",
         "institute appoints")
    ]
