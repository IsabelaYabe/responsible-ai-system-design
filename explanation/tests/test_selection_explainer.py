"""Unit tests for SelectionExplainer (word / sentence / excerpt selection)."""

from unittest.mock import MagicMock

import pytest

from explanation.inference.selection_explainer import SelectionExplainer
from explanation.schemas import DifficultWord, ExplanationResult

PARA_A = "The committee reviews every application. It deliberates for several weeks before voting."
PARA_B = "The chairman announces the final decision. Journalists gather outside to report the result."
TEXT = PARA_A + "\n\n" + PARA_B

SENT_A1 = "The committee reviews every application."
SENT_A2 = "It deliberates for several weeks before voting."
SENT_B1 = "The chairman announces the final decision."
SENT_B2 = "Journalists gather outside to report the result."


class _FakeIdentifier:
    instances = []
    detected_by_sentence = {}

    def __init__(self, checkpoint_dir, max_length=128, device=None, threshold=None):
        self.checkpoint_dir = checkpoint_dir
        self.max_length = max_length
        self.device = device
        self.threshold = threshold
        self.calls = []
        _FakeIdentifier.instances.append(self)

    def identify(self, sentence):
        self.calls.append(sentence)
        return list(_FakeIdentifier.detected_by_sentence.get(sentence, []))


def _patch_identifier(monkeypatch, detected_by_sentence):
    _FakeIdentifier.instances = []
    _FakeIdentifier.detected_by_sentence = detected_by_sentence
    monkeypatch.setattr(
        "explanation.inference.selection_explainer.EditPredictorIdentifier", _FakeIdentifier
    )


def _mock_client(explain_response):
    mock_client = MagicMock()
    mock_client.explain_difficult_words.return_value = explain_response
    return mock_client


def test_single_word_selection_is_scoped_to_its_own_paragraph(monkeypatch):
    _patch_identifier(monkeypatch, {})
    mock_client = _mock_client({
        "difficult_words": [
            {"word": "chairman", "span": [0, 0], "meaning_in_context": "the head of a committee"}
        ]
    })

    explainer = SelectionExplainer(edit_predictor_checkpoint="checkpoint", client=mock_client)
    result = explainer.explain(TEXT, "chairman")

    # Identification never runs for a single word.
    assert _FakeIdentifier.instances[0].calls == []
    # Only paragraph B (which contains "chairman") is sent as context -- not
    # the whole two-paragraph text.
    mock_client.explain_difficult_words.assert_called_once_with(PARA_B, ["chairman"])
    start = PARA_B.index("chairman")
    assert result == ExplanationResult(
        sentence=PARA_B,
        difficult_words=[
            DifficultWord(
                word="chairman",
                span=(start, start + len("chairman")),
                meaning_in_context="the head of a committee",
            )
        ],
    )


def test_full_sentence_selection_is_scoped_to_its_own_paragraph(monkeypatch):
    _patch_identifier(monkeypatch, {SENT_A2: [{"word": "deliberates", "span": (3, 14)}]})
    mock_client = _mock_client({
        "difficult_words": [
            {"word": "deliberates", "span": [0, 0], "meaning_in_context": "discusses at length"}
        ]
    })

    explainer = SelectionExplainer(edit_predictor_checkpoint="checkpoint", client=mock_client)
    result = explainer.explain(TEXT, SENT_A2)

    assert _FakeIdentifier.instances[0].calls == [SENT_A2]
    # Only paragraph A (which contains this sentence) is sent as context.
    mock_client.explain_difficult_words.assert_called_once_with(PARA_A, ["deliberates"])
    start = PARA_A.index("deliberates")
    assert result.sentence == PARA_A
    assert result.difficult_words == [
        DifficultWord(
            word="deliberates",
            span=(start, start + len("deliberates")),
            meaning_in_context="discusses at length",
        )
    ]


def test_excerpt_within_one_paragraph_interpolates_and_filters_to_selection(monkeypatch):
    _patch_identifier(
        monkeypatch,
        {
            SENT_A1: [{"word": "committee", "span": (4, 13)}],
            SENT_A2: [{"word": "deliberates", "span": (3, 14)}],
        },
    )
    mock_client = _mock_client({
        "difficult_words": [
            {"word": "deliberates", "span": [0, 0], "meaning_in_context": "discusses at length"}
        ]
    })

    # Cuts into the tail of sentence A1 and the head of sentence A2; neither
    # is fully selected.
    excerpt_start = PARA_A.index("application")
    excerpt_end = PARA_A.index("deliberates") + len("deliberates")
    excerpt = PARA_A[excerpt_start:excerpt_end]

    explainer = SelectionExplainer(edit_predictor_checkpoint="checkpoint", client=mock_client)
    result = explainer.explain(TEXT, excerpt)

    assert set(_FakeIdentifier.instances[0].calls) == {SENT_A1, SENT_A2}
    # "committee" (sentence A1) falls outside the excerpt and must be dropped.
    mock_client.explain_difficult_words.assert_called_once_with(PARA_A, ["deliberates"])
    start = PARA_A.index("deliberates")
    assert result.difficult_words == [
        DifficultWord(
            word="deliberates",
            span=(start, start + len("deliberates")),
            meaning_in_context="discusses at length",
        )
    ]


def test_excerpt_crossing_paragraph_boundary_uses_both_paragraphs_as_context(monkeypatch):
    _patch_identifier(
        monkeypatch,
        {
            SENT_A2: [{"word": "deliberates", "span": (3, 14)}],
            SENT_B1: [{"word": "chairman", "span": (4, 12)}],
        },
    )
    mock_client = _mock_client({
        "difficult_words": [
            {"word": "deliberates", "span": [0, 0], "meaning_in_context": "discusses at length"},
            {"word": "chairman", "span": [0, 0], "meaning_in_context": "the head of a committee"},
        ]
    })

    # Starts inside sentence A2 (paragraph A) and ends inside sentence B1
    # (paragraph B) -- crosses the paragraph break entirely.
    excerpt_start = TEXT.index("deliberates")
    excerpt_end = TEXT.index("announces") + len("announces")
    excerpt = TEXT[excerpt_start:excerpt_end]
    assert "\n\n" in excerpt  # sanity check: it really crosses the paragraph break

    explainer = SelectionExplainer(edit_predictor_checkpoint="checkpoint", client=mock_client)
    result = explainer.explain(TEXT, excerpt)

    assert set(_FakeIdentifier.instances[0].calls) == {SENT_A2, SENT_B1}

    expected_context = PARA_A + "\n\n" + PARA_B
    mock_client.explain_difficult_words.assert_called_once_with(
        expected_context, ["deliberates", "chairman"]
    )
    assert result.sentence == expected_context

    deliberates_start = expected_context.index("deliberates")
    chairman_start = expected_context.index("chairman")
    assert result.difficult_words == [
        DifficultWord(
            word="deliberates",
            span=(deliberates_start, deliberates_start + len("deliberates")),
            meaning_in_context="discusses at length",
        ),
        DifficultWord(
            word="chairman",
            span=(chairman_start, chairman_start + len("chairman")),
            meaning_in_context="the head of a committee",
        ),
    ]


def test_returns_empty_when_no_difficult_words_found_in_touched_sentences(monkeypatch):
    _patch_identifier(monkeypatch, {SENT_A1: [], SENT_A2: []})
    mock_client = _mock_client({})

    explainer = SelectionExplainer(edit_predictor_checkpoint="checkpoint", client=mock_client)
    result = explainer.explain(TEXT, PARA_A)

    assert result == ExplanationResult(sentence=PARA_A, difficult_words=[])
    mock_client.explain_difficult_words.assert_not_called()


def test_blank_selection_returns_empty_without_calling_llm(monkeypatch):
    _patch_identifier(monkeypatch, {})
    mock_client = _mock_client({})

    explainer = SelectionExplainer(edit_predictor_checkpoint="checkpoint", client=mock_client)
    result = explainer.explain(TEXT, "   ")

    assert result == ExplanationResult(sentence="", difficult_words=[])
    mock_client.explain_difficult_words.assert_not_called()


def test_selection_not_found_raises_value_error(monkeypatch):
    _patch_identifier(monkeypatch, {})
    mock_client = _mock_client({})

    explainer = SelectionExplainer(edit_predictor_checkpoint="checkpoint", client=mock_client)
    with pytest.raises(ValueError, match="not found"):
        explainer.explain(TEXT, "the dog")


def test_selection_span_overrides_ambiguous_find(monkeypatch):
    text = "The cat sat. The cat slept."
    _patch_identifier(monkeypatch, {})
    mock_client = _mock_client({
        "difficult_words": [{"word": "cat", "span": [0, 0], "meaning_in_context": "a feline"}]
    })

    explainer = SelectionExplainer(edit_predictor_checkpoint="checkpoint", client=mock_client)
    second_cat_start = text.rindex("cat")
    result = explainer.explain(
        text, "cat", selection_span=(second_cat_start, second_cat_start + 3)
    )

    assert result.difficult_words[0].span == (second_cat_start, second_cat_start + 3)


def test_constructor_forwards_edit_predictor_options(monkeypatch):
    _patch_identifier(monkeypatch, {})
    mock_client = _mock_client({})

    SelectionExplainer(
        edit_predictor_checkpoint="checkpoint",
        client=mock_client,
        edit_predictor_threshold=0.5,
        max_length=64,
        device="cpu",
    )

    assert _FakeIdentifier.instances[0].checkpoint_dir == "checkpoint"
    assert _FakeIdentifier.instances[0].max_length == 64
    assert _FakeIdentifier.instances[0].device == "cpu"
    assert _FakeIdentifier.instances[0].threshold == 0.5
