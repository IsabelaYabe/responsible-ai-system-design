"""Unit tests for the maintained explanation pipeline."""

import json
from unittest.mock import MagicMock, patch

import pytest

from explanation.inference.pipeline import ExplanationPipeline, _coerce_span, _explanations_by_word
from explanation.schemas import DifficultWord, ExplanationResult


SENTENCE = (
    "A committee of the institute appoints the laureates "
    "for the Nobel Prize in Physiology or Medicine."
)

EXPLAIN_RESPONSE = {
    "sentence": SENTENCE,
    "difficult_words": [
        {
            "word": "laureates",
            "span": [0, 3],
            "meaning_in_context": "people who receive an important award",
        }
    ],
}


class _FakeIdentifier:
    instances = []
    detected = []

    def __init__(
        self,
        checkpoint_dir,
        max_length=128,
        device=None,
        threshold=None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_length = max_length
        self.device = device
        self.threshold = threshold
        _FakeIdentifier.instances.append(self)

    def identify(self, sentence):
        self.sentence = sentence
        return list(_FakeIdentifier.detected)


def _patch_identifier(monkeypatch, detected):
    _FakeIdentifier.instances = []
    _FakeIdentifier.detected = detected
    monkeypatch.setattr("explanation.inference.pipeline.EditPredictorIdentifier", _FakeIdentifier)


def test_pipeline_uses_local_identifier_then_explains(monkeypatch):
    _patch_identifier(
        monkeypatch,
        [{"word": "laureates", "span": (41, 50), "score": 0.9}],
    )
    mock_client = MagicMock()
    mock_client.explain_difficult_words.return_value = EXPLAIN_RESPONSE

    pipeline = ExplanationPipeline(
        edit_predictor_checkpoint="checkpoint",
        client=mock_client,
        edit_predictor_threshold=0.5,
        max_length=64,
        device="cpu",
    )
    result = pipeline.run(SENTENCE)

    assert isinstance(result, ExplanationResult)
    assert result.difficult_words == [
        DifficultWord(
            word="laureates",
            span=(41, 50),
            meaning_in_context="people who receive an important award",
        )
    ]
    mock_client.explain_difficult_words.assert_called_once_with(SENTENCE, ["laureates"])
    assert not hasattr(mock_client, "identify_difficult_words") or not mock_client.identify_difficult_words.called
    assert _FakeIdentifier.instances[0].checkpoint_dir == "checkpoint"
    assert _FakeIdentifier.instances[0].max_length == 64
    assert _FakeIdentifier.instances[0].device == "cpu"
    assert _FakeIdentifier.instances[0].threshold == 0.5


def test_pipeline_returns_empty_when_local_identifier_finds_nothing(monkeypatch):
    _patch_identifier(monkeypatch, [])
    mock_client = MagicMock()

    pipeline = ExplanationPipeline(edit_predictor_checkpoint="checkpoint", client=mock_client)
    result = pipeline.run("The cat sat on the mat.")

    assert result == ExplanationResult(sentence="The cat sat on the mat.", difficult_words=[])
    mock_client.explain_difficult_words.assert_not_called()


def test_pipeline_strips_blank_sentence_without_loading_llm(monkeypatch):
    _patch_identifier(monkeypatch, [])
    mock_client = MagicMock()

    pipeline = ExplanationPipeline(edit_predictor_checkpoint="checkpoint", client=mock_client)
    result = pipeline.run("   ")

    assert result == ExplanationResult(sentence="", difficult_words=[])
    mock_client.explain_difficult_words.assert_not_called()


def test_pipeline_preserves_local_span_when_llm_disagrees(monkeypatch):
    _patch_identifier(monkeypatch, [{"word": "laureates", "span": (41, 50)}])
    mock_client = MagicMock()
    mock_client.explain_difficult_words.return_value = EXPLAIN_RESPONSE

    pipeline = ExplanationPipeline(edit_predictor_checkpoint="checkpoint", client=mock_client)
    result = pipeline.run(SENTENCE)

    assert result.difficult_words[0].span == (41, 50)


def test_pipeline_preserves_local_word_when_llm_omits_it(monkeypatch):
    _patch_identifier(monkeypatch, [{"word": "laureates", "span": (41, 50)}])
    mock_client = MagicMock()
    mock_client.explain_difficult_words.return_value = {
        "sentence": SENTENCE,
        "difficult_words": [],
    }

    pipeline = ExplanationPipeline(edit_predictor_checkpoint="checkpoint", client=mock_client)
    result = pipeline.run(SENTENCE)

    assert result.difficult_words == [
        DifficultWord(word="laureates", span=(41, 50), meaning_in_context="")
    ]


def test_coerce_span_rejects_invalid_spans():
    with pytest.raises(ValueError, match="Invalid span"):
        _coerce_span((-1, 3))
    with pytest.raises(ValueError, match="Invalid span"):
        _coerce_span((5, 3))
    with pytest.raises(ValueError, match="Invalid span"):
        _coerce_span((1,))


def test_explanations_by_word_ignores_invalid_entries_and_duplicates():
    entries = [
        {"word": "laureates", "meaning_in_context": "award recipients"},
        {"word": "laureates", "meaning_in_context": "duplicate"},
        {"word": "", "meaning_in_context": "blank"},
        "not a dict",
    ]

    assert _explanations_by_word(entries) == {"laureates": "award recipients"}
    assert _explanations_by_word("not a list") == {}


def test_openrouter_client_explain_strips_markdown_fences():
    from explanation.llm.openrouter_client import OpenRouterClient

    payload = json.dumps({"sentence": "S", "difficult_words": []})
    fenced = f"```json\n{payload}\n```"

    with patch.object(OpenRouterClient, "__init__", lambda s, **kw: None):
        client = OpenRouterClient.__new__(OpenRouterClient)
        client._chat = MagicMock(return_value=fenced)

        result = client.explain_difficult_words("S", [])
        assert result["difficult_words"] == []


def test_openrouter_client_explain_raises_on_invalid_json():
    from explanation.llm.openrouter_client import OpenRouterClient, OpenRouterClientError

    with patch.object(OpenRouterClient, "__init__", lambda s, **kw: None):
        client = OpenRouterClient.__new__(OpenRouterClient)
        client._chat = MagicMock(return_value="not json at all")

        with pytest.raises(OpenRouterClientError, match="non-JSON"):
            client.explain_difficult_words("S", [])
