"""Tests for local Edit Predictor difficult-word identification."""

from types import SimpleNamespace

import pytest

from explanation.inference import edit_predictor_identifier
from explanation.inference.edit_predictor_identifier import EditPredictorIdentifier


class _FakeModel:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _FakePredictor:
    def __init__(self, spans):
        self.model = _FakeModel()
        self.spans = spans
        self.calls = []

    def predict_difficult_spans_with_scores(
        self,
        sentences,
        max_length=128,
        device=None,
        threshold=None,
    ):
        self.calls.append({
            "sentences": sentences,
            "max_length": max_length,
            "device": device,
            "threshold": threshold,
        })
        return [self.spans]


def test_identifier_extracts_word_text_from_original_sentence(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    sentence = "A committee of the institute appoints the laureates for the Nobel Prize."
    start = sentence.index("laureates")
    end = start + len("laureates")
    fake_predictor = _FakePredictor([
        {"span": (start, end), "score": 0.91},
    ])

    monkeypatch.setattr(
        edit_predictor_identifier.EditPredictor,
        "from_checkpoint",
        lambda path: fake_predictor,
    )

    identifier = EditPredictorIdentifier(
        checkpoint_dir,
        max_length=64,
        device="cpu",
        threshold=0.5,
    )
    result = identifier.identify(sentence)

    assert result == [{"word": "laureates", "span": (start, end), "score": 0.91}]
    assert fake_predictor.model.device == "cpu"
    assert fake_predictor.calls == [{
        "sentences": [sentence],
        "max_length": 64,
        "device": "cpu",
        "threshold": 0.5,
    }]


def test_identifier_filters_invalid_spans(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    fake_predictor = _FakePredictor([
        {"span": (-1, -1), "score": 0.9},
        {"span": (0, 500), "score": 0.9},
        {"span": (2, 2), "score": 0.9},
        {"span": (0, 4), "score": 0.8},
    ])

    monkeypatch.setattr(
        edit_predictor_identifier.EditPredictor,
        "from_checkpoint",
        lambda path: fake_predictor,
    )

    identifier = EditPredictorIdentifier(checkpoint_dir)

    assert identifier.identify("word") == [{"word": "word", "span": (0, 4), "score": 0.8}]


def test_identifier_missing_checkpoint_raises_clear_error(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Edit Predictor checkpoint not found"):
        EditPredictorIdentifier(missing)


def test_identifier_rejects_invalid_threshold(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    with pytest.raises(ValueError, match="threshold"):
        EditPredictorIdentifier(checkpoint_dir, threshold=1.5)
