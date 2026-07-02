"""Tests for the EditPredictor token-classifier wrapper.

No real Hugging Face download happens here: `from_base_model` and
`from_checkpoint` are exercised against a fake `transformers` module injected
into `sys.modules`, and prediction/freeze tests use small fake model/tokenizer
doubles with real torch tensors.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from explanation.model.edit_predictor import (
    LABEL_IGNORE,
    LABEL_KEEP,
    LABEL_MASK,
    LABEL_NAMES,
    EditPredictor,
)


# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------


def test_label_constants_match_edit_predictor_dataset_convention():
    assert LABEL_KEEP == 0
    assert LABEL_MASK == 1
    assert LABEL_IGNORE == -100
    assert LABEL_NAMES == {LABEL_KEEP: "K", LABEL_MASK: "M"}


# ---------------------------------------------------------------------------
# from_base_model / from_checkpoint (mocked transformers)
# ---------------------------------------------------------------------------


def test_from_base_model_creates_correct_label_maps(monkeypatch):
    calls = {}

    class _FakeModelClass:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["model_args"] = (model_name, kwargs)
            return SimpleNamespace(kind="model")

    class _FakeTokenizerClass:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["tokenizer_args"] = (model_name, kwargs)
            return SimpleNamespace(kind="tokenizer")

    fake_module = types.SimpleNamespace(
        AutoModelForTokenClassification=_FakeModelClass,
        AutoTokenizer=_FakeTokenizerClass,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_module)

    predictor = EditPredictor.from_base_model("fake-model")

    assert predictor.label_names == {LABEL_KEEP: "K", LABEL_MASK: "M"}
    assert predictor.model.kind == "model"
    assert predictor.tokenizer.kind == "tokenizer"

    model_name, model_kwargs = calls["model_args"]
    assert model_name == "fake-model"
    assert model_kwargs["num_labels"] == 2
    assert model_kwargs["id2label"] == {LABEL_KEEP: "K", LABEL_MASK: "M"}
    assert model_kwargs["label2id"] == {"K": LABEL_KEEP, "M": LABEL_MASK}

    tokenizer_name, _ = calls["tokenizer_args"]
    assert tokenizer_name == "fake-model"


def test_from_checkpoint_calls_expected_hf_loader(monkeypatch, tmp_path):
    calls = {}
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    class _FakeModelClass:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["model_path"] = path
            calls["model_kwargs"] = kwargs
            return SimpleNamespace(kind="model")

    class _FakeTokenizerClass:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["tokenizer_path"] = path
            return SimpleNamespace(kind="tokenizer")

    fake_module = types.SimpleNamespace(
        AutoModelForTokenClassification=_FakeModelClass,
        AutoTokenizer=_FakeTokenizerClass,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_module)

    predictor = EditPredictor.from_checkpoint(checkpoint_dir)

    assert calls["model_path"] == str(checkpoint_dir)
    assert calls["tokenizer_path"] == str(checkpoint_dir)
    assert calls["model_kwargs"] == {}
    assert predictor.label_names == {LABEL_KEEP: "K", LABEL_MASK: "M"}


def test_from_checkpoint_freeze_true_freezes_model(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    class _FakeParameter:
        def __init__(self):
            self.requires_grad = True

    class _FakeModel:
        def __init__(self):
            self.eval_called = False
            self._params = [_FakeParameter(), _FakeParameter()]

        def eval(self):
            self.eval_called = True

        def parameters(self):
            return iter(self._params)

    class _FakeModelClass:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return _FakeModel()

    class _FakeTokenizerClass:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return SimpleNamespace(kind="tokenizer")

    fake_module = types.SimpleNamespace(
        AutoModelForTokenClassification=_FakeModelClass,
        AutoTokenizer=_FakeTokenizerClass,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_module)

    predictor = EditPredictor.from_checkpoint(checkpoint_dir, freeze=True)

    assert predictor.model.eval_called is True
    assert all(p.requires_grad is False for p in predictor.model._params)


# ---------------------------------------------------------------------------
# freeze() on a directly constructed instance
# ---------------------------------------------------------------------------


class _FakeParameter:
    def __init__(self):
        self.requires_grad = True


class _FakeModelForFreeze:
    def __init__(self, parameters):
        self._parameters = parameters
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def parameters(self):
        return iter(self._parameters)


def test_freeze_disables_gradients_and_sets_eval():
    params = [_FakeParameter(), _FakeParameter()]
    model = _FakeModelForFreeze(params)
    predictor = EditPredictor(model=model, tokenizer=None, label_names=dict(LABEL_NAMES))

    result = predictor.freeze()

    assert model.eval_called is True
    assert all(p.requires_grad is False for p in params)
    assert result is predictor


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


def test_save_calls_model_and_tokenizer_save_pretrained(tmp_path):
    calls = {}

    class _FakeModel:
        def save_pretrained(self, path):
            calls["model_path"] = Path(path)

    class _FakeTokenizer:
        def save_pretrained(self, path):
            calls["tokenizer_path"] = Path(path)

    predictor = EditPredictor(
        model=_FakeModel(), tokenizer=_FakeTokenizer(), label_names=dict(LABEL_NAMES)
    )
    output_dir = tmp_path / "out"

    predictor.save(output_dir)

    assert output_dir.exists()
    assert calls["model_path"] == output_dir
    assert calls["tokenizer_path"] == output_dir


# ---------------------------------------------------------------------------
# predict_token_labels
# ---------------------------------------------------------------------------


class _FakeEncoding(dict):
    def __init__(self, data, seq_ids_per_row):
        super().__init__(data)
        self._seq_ids_per_row = seq_ids_per_row

    def sequence_ids(self, row_index):
        return self._seq_ids_per_row[row_index]


class _FakeTokenizerForPredict:
    def __call__(
        self,
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors=None,
        return_offsets_mapping=False,
    ):
        batch_size = len(sentences)
        input_ids = torch.tensor([[101, 5, 6, 102]] * batch_size)
        attention_mask = torch.ones((batch_size, 4), dtype=torch.long)
        seq_ids = [[None, 0, 0, None] for _ in range(batch_size)]
        data = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_offsets_mapping:
            data["offset_mapping"] = torch.tensor(
                [[[0, 0], [0, 3], [4, 7], [0, 0]]] * batch_size
            )
        return _FakeEncoding(data, seq_ids)

    def convert_ids_to_tokens(self, ids):
        mapping = {101: "[CLS]", 102: "[SEP]", 5: "cat", 6: "sat"}
        return [mapping.get(int(i), str(int(i))) for i in ids]


class _FakeModelForPredict:
    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, 2))
        logits[:, 1, LABEL_KEEP] = 5.0
        logits[:, 2, LABEL_MASK] = 5.0
        return SimpleNamespace(logits=logits)


def test_predict_token_labels_maps_ids_to_km():
    predictor = EditPredictor(
        model=_FakeModelForPredict(),
        tokenizer=_FakeTokenizerForPredict(),
        label_names=dict(LABEL_NAMES),
    )

    predictions = predictor.predict_token_labels(["The cat sat."])

    assert len(predictions) == 1
    assert predictions[0] == [
        {"token": "cat", "label_id": LABEL_KEEP, "label": "K"},
        {"token": "sat", "label_id": LABEL_MASK, "label": "M"},
    ]


def test_predict_token_labels_excludes_padding():
    class _FakeTokenizerWithPadding(_FakeTokenizerForPredict):
        def __call__(self, sentences, **kwargs):
            encoded = super().__call__(sentences, **kwargs)
            # Second row's second real token ("sat") is padding.
            encoded["attention_mask"] = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
            encoded._seq_ids_per_row = [[None, 0, 0, None], [None, 0, 0, None]]
            return encoded

    predictor = EditPredictor(
        model=_FakeModelForPredict(),
        tokenizer=_FakeTokenizerWithPadding(),
        label_names=dict(LABEL_NAMES),
    )

    predictions = predictor.predict_token_labels(["The cat sat.", "Short."])

    assert len(predictions[1]) == 1
    assert predictions[1][0]["token"] == "cat"


# ---------------------------------------------------------------------------
# predict_difficult_spans
# ---------------------------------------------------------------------------


class _FakeTokenizerForSpans:
    def __call__(
        self,
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors=None,
        return_offsets_mapping=False,
    ):
        batch_size = len(sentences)
        input_ids = torch.tensor([[101, 5, 6, 7, 102]] * batch_size)
        attention_mask = torch.ones((batch_size, 5), dtype=torch.long)
        seq_ids = [[None, 0, 0, 0, None] for _ in range(batch_size)]
        offsets = torch.tensor([[[0, 0], [0, 2], [2, 9], [10, 13], [0, 0]]] * batch_size)
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offsets,
        }
        return _FakeEncoding(data, seq_ids)

    def convert_ids_to_tokens(self, ids):
        return [str(int(i)) for i in ids]


class _FakeModelForSpans:
    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros((batch_size, seq_len, 2))
        logits[:, 1, LABEL_MASK] = 5.0
        logits[:, 2, LABEL_MASK] = 5.0
        logits[:, 3, LABEL_KEEP] = 5.0
        return SimpleNamespace(logits=logits)


def test_predict_difficult_spans_merges_touching_subword_tokens():
    predictor = EditPredictor(
        model=_FakeModelForSpans(),
        tokenizer=_FakeTokenizerForSpans(),
        label_names=dict(LABEL_NAMES),
    )

    spans = predictor.predict_difficult_spans(["laureates smiled."])

    assert spans == [[(0, 9)]]


def test_predict_difficult_spans_with_scores_returns_average_and_max_score():
    predictor = EditPredictor(
        model=_FakeModelForSpans(),
        tokenizer=_FakeTokenizerForSpans(),
        label_names=dict(LABEL_NAMES),
    )

    rows = predictor.predict_difficult_spans_with_scores(["laureates smiled."])

    assert len(rows) == 1
    assert len(rows[0]) == 1
    span = rows[0][0]
    assert span["span"] == (0, 9)
    assert span["score"] == pytest.approx(span["average_score"])
    assert span["score"] > 0.9
    assert span["max_score"] > 0.9


def test_predict_difficult_spans_with_scores_respects_threshold():
    predictor = EditPredictor(
        model=_FakeModelForSpans(),
        tokenizer=_FakeTokenizerForSpans(),
        label_names=dict(LABEL_NAMES),
    )

    rows = predictor.predict_difficult_spans_with_scores(
        ["laureates smiled."],
        threshold=0.999,
    )

    assert rows == [[]]
