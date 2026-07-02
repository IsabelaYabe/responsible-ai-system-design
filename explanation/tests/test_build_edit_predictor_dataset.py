"""Tests for the Edit Predictor token dataset build CLI helpers."""

import json
import pickle
import re

from explanation.scripts.build_edit_predictor_dataset import (
    build_edit_predictor_dataset_artifact,
    build_parser,
)


class FakeTensor:
    def __init__(self, data=None, shape=None, dtype=None):
        self.data = data
        self.shape = shape
        self.dtype = dtype


class FakeTorch:
    long = "long"

    def tensor(self, data, dtype=None):
        return FakeTensor(data=data, dtype=dtype)

    def empty(self, shape, dtype=None):
        return FakeTensor(data=[], shape=shape, dtype=dtype)

    def save(self, obj, path):
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)


class FakeTokenizer:
    def __call__(
        self,
        sentences,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=128,
    ):
        input_ids = []
        attention_mask = []
        offset_mapping = []
        sequence_ids = []

        for sentence in sentences:
            offsets = [match.span() for match in re.finditer(r"\S+", sentence)]
            offsets = offsets[: max(0, max_length - 2)]
            ids = [101] + list(range(2000, 2000 + len(offsets))) + [102]
            masks = [1] * len(ids)
            seq_ids = [None] + [0] * len(offsets) + [None]
            token_offsets = [(0, 0)] + offsets + [(0, 0)]

            pad = max_length - len(ids)
            ids.extend([0] * pad)
            masks.extend([0] * pad)
            seq_ids.extend([None] * pad)
            token_offsets.extend([(0, 0)] * pad)

            input_ids.append(ids)
            attention_mask.append(masks)
            offset_mapping.append(token_offsets)
            sequence_ids.append(seq_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
            "sequence_ids": sequence_ids,
        }


def test_build_parser_defaults():
    args = build_parser().parse_args([])

    assert args.input_file.endswith("complex_words_pseudo_labels.jsonl")
    assert args.output_file.endswith(
        "edit_predictor_token_labels_complex_words_distilbert_max256.pt"
    )
    assert args.model_name == "distilbert-base-uncased"
    assert args.max_length == 256
    assert args.train_ratio == 0.8
    assert args.validation_ratio == 0.1
    assert args.test_ratio == 0.1
    assert args.seed == 13
    assert args.keep_invalid_rows is False
    assert args.keep_no_difficult_rows is False


def test_build_edit_predictor_dataset_artifact_writes_pt_file(tmp_path):
    input_file = tmp_path / "pseudo_labels.jsonl"
    output_file = tmp_path / "token_labels.pt"
    rows = [
        {
            "sentence": "The laureates smiled.",
            "difficult_words": ["laureates"],
            "spans": [[4, 13]],
            "source": "openrouter",
            "model": "m",
        },
        {
            "sentence": "A plain sentence.",
            "difficult_words": [],
            "spans": [],
            "source": "openrouter",
            "model": "m",
        },
        {
            "sentence": "Another plain sentence.",
            "difficult_words": [],
            "spans": [],
            "source": "groq",
            "model": "m",
        },
        {
            "sentence": "Bad span here.",
            "difficult_words": ["missing"],
            "spans": [[-1, -1]],
            "source": "groq",
            "model": "m",
        },
    ]
    input_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    artifact = build_edit_predictor_dataset_artifact(
        input_file=input_file,
        output_file=output_file,
        tokenizer=FakeTokenizer(),
        model_name="fake-model",
        max_length=8,
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
        seed=13,
        torch_module=FakeTorch(),
    )

    assert output_file.exists()
    with output_file.open("rb") as fh:
        saved = pickle.load(fh)

    assert saved.keys() == {"train", "validation", "test", "metadata"}
    assert artifact["metadata"] == saved["metadata"]
    assert saved["metadata"]["input_file"] == str(input_file)
    assert saved["metadata"]["model_name"] == "fake-model"
    assert saved["metadata"]["max_length"] == 8
    assert saved["metadata"]["rows_read"] == 4
    assert saved["metadata"]["rows_written"] == 1
    assert saved["metadata"]["invalid_spans_ignored"] == 1
    assert saved["metadata"]["examples_no_difficult_words"] == 2
    assert saved["metadata"]["train_examples"] == 0
    assert saved["metadata"]["validation_examples"] == 0
    assert saved["metadata"]["test_examples"] == 1
    assert "input_ids" in saved["train"]
    assert "attention_mask" in saved["train"]
    assert "labels" in saved["train"]
    assert "sentences" in saved["train"]
