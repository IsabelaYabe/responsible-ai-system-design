"""Tests for the supervised Edit Predictor manager CLI."""

from types import SimpleNamespace

import pytest

from explanation.model import manage_edit_predictor


def test_build_parser_defaults_to_reference_checkpoint():
    args = manage_edit_predictor.build_parser().parse_args([])

    assert args.mode is None
    assert args.dataset_file.endswith(
        "edit_predictor_token_labels_complex_words_distilbert_max256.pt"
    )
    assert args.checkpoint_dir.endswith(
        "edit_predictor_complex_words_distilbert_max256_v1"
    )
    assert args.output_dir.endswith(
        "edit_predictor_complex_words_distilbert_max256_v1"
    )
    assert args.model_name == "distilbert-base-uncased"
    assert args.epochs == 10
    assert args.batch_size == 16
    assert args.use_class_weights is False


def test_choose_mode_interactively_maps_choices():
    assert manage_edit_predictor.choose_mode_interactively(lambda _: "1") == "train"
    assert manage_edit_predictor.choose_mode_interactively(lambda _: "2") == "checkpoint"
    assert manage_edit_predictor.choose_mode_interactively(lambda _: "3") == "continue"
    assert manage_edit_predictor.choose_mode_interactively(lambda _: "4") == "predict"


def test_choose_mode_interactively_rejects_invalid_choice():
    with pytest.raises(ValueError, match="Invalid choice"):
        manage_edit_predictor.choose_mode_interactively(lambda _: "x")


def test_run_train_mode_calls_supervised_train_function(monkeypatch, tmp_path):
    dataset_file = tmp_path / "dataset.pt"
    dataset_file.write_bytes(b"not-a-real-torch-file")
    output_dir = tmp_path / "out"
    calls = {}

    expected_result = SimpleNamespace(
        best_epoch=1,
        best_validation_f1_M=0.5,
        test_metrics={"f1_M": 0.4, "token_accuracy": 0.8},
        baseline_all_keep={"f1_M": 0.0},
    )

    def fake_train_edit_predictor(**kwargs):
        calls.update(kwargs)
        return expected_result

    monkeypatch.setattr(
        manage_edit_predictor, "train_edit_predictor", fake_train_edit_predictor
    )

    args = manage_edit_predictor.build_parser().parse_args([
        "--mode", "train",
        "--dataset-file", str(dataset_file),
        "--output-dir", str(output_dir),
        "--epochs", "2",
        "--batch-size", "4",
        "--use-class-weights",
    ])

    result = manage_edit_predictor.run_train_mode(args)

    assert result is expected_result
    assert calls["dataset_file"] == dataset_file
    assert calls["output_dir"] == output_dir
    assert calls["load_mode"] == "base"
    assert calls["epochs"] == 2
    assert calls["batch_size"] == 4
    assert calls["use_class_weights"] is True


def test_run_continue_mode_requires_checkpoint_and_uses_checkpoint_load(
    monkeypatch, tmp_path
):
    dataset_file = tmp_path / "dataset.pt"
    dataset_file.write_bytes(b"not-a-real-torch-file")
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    output_dir = tmp_path / "continued"
    calls = {}

    expected_result = SimpleNamespace(
        best_epoch=1,
        best_validation_f1_M=0.5,
        test_metrics={"f1_M": 0.4, "token_accuracy": 0.8},
        baseline_all_keep={"f1_M": 0.0},
    )

    def fake_train_edit_predictor(**kwargs):
        calls.update(kwargs)
        return expected_result

    monkeypatch.setattr(
        manage_edit_predictor, "train_edit_predictor", fake_train_edit_predictor
    )

    args = manage_edit_predictor.build_parser().parse_args([
        "--mode", "continue",
        "--dataset-file", str(dataset_file),
        "--checkpoint-dir", str(checkpoint_dir),
        "--output-dir", str(output_dir),
    ])

    result = manage_edit_predictor.run_continue_mode(args)

    assert result is expected_result
    assert calls["dataset_file"] == dataset_file
    assert calls["checkpoint_dir"] == checkpoint_dir
    assert calls["output_dir"] == output_dir
    assert calls["load_mode"] == "checkpoint"


def test_run_predict_mode_uses_checkpoint_and_sentences(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    calls = {}
    fake_predictions = [[{"token": "word", "label": "M"}]]

    class FakePredictor:
        def predict_token_labels(self, sentences, max_length):
            calls["sentences"] = sentences
            calls["max_length"] = max_length
            return fake_predictions

    class FakeEditPredictor:
        @staticmethod
        def from_checkpoint(path):
            calls["checkpoint"] = path
            return FakePredictor()

    monkeypatch.setattr(manage_edit_predictor, "EditPredictor", FakeEditPredictor)

    args = manage_edit_predictor.build_parser().parse_args([
        "--mode", "predict",
        "--checkpoint-dir", str(checkpoint_dir),
        "--sentence", "A complex sentence.",
        "--max-length", "32",
    ])

    result = manage_edit_predictor.run_predict_mode(args)

    assert result == fake_predictions
    assert calls["checkpoint"] == checkpoint_dir
    assert calls["sentences"] == ["A complex sentence."]
    assert calls["max_length"] == 32


def test_format_equivalent_command_quotes_predict_sentences():
    args = manage_edit_predictor.build_parser().parse_args([
        "--mode", "predict",
        "--checkpoint-dir", "some checkpoint",
        "--sentence", "A complex sentence.",
    ])

    command = manage_edit_predictor.format_equivalent_command(args, "predict")

    assert "explanation.model.manage_edit_predictor" in command
    assert "--mode predict" in command
    assert "'some checkpoint'" in command
    assert "'A complex sentence.'" in command


def test_require_file_reports_missing_path(tmp_path):
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="Missing token-level dataset file"):
        manage_edit_predictor._require_file(str(missing), "token-level dataset")


def test_require_checkpoint_dir_reports_missing_path(tmp_path):
    missing = tmp_path / "missing-checkpoint"

    with pytest.raises(FileNotFoundError, match="Missing Edit Predictor checkpoint"):
        manage_edit_predictor._require_checkpoint_dir(str(missing))
