"""Tests for Edit Predictor training helpers.

Heavy training (torch, transformers) is not exercised here -- only the
pure-Python helpers (parser, metrics, class weights, checkpoint selection,
CSV writer) that work without any model download or GPU.
"""

import csv

import pytest

from explanation.model import train_edit_predictor as train_module
from explanation.model.edit_predictor import LABEL_IGNORE, LABEL_KEEP, LABEL_MASK
from explanation.model.train_edit_predictor import (
    HISTORY_FIELDNAMES,
    build_parser,
    compute_all_keep_baseline,
    compute_class_weights,
    compute_token_metrics,
    load_token_dataset,
    select_best_epoch,
    _write_history,
)


# ---------------------------------------------------------------------------
# load_token_dataset
# ---------------------------------------------------------------------------


def test_load_token_dataset_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="Token dataset not found"):
        load_token_dataset(missing)


# ---------------------------------------------------------------------------
# compute_token_metrics
# ---------------------------------------------------------------------------


def test_compute_token_metrics_ignores_label_ignore_positions():
    true_with_ignore = [LABEL_KEEP, LABEL_MASK, LABEL_IGNORE, LABEL_KEEP]
    pred_with_ignore = [LABEL_KEEP, LABEL_MASK, LABEL_MASK, LABEL_KEEP]

    true_without_ignore = [LABEL_KEEP, LABEL_MASK, LABEL_KEEP]
    pred_without_ignore = [LABEL_KEEP, LABEL_MASK, LABEL_KEEP]

    metrics_with = compute_token_metrics(true_with_ignore, pred_with_ignore)
    metrics_without = compute_token_metrics(true_without_ignore, pred_without_ignore)

    assert metrics_with == metrics_without
    assert metrics_with["gold_M_tokens"] == 1
    assert metrics_with["predicted_M_tokens"] == 1


def test_compute_token_metrics_empty_returns_zeros():
    metrics = compute_token_metrics([], [])

    assert metrics["token_accuracy"] == 0.0
    assert metrics["f1_M"] == 0.0
    assert metrics["gold_M_tokens"] == 0


def test_compute_token_metrics_all_ignore_returns_zeros():
    metrics = compute_token_metrics([LABEL_IGNORE, LABEL_IGNORE], [LABEL_KEEP, LABEL_MASK])

    assert metrics["token_accuracy"] == 0.0
    assert metrics["gold_M_tokens"] == 0
    assert metrics["predicted_M_tokens"] == 0


def test_f1_m_positive_when_model_predicts_m_correctly():
    true = [LABEL_KEEP, LABEL_MASK, LABEL_MASK, LABEL_KEEP]
    pred = [LABEL_KEEP, LABEL_MASK, LABEL_MASK, LABEL_KEEP]

    metrics = compute_token_metrics(true, pred)

    assert metrics["f1_M"] == 1.0
    assert metrics["precision_M"] == 1.0
    assert metrics["recall_M"] == 1.0


def test_f1_m_zero_when_model_predicts_only_keep():
    true = [LABEL_KEEP, LABEL_MASK, LABEL_MASK, LABEL_KEEP]
    pred = [LABEL_KEEP, LABEL_KEEP, LABEL_KEEP, LABEL_KEEP]

    metrics = compute_token_metrics(true, pred)

    assert metrics["f1_M"] == 0.0
    assert metrics["recall_M"] == 0.0
    assert metrics["predicted_M_tokens"] == 0


def test_compute_token_metrics_includes_loss_when_provided():
    metrics = compute_token_metrics([LABEL_KEEP], [LABEL_KEEP], loss=0.1234567)

    assert metrics["loss"] == pytest.approx(0.123457)


def test_compute_token_metrics_omits_loss_when_not_provided():
    metrics = compute_token_metrics([LABEL_KEEP], [LABEL_KEEP])

    assert "loss" not in metrics


# ---------------------------------------------------------------------------
# compute_all_keep_baseline
# ---------------------------------------------------------------------------


def test_compute_all_keep_baseline_matches_manual_all_keep_predictions():
    true = [LABEL_KEEP, LABEL_MASK, LABEL_MASK, LABEL_KEEP, LABEL_IGNORE]

    baseline = compute_all_keep_baseline(true)
    manual = compute_token_metrics(true, [LABEL_KEEP] * len(true))

    assert baseline == manual
    assert baseline["f1_M"] == 0.0
    assert baseline["gold_M_tokens"] == 2


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------


def test_compute_class_weights_excludes_ignore_label():
    labels_with_ignore = [
        LABEL_KEEP, LABEL_KEEP, LABEL_KEEP, LABEL_MASK, LABEL_IGNORE, LABEL_IGNORE,
    ]
    labels_without_ignore = [LABEL_KEEP, LABEL_KEEP, LABEL_KEEP, LABEL_MASK]

    weights_with = compute_class_weights(labels_with_ignore)
    weights_without = compute_class_weights(labels_without_ignore)

    assert weights_with == weights_without
    assert weights_with[0] == pytest.approx(4 / 6)
    assert weights_with[1] == pytest.approx(2.0)


def test_compute_class_weights_mask_override_keeps_computed_keep_weight():
    labels = [LABEL_KEEP, LABEL_KEEP, LABEL_MASK]

    default_weights = compute_class_weights(labels)
    overridden_weights = compute_class_weights(labels, mask_class_weight=10.0)

    assert overridden_weights[1] == pytest.approx(10.0)
    assert overridden_weights[0] == pytest.approx(default_weights[0])


def test_compute_class_weights_empty_returns_ones():
    assert compute_class_weights([]) == [1.0, 1.0]
    assert compute_class_weights([LABEL_IGNORE, LABEL_IGNORE]) == [1.0, 1.0]


# ---------------------------------------------------------------------------
# select_best_epoch
# ---------------------------------------------------------------------------


def test_select_best_epoch_picks_highest_f1_m():
    history = [
        {"epoch": 1, "validation_f1_M": 0.1},
        {"epoch": 2, "validation_f1_M": 0.5},
        {"epoch": 3, "validation_f1_M": 0.3},
    ]

    best_epoch, best_f1_m = select_best_epoch(history)

    assert best_epoch == 2
    assert best_f1_m == pytest.approx(0.5)


def test_select_best_epoch_respects_min_delta():
    history = [
        {"epoch": 1, "validation_f1_M": 0.500},
        {"epoch": 2, "validation_f1_M": 0.5005},
        {"epoch": 3, "validation_f1_M": 0.503},
    ]

    best_epoch, best_f1_m = select_best_epoch(history, min_delta=0.001)

    assert best_epoch == 3
    assert best_f1_m == pytest.approx(0.503)


def test_select_best_epoch_empty_history():
    best_epoch, best_f1_m = select_best_epoch([])

    assert best_epoch == 0
    assert best_f1_m == float("-inf")


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------


def test_build_parser_defaults_match_reference_command():
    args = build_parser().parse_args([])

    assert args.dataset_file.endswith(
        "edit_predictor_token_labels_complex_words_distilbert_max256.pt"
    )
    assert args.output_dir.endswith(
        "edit_predictor_complex_words_distilbert_max256_v1"
    )
    assert args.model_name == "distilbert-base-uncased"
    assert args.load_mode == "base"
    assert args.checkpoint_dir is None
    assert args.epochs == 10
    assert args.batch_size == 16
    assert args.learning_rate == pytest.approx(3e-5)
    assert args.weight_decay == pytest.approx(0.01)
    assert args.early_stopping_patience == 2
    assert args.early_stopping_min_delta == pytest.approx(0.001)
    assert args.seed == 13
    assert args.use_class_weights is False
    assert args.mask_class_weight is None


def test_build_parser_accepts_class_weight_options():
    args = build_parser().parse_args([
        "--use-class-weights",
        "--mask-class-weight", "5.0",
    ])

    assert args.use_class_weights is True
    assert args.mask_class_weight == pytest.approx(5.0)


def test_build_parser_accepts_checkpoint_load_mode():
    args = build_parser().parse_args([
        "--load-mode", "checkpoint",
        "--checkpoint-dir", "some/checkpoint",
    ])

    assert args.load_mode == "checkpoint"
    assert args.checkpoint_dir == "some/checkpoint"


# ---------------------------------------------------------------------------
# _write_history
# ---------------------------------------------------------------------------


def test_write_history_creates_file_with_expected_columns(tmp_path):
    history = [{
        "epoch": 1,
        "train_loss": 0.5,
        "validation_loss": 0.4,
        "validation_token_accuracy": 0.9,
        "validation_precision_M": 0.6,
        "validation_recall_M": 0.5,
        "validation_f1_M": 0.545,
        "validation_precision_K": 0.95,
        "validation_recall_K": 0.97,
        "validation_f1_K": 0.96,
        "validation_macro_f1": 0.75,
    }]

    _write_history(tmp_path, history)

    path = tmp_path / "history.csv"
    assert path.exists()
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert set(rows[0].keys()) == set(HISTORY_FIELDNAMES)
    assert rows[0]["epoch"] == "1"


def test_write_history_empty_list_writes_header_only(tmp_path):
    _write_history(tmp_path, [])

    with open(tmp_path / "history.csv", encoding="utf-8") as fh:
        content = fh.read()

    assert "validation_f1_M" in content
    assert len(content.strip().splitlines()) == 1


# ---------------------------------------------------------------------------
# train_edit_predictor argument validation (before dataset load / torch import)
# ---------------------------------------------------------------------------


def test_train_edit_predictor_rejects_invalid_epochs_before_dataset_load(tmp_path):
    with pytest.raises(ValueError, match="epochs"):
        train_module.train_edit_predictor(
            dataset_file=tmp_path / "missing.pt",
            output_dir=tmp_path / "out",
            epochs=0,
        )


def test_train_edit_predictor_rejects_invalid_batch_size_before_dataset_load(tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        train_module.train_edit_predictor(
            dataset_file=tmp_path / "missing.pt",
            output_dir=tmp_path / "out",
            batch_size=0,
        )


def test_train_edit_predictor_rejects_negative_weight_decay_before_dataset_load(tmp_path):
    with pytest.raises(ValueError, match="weight_decay"):
        train_module.train_edit_predictor(
            dataset_file=tmp_path / "missing.pt",
            output_dir=tmp_path / "out",
            weight_decay=-0.01,
        )


def test_train_edit_predictor_rejects_negative_early_stopping_patience_before_dataset_load(tmp_path):
    with pytest.raises(ValueError, match="early_stopping_patience"):
        train_module.train_edit_predictor(
            dataset_file=tmp_path / "missing.pt",
            output_dir=tmp_path / "out",
            early_stopping_patience=-1,
        )


def test_train_edit_predictor_rejects_negative_early_stopping_min_delta_before_dataset_load(tmp_path):
    with pytest.raises(ValueError, match="early_stopping_min_delta"):
        train_module.train_edit_predictor(
            dataset_file=tmp_path / "missing.pt",
            output_dir=tmp_path / "out",
            early_stopping_min_delta=-0.001,
        )


def test_train_edit_predictor_raises_for_missing_dataset_after_hyperparam_validation(tmp_path):
    with pytest.raises(FileNotFoundError, match="Token dataset not found"):
        train_module.train_edit_predictor(
            dataset_file=tmp_path / "missing.pt",
            output_dir=tmp_path / "out",
        )
