"""Interactive manager for Edit Predictor training and checkpoint use.

This CLI keeps the Edit Predictor workflows explicit:
- train: train from a base Hugging Face model
- checkpoint: load a saved checkpoint and evaluate it on the dataset's test split
- continue: continue training from a saved checkpoint
- predict: load a checkpoint and predict K/M labels for sample/provided sentences

This is the supervised baseline manager only. It never calls external APIs.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Callable, Sequence

from explanation.model.edit_predictor import EditPredictor
from explanation.model.train_edit_predictor import (
    DEFAULT_DATASET_FILE,
    DEFAULT_MODEL_NAME,
    EditPredictorTrainingResult,
    LABEL_IGNORE,
    compute_all_keep_baseline,
    load_token_dataset,
    train_edit_predictor,
)

DEFAULT_CHECKPOINT_DIR = Path(
    "explanation/model/checkpoints/edit_predictor_complex_words_distilbert_max256_v1"
)
DEFAULT_SAMPLE_SENTENCES = [
    "The laureates were appointed by the institute.",
    "The group talked for a long time before they agreed on the ultimate decision.",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage Edit Predictor training, checkpoint evaluation, continued training, and prediction."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "checkpoint", "continue", "predict"],
        default=None,
        help="Workflow mode. If omitted, an interactive menu is shown.",
    )
    parser.add_argument(
        "--dataset-file",
        default=str(DEFAULT_DATASET_FILE),
        help="Path to the token-level .pt dataset (train/continue/checkpoint modes).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Existing Edit Predictor checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Output directory for trained model/tokenizer/metrics.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face base model name for train mode.",
    )
    parser.add_argument(
        "--sentence",
        action="append",
        default=None,
        help="Sentence to run through predict mode. Can be passed more than once.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--learning-rate", type=float, default=3e-5, help="AdamW learning rate."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="AdamW weight decay."
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Stop after this many epochs without validation f1_M improvement.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.001,
        help="Minimum validation f1_M improvement required to reset early stopping.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Deterministic seed.")
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Weight the loss by inverse class frequency (K vs M) from the train split.",
    )
    parser.add_argument(
        "--mask-class-weight",
        type=float,
        default=None,
        help="Optional manual override for the M class weight (requires --use-class-weights).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Tokenizer max sequence length for predict mode.",
    )
    return parser


def choose_mode_interactively(input_func: Callable[[str], str] = input) -> str:
    """Ask the user which Edit Predictor workflow to run."""
    print("Choose Edit Predictor mode:")
    print("1 - Train/retrain from base model")
    print("2 - Load checkpoint and evaluate on the dataset's test split")
    print("3 - Continue training from existing checkpoint")
    print("4 - Load checkpoint and predict K/M labels for sample/provided sentences")
    choice = input_func("Enter 1, 2, 3, or 4: ").strip()
    choices = {
        "1": "train",
        "2": "checkpoint",
        "3": "continue",
        "4": "predict",
    }
    if choice not in choices:
        raise ValueError("Invalid choice. Expected 1, 2, 3, or 4.")
    return choices[choice]


def run(args: argparse.Namespace) -> None:
    """Run the selected Edit Predictor manager workflow."""
    mode = args.mode or choose_mode_interactively()
    if mode == "train":
        result = run_train_mode(args)
        _print_training_summary(result, args.output_dir)
    elif mode == "checkpoint":
        run_checkpoint_mode(args)
    elif mode == "continue":
        result = run_continue_mode(args)
        _print_training_summary(result, args.output_dir)
    elif mode == "predict":
        run_predict_mode(args)
    else:  # pragma: no cover - argparse prevents this outside direct calls
        raise ValueError(f"Unknown mode: {mode}")

    print()
    print("Equivalent command:")
    print(format_equivalent_command(args, mode))


def run_train_mode(args: argparse.Namespace) -> EditPredictorTrainingResult:
    """Train a fresh Edit Predictor from a Hugging Face base model."""
    dataset_file = _require_file(args.dataset_file, "token-level dataset")
    return train_edit_predictor(
        dataset_file=dataset_file,
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        seed=args.seed,
        use_class_weights=args.use_class_weights,
        mask_class_weight=args.mask_class_weight,
        load_mode="base",
    )


def run_continue_mode(args: argparse.Namespace) -> EditPredictorTrainingResult:
    """Continue Edit Predictor training from an existing checkpoint."""
    dataset_file = _require_file(args.dataset_file, "token-level dataset")
    checkpoint_dir = _require_checkpoint_dir(args.checkpoint_dir)
    return train_edit_predictor(
        dataset_file=dataset_file,
        output_dir=Path(args.output_dir),
        model_name=str(checkpoint_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        seed=args.seed,
        use_class_weights=args.use_class_weights,
        mask_class_weight=args.mask_class_weight,
        load_mode="checkpoint",
        checkpoint_dir=checkpoint_dir,
    )


def run_checkpoint_mode(args: argparse.Namespace) -> dict:
    """Load a checkpoint and evaluate it on the dataset's test split."""
    checkpoint_dir = _require_checkpoint_dir(args.checkpoint_dir)
    dataset_file = _require_file(args.dataset_file, "token-level dataset")

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from explanation.model.train_edit_predictor import _evaluate

    dataset = load_token_dataset(dataset_file)
    predictor = EditPredictor.from_checkpoint(checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor.model.to(device)

    test_split = dataset["test"]
    test_dataset = TensorDataset(
        test_split["input_ids"], test_split["attention_mask"], test_split["labels"]
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=LABEL_IGNORE)

    metrics, test_true, _ = _evaluate(predictor.model, test_loader, device, criterion, torch)
    baseline = compute_all_keep_baseline(test_true)

    print("=== Edit Predictor checkpoint evaluation (test split) ===")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Dataset   : {dataset_file}")
    for key, value in metrics.items():
        print(f"  {key:<20}: {value}")
    print(f"All-K baseline f1_M: {baseline['f1_M']}")
    return {"metrics": metrics, "baseline_all_keep": baseline}


def run_predict_mode(args: argparse.Namespace) -> list[list[dict]]:
    """Load a checkpoint and predict K/M labels for provided/sample sentences."""
    checkpoint_dir = _require_checkpoint_dir(args.checkpoint_dir)
    sentences = args.sentence or DEFAULT_SAMPLE_SENTENCES
    predictor = EditPredictor.from_checkpoint(checkpoint_dir)
    predictions = predictor.predict_token_labels(sentences, max_length=args.max_length)

    print("=== Edit Predictor checkpoint predictions ===")
    print(f"Checkpoint: {checkpoint_dir}")
    for sentence, tokens in zip(sentences, predictions):
        print()
        print(f"Sentence: {sentence}")
        rendered = " ".join(
            f"{item['token']}[{item['label']}]" if item["label"] == "M" else item["token"]
            for item in tokens
        )
        print(f"Tokens: {rendered}")
    return predictions


def format_equivalent_command(args: argparse.Namespace, mode: str) -> str:
    """Return a one-line shell-friendly command for reproducibility."""
    parts = [
        "python",
        "-m",
        "explanation.model.manage_edit_predictor",
        "--mode",
        mode,
    ]

    if mode in {"train", "continue"}:
        parts.extend([
            "--dataset-file", args.dataset_file,
            "--output-dir", args.output_dir,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
            "--early-stopping-patience", str(args.early_stopping_patience),
            "--early-stopping-min-delta", str(args.early_stopping_min_delta),
            "--seed", str(args.seed),
        ])
        if args.use_class_weights:
            parts.append("--use-class-weights")
        if args.mask_class_weight is not None:
            parts.extend(["--mask-class-weight", str(args.mask_class_weight)])
        if mode == "train":
            parts.extend(["--model-name", args.model_name])
        else:
            parts.extend(["--checkpoint-dir", args.checkpoint_dir])
    elif mode == "checkpoint":
        parts.extend([
            "--checkpoint-dir", args.checkpoint_dir,
            "--dataset-file", args.dataset_file,
        ])
    elif mode == "predict":
        parts.extend(["--checkpoint-dir", args.checkpoint_dir])
        for sentence in args.sentence or []:
            parts.extend(["--sentence", sentence])

    return " ".join(shlex.quote(part) for part in parts if part != "")


def _require_file(path_value: str, description: str) -> Path:
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Missing {description} file: {path}")
    return path


def _require_checkpoint_dir(path_value: str) -> Path:
    path = Path(path_value)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Missing Edit Predictor checkpoint directory: {path}")
    return path


def _print_training_summary(result: EditPredictorTrainingResult, output_dir: str) -> None:
    print("=== Edit Predictor training complete ===")
    print(f"Model/tokenizer saved to  : {output_dir}")
    print(f"Best epoch                : {result.best_epoch}")
    print(f"Best validation f1_M      : {result.best_validation_f1_M:.4f}")
    print(f"Test f1_M                 : {result.test_metrics['f1_M']:.4f}")
    print(f"Test token accuracy       : {result.test_metrics['token_accuracy']:.4f}")
    print(f"All-K baseline f1_M       : {result.baseline_all_keep['f1_M']:.4f}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
