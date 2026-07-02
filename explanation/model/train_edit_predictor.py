"""Train the supervised baseline token-level Edit Predictor (K/M classifier).

This trains only the supervised baseline that corresponds to local CEFR
pseudo-label supervision (see `explanation/model/edit_predictor_dataset.py`
and `explanation/scripts/build_edit_predictor_dataset.py`). It never calls
external APIs and does not generate explanations.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from explanation.model.edit_predictor import EditPredictor, LABEL_IGNORE, LABEL_KEEP, LABEL_MASK

DEFAULT_DATASET_FILE = Path(
    "explanation/data/processed/"
    "edit_predictor_token_labels_complex_words_distilbert_max256.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "explanation/model/checkpoints/edit_predictor_complex_words_distilbert_max256_v1"
)
DEFAULT_MODEL_NAME = "distilbert-base-uncased"

HISTORY_FIELDNAMES = [
    "epoch",
    "train_loss",
    "validation_loss",
    "validation_token_accuracy",
    "validation_precision_M",
    "validation_recall_M",
    "validation_f1_M",
    "validation_precision_K",
    "validation_recall_K",
    "validation_f1_K",
    "validation_macro_f1",
]


@dataclass(frozen=True)
class EditPredictorTrainingResult:
    """Everything saved to metrics.json after training."""

    dataset_file: str
    output_dir: str
    model_name: str
    load_mode: str
    epochs_run: int
    best_epoch: int
    best_validation_f1_M: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    baseline_all_keep: dict[str, float]
    train_examples: int
    validation_examples: int
    test_examples: int
    use_class_weights: bool
    mask_class_weight: float | None
    class_weights: list[float] | None
    seed: int


# ---------------------------------------------------------------------------
# Pure-Python helpers (no torch/transformers required)
# ---------------------------------------------------------------------------


def compute_token_metrics(
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
    loss: float | None = None,
) -> dict[str, float]:
    """Compute token classification metrics, ignoring ``LABEL_IGNORE`` positions.

    Args:
        true_labels: Gold label ids, possibly including ``LABEL_IGNORE`` (-100).
        pred_labels: Predicted label ids, same length as ``true_labels``.
        loss: Optional average loss to include in the returned dict.

    Returns:
        Dict with ``token_accuracy``, per-class precision/recall/f1 for K and
        M, ``macro_f1``, and gold/predicted token counts for K and M.
    """
    pairs = [(t, p) for t, p in zip(true_labels, pred_labels) if t != LABEL_IGNORE]
    total = len(pairs)

    if total == 0:
        metrics = {
            "token_accuracy": 0.0,
            "precision_M": 0.0,
            "recall_M": 0.0,
            "f1_M": 0.0,
            "precision_K": 0.0,
            "recall_K": 0.0,
            "f1_K": 0.0,
            "macro_f1": 0.0,
            "gold_M_tokens": 0,
            "predicted_M_tokens": 0,
            "gold_K_tokens": 0,
            "predicted_K_tokens": 0,
        }
    else:
        correct = sum(1 for t, p in pairs if t == p)
        token_accuracy = correct / total

        def _class_prf(cls: int) -> tuple[float, float, float]:
            tp = sum(1 for t, p in pairs if t == cls and p == cls)
            fp = sum(1 for t, p in pairs if t != cls and p == cls)
            fn = sum(1 for t, p in pairs if t == cls and p != cls)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            return precision, recall, f1

        precision_m, recall_m, f1_m = _class_prf(LABEL_MASK)
        precision_k, recall_k, f1_k = _class_prf(LABEL_KEEP)

        metrics = {
            "token_accuracy": round(token_accuracy, 6),
            "precision_M": round(precision_m, 6),
            "recall_M": round(recall_m, 6),
            "f1_M": round(f1_m, 6),
            "precision_K": round(precision_k, 6),
            "recall_K": round(recall_k, 6),
            "f1_K": round(f1_k, 6),
            "macro_f1": round((f1_m + f1_k) / 2, 6),
            "gold_M_tokens": sum(1 for t, _ in pairs if t == LABEL_MASK),
            "predicted_M_tokens": sum(1 for _, p in pairs if p == LABEL_MASK),
            "gold_K_tokens": sum(1 for t, _ in pairs if t == LABEL_KEEP),
            "predicted_K_tokens": sum(1 for _, p in pairs if p == LABEL_KEEP),
        }

    if loss is not None:
        metrics["loss"] = round(loss, 6)
    return metrics


def compute_all_keep_baseline(true_labels: Sequence[int]) -> dict[str, float]:
    """Metrics if the model predicted ``LABEL_KEEP`` for every valid token.

    Token accuracy on this dataset is high even for a model that never
    detects M, because K tokens vastly outnumber M tokens. This baseline
    makes that failure mode visible: a real model must beat this on f1_M.
    """
    predicted = [LABEL_KEEP] * len(true_labels)
    return compute_token_metrics(true_labels, predicted)


def compute_class_weights(
    flat_labels: Sequence[int],
    mask_class_weight: float | None = None,
) -> list[float]:
    """Compute inverse-frequency ``[weight_K, weight_M]``, ignoring ``LABEL_IGNORE``.

    Pass ``mask_class_weight`` to override the computed M weight directly;
    the K weight is left at its computed value.
    """
    counts = Counter(label for label in flat_labels if label != LABEL_IGNORE)
    count_k = counts.get(LABEL_KEEP, 0)
    count_m = counts.get(LABEL_MASK, 0)
    total = count_k + count_m

    weight_k = total / (2 * count_k) if total and count_k else 1.0
    weight_m = total / (2 * count_m) if total and count_m else 1.0

    if mask_class_weight is not None:
        weight_m = mask_class_weight

    return [weight_k, weight_m]


def select_best_epoch(
    history: Sequence[dict[str, Any]],
    min_delta: float = 0.0,
) -> tuple[int, float]:
    """Return ``(best_epoch, best_validation_f1_M)`` using early-stopping semantics.

    An epoch only becomes the new best if its ``validation_f1_M`` exceeds the
    previous best by more than ``min_delta``. Mirrors the improvement check
    used to decide when to save a checkpoint during training.
    """
    best_epoch = 0
    best_f1_m = float("-inf")
    for row in history:
        f1_m = row["validation_f1_M"]
        if f1_m > best_f1_m + min_delta:
            best_f1_m = f1_m
            best_epoch = row["epoch"]
    return best_epoch, best_f1_m


def load_token_dataset(dataset_file: str | Path) -> dict[str, Any]:
    """Load the token-level ``.pt`` dataset built by ``build_edit_predictor_dataset.py``."""
    dataset_file = Path(dataset_file)
    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Token dataset not found: {dataset_file}. Build it first with: "
            "python -m explanation.scripts.build_edit_predictor_dataset"
        )
    return torch.load(dataset_file, weights_only=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the supervised Edit Predictor (token-level K/M "
            "classifier) on a pre-tokenized .pt dataset. This is the "
            "local CEFR pseudo-label supervised baseline only."
        )
    )
    parser.add_argument(
        "--dataset-file",
        default=str(DEFAULT_DATASET_FILE),
        help="Path to the token-level .pt dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where checkpoint, metrics, and history are saved.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face base model name.",
    )
    parser.add_argument(
        "--load-mode",
        choices=["base", "checkpoint"],
        default="base",
        help="Initialize from a base model or an existing Edit Predictor checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint directory used when --load-mode checkpoint.",
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
    return parser


# ---------------------------------------------------------------------------
# Training (requires torch/transformers)
# ---------------------------------------------------------------------------


def train_edit_predictor(
    dataset_file: str | Path,
    output_dir: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    early_stopping_patience: int = 2,
    early_stopping_min_delta: float = 0.001,
    seed: int = 13,
    use_class_weights: bool = False,
    mask_class_weight: float | None = None,
    load_mode: Literal["base", "checkpoint"] = "base",
    checkpoint_dir: str | Path | None = None,
) -> EditPredictorTrainingResult:
    """Train and save the supervised baseline Edit Predictor.

    Loads pre-tokenized train/validation/test tensors from ``dataset_file``
    (no re-tokenization happens here). Checkpoints are selected by
    validation ``f1_M``, not token accuracy, because K/M class imbalance
    makes accuracy a misleading signal on its own (see ``baseline_all_keep``
    in the returned/saved metrics).
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be non-negative")

    dataset_file = Path(dataset_file)
    output_dir = Path(output_dir)
    dataset = load_token_dataset(dataset_file)

    _set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")

    if load_mode == "base":
        predictor = EditPredictor.from_base_model(model_name)
    elif load_mode == "checkpoint":
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required when load_mode='checkpoint'")
        predictor = EditPredictor.from_checkpoint(checkpoint_dir)
    else:
        raise ValueError("load_mode must be either 'base' or 'checkpoint'")

    predictor.model.to(device)

    train_split = dataset["train"]
    validation_split = dataset["validation"]
    test_split = dataset["test"]

    train_dataset = TensorDataset(
        train_split["input_ids"], train_split["attention_mask"], train_split["labels"]
    )
    validation_dataset = TensorDataset(
        validation_split["input_ids"],
        validation_split["attention_mask"],
        validation_split["labels"],
    )
    test_dataset = TensorDataset(
        test_split["input_ids"], test_split["attention_mask"], test_split["labels"]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_weights: list[float] | None = None
    weight_tensor = None
    if use_class_weights:
        flat_train_labels = train_split["labels"].reshape(-1).tolist()
        class_weights = compute_class_weights(
            flat_train_labels, mask_class_weight=mask_class_weight
        )
        weight_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)
        print(f"Using class weights [K, M]: {class_weights}")

    criterion = torch.nn.CrossEntropyLoss(ignore_index=LABEL_IGNORE, weight=weight_tensor)
    optimizer = AdamW(predictor.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: list[dict[str, Any]] = []
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        predictor.model.train()
        total_loss = 0.0
        batch_count = 0
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = predictor.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += float(loss.item())
            batch_count += 1
        train_loss = total_loss / batch_count if batch_count else 0.0

        validation_metrics, _, _ = _evaluate(
            predictor.model, validation_loader, device, criterion, torch
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "validation_loss": validation_metrics["loss"],
            "validation_token_accuracy": validation_metrics["token_accuracy"],
            "validation_precision_M": validation_metrics["precision_M"],
            "validation_recall_M": validation_metrics["recall_M"],
            "validation_f1_M": validation_metrics["f1_M"],
            "validation_precision_K": validation_metrics["precision_K"],
            "validation_recall_K": validation_metrics["recall_K"],
            "validation_f1_K": validation_metrics["f1_K"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        })

        current_best_epoch, _ = select_best_epoch(history, early_stopping_min_delta)
        if current_best_epoch == epoch:
            best_epoch = epoch
            epochs_without_improvement = 0
            predictor.save(output_dir)
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_f1_M={validation_metrics['f1_M']:.4f} "
            f"val_token_acc={validation_metrics['token_accuracy']:.4f}"
        )

        if (
            early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_epoch > 0:
        predictor = EditPredictor.from_checkpoint(output_dir)
        predictor.model.to(device)

    validation_final_metrics, _, _ = _evaluate(
        predictor.model, validation_loader, device, criterion, torch
    )
    test_final_metrics, test_true, _ = _evaluate(
        predictor.model, test_loader, device, criterion, torch
    )
    baseline_all_keep = compute_all_keep_baseline(test_true)

    best_history_row = next(
        (row for row in history if row["epoch"] == best_epoch),
        history[-1] if history else None,
    )

    result = EditPredictorTrainingResult(
        dataset_file=str(dataset_file),
        output_dir=str(output_dir),
        model_name=model_name,
        load_mode=load_mode,
        epochs_run=len(history),
        best_epoch=best_epoch,
        best_validation_f1_M=(
            best_history_row["validation_f1_M"] if best_history_row else 0.0
        ),
        validation_metrics=validation_final_metrics,
        test_metrics=test_final_metrics,
        baseline_all_keep=baseline_all_keep,
        train_examples=len(train_dataset),
        validation_examples=len(validation_dataset),
        test_examples=len(test_dataset),
        use_class_weights=use_class_weights,
        mask_class_weight=mask_class_weight,
        class_weights=class_weights,
        seed=seed,
    )

    _write_metrics(output_dir, result)
    _write_history(output_dir, history)
    _write_training_config(
        output_dir,
        dataset_file=dataset_file,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        seed=seed,
        use_class_weights=use_class_weights,
        mask_class_weight=mask_class_weight,
        load_mode=load_mode,
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
    )

    return result


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(
    model: Any,
    loader: Any,
    device: Any,
    criterion: Any,
    torch_module: Any,
) -> tuple[dict[str, float], list[int], list[int]]:
    """Run *model* over *loader* and return (metrics, flat true, flat pred)."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    with torch_module.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
            total_loss += float(loss.item())
            total_batches += 1

            preds = logits.argmax(dim=-1)
            all_true.extend(labels.view(-1).tolist())
            all_pred.extend(preds.view(-1).tolist())

    avg_loss = total_loss / total_batches if total_batches else 0.0
    metrics = compute_token_metrics(all_true, all_pred, loss=avg_loss)
    return metrics, all_true, all_pred


def _write_metrics(output_dir: Path, result: EditPredictorTrainingResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_history(output_dir: Path, history: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "history.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(history)


def _write_training_config(
    output_dir: Path,
    dataset_file: str | Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    seed: int,
    use_class_weights: bool,
    mask_class_weight: float | None,
    load_mode: str,
    checkpoint_dir: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_file": str(dataset_file),
        "output_dir": str(output_dir),
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "seed": seed,
        "use_class_weights": use_class_weights,
        "mask_class_weight": mask_class_weight,
        "load_mode": load_mode,
        "checkpoint_dir": checkpoint_dir,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = train_edit_predictor(
        dataset_file=Path(args.dataset_file),
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
        load_mode=args.load_mode,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
    )

    print("=== Edit Predictor training complete ===")
    print(f"Model/tokenizer saved to  : {args.output_dir}")
    print(f"Best epoch                : {result.best_epoch}")
    print(f"Best validation f1_M      : {result.best_validation_f1_M:.4f}")
    print(f"Test f1_M                 : {result.test_metrics['f1_M']:.4f}")
    print(f"Test token accuracy       : {result.test_metrics['token_accuracy']:.4f}")
    print(f"All-K baseline f1_M       : {result.baseline_all_keep['f1_M']:.4f}")


if __name__ == "__main__":
    main()
