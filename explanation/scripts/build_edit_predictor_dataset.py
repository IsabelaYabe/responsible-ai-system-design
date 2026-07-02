"""Build token-level Edit Predictor pseudo-label datasets.

This script converts sentence-level difficult-word pseudo-label JSONL rows into
token-level K/M labels. It does not train an Edit Predictor model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch as _torch
from transformers import AutoTokenizer

from explanation.model.edit_predictor_dataset import (
    LABEL_IGNORE,
    LABEL_KEEP,
    LABEL_MASK,
    EditPredictorConversionSummary,
    TokenizedEditPredictorExample,
    convert_examples_to_token_dataset,
    read_pseudo_label_jsonl,
    split_token_dataset_train_validation_test,
)

DEFAULT_INPUT_FILE = Path(
    "explanation/data/processed/"
    "complex_words_pseudo_labels.jsonl"
)
DEFAULT_OUTPUT_FILE = Path(
    "explanation/data/processed/"
    "edit_predictor_token_labels_complex_words_distilbert_max256.pt"
)
DEFAULT_MODEL_NAME = "distilbert-base-uncased"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert sentence-level difficult-word pseudo-labels into "
            "token-level K/M labels for the supervised Edit Predictor."
        )
    )
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_INPUT_FILE),
        help="Path to sentence-level pseudo-label JSONL.",
    )
    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Path to write the token-level .pt dataset.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face tokenizer/model name.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokenizer max sequence length.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of examples used for training.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of examples used for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of examples used for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Deterministic split seed.",
    )
    parser.add_argument(
        "--keep-invalid-rows",
        action="store_true",
        help=(
            "Keep rows with invalid or out-of-bounds spans as all-K examples. "
            "By default they are dropped."
        ),
    )
    return parser


def build_edit_predictor_dataset_artifact(
    input_file: Path,
    output_file: Path,
    tokenizer: Any,
    model_name: str,
    max_length: int = 128,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 13,
    drop_invalid_rows: bool = True,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Build and save the token-level dataset artifact."""
    torch = torch_module if torch_module is not None else _torch

    examples = read_pseudo_label_jsonl(input_file)
    token_examples, summary = convert_examples_to_token_dataset(
        examples=examples,
        tokenizer=tokenizer,
        max_length=max_length,
        drop_invalid_rows=drop_invalid_rows,
    )
    splits = split_token_dataset_train_validation_test(
        token_examples,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    metadata = {
        "input_file": str(input_file),
        "model_name": model_name,
        "max_length": max_length,
        "label_keep": LABEL_KEEP,
        "label_mask": LABEL_MASK,
        "label_ignore": LABEL_IGNORE,
        **asdict(summary),
        "rows_skipped": summary.rows_read - summary.rows_written,
        "train_examples": len(splits.train),
        "validation_examples": len(splits.validation),
        "test_examples": len(splits.test),
    }
    artifact = {
        "train": _pack_split(splits.train, torch, max_length),
        "validation": _pack_split(splits.validation, torch, max_length),
        "test": _pack_split(splits.test, torch, max_length),
        "metadata": metadata,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_file)
    _print_summary(summary, metadata, output_file)
    return artifact


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    build_edit_predictor_dataset_artifact(
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        tokenizer=tokenizer,
        model_name=args.model_name,
        max_length=args.max_length,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        drop_invalid_rows=not args.keep_invalid_rows
    )


def _pack_split(
    examples: Sequence[TokenizedEditPredictorExample],
    torch: Any,
    max_length: int,
) -> dict[str, Any]:
    return {
        "input_ids": _tensor_2d(
            [example.input_ids for example in examples], torch, max_length
        ),
        "attention_mask": _tensor_2d(
            [example.attention_mask for example in examples], torch, max_length
        ),
        "labels": _tensor_2d([example.labels for example in examples], torch, max_length),
        "sentences": [example.sentence for example in examples],
        "words": [example.words for example in examples],
        "spans": [example.spans for example in examples],
        "source": [example.source for example in examples],
        "model": [example.model for example in examples],
    }


def _tensor_2d(rows: list[list[int]], torch: Any, max_length: int) -> Any:
    if rows:
        return torch.tensor(rows, dtype=torch.long)
    return torch.empty((0, max_length), dtype=torch.long)


def _print_summary(
    summary: EditPredictorConversionSummary,
    metadata: dict[str, Any],
    output_file: Path,
) -> None:
    print("=== Edit Predictor token dataset build complete ===")
    print(f"Rows read                         : {summary.rows_read}")
    print(f"Rows written                      : {summary.rows_written}")
    print(f"Rows skipped                      : {summary.rows_read - summary.rows_written}")
    print(f"Rows skipped malformed            : {summary.rows_skipped_malformed}")
    print(f"Invalid spans ignored             : {summary.invalid_spans_ignored}")
    print(f"Out-of-bounds spans ignored       : {summary.out_of_bounds_spans_ignored}")
    print(f"Examples with no difficult words  : {summary.examples_no_difficult_words}")
    print(f"Tokens K / M / -100               : {summary.tokens_keep} / {summary.tokens_mask} / {summary.tokens_ignore}")
    print(
        "Train / validation / test examples: "
        f"{metadata['train_examples']} / "
        f"{metadata['validation_examples']} / "
        f"{metadata['test_examples']}"
    )
    print(f"Output path                       : {output_file}")


if __name__ == "__main__":
    main()
