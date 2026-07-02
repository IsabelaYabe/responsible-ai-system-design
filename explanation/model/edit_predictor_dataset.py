"""Dataset helpers for future Edit Predictor token labels.

This module converts sentence-level difficult-word pseudo-label JSONL rows into
token-level K/M labels. It does not define or train an Edit Predictor model.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

LABEL_KEEP = 0
LABEL_MASK = 1
LABEL_IGNORE = -100
LABEL_NAMES = {
    LABEL_KEEP: "K",
    LABEL_MASK: "M",
}


@dataclass(frozen=True)
class PseudoLabelExample:
    """One sentence-level pseudo-label row."""

    sentence: str
    difficult_words: list[str]
    spans: list[list[int]]
    source: str
    model: str


@dataclass(frozen=True)
class TokenizedEditPredictorExample:
    """One tokenized example with token-level Edit Predictor labels."""

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    sentence: str
    words: list[str]
    spans: list[list[int]]
    source: str
    model: str


@dataclass(frozen=True)
class EditPredictorConversionSummary:
    """Counters collected while converting sentence labels to token labels."""

    rows_read: int
    rows_written: int
    rows_skipped_malformed: int
    invalid_spans_ignored: int
    out_of_bounds_spans_ignored: int
    examples_no_difficult_words: int
    tokens_keep: int
    tokens_mask: int
    tokens_ignore: int


@dataclass(frozen=True)
class EditPredictorSplits:
    """Container for train/validation/test tokenized examples."""

    train: list[TokenizedEditPredictorExample]
    validation: list[TokenizedEditPredictorExample]
    test: list[TokenizedEditPredictorExample]


def read_pseudo_label_jsonl(path: Path) -> list[PseudoLabelExample]:
    """Read sentence-level pseudo-label examples from *path*."""
    examples: list[PseudoLabelExample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(
                PseudoLabelExample(
                    sentence=str(row.get("sentence", "")),
                    difficult_words=[str(word) for word in row.get("difficult_words", [])],
                    spans=[list(span) for span in row.get("spans", [])],
                    source=str(row.get("source", "")),
                    model=str(row.get("model", "")),
                )
            )
    return examples


def token_overlaps_span(
    token_start: int,
    token_end: int,
    span_start: int,
    span_end: int,
) -> bool:
    """Return True when a token character interval overlaps a span interval."""
    return token_start < span_end and span_start < token_end


def build_token_labels_from_offsets(
    offsets: Sequence[Sequence[int]],
    spans: Sequence[Sequence[int]],
    sequence_ids: Sequence[int | None] | None = None,
) -> list[int]:
    """Build K/M/-100 labels from tokenizer offsets and valid difficult spans.

    Special tokens and padding are labeled ``LABEL_IGNORE``. For single-sentence
    inputs, any token whose offset overlaps at least one valid difficult-word
    span is labeled ``LABEL_MASK``; all other normal tokens are ``LABEL_KEEP``.
    """
    labels: list[int] = []
    for index, offset in enumerate(offsets):
        token_start, token_end = int(offset[0]), int(offset[1])
        sequence_id = sequence_ids[index] if sequence_ids is not None else 0

        if sequence_id is not None and sequence_id != 0:
            labels.append(LABEL_IGNORE)
            continue
        if token_start == token_end:
            labels.append(LABEL_IGNORE)
            continue

        if any(
            token_overlaps_span(token_start, token_end, int(span[0]), int(span[1]))
            for span in spans
        ):
            labels.append(LABEL_MASK)
        else:
            labels.append(LABEL_KEEP)
    return labels


def convert_examples_to_token_dataset(
    examples: Sequence[PseudoLabelExample],
    tokenizer: Any,
    max_length: int = 128,
    drop_invalid_rows: bool = True,
    keep_no_difficult_rows: bool = False,
) -> tuple[list[TokenizedEditPredictorExample], EditPredictorConversionSummary]:
    """Convert pseudo-label examples into token-level K/M examples."""
    if max_length < 1:
        raise ValueError("max_length must be at least 1")

    rows_skipped_malformed = 0
    invalid_spans_ignored = 0
    out_of_bounds_spans_ignored = 0
    examples_no_difficult_words = 0

    kept_examples: list[PseudoLabelExample] = []
    valid_spans_by_example: list[list[list[int]]] = []

    for example in examples:
        if len(example.difficult_words) != len(example.spans):
            rows_skipped_malformed += 1
            continue

        if not example.difficult_words and not example.spans:
            examples_no_difficult_words += 1
            if not keep_no_difficult_rows:
                continue

        valid_spans: list[list[int]] = []
        ignored_for_row = 0
        for span in example.spans:
            status = _validate_span(span, len(example.sentence))
            if status == "valid":
                valid_spans.append([int(span[0]), int(span[1])])
            elif status == "out_of_bounds":
                out_of_bounds_spans_ignored += 1
                ignored_for_row += 1
            else:
                invalid_spans_ignored += 1
                ignored_for_row += 1

        if drop_invalid_rows and ignored_for_row:
            rows_skipped_malformed += 1
            continue

        kept_examples.append(example)
        valid_spans_by_example.append(valid_spans)

    if not kept_examples:
        summary = EditPredictorConversionSummary(
            rows_read=len(examples),
            rows_written=0,
            rows_skipped_malformed=rows_skipped_malformed,
            invalid_spans_ignored=invalid_spans_ignored,
            out_of_bounds_spans_ignored=out_of_bounds_spans_ignored,
            examples_no_difficult_words=examples_no_difficult_words,
            tokens_keep=0,
            tokens_mask=0,
            tokens_ignore=0,
        )
        return [], summary

    sentences = [example.sentence for example in kept_examples]
    encodings = tokenizer(
        sentences,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    tokenized_examples: list[TokenizedEditPredictorExample] = []
    tokens_keep = 0
    tokens_mask = 0
    tokens_ignore = 0

    for index, example in enumerate(kept_examples):
        offsets = _get_batch_item(encodings, "offset_mapping", index)
        sequence_ids = _get_sequence_ids(encodings, index)
        labels = build_token_labels_from_offsets(
            offsets=offsets,
            spans=valid_spans_by_example[index],
            sequence_ids=sequence_ids,
        )
        tokens_keep += labels.count(LABEL_KEEP)
        tokens_mask += labels.count(LABEL_MASK)
        tokens_ignore += labels.count(LABEL_IGNORE)

        tokenized_examples.append(
            TokenizedEditPredictorExample(
                input_ids=list(_get_batch_item(encodings, "input_ids", index)),
                attention_mask=list(_get_batch_item(encodings, "attention_mask", index)),
                labels=labels,
                sentence=example.sentence,
                words=list(example.difficult_words),
                spans=[list(span) for span in example.spans],
                source=example.source,
                model=example.model,
            )
        )

    summary = EditPredictorConversionSummary(
        rows_read=len(examples),
        rows_written=len(tokenized_examples),
        rows_skipped_malformed=rows_skipped_malformed,
        invalid_spans_ignored=invalid_spans_ignored,
        out_of_bounds_spans_ignored=out_of_bounds_spans_ignored,
        examples_no_difficult_words=examples_no_difficult_words,
        tokens_keep=tokens_keep,
        tokens_mask=tokens_mask,
        tokens_ignore=tokens_ignore,
    )
    return tokenized_examples, summary


def split_token_dataset_train_validation_test(
    examples: Sequence[TokenizedEditPredictorExample],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 13,
    shuffle: bool = True,
) -> EditPredictorSplits:
    """Deterministically split tokenized examples into train/validation/test."""
    _validate_ratios(train_ratio, validation_ratio, test_ratio)

    items = list(examples)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)

    n_total = len(items)
    n_train = int(n_total * train_ratio)
    n_validation = int(n_total * validation_ratio)

    return EditPredictorSplits(
        train=items[:n_train],
        validation=items[n_train : n_train + n_validation],
        test=items[n_train + n_validation :],
    )


def _validate_span(span: Sequence[int], sentence_length: int) -> str:
    if len(span) != 2:
        return "invalid"
    start, end = span
    if not isinstance(start, int) or not isinstance(end, int):
        return "invalid"
    if start == -1 and end == -1:
        return "invalid"
    if end <= start:
        return "invalid"
    if start < 0 or start >= sentence_length or end > sentence_length:
        return "out_of_bounds"
    return "valid"


def _get_batch_item(encodings: Any, key: str, index: int) -> Any:
    return encodings[key][index]


def _get_sequence_ids(encodings: Any, index: int) -> list[int | None] | None:
    if hasattr(encodings, "sequence_ids"):
        return list(encodings.sequence_ids(index))
    if isinstance(encodings, dict) and "sequence_ids" in encodings:
        return list(encodings["sequence_ids"][index])
    return None


def _validate_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    if train_ratio < 0 or validation_ratio < 0 or test_ratio < 0:
        raise ValueError("All split ratios must be non-negative.")

    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "train_ratio + validation_ratio + test_ratio must equal 1.0. "
            f"Got {total:.6f}."
        )
