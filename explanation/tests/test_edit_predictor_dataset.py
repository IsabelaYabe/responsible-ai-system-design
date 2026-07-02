"""Tests for Edit Predictor pseudo-label token dataset conversion."""

import json
import re

from explanation.model.edit_predictor_dataset import (
    LABEL_IGNORE,
    LABEL_KEEP,
    LABEL_MASK,
    PseudoLabelExample,
    build_token_labels_from_offsets,
    convert_examples_to_token_dataset,
    read_pseudo_label_jsonl,
    split_token_dataset_train_validation_test,
    token_overlaps_span,
)


class FakeTokenizer:
    """Whitespace tokenizer that returns Hugging Face-like batch encodings."""

    def __call__(
        self,
        sentences,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=128,
    ):
        assert return_offsets_mapping is True
        assert padding == "max_length"
        assert truncation is True

        all_input_ids = []
        all_attention_mask = []
        all_offsets = []
        all_sequence_ids = []

        for sentence in sentences:
            token_offsets = [match.span() for match in re.finditer(r"\S+", sentence)]
            max_normal_tokens = max(0, max_length - 2)
            token_offsets = token_offsets[:max_normal_tokens]

            input_ids = [101] + list(range(1000, 1000 + len(token_offsets))) + [102]
            offsets = [(0, 0)] + token_offsets + [(0, 0)]
            sequence_ids = [None] + [0] * len(token_offsets) + [None]
            attention_mask = [1] * len(input_ids)

            pad_count = max_length - len(input_ids)
            input_ids.extend([0] * pad_count)
            offsets.extend([(0, 0)] * pad_count)
            sequence_ids.extend([None] * pad_count)
            attention_mask.extend([0] * pad_count)

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_offsets.append(offsets)
            all_sequence_ids.append(sequence_ids)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "offset_mapping": all_offsets,
            "sequence_ids": all_sequence_ids,
        }


def _example(sentence, words, spans):
    return PseudoLabelExample(
        sentence=sentence,
        difficult_words=words,
        spans=spans,
        source="groq",
        model="m",
    )


def test_token_overlaps_span():
    assert token_overlaps_span(4, 13, 4, 13)
    assert token_overlaps_span(4, 13, 8, 20)
    assert not token_overlaps_span(0, 3, 4, 13)


def test_build_token_labels_marks_special_tokens_and_padding_ignore():
    labels = build_token_labels_from_offsets(
        offsets=[(0, 0), (0, 3), (4, 13), (0, 0)],
        spans=[(4, 13)],
        sequence_ids=[None, 0, 0, None],
    )

    assert labels == [LABEL_IGNORE, LABEL_KEEP, LABEL_MASK, LABEL_IGNORE]


def test_single_difficult_word_maps_to_correct_token():
    sentence = "The laureates smiled."
    examples = [_example(sentence, ["laureates"], [[4, 13]])]

    token_examples, summary = convert_examples_to_token_dataset(
        examples, FakeTokenizer(), max_length=8
    )

    assert token_examples[0].labels[:5] == [
        LABEL_IGNORE,
        LABEL_KEEP,
        LABEL_MASK,
        LABEL_KEEP,
        LABEL_IGNORE,
    ]
    assert summary.tokens_mask == 1


def test_multiple_difficult_spans_map_to_multiple_mask_labels():
    sentence = "The laureates accepted honors."
    examples = [_example(sentence, ["laureates", "honors"], [[4, 13], [23, 29]])]

    token_examples, summary = convert_examples_to_token_dataset(
        examples, FakeTokenizer(), max_length=8
    )

    assert token_examples[0].labels[:6] == [
        LABEL_IGNORE,
        LABEL_KEEP,
        LABEL_MASK,
        LABEL_KEEP,
        LABEL_MASK,
        LABEL_IGNORE,
    ]
    assert summary.tokens_mask == 2


def test_repeated_word_occurrences_with_different_spans_are_both_masked():
    sentence = "parish roads crossed parish fields"
    second_start = sentence.rfind("parish")
    examples = [_example(sentence, ["parish", "parish"], [[0, 6], [second_start, second_start + 6]])]

    token_examples, summary = convert_examples_to_token_dataset(
        examples, FakeTokenizer(), max_length=8
    )

    assert token_examples[0].labels[:6] == [
        LABEL_IGNORE,
        LABEL_MASK,
        LABEL_KEEP,
        LABEL_KEEP,
        LABEL_MASK,
        LABEL_KEEP,
    ]
    assert summary.tokens_mask == 2


def test_no_difficult_words_row_is_dropped_by_default():
    examples = [_example("A plain sentence.", [], [])]

    token_examples, summary = convert_examples_to_token_dataset(
        examples, FakeTokenizer(), max_length=7
    )

    assert token_examples == []
    assert summary.rows_written == 0
    assert summary.examples_no_difficult_words == 1


def test_no_difficult_words_can_be_kept_as_all_keep_example():
    examples = [_example("A plain sentence.", [], [])]

    token_examples, summary = convert_examples_to_token_dataset(
        examples, FakeTokenizer(), max_length=7, keep_no_difficult_rows=True
    )

    assert token_examples[0].labels[:5] == [
        LABEL_IGNORE,
        LABEL_KEEP,
        LABEL_KEEP,
        LABEL_KEEP,
        LABEL_IGNORE,
    ]
    assert LABEL_MASK not in token_examples[0].labels
    assert summary.examples_no_difficult_words == 1


def test_invalid_span_row_is_dropped_by_default():
    examples = [_example("A plain sentence.", ["missing"], [[-1, -1]])]

    token_examples, summary = convert_examples_to_token_dataset(
        examples, FakeTokenizer(), max_length=7
    )

    assert summary.invalid_spans_ignored == 1
    assert summary.rows_written == 0
    assert summary.rows_skipped_malformed == 1
    assert token_examples == []


def test_invalid_span_can_be_kept_as_all_keep_example():
    examples = [_example("A plain sentence.", ["missing"], [[-1, -1]])]

    token_examples, summary = convert_examples_to_token_dataset(
        examples, FakeTokenizer(), max_length=7, drop_invalid_rows=False
    )

    assert summary.invalid_spans_ignored == 1
    assert summary.rows_written == 1
    assert LABEL_MASK not in token_examples[0].labels


def test_out_of_bounds_span_row_is_dropped_by_default():
    examples = [_example("Short.", ["missing"], [[20, 30]])]

    token_examples, summary = convert_examples_to_token_dataset(
        examples, FakeTokenizer(), max_length=6
    )

    assert summary.out_of_bounds_spans_ignored == 1
    assert summary.rows_written == 0
    assert summary.rows_skipped_malformed == 1
    assert token_examples == []


def test_malformed_row_with_word_span_length_mismatch_is_skipped():
    examples = [_example("The laureates smiled.", ["laureates"], [])]

    token_examples, summary = convert_examples_to_token_dataset(
        examples, FakeTokenizer(), max_length=8
    )

    assert token_examples == []
    assert summary.rows_skipped_malformed == 1


def test_truncation_does_not_crash():
    examples = [_example("one two three four five", ["five"], [[19, 23]])]

    token_examples, summary = convert_examples_to_token_dataset(
        examples, FakeTokenizer(), max_length=4
    )

    assert len(token_examples[0].labels) == 4
    assert summary.rows_written == 1


def test_train_validation_test_split_is_deterministic():
    examples = [
        _example(f"Sentence {index}.", [], [])
        for index in range(10)
    ]
    token_examples, _ = convert_examples_to_token_dataset(
        examples, FakeTokenizer(), max_length=6, keep_no_difficult_rows=True
    )

    first = split_token_dataset_train_validation_test(token_examples, seed=7)
    second = split_token_dataset_train_validation_test(token_examples, seed=7)

    assert first == second
    assert len(first.train) == 8
    assert len(first.validation) == 1
    assert len(first.test) == 1


def test_read_pseudo_label_jsonl(tmp_path):
    input_file = tmp_path / "labels.jsonl"
    row = {
        "sentence": "The laureates smiled.",
        "difficult_words": ["laureates"],
        "spans": [[4, 13]],
        "source": "openrouter",
        "model": "m",
    }
    input_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

    examples = read_pseudo_label_jsonl(input_file)

    assert examples == [
        PseudoLabelExample(
            sentence="The laureates smiled.",
            difficult_words=["laureates"],
            spans=[[4, 13]],
            source="openrouter",
            model="m",
        )
    ]
