"""Tests for the compatibility CEFR CSV pseudo-label CLI."""

import json
from pathlib import Path

import pytest

from explanation.scripts.generate_pseudo_labels import (
    build_parser,
    build_pseudo_label_row,
    find_all_spans,
    generate_pseudo_labels,
    load_processed_sentences,
    read_cefr_complex_words_csv,
)


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_build_parser_defaults():
    args = build_parser().parse_args([])

    assert args.input_file.endswith("sentences.txt")
    assert args.complex_words_csv.endswith("complex_words.csv")
    assert args.output_file.endswith("complex_words_pseudo_labels.jsonl")
    assert args.limit is None
    assert args.resume is False
    assert args.seed == 13
    assert args.skip_mojibake is False
    assert args.include_phrases is False
    assert args.cefr_levels == ["C1", "C2"]
    assert args.source == "complex_words_csv"
    assert args.model == "lexicon"


def test_find_all_spans_uses_full_word_matching_only():
    assert find_all_spans("The attractor moved.", "tract") == []
    assert find_all_spans("It actually happened.", "ally") == []
    assert find_all_spans("Management improved.", "man") == []
    assert find_all_spans("Management improved.", "gem") == []
    assert find_all_spans("The tract was long.", "tract") == [(4, 9)]
    assert find_all_spans("She was an ally.", "ally") == [(11, 15)]
    assert find_all_spans("He felt self-conscious.", "self-conscious") == [(8, 22)]


def test_read_cefr_complex_words_csv_keeps_c1_c2_by_default(tmp_path):
    csv_path = tmp_path / "complex_words.csv"
    _write_csv(csv_path, "headword,pos,CEFR,notes", [
        "dog,noun,A1,",
        "decide,verb,B2,",
        "tract,noun,C1,",
        "ally,noun,C2,",
    ])

    assert read_cefr_complex_words_csv(csv_path) == {"tract", "ally"}


def test_read_cefr_complex_words_csv_rejects_custom_levels(tmp_path):
    csv_path = tmp_path / "complex_words.csv"
    _write_csv(csv_path, "headword,pos,CEFR,notes", ["dog,noun,A1,"])

    with pytest.raises(ValueError, match="Custom CEFR levels"):
        read_cefr_complex_words_csv(csv_path, accepted_levels={"A1"})


def test_build_pseudo_label_row_does_not_emit_false_substrings():
    row = build_pseudo_label_row(
        "Management actually attracted investors.",
        {"man", "gem", "ally", "tract"},
    )

    assert row["difficult_words"] == []
    assert row["spans"] == []


def test_build_pseudo_label_row_keeps_full_word_matches():
    sentence = "The tract was long, and she was an ally."
    row = build_pseudo_label_row(sentence, {"tract", "ally"})

    assert row["difficult_words"] == ["tract", "ally"]
    assert row["spans"] == [[4, 9], [35, 39]]


def test_load_processed_sentences_missing_file_returns_empty_set(tmp_path):
    assert load_processed_sentences(tmp_path / "missing.jsonl") == set()


def test_generate_pseudo_labels_writes_local_csv_rows(tmp_path):
    input_file = tmp_path / "sentences.txt"
    input_file.write_text(
        "The tract was long.\nManagement actually attracted investors.\n",
        encoding="utf-8",
    )
    csv_path = tmp_path / "complex_words.csv"
    _write_csv(csv_path, "headword,pos,CEFR,notes", [
        "tract,noun,C1,",
        "ally,noun,C1,",
        "man,noun,C1,",
        "gem,noun,C1,",
    ])
    output_file = tmp_path / "out.jsonl"

    summary = generate_pseudo_labels(
        input_file=input_file,
        output_file=output_file,
        complex_words_csv=csv_path,
    )

    rows = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
    ]
    by_sentence = {row["sentence"]: row for row in rows}

    assert by_sentence["The tract was long."]["difficult_words"] == ["tract"]
    assert by_sentence["Management actually attracted investors."]["difficult_words"] == []
    assert summary["sentences_written"] == 2
    assert summary["sentences_with_complex_words"] == 1
    assert summary["sentences_without_complex_words"] == 1


def test_generate_pseudo_labels_resume_skips_existing_rows(tmp_path):
    input_file = tmp_path / "sentences.txt"
    input_file.write_text("The tract was long.\nShe was an ally.\n", encoding="utf-8")
    csv_path = tmp_path / "complex_words.csv"
    _write_csv(csv_path, "headword,pos,CEFR,notes", [
        "tract,noun,C1,",
        "ally,noun,C1,",
    ])
    output_file = tmp_path / "out.jsonl"

    generate_pseudo_labels(
        input_file=input_file,
        output_file=output_file,
        complex_words_csv=csv_path,
        limit=1,
    )
    generate_pseudo_labels(
        input_file=input_file,
        output_file=output_file,
        complex_words_csv=csv_path,
        resume=True,
    )

    rows = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["sentence"] for row in rows} == {
        "The tract was long.",
        "She was an ally.",
    }
