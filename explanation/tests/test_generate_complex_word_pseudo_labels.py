"""Tests for the CEFR complex-word lexicon pseudo-label generator.

These tests never call an external API and never touch the network: they
use only temporary files and the standard library.
"""

import ast
import json
from pathlib import Path

import pytest

from explanation.scripts.generate_complex_word_pseudo_labels import (
    build_parser,
    build_pseudo_label_row,
    find_all_spans,
    generate_complex_word_pseudo_labels,
    has_mojibake,
    read_complex_word_csv,
    read_sentence_file,
)


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 1-6. read_complex_word_csv
# ---------------------------------------------------------------------------


def test_reads_headword_pos_cefr_notes_csv(tmp_path):
    csv_path = tmp_path / "complex_words.csv"
    _write_csv(csv_path, "headword,pos,CEFR,notes", [
        "exterior,noun,C1,",
        "cloak,noun,C1,",
        "timid,adjective,C1,",
    ])

    words = read_complex_word_csv(csv_path)

    assert words == {"exterior", "cloak", "timid"}


def test_keeps_only_c1_and_c2_rows(tmp_path):
    csv_path = tmp_path / "complex_words.csv"
    _write_csv(csv_path, "headword,pos,CEFR,notes", [
        "exterior,noun,C1,",
        "cloak,noun,C2,",
    ])

    words = read_complex_word_csv(csv_path)

    assert words == {"exterior", "cloak"}


def test_ignores_a1_and_b1_rows_keeps_b2_c1_c2(tmp_path):
    csv_path = tmp_path / "complex_words.csv"
    _write_csv(csv_path, "headword,pos,CEFR,notes", [
        "dog,noun,A1,",
        "run,verb,B1,",
        "decide,verb,B2,",
        "exterior,noun,C1,",
    ])

    words = read_complex_word_csv(csv_path)

    assert words == {"decide", "exterior"}
    assert "dog" not in words
    assert "run" not in words


def test_supports_fallback_word_column(tmp_path):
    csv_path = tmp_path / "complex_words.csv"
    _write_csv(csv_path, "word,pos,CEFR,notes", [
        "timid,adjective,C1,",
    ])

    words = read_complex_word_csv(csv_path)

    assert words == {"timid"}


def test_ignores_empty_words(tmp_path):
    csv_path = tmp_path / "complex_words.csv"
    _write_csv(csv_path, "headword,pos,CEFR,notes", [
        ",noun,C1,",
        "   ,noun,C1,",
        "cloak,noun,C1,",
    ])

    words = read_complex_word_csv(csv_path)

    assert words == {"cloak"}


def test_ignores_phrases_unless_include_phrases_true(tmp_path):
    csv_path = tmp_path / "complex_words.csv"
    _write_csv(csv_path, "headword,pos,CEFR,notes", [
        "make up,phrase,C1,",
        "cloak,noun,C1,",
    ])

    default_words = read_complex_word_csv(csv_path)
    phrase_words = read_complex_word_csv(csv_path, include_phrases=True)

    assert default_words == {"cloak"}
    assert phrase_words == {"cloak", "make up"}


def test_keeps_hyphenated_words(tmp_path):
    csv_path = tmp_path / "complex_words.csv"
    _write_csv(csv_path, "headword,pos,CEFR,notes", [
        "self-conscious,adjective,C1,",
    ])

    words = read_complex_word_csv(csv_path)

    assert words == {"self-conscious"}


def test_deduplicates_words(tmp_path):
    csv_path = tmp_path / "complex_words.csv"
    _write_csv(csv_path, "headword,pos,CEFR,notes", [
        "juvenile,noun,C1,",
        "juvenile,adjective,C1,",
        "Juvenile,adjective,C1,",
    ])

    words = read_complex_word_csv(csv_path)

    assert words == {"juvenile"}


def test_no_cefr_column_keeps_all_rows(tmp_path):
    csv_path = tmp_path / "complex_words.csv"
    _write_csv(csv_path, "headword,pos,notes", [
        "cloak,noun,",
        "dog,noun,",
    ])

    words = read_complex_word_csv(csv_path)

    assert words == {"cloak", "dog"}


def test_missing_csv_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Complex word CSV not found"):
        read_complex_word_csv(tmp_path / "missing.csv")


def test_missing_word_column_raises_value_error(tmp_path):
    csv_path = tmp_path / "complex_words.csv"
    _write_csv(csv_path, "foo,bar", ["1,2"])

    with pytest.raises(ValueError, match="Could not find a headword column"):
        read_complex_word_csv(csv_path)


# ---------------------------------------------------------------------------
# read_sentence_file
# ---------------------------------------------------------------------------


def test_read_sentence_file_strips_and_skips_blank_lines(tmp_path):
    path = tmp_path / "complex.txt"
    path.write_text("  The cloak was dark.  \n\n\nTimid animals hide.\n", encoding="utf-8")

    sentences = read_sentence_file(path)

    assert sentences == ["The cloak was dark.", "Timid animals hide."]


def test_read_sentence_file_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Sentence file not found"):
        read_sentence_file(tmp_path / "missing.txt")


# ---------------------------------------------------------------------------
# has_mojibake
# ---------------------------------------------------------------------------


def test_has_mojibake_true_for_non_ascii():
    assert has_mojibake("QuakenbrÃ¼ck is a town.")


def test_has_mojibake_false_for_plain_ascii():
    assert not has_mojibake("The cloak was dark.")


# ---------------------------------------------------------------------------
# 7-8. find_all_spans
# ---------------------------------------------------------------------------


def test_finds_standalone_span_for_cloak():
    sentence = "The cloak was dark."

    spans = find_all_spans(sentence, "cloak")

    assert spans == [(4, 9)]
    assert sentence[4:9] == "cloak"


def test_does_not_match_art_inside_article_before_standalone():
    sentence = "This article discusses fine art in detail."

    spans = find_all_spans(sentence, "art")

    # Only the standalone "art" matches; "article" is never matched.
    assert spans == [(28, 31)]
    assert sentence[28:31] == "art"


def test_does_not_match_substrings_inside_larger_words():
    assert find_all_spans("The attractor moved.", "tract") == []
    assert find_all_spans("It actually happened.", "ally") == []
    assert find_all_spans("Management improved.", "man") == []
    assert find_all_spans("Management improved.", "gem") == []


def test_matches_full_words_after_substring_fix():
    assert find_all_spans("The tract was long.", "tract") == [(4, 9)]
    assert find_all_spans("She was an ally.", "ally") == [(11, 15)]
    assert find_all_spans("He felt self-conscious.", "self-conscious") == [(8, 22)]


# ---------------------------------------------------------------------------
# 9-10. build_pseudo_label_row
# ---------------------------------------------------------------------------


def test_keeps_multiple_occurrences():
    sentence = "The timid cloak hid the timid animal."
    row = build_pseudo_label_row(
        sentence, {"timid", "cloak"}, source="complex_words_csv", model="lexicon"
    )

    assert row["difficult_words"] == ["timid", "cloak", "timid"]
    assert row["spans"] == [[4, 9], [10, 15], [24, 29]]
    assert len(row["difficult_words"]) == len(row["spans"])


def test_empty_difficult_words_and_spans_when_no_lexicon_word_matches():
    row = build_pseudo_label_row(
        "A plain sentence with nothing special.",
        {"cloak", "timid"},
        source="complex_words_csv",
        model="lexicon",
    )

    assert row["difficult_words"] == []
    assert row["spans"] == []
    assert row["source"] == "complex_words_csv"
    assert row["model"] == "lexicon"


def test_spans_sorted_by_start_index():
    sentence = "cloak first, timid second."
    row = build_pseudo_label_row(
        sentence, {"timid", "cloak"}, source="complex_words_csv", model="lexicon"
    )

    starts = [span[0] for span in row["spans"]]
    assert starts == sorted(starts)


def test_uses_sentence_surface_form_not_lowercase_lexicon_form():
    sentence = "Cloak and TIMID were both used."
    row = build_pseudo_label_row(
        sentence, {"cloak", "timid"}, source="complex_words_csv", model="lexicon"
    )

    assert row["difficult_words"] == ["Cloak", "TIMID"]


def test_build_row_does_not_emit_false_substring_labels():
    row = build_pseudo_label_row(
        "Management actually attracted investors.",
        {"man", "gem", "ally", "tract"},
        source="complex_words_csv",
        model="lexicon",
    )

    assert row["difficult_words"] == []
    assert row["spans"] == []


# ---------------------------------------------------------------------------
# 11. generate_complex_word_pseudo_labels writes valid JSONL
# ---------------------------------------------------------------------------


def test_generate_writes_valid_jsonl_compatible_with_token_dataset_builder(tmp_path):
    from explanation.model.edit_predictor_dataset import read_pseudo_label_jsonl

    input_file = tmp_path / "complex.txt"
    input_file.write_text(
        "The cloak was dark.\nA plain sentence.\nTimid animals hide well.\n",
        encoding="utf-8",
    )
    csv_path = tmp_path / "complex_words.csv"
    _write_csv(csv_path, "headword,pos,CEFR,notes", [
        "cloak,noun,C1,",
        "timid,adjective,C1,",
    ])
    output_file = tmp_path / "out.jsonl"

    summary = generate_complex_word_pseudo_labels(
        input_file=input_file,
        complex_words_csv=csv_path,
        output_file=output_file,
        source="complex_words_csv",
        model="lexicon",
    )

    assert output_file.exists()
    lines = output_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        row = json.loads(line)
        assert set(row.keys()) == {"sentence", "difficult_words", "spans", "source", "model"}
        assert len(row["difficult_words"]) == len(row["spans"])

    examples = read_pseudo_label_jsonl(output_file)
    assert len(examples) == 3
    sentences_with_matches = {e.sentence: e.difficult_words for e in examples}
    assert sentences_with_matches["The cloak was dark."] == ["cloak"]
    assert sentences_with_matches["Timid animals hide well."] == ["Timid"]
    assert sentences_with_matches["A plain sentence."] == []

    assert summary["lexicon_size"] == 2
    assert summary["sentences_read"] == 3
    assert summary["sentences_written"] == 3
    assert summary["sentences_with_complex_words"] == 2
    assert summary["sentences_without_complex_words"] == 1
    assert summary["total_complex_word_matches"] == 2
    assert summary["input_file"] == str(input_file)
    assert summary["complex_words_csv"] == str(csv_path)
    assert summary["output_file"] == str(output_file)


def test_generate_respects_limit_and_skip_mojibake(tmp_path):
    input_file = tmp_path / "complex.txt"
    input_file.write_text(
        "\n".join(f"Sentence number {i} with cloak." for i in range(10))
        + "\nQuakenbrÃ¼ck cloak sentence.\n",
        encoding="utf-8",
    )
    csv_path = tmp_path / "complex_words.csv"
    _write_csv(csv_path, "headword,pos,CEFR,notes", ["cloak,noun,C1,"])
    output_file = tmp_path / "out.jsonl"

    summary = generate_complex_word_pseudo_labels(
        input_file=input_file,
        complex_words_csv=csv_path,
        output_file=output_file,
        limit=3,
        skip_mojibake=True,
    )

    lines = output_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert summary["sentences_written"] == 3
    assert summary["sentences_read"] == 11
    for line in lines:
        row = json.loads(line)
        assert "Ã" not in row["sentence"]


def test_generate_is_deterministic_across_runs_with_same_seed(tmp_path):
    input_file = tmp_path / "complex.txt"
    input_file.write_text(
        "\n".join(f"Sentence number {i} with cloak." for i in range(20)) + "\n",
        encoding="utf-8",
    )
    csv_path = tmp_path / "complex_words.csv"
    _write_csv(csv_path, "headword,pos,CEFR,notes", ["cloak,noun,C1,"])

    output_file_1 = tmp_path / "out1.jsonl"
    output_file_2 = tmp_path / "out2.jsonl"

    generate_complex_word_pseudo_labels(
        input_file=input_file, complex_words_csv=csv_path,
        output_file=output_file_1, limit=5, seed=13,
    )
    generate_complex_word_pseudo_labels(
        input_file=input_file, complex_words_csv=csv_path,
        output_file=output_file_2, limit=5, seed=13,
    )

    assert output_file_1.read_text(encoding="utf-8") == output_file_2.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def test_build_parser_defaults():
    args = build_parser().parse_args([])

    assert args.input_file.endswith("sentences.txt")
    assert args.complex_words_csv.endswith("cefr_vocabularies.csv")
    assert args.output_file.endswith("complex_words_pseudo_labels.jsonl")
    assert args.limit is None
    assert args.seed == 13
    assert args.skip_mojibake is False
    assert args.include_phrases is False
    assert args.source == "complex_words_csv"
    assert args.model == "lexicon"


def test_build_parser_accepts_overrides():
    args = build_parser().parse_args([
        "--input-file", "custom.txt",
        "--complex-words-csv", "custom_lexicon.csv",
        "--output-file", "custom_out.jsonl",
        "--limit", "50",
        "--seed", "7",
        "--skip-mojibake",
        "--include-phrases",
        "--source", "custom_source",
        "--model", "custom_model",
    ])

    assert args.input_file == "custom.txt"
    assert args.complex_words_csv == "custom_lexicon.csv"
    assert args.output_file == "custom_out.jsonl"
    assert args.limit == 50
    assert args.seed == 7
    assert args.skip_mojibake is True
    assert args.include_phrases is True
    assert args.source == "custom_source"
    assert args.model == "custom_model"


# ---------------------------------------------------------------------------
# 12-13. no forbidden imports
# ---------------------------------------------------------------------------


def _module_source_path() -> Path:
    import explanation.scripts.generate_complex_word_pseudo_labels as module

    return Path(module.__file__)


def _imported_module_names(source: str) -> list[str]:
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_does_not_import_external_llm_clients():
    source = _module_source_path().read_text(encoding="utf-8")
    imported = [name.lower() for name in _imported_module_names(source)]

    assert not any("openrouter" in name for name in imported)


def test_does_not_import_raw_dataset_or_build_raw_from_aligned():
    source = _module_source_path().read_text(encoding="utf-8")
    imported = [name.lower() for name in _imported_module_names(source)]

    assert not any("raw_dataset" in name for name in imported)
    assert not any("build_raw_from_aligned" in name for name in imported)


def test_only_uses_standard_library_imports():
    source = _module_source_path().read_text(encoding="utf-8")
    imported = _imported_module_names(source)

    allowed_top_level = {
        "__future__", "argparse", "csv", "functools", "json", "random", "re", "pathlib",
    }
    top_level_names = {name.split(".")[0] for name in imported}

    assert top_level_names <= allowed_top_level
