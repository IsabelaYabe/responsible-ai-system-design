"""Generate difficult-word pseudo-labels from a local CEFR complex-word lexicon.

This is a lexical baseline, not contextual LLM labeling. A sentence's difficult
words are the sentence spans whose lowercase spaCy lemmas match entries in
``explanation/data/cefr_vocabularies.csv``.

The output rows use the same sentence-level pseudo-label JSONL schema consumed
by ``build_edit_predictor_dataset.py`` and ``train_edit_predictor.py``.

Requirements:
    pip install spacy
    python -m spacy download en_core_web_sm

Usage:
    python -m explanation.scripts.generate_complex_word_pseudo_labels \
        --input-file explanation/data/sentences.txt \
        --complex-words-csv explanation/data/cefr_vocabularies.csv \
        --output-file explanation/data/processed/complex_words_pseudo_labels.jsonl \
        --cefr-levels "B2,C1,C2" \
        --skip-mojibake
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import spacy

DEFAULT_INPUT_FILE = Path("explanation/data/sentences.txt")
DEFAULT_COMPLEX_WORDS_CSV = Path("explanation/data/cefr_vocabularies.csv")
DEFAULT_OUTPUT_FILE = Path("explanation/data/processed/complex_words_pseudo_labels.jsonl")
DEFAULT_SOURCE = "complex_words_csv"
DEFAULT_MODEL = "spacy_lemma_lexicon"
DEFAULT_SPACY_MODEL = "en_core_web_sm"

VALID_CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
DEFAULT_CEFR_LEVELS = ("B2", "C1", "C2")
ACCEPTED_CEFR_LEVELS = set(DEFAULT_CEFR_LEVELS)

# Checked in order; the first column present in the CSV header wins.
WORD_COLUMN_CANDIDATES = ("lemma", "headword", "word", "item", "vocabulary", "text")
CEFR_COLUMN_CANDIDATES = ("label", "CEFR", "cefr", "level", "cefr_level", "CEFR_Level")

MOJIBAKE_MARKERS = (
    "Â", "Ã", "Ä", "Å", "Ç", "É", "Ê", "Ë", "Î", "Ï",
    "â", "ã", "ä", "å", "æ", "ç", "è", "é", "ë", "ì",
    "Ù", "Ú", "Û", "Ø",
    "�",
)

_NLP_CACHE: dict[str, spacy.language.Language] = {}


@dataclass(frozen=True)
class LemmaSpan:
    """One surface span from a sentence with its normalized lemma."""

    lemma: str
    start: int
    end: int


def get_nlp(model_name: str = DEFAULT_SPACY_MODEL):
    """Load a spaCy model once and reuse it."""
    if model_name in _NLP_CACHE:
        return _NLP_CACHE[model_name]

    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model not found: {model_name!r}. Install it with:\n"
            f"python -m spacy download {model_name}"
        ) from exc

    _NLP_CACHE[model_name] = nlp
    return nlp


def has_mojibake(text: str) -> bool:
    """Return True when text contains likely encoding-corrupted characters."""
    return (not text.isascii()) or any(marker in text for marker in MOJIBAKE_MARKERS)


def read_sentence_file(path: Path) -> list[str]:
    """Read one sentence per line from *path*, stripped, skipping blank lines."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sentence file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def parse_cefr_levels(values: Sequence[str] | str | None) -> set[str]:
    """Parse CEFR levels from CLI-friendly values.

    Accepted forms:
        --cefr-levels "B2,C1,C2"
        --cefr-levels "B2 C1 C2"
        --cefr-levels B2 C1 C2
    """
    if values is None:
        return set(DEFAULT_CEFR_LEVELS)

    if isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = list(values)

    levels: set[str] = set()
    for value in raw_values:
        for part in re.split(r"[,\s]+", str(value).strip()):
            if part:
                levels.add(part.upper())

    if not levels:
        raise ValueError("At least one CEFR level must be provided.")

    invalid = sorted(level for level in levels if level not in VALID_CEFR_LEVELS)
    if invalid:
        raise ValueError(
            f"Invalid CEFR level(s): {', '.join(invalid)}. "
            f"Expected only: {', '.join(VALID_CEFR_LEVELS)}."
        )

    return levels


def _find_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> str | None:
    """Find a CSV column by exact name first, then case-insensitively."""
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate

    lowered = {field.lower(): field for field in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return None


def _clean_text(text: str) -> str:
    """Normalize whitespace, hyphen spacing, and lowercase."""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return text.strip()


def _token_lemma(token) -> str:
    """Return a lowercase lemma for one spaCy token."""
    lemma = (token.lemma_ or token.text).strip().lower()
    if not lemma or lemma == "-pron-":
        lemma = token.text.strip().lower()
    return lemma


def lemmatize_text(text: str, spacy_model: str = DEFAULT_SPACY_MODEL) -> str:
    """Lemmatize a lexicon entry into the same normalized form used for matching.

    Hyphenated lexical entries are preserved as hyphenated lemmas when spaCy
    tokenizes them as multiple adjacent pieces.
    """
    text = _clean_text(text)
    if not text or text == "nan":
        return ""

    nlp = get_nlp(spacy_model)
    doc = nlp(text)
    units = _lemma_units_from_doc(doc)
    lemma = " ".join(unit.lemma for unit in units)
    return _clean_text(lemma)


def expand_slash_variants_to_lemmas(
    raw_value: str,
    spacy_model: str = DEFAULT_SPACY_MODEL,
) -> list[str]:
    """Split slash-separated variants into independent lowercase lemmas.

    Examples:
        "T-shirt/tee-shirt" -> ["t-shirt", "tee-shirt"]
        "mice/men" -> ["mouse", "man"]
        "running/runs" -> ["run"]
    """
    if not isinstance(raw_value, str):
        return []

    raw_value = raw_value.strip()
    if not raw_value or raw_value.lower() == "nan":
        return []

    lemmas: list[str] = []
    seen: set[str] = set()
    for part in raw_value.split("/"):
        lemma = lemmatize_text(part, spacy_model=spacy_model)
        if lemma and lemma != "nan" and lemma not in seen:
            seen.add(lemma)
            lemmas.append(lemma)

    return lemmas


def read_complex_word_csv(
    path: Path,
    accepted_cefr_levels: set[str] | None = None,
    include_phrases: bool = False,
    spacy_model: str = DEFAULT_SPACY_MODEL,
) -> set[str]:
    """Load the local complex-word lexicon as a lowercase lemma set.

    If a CEFR/label column is present, only rows whose label is in
    ``accepted_cefr_levels`` are kept. By default, B2, C1, and C2 are kept.
    Entries are lowercased and lemmatized with spaCy. Multi-word phrases are
    dropped unless ``include_phrases=True``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Complex word CSV not found: {path}")

    levels = accepted_cefr_levels or set(DEFAULT_CEFR_LEVELS)

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []

        word_column = _find_column(fieldnames, WORD_COLUMN_CANDIDATES)
        if word_column is None:
            raise ValueError(
                f"Could not find a lemma/word column in {path}. "
                f"Expected one of: {', '.join(WORD_COLUMN_CANDIDATES)}. "
                f"Found columns: {', '.join(fieldnames)}"
            )

        cefr_column = _find_column(fieldnames, CEFR_COLUMN_CANDIDATES)

        lemmas: set[str] = set()
        for row in reader:
            if cefr_column is not None:
                level = (row.get(cefr_column) or "").strip().upper()
                if level not in levels:
                    continue

            raw_word = (row.get(word_column) or "").strip()
            for lemma in expand_slash_variants_to_lemmas(raw_word, spacy_model=spacy_model):
                if not lemma:
                    continue
                if " " in lemma and not include_phrases:
                    continue
                lemmas.add(lemma)

    return lemmas


def _lemma_units_from_doc(doc) -> list[LemmaSpan]:
    """Return sentence units used for lemma matching.

    Adjacent hyphenated tokens are merged into one lemma span, so forms such as
    ``T-shirt`` and ``self-conscious`` can match hyphenated lexicon lemmas.
    Other punctuation is ignored.
    """
    units: list[LemmaSpan] = []
    current: list = []

    def flush_current() -> None:
        nonlocal current
        if not current:
            return

        # Drop dangling hyphens.
        while current and current[0].text == "-":
            current = current[1:]
        while current and current[-1].text == "-":
            current = current[:-1]

        if not current:
            return

        if any(token.text == "-" for token in current):
            lemma_parts = ["-" if token.text == "-" else _token_lemma(token) for token in current]
            lemma = "".join(lemma_parts)
        else:
            lemma = " ".join(_token_lemma(token) for token in current)

        lemma = _clean_text(lemma)
        if lemma:
            units.append(
                LemmaSpan(
                    lemma=lemma,
                    start=int(current[0].idx),
                    end=int(current[-1].idx + len(current[-1].text)),
                )
            )
        current = []

    for token in doc:
        if token.is_space:
            flush_current()
            continue

        if token.is_punct and token.text != "-":
            flush_current()
            continue

        if token.text == "-":
            if current and token.idx == current[-1].idx + len(current[-1].text):
                current.append(token)
            else:
                flush_current()
            continue

        if not current:
            current = [token]
            continue

        previous_end = current[-1].idx + len(current[-1].text)
        previous_is_hyphen = current[-1].text == "-"
        current_has_hyphen = any(part.text == "-" for part in current)

        if token.idx == previous_end and (previous_is_hyphen or current_has_hyphen):
            current.append(token)
        else:
            flush_current()
            current = [token]

    flush_current()
    return units


def _build_lemma_index(complex_lemmas: set[str]) -> dict[int, set[tuple[str, ...]]]:
    """Group complex lemmas by token length for efficient n-gram matching."""
    index: dict[int, set[tuple[str, ...]]] = {}
    for lemma in complex_lemmas:
        parts = tuple(part for part in lemma.split(" ") if part)
        if not parts:
            continue
        index.setdefault(len(parts), set()).add(parts)
    return index


def find_lemma_spans(
    sentence: str,
    complex_lemmas: set[str],
    include_phrases: bool = False,
    spacy_model: str = DEFAULT_SPACY_MODEL,
) -> list[tuple[int, int]]:
    """Return sentence spans whose lowercase spaCy lemma matches the lexicon."""
    if not sentence or not complex_lemmas:
        return []

    nlp = get_nlp(spacy_model)
    units = _lemma_units_from_doc(nlp(sentence))
    lemma_index = _build_lemma_index(complex_lemmas)

    candidates: list[tuple[int, int]] = []

    # Single-token lemma matches.
    for unit in units:
        if (unit.lemma,) in lemma_index.get(1, set()):
            candidates.append((unit.start, unit.end))

    # Optional multi-word lemma phrase matches.
    if include_phrases:
        for length, phrase_set in lemma_index.items():
            if length <= 1:
                continue
            for start_index in range(0, len(units) - length + 1):
                window = tuple(unit.lemma for unit in units[start_index : start_index + length])
                if window in phrase_set:
                    candidates.append((units[start_index].start, units[start_index + length - 1].end))

    return _deduplicate_non_overlapping_spans(candidates)


def _deduplicate_non_overlapping_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Deduplicate spans, preferring longer spans when overlaps occur."""
    unique = set(spans)
    ordered = sorted(unique, key=lambda span: (span[0], -(span[1] - span[0]), span[1]))

    kept: list[tuple[int, int]] = []
    for start, end in ordered:
        if any(start < kept_end and kept_start < end for kept_start, kept_end in kept):
            continue
        kept.append((start, end))

    return sorted(kept, key=lambda span: (span[0], span[1]))


@functools.lru_cache(maxsize=None)
def _compiled_pattern(word: str, ignorecase: bool) -> re.Pattern:
    """Compile and cache a full-word/term matching regex.

    Kept for backward compatibility with older tests/imports. The maintained
    pseudo-label path uses ``find_lemma_spans`` instead.
    """
    pattern = rf"(?<![A-Za-z0-9_-]){re.escape(word)}(?![A-Za-z0-9_-])"
    flags = re.IGNORECASE if ignorecase else 0
    return re.compile(pattern, flags)


def find_all_spans(sentence: str, word: str) -> list[tuple[int, int]]:
    """Return full-word raw string spans for backward compatibility."""
    if not word:
        return []

    if word.lower() not in sentence.lower():
        return []

    standalone_ci = _compiled_pattern(word, ignorecase=True)
    return [match.span() for match in standalone_ci.finditer(sentence)]


def build_pseudo_label_row(
    sentence: str,
    complex_words: set[str],
    source: str,
    model: str,
    include_phrases: bool = False,
    spacy_model: str = DEFAULT_SPACY_MODEL,
) -> dict:
    """Build one pseudo-label JSONL row for *sentence*.

    Matching is performed by lowercase spaCy lemma equality. The JSONL output
    still stores the original surface form and character span from the sentence.
    """
    matches = find_lemma_spans(
        sentence=sentence,
        complex_lemmas=complex_words,
        include_phrases=include_phrases,
        spacy_model=spacy_model,
    )

    return {
        "sentence": sentence,
        "difficult_words": [sentence[start:end] for start, end in matches],
        "spans": [[start, end] for start, end in matches],
        "source": source,
        "model": model,
    }


def generate_complex_word_pseudo_labels(
    input_file: Path,
    complex_words_csv: Path,
    output_file: Path,
    limit: int | None = None,
    seed: int = 13,
    skip_mojibake: bool = False,
    include_phrases: bool = False,
    accepted_cefr_levels: set[str] | None = None,
    spacy_model: str = DEFAULT_SPACY_MODEL,
    source: str = DEFAULT_SOURCE,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Generate lexicon-based pseudo-labels for sentences in *input_file*."""
    input_file = Path(input_file)
    complex_words_csv = Path(complex_words_csv)
    output_file = Path(output_file)
    levels = accepted_cefr_levels or set(DEFAULT_CEFR_LEVELS)

    sentences = read_sentence_file(input_file)
    sentences_read = len(sentences)

    if skip_mojibake:
        sentences = [sentence for sentence in sentences if not has_mojibake(sentence)]

    random.Random(seed).shuffle(sentences)
    if limit is not None:
        sentences = sentences[:limit]

    complex_words = read_complex_word_csv(
        complex_words_csv,
        accepted_cefr_levels=levels,
        include_phrases=include_phrases,
        spacy_model=spacy_model,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    sentences_with_complex_words = 0
    sentences_without_complex_words = 0
    total_complex_word_matches = 0

    with output_file.open("w", encoding="utf-8") as out_fh:
        for sentence in sentences:
            row = build_pseudo_label_row(
                sentence,
                complex_words,
                source=source,
                model=model,
                include_phrases=include_phrases,
                spacy_model=spacy_model,
            )
            if row["difficult_words"]:
                sentences_with_complex_words += 1
                total_complex_word_matches += len(row["difficult_words"])
            else:
                sentences_without_complex_words += 1
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "input_file": str(input_file),
        "complex_words_csv": str(complex_words_csv),
        "output_file": str(output_file),
        "accepted_cefr_levels": sorted(levels),
        "spacy_model": spacy_model,
        "lexicon_size": len(complex_words),
        "sentences_read": sentences_read,
        "sentences_written": len(sentences),
        "sentences_with_complex_words": sentences_with_complex_words,
        "sentences_without_complex_words": sentences_without_complex_words,
        "total_complex_word_matches": total_complex_word_matches,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate difficult-word pseudo-labels from a local CEFR lexicon "
            "using lowercase spaCy lemma matching. No external API calls."
        )
    )
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_INPUT_FILE),
        help="Path to local sentence file (one sentence per line).",
    )
    parser.add_argument(
        "--complex-words-csv",
        default=str(DEFAULT_COMPLEX_WORDS_CSV),
        help="Path to the CEFR complex-word lexicon CSV.",
    )
    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Path to write pseudo-label JSONL rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of sentences to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Deterministic shuffle seed used before applying --limit.",
    )
    parser.add_argument(
        "--skip-mojibake",
        action="store_true",
        help="Skip sentences containing obvious mojibake/encoding artifacts.",
    )
    parser.add_argument(
        "--include-phrases",
        action="store_true",
        help="Keep and match multi-word lexicon entries.",
    )
    parser.add_argument(
        "--cefr-levels",
        nargs="+",
        default=list(DEFAULT_CEFR_LEVELS),
        help=(
            "CEFR levels to treat as difficult. Accepts quoted comma/space lists, "
            "e.g. --cefr-levels \"B2,C1,C2\". Default: B2 C1 C2."
        ),
    )
    parser.add_argument(
        "--spacy-model",
        default=DEFAULT_SPACY_MODEL,
        help="spaCy English model used for sentence and lexicon lemmatization.",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Value written to the JSONL 'source' field.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Value written to the JSONL 'model' field.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        accepted_cefr_levels = parse_cefr_levels(args.cefr_levels)
    except ValueError as exc:
        parser.error(str(exc))

    summary = generate_complex_word_pseudo_labels(
        input_file=Path(args.input_file),
        complex_words_csv=Path(args.complex_words_csv),
        output_file=Path(args.output_file),
        limit=args.limit,
        seed=args.seed,
        skip_mojibake=args.skip_mojibake,
        include_phrases=args.include_phrases,
        accepted_cefr_levels=accepted_cefr_levels,
        spacy_model=args.spacy_model,
        source=args.source,
        model=args.model,
    )

    print("=== Complex-word lexicon pseudo-label generation complete ===")
    for key, value in summary.items():
        print(f"{key:<32}: {value}")


if __name__ == "__main__":
    main()
