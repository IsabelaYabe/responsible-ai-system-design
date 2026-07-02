"""
Concatenate CEFR vocabulary CSV files into one normalized CSV with only:
    lemma,label

Default behavior:
- Reads vocabulary-style files with columns like lemma/headword/word and CEFR/label.
- Skips text-level corpora such as cefr_leveled_texts.csv unless --include-text-files
  is passed.
- Keeps only labels A1, A2, B1, B2, C1, C2.
- Converts every lexical item to lowercase.
- Uses spaCy to lemmatize English forms.
- Splits slash-separated variants into separate rows.
  Example: T-shirt/tee-shirt,A2 -> t-shirt,A2 and tee-shirt,A2.
- Deduplicates by normalized lowercase lemma and keeps the highest CEFR level by
  default. Use --duplicate-policy first/lowest/all to change this.

Requirements:
    pip install spacy
    python -m spacy download en_core_web_sm

Example:
    python explanation/data/concat_cefr_vocabularies.py \
      --inputs explanation/data/raw/cefrj-vocabulary-profile-1.5.csv explanation/data/raw/ENGLISH_CERF_WORDS.csv \
      --output explanation/data/complex_words.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import spacy


VALID_LABELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LABEL_RANK = {label: index for index, label in enumerate(VALID_LABELS)}

WORD_COLUMNS = ["lemma", "headword", "word", "item", "vocabulary"]
LABEL_COLUMNS = ["label", "CEFR", "cefr", "level", "cefr_level", "CEFR_Level"]
TEXT_COLUMNS = ["text"]

_NLP = None


def get_nlp(model_name: str):
    """Load spaCy model once and reuse it."""
    global _NLP

    if _NLP is not None:
        return _NLP

    try:
        _NLP = spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model not found: {model_name!r}. Install it with:\n"
            f"python -m spacy download {model_name}"
        ) from exc

    return _NLP


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first candidate that exists in columns, case-sensitive first."""
    for candidate in candidates:
        if candidate in columns:
            return candidate

    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return None


def clean_lemma_text(text: str) -> str:
    """Normalize spacing, hyphen spacing, and lowercase."""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return text.strip()


def lemmatize_variant(value: str, model_name: str) -> str:
    """Convert one lexical variant to its lowercase spaCy lemma form.

    Examples:
        mice -> mouse
        went -> go
        feet -> foot
        men -> man
        T-shirt -> t-shirt
        t-shirts -> t-shirt
    """
    value = clean_lemma_text(value)

    if not value or value == "nan":
        return ""

    nlp = get_nlp(model_name)
    doc = nlp(value)

    lemmas: list[str] = []
    for token in doc:
        if token.is_space:
            continue

        lemma = token.lemma_.strip().lower()

        if not lemma or lemma == "-pron-":
            lemma = token.text.strip().lower()

        lemmas.append(lemma)

    lemma_text = " ".join(lemmas)
    lemma_text = clean_lemma_text(lemma_text)

    if not lemma_text or lemma_text == "nan":
        return ""

    return lemma_text


def expand_slash_variants_to_lemmas(value: str, model_name: str) -> list[str]:
    """Split slash-separated variants into independent lowercase lemmas.

    Examples:
        "T-shirt/tee-shirt" -> ["t-shirt", "tee-shirt"]
        "mice/men" -> ["mouse", "man"]
        "running/runs" -> ["run"]
    """
    if not isinstance(value, str):
        return []

    value = value.strip()
    if not value or value.lower() == "nan":
        return []

    lemmas: list[str] = []

    for part in value.split("/"):
        lemma = lemmatize_variant(part, model_name=model_name)
        if lemma and lemma != "nan":
            lemmas.append(lemma)

    return lemmas


def read_cefr_file(
    path: Path,
    include_text_files: bool = False,
    spacy_model: str = "en_core_web_sm",
) -> pd.DataFrame:
    """Read one CSV and normalize it to lemma,label."""
    df = pd.read_csv(path, encoding="utf-8-sig")

    word_column = find_column(list(df.columns), WORD_COLUMNS)
    label_column = find_column(list(df.columns), LABEL_COLUMNS)

    if word_column is None and include_text_files:
        word_column = find_column(list(df.columns), TEXT_COLUMNS)

    if word_column is None:
        print(f"SKIP {path}: no vocabulary/lemma column found. Columns: {list(df.columns)}")
        return pd.DataFrame(columns=["lemma", "label"])

    if label_column is None:
        print(f"SKIP {path}: no CEFR/label column found. Columns: {list(df.columns)}")
        return pd.DataFrame(columns=["lemma", "label"])

    out = df[[word_column, label_column]].copy()
    out.columns = ["raw_word", "label"]

    out["label"] = out["label"].astype(str).str.strip().str.upper()
    out = out[out["label"].isin(VALID_LABELS)]

    rows: list[dict[str, str]] = []

    for _, row in out.iterrows():
        label = str(row["label"]).strip().upper()
        lemmas = expand_slash_variants_to_lemmas(
            str(row["raw_word"]),
            model_name=spacy_model,
        )

        for lemma in lemmas:
            rows.append({"lemma": lemma, "label": label})

    if not rows:
        return pd.DataFrame(columns=["lemma", "label"])

    normalized = pd.DataFrame(rows, columns=["lemma", "label"])

    normalized["lemma"] = normalized["lemma"].astype(str).str.strip().str.lower()
    normalized["label"] = normalized["label"].astype(str).str.strip().str.upper()

    normalized = normalized[normalized["lemma"].ne("")]
    normalized = normalized[normalized["lemma"].str.lower().ne("nan")]
    normalized = normalized[normalized["label"].isin(VALID_LABELS)]

    return normalized[["lemma", "label"]]


def deduplicate(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Deduplicate lemmas according to the chosen policy."""
    work = df.copy()

    work["lemma"] = work["lemma"].astype(str).str.strip().str.lower()
    work["label"] = work["label"].astype(str).str.strip().str.upper()

    work = work[work["lemma"].ne("")]
    work = work[work["lemma"].str.lower().ne("nan")]
    work = work[work["label"].isin(VALID_LABELS)]

    if policy == "all":
        return (
            work.drop_duplicates(subset=["lemma", "label"])
            .sort_values(["label", "lemma"])
            .reset_index(drop=True)
        )

    work["_lemma_norm"] = work["lemma"].str.lower()
    work["_rank"] = work["label"].map(LABEL_RANK)

    if policy == "first":
        out = work.drop_duplicates(subset=["_lemma_norm"], keep="first")
    elif policy == "lowest":
        out = (
            work.sort_values(["_lemma_norm", "_rank"], ascending=[True, True])
            .drop_duplicates(subset=["_lemma_norm"], keep="first")
        )
    elif policy == "highest":
        out = (
            work.sort_values(["_lemma_norm", "_rank"], ascending=[True, False])
            .drop_duplicates(subset=["_lemma_norm"], keep="first")
        )
    else:
        raise ValueError(f"Invalid duplicate policy: {policy}")

    return out[["lemma", "label"]].sort_values(["label", "lemma"]).reset_index(drop=True)


def concatenate_cefr_files(
    input_paths: list[Path],
    output_path: Path,
    duplicate_policy: str = "highest",
    include_text_files: bool = False,
    spacy_model: str = "en_core_web_sm",
) -> pd.DataFrame:
    """Concatenate CEFR files and save lemma,label CSV."""
    frames = [
        read_cefr_file(
            path,
            include_text_files=include_text_files,
            spacy_model=spacy_model,
        )
        for path in input_paths
    ]
    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        raise ValueError("No valid rows found in the provided input files.")

    merged_raw = pd.concat(frames, ignore_index=True)
    merged = deduplicate(merged_raw, duplicate_policy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8")

    print("=== CEFR lemma merge complete ===")
    print(f"Input files       : {len(input_paths)}")
    print(f"Rows read         : {len(merged_raw)}")
    print(f"Rows written      : {len(merged)}")
    print(f"Duplicate policy  : {duplicate_policy}")
    print(f"spaCy model       : {spacy_model}")
    print(f"Output            : {output_path}")

    print("\nLabel distribution:")
    print(merged["label"].value_counts().reindex(VALID_LABELS, fill_value=0).to_string())

    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concatenate CEFR vocabulary bases into a two-column lemma,label CSV."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input CSV files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=["highest", "lowest", "first", "all"],
        default="highest",
        help=(
            "How to handle duplicated lemmas. "
            "highest keeps the highest CEFR level; lowest keeps the lowest; "
            "first keeps first occurrence; all keeps unique lemma,label pairs."
        ),
    )
    parser.add_argument(
        "--include-text-files",
        action="store_true",
        help="Allow files with text,label columns by mapping text -> lemma.",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="spaCy English model used for lemmatization.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    concatenate_cefr_files(
        input_paths=[Path(path) for path in args.inputs],
        output_path=Path(args.output),
        duplicate_policy=args.duplicate_policy,
        include_text_files=args.include_text_files,
        spacy_model=args.spacy_model,
    )


if __name__ == "__main__":
    main()