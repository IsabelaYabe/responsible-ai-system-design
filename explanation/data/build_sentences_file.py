"""
Build a shuffled sentences.txt file from complex/simple sentence corpora.

Each input file must contain one sentence per line. The script reads both files,
optionally deduplicates exact repeated sentences, shuffles deterministically, and
writes the requested number of sentences to the output file.

Default paths match the explanation module layout:
    explanation/data/raw/complex.txt
    explanation/data/raw/simple.txt
    -> explanation/data/sentences.txt

Example:
    python explanation/data/build_sentences_file.py --size 6000

Balanced example, taking half from each file before shuffling:
    python explanation/data/build_sentences_file.py --size 6000 --balanced
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


DEFAULT_COMPLEX_FILE = Path("explanation/data/raw/complex.txt")
DEFAULT_SIMPLE_FILE = Path("explanation/data/raw/simple.txt")
DEFAULT_OUTPUT_FILE = Path("explanation/data/sentences.txt")
DEFAULT_SIZE = 6000
DEFAULT_SEED = 13


def read_sentences(path: Path) -> list[str]:
    """Read one stripped sentence per non-empty line."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Input path is not a file: {path}")

    with path.open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def deduplicate_preserving_order(sentences: list[str]) -> list[str]:
    """Remove exact duplicate sentences while preserving the first occurrence."""
    seen: set[str] = set()
    unique: list[str] = []

    for sentence in sentences:
        if sentence in seen:
            continue
        seen.add(sentence)
        unique.append(sentence)

    return unique


def sample_mixed(
    complex_sentences: list[str],
    simple_sentences: list[str],
    size: int,
    rng: random.Random,
) -> list[str]:
    """Concatenate both corpora, shuffle, and take *size* sentences."""
    pool = list(complex_sentences) + list(simple_sentences)

    if size > len(pool):
        raise ValueError(
            f"Requested size {size}, but only {len(pool)} sentences are available."
        )

    rng.shuffle(pool)
    return pool[:size]


def sample_balanced(
    complex_sentences: list[str],
    simple_sentences: list[str],
    size: int,
    rng: random.Random,
) -> list[str]:
    """Sample approximately half from complex and half from simple, then shuffle."""
    complex_count = size // 2
    simple_count = size - complex_count

    if complex_count > len(complex_sentences):
        raise ValueError(
            f"Requested {complex_count} complex sentences, but only "
            f"{len(complex_sentences)} are available."
        )
    if simple_count > len(simple_sentences):
        raise ValueError(
            f"Requested {simple_count} simple sentences, but only "
            f"{len(simple_sentences)} are available."
        )

    complex_pool = list(complex_sentences)
    simple_pool = list(simple_sentences)
    rng.shuffle(complex_pool)
    rng.shuffle(simple_pool)

    selected = complex_pool[:complex_count] + simple_pool[:simple_count]
    rng.shuffle(selected)
    return selected


def build_sentences_file(
    complex_file: Path,
    simple_file: Path,
    output_file: Path,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
    balanced: bool = False,
    keep_duplicates: bool = False,
) -> dict[str, int | str | bool]:
    """Build the shuffled output file and return a summary."""
    if size < 1:
        raise ValueError("--size must be at least 1")

    complex_sentences = read_sentences(complex_file)
    simple_sentences = read_sentences(simple_file)

    complex_read = len(complex_sentences)
    simple_read = len(simple_sentences)

    if not keep_duplicates:
        complex_sentences = deduplicate_preserving_order(complex_sentences)
        simple_sentences = deduplicate_preserving_order(simple_sentences)

        # Remove sentences that occur in both files, keeping the complex-side copy.
        complex_set = set(complex_sentences)
        simple_sentences = [s for s in simple_sentences if s not in complex_set]

    rng = random.Random(seed)
    if balanced:
        selected = sample_balanced(complex_sentences, simple_sentences, size, rng)
    else:
        selected = sample_mixed(complex_sentences, simple_sentences, size, rng)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="\n") as fh:
        for sentence in selected:
            fh.write(sentence + "\n")

    return {
        "complex_file": str(complex_file),
        "simple_file": str(simple_file),
        "output_file": str(output_file),
        "size_requested": size,
        "size_written": len(selected),
        "seed": seed,
        "balanced": balanced,
        "keep_duplicates": keep_duplicates,
        "complex_rows_read": complex_read,
        "simple_rows_read": simple_read,
        "complex_rows_available": len(complex_sentences),
        "simple_rows_available": len(simple_sentences),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a shuffled sentences.txt file from complex.txt and simple.txt."
    )
    parser.add_argument(
        "--complex-file",
        default=str(DEFAULT_COMPLEX_FILE),
        help="Path to complex.txt, one sentence per line.",
    )
    parser.add_argument(
        "--simple-file",
        default=str(DEFAULT_SIMPLE_FILE),
        help="Path to simple.txt, one sentence per line.",
    )
    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Path to write the shuffled sentences.txt file.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help="Number of sentences to write.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic shuffle seed.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Take approximately half from complex.txt and half from simple.txt.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep exact duplicate sentences instead of removing them.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_sentences_file(
        complex_file=Path(args.complex_file),
        simple_file=Path(args.simple_file),
        output_file=Path(args.output_file),
        size=args.size,
        seed=args.seed,
        balanced=args.balanced,
        keep_duplicates=args.keep_duplicates,
    )

    print("=== sentences.txt build complete ===")
    for key, value in summary.items():
        print(f"{key:<24}: {value}")


if __name__ == "__main__":
    main()
