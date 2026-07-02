"""CLI entry point: run the maintained explanation pipeline on one sentence.

Usage:
    python -m explanation.scripts.run_inference \
        --sentence "A committee of the institute appoints the laureates for the Nobel Prize." \
        --edit-predictor-checkpoint explanation/model/checkpoints/edit_predictor_complex_words_distilbert_max256_big_final \
        --max-length 256

OPENROUTER_API_KEY must be set in your environment or in a .env file at the
repo root when explanations need to be generated.
"""

import argparse
import json
from dataclasses import asdict

from explanation.inference.pipeline import ExplanationPipeline
from explanation.llm.openrouter_client import DEFAULT_MODEL

DEFAULT_EDIT_PREDICTOR_CHECKPOINT = (
    "explanation/model/checkpoints/edit_predictor_complex_words_distilbert_max256_big_final"
)


class _LazyOpenRouterClient:
    """Create the real OpenRouter client only when an explanation is needed."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from explanation.llm.openrouter_client import OpenRouterClient

            self._client = OpenRouterClient(model=self.model)
        return self._client

    def explain_difficult_words(self, sentence: str, words: list[str]) -> dict:
        return self._get_client().explain_difficult_words(sentence, words)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Identify and explain difficult words in an English sentence."
    )
    parser.add_argument("--sentence", required=True, help="English sentence to process.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model to use for explanations (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--edit-predictor-checkpoint",
        default=DEFAULT_EDIT_PREDICTOR_CHECKPOINT,
        help="Supervised Edit Predictor checkpoint used for local identification.",
    )
    parser.add_argument(
        "--edit-predictor-threshold",
        type=float,
        default=0.5,
        help="M-class probability threshold for local Edit Predictor identification.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokenizer max sequence length for local Edit Predictor identification.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device for local Edit Predictor inference, e.g. cuda or cpu.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    pipeline = ExplanationPipeline(
        client=_LazyOpenRouterClient(model=args.model),
        edit_predictor_checkpoint=args.edit_predictor_checkpoint,
        edit_predictor_threshold=args.edit_predictor_threshold,
        max_length=args.max_length,
        device=args.device,
    )

    result = pipeline.run(args.sentence)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
