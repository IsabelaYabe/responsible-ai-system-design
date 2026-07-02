"""Tests for the run_inference CLI argument wiring."""

import json

from explanation.scripts import run_inference
from explanation.schemas import ExplanationResult


def test_build_parser_accepts_edit_predictor_arguments():
    args = run_inference.build_parser().parse_args([
        "--sentence", "A sentence.",
        "--edit-predictor-checkpoint", "checkpoint",
        "--edit-predictor-threshold", "0.5",
        "--max-length", "64",
        "--device", "cpu",
    ])

    assert "identifier" not in vars(args)
    assert args.edit_predictor_checkpoint == "checkpoint"
    assert args.edit_predictor_threshold == 0.5
    assert args.max_length == 64
    assert args.device == "cpu"


def test_build_parser_help_describes_current_architecture_only():
    help_text = run_inference.build_parser().format_help()

    assert "identifier-mode" not in help_text
    assert "LLM identification" not in help_text
    legacy_key = "GROQ" + "_API_KEY"
    assert legacy_key not in help_text
    assert "OpenRouter" in help_text


def test_main_passes_edit_predictor_options_to_pipeline(monkeypatch, capsys):
    calls = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def run(self, sentence):
            calls["sentence"] = sentence
            return ExplanationResult(sentence=sentence, difficult_words=[])

    monkeypatch.setattr(run_inference, "ExplanationPipeline", FakePipeline)

    run_inference.main([
        "--sentence", "A sentence.",
        "--edit-predictor-checkpoint", "checkpoint",
        "--edit-predictor-threshold", "0.5",
        "--max-length", "64",
        "--device", "cpu",
    ])

    assert calls["sentence"] == "A sentence."
    assert calls["init"]["edit_predictor_checkpoint"] == "checkpoint"
    assert calls["init"]["edit_predictor_threshold"] == 0.5
    assert calls["init"]["max_length"] == 64
    assert calls["init"]["device"] == "cpu"

    output = json.loads(capsys.readouterr().out)
    assert output == {"sentence": "A sentence.", "difficult_words": []}
