"""Local difficult-word identification with the supervised Edit Predictor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from explanation.model.edit_predictor import EditPredictor


class EditPredictorIdentifier:
    """Identify difficult words locally with a trained Edit Predictor checkpoint."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_length: int = 128,
        device: str | None = None,
        threshold: float | None = None,
    ) -> None:
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists() or not checkpoint_path.is_dir():
            raise FileNotFoundError(
                f"Edit Predictor checkpoint not found: {checkpoint_path}"
            )
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        self.checkpoint_dir = checkpoint_path
        self.max_length = max_length
        self.device = device
        self.threshold = threshold
        self.predictor = EditPredictor.from_checkpoint(checkpoint_path)
        if device is not None:
            self.predictor.model.to(device)

    def identify(self, sentence: str) -> list[dict[str, Any]]:
        """Return difficult words with local spans and optional confidence score."""
        if not sentence:
            return []

        if hasattr(self.predictor, "predict_difficult_spans_with_scores"):
            span_rows = self.predictor.predict_difficult_spans_with_scores(
                [sentence],
                max_length=self.max_length,
                device=self.device,
                threshold=self.threshold,
            )
        else:  # pragma: no cover - compatibility fallback
            span_rows = [[
                {"span": span}
                for span in self.predictor.predict_difficult_spans(
                    [sentence],
                    max_length=self.max_length,
                    device=self.device,
                )[0]
            ]]

        results: list[dict[str, Any]] = []
        for item in span_rows[0] if span_rows else []:
            raw_span = item.get("span")
            if not isinstance(raw_span, (list, tuple)) or len(raw_span) != 2:
                continue
            start, end = int(raw_span[0]), int(raw_span[1])
            if start < 0 or end <= start or end > len(sentence):
                continue

            result: dict[str, Any] = {
                "word": sentence[start:end],
                "span": (start, end),
            }
            if "score" in item:
                result["score"] = float(item["score"])
            elif "average_score" in item:
                result["score"] = float(item["average_score"])
            results.append(result)
        return results
