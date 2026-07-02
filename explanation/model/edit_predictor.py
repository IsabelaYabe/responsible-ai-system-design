"""Transformer wrapper for the token-level Edit Predictor (K/M classifier).

This is the supervised baseline Edit Predictor only: it predicts per-token
K (keep) / M (mask/difficult) labels. It does not generate text or
contextual explanations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

LABEL_KEEP = 0
LABEL_MASK = 1
LABEL_IGNORE = -100
LABEL_NAMES = {
    LABEL_KEEP: "K",
    LABEL_MASK: "M",
}


@dataclass
class EditPredictor:
    """Small wrapper around a Hugging Face token classifier for K/M prediction."""

    model: Any
    tokenizer: Any
    label_names: dict[int, str]

    @classmethod
    def from_base_model(cls, model_name: str = "distilbert-base-uncased") -> "EditPredictor":
        """Create a fresh Edit Predictor from a base Hugging Face model."""
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={LABEL_KEEP: "K", LABEL_MASK: "M"},
            label2id={"K": LABEL_KEEP, "M": LABEL_MASK},
        )
        return cls(model=model, tokenizer=tokenizer, label_names=dict(LABEL_NAMES))

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: str | Path, freeze: bool = False
    ) -> "EditPredictor":
        """Load an already trained Edit Predictor checkpoint.

        Args:
            checkpoint_dir: Path to the saved checkpoint directory.
            freeze: If True, call :meth:`freeze` before returning. Note that
                ``requires_grad=False`` is not persisted in HuggingFace checkpoints, so
                ``freeze=True`` must be passed again on every load.
        """
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        checkpoint_dir = str(checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
        instance = cls(model=model, tokenizer=tokenizer, label_names=dict(LABEL_NAMES))
        if freeze:
            instance.freeze()
        return instance

    def freeze(self) -> "EditPredictor":
        """Freeze model weights (eval mode, no gradients).

        Note: ``requires_grad=False`` is not persisted in HuggingFace checkpoints.
        Re-apply by calling ``from_checkpoint(path, freeze=True)`` on each load.

        Returns:
            ``self``, for method chaining.
        """
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        return self

    def tokenize(
        self,
        sentences: Sequence[str],
        max_length: int = 128,
        device: Any | None = None,
    ) -> dict[str, Any]:
        """Tokenize a batch of sentences for the token classifier."""
        encoded = self.tokenizer(
            list(sentences),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        return encoded

    def predict_token_labels(
        self,
        sentences: Sequence[str],
        max_length: int = 128,
        device: Any | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Predict per-token K/M labels for each sentence.

        Returns, for each sentence, a list of dicts with keys ``token``,
        ``label_id``, and ``label`` for every non-special, non-padding token
        (as determined by the fast tokenizer's ``sequence_ids``).
        """
        self.model.eval()
        sentences = list(sentences)
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        model_inputs = {key: value for key, value in encoded.items()}
        if device is not None:
            model_inputs = {key: value.to(device) for key, value in model_inputs.items()}

        with torch.no_grad():
            outputs = self.model(**model_inputs)
        predicted_ids = outputs.logits.argmax(dim=-1)

        results: list[list[dict[str, Any]]] = []
        for row_index in range(len(sentences)):
            tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][row_index])
            sequence_ids = (
                encoded.sequence_ids(row_index) if hasattr(encoded, "sequence_ids") else None
            )
            attention_row = encoded["attention_mask"][row_index]

            row_result: list[dict[str, Any]] = []
            for token_index, token in enumerate(tokens):
                if sequence_ids is not None and sequence_ids[token_index] is None:
                    continue
                if int(attention_row[token_index]) == 0:
                    continue
                label_id = int(predicted_ids[row_index][token_index].item())
                row_result.append({
                    "token": token,
                    "label_id": label_id,
                    "label": self.label_names.get(label_id, str(label_id)),
                })
            results.append(row_result)
        return results

    def predict_difficult_spans(
        self,
        sentences: Sequence[str],
        max_length: int = 128,
        device: Any | None = None,
    ) -> list[list[tuple[int, int]]]:
        """Predict character-level difficult-word spans per sentence.

        Adjacent tokens predicted as ``M`` are merged into a single span using
        the fast tokenizer's offset mapping. Requires a fast tokenizer.
        """
        scored_results = self.predict_difficult_spans_with_scores(
            sentences=sentences,
            max_length=max_length,
            device=device,
            threshold=None,
        )
        return [[tuple(item["span"]) for item in row] for row in scored_results]

    def predict_difficult_spans_with_scores(
        self,
        sentences: Sequence[str],
        max_length: int = 128,
        device: Any | None = None,
        threshold: float | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Predict difficult character spans with M-class probabilities.
    
        Uses tokenizer offset mappings as the source of truth for character
        spans. Adjacent or overlapping tokens predicted as M are merged into one
        span. If ``threshold`` is provided, a token is treated as M when its
        softmax probability for class M is greater than or equal to the
        threshold; otherwise the argmax label must be M.
    
        After merging M-token spans, each span is expanded to full word boundaries.
        This avoids partial words such as "laureate" instead of "laureates".
        """
        self.model.eval()
        sentences = list(sentences)
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
    
        offset_mapping = encoded["offset_mapping"]
        model_inputs = {
            key: value for key, value in encoded.items() if key != "offset_mapping"
        }
    
        if device is not None:
            model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
    
        with torch.no_grad():
            outputs = self.model(**model_inputs)
    
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_ids = outputs.logits.argmax(dim=-1)
    
        results: list[list[dict[str, Any]]] = []
    
        for row_index in range(len(sentences)):
            sentence = sentences[row_index]
            sequence_ids = (
                encoded.sequence_ids(row_index) if hasattr(encoded, "sequence_ids") else None
            )
            attention_row = encoded["attention_mask"][row_index]
    
            spans: list[dict[str, Any]] = []
            current_start: int | None = None
            current_end: int | None = None
            current_scores: list[float] = []
    
            num_tokens = offset_mapping.shape[1]
    
            for token_index in range(num_tokens):
                is_special = sequence_ids is not None and sequence_ids[token_index] is None
                attended = int(attention_row[token_index]) == 1
    
                token_start = int(offset_mapping[row_index][token_index][0])
                token_end = int(offset_mapping[row_index][token_index][1])
    
                mask_score = float(
                    probabilities[row_index][token_index][LABEL_MASK].item()
                )
    
                if threshold is None:
                    is_mask = (
                        int(predicted_ids[row_index][token_index].item()) == LABEL_MASK
                    )
                else:
                    is_mask = mask_score >= threshold
    
                is_real_token = token_end > token_start
    
                if attended and not is_special and is_mask and is_real_token:
                    if current_start is None:
                        current_start = token_start
                        current_end = token_end
                        current_scores = [mask_score]
    
                    elif current_end is not None and token_start <= current_end:
                        current_end = max(current_end, token_end)
                        current_scores.append(mask_score)
    
                    else:
                        assert current_end is not None
    
                        expanded_start, expanded_end = _expand_to_word_boundaries(
                            sentence,
                            current_start,
                            current_end,
                        )
                        spans.append(
                            _scored_span(
                                expanded_start,
                                expanded_end,
                                current_scores,
                            )
                        )
    
                        current_start = token_start
                        current_end = token_end
                        current_scores = [mask_score]
    
                else:
                    if current_start is not None:
                        assert current_end is not None
    
                        expanded_start, expanded_end = _expand_to_word_boundaries(
                            sentence,
                            current_start,
                            current_end,
                        )
                        spans.append(
                            _scored_span(
                                expanded_start,
                                expanded_end,
                                current_scores,
                            )
                        )
    
                        current_start = None
                        current_end = None
                        current_scores = []
    
            if current_start is not None:
                assert current_end is not None
    
                expanded_start, expanded_end = _expand_to_word_boundaries(
                    sentence,
                    current_start,
                    current_end,
                )
                spans.append(
                    _scored_span(
                        expanded_start,
                        expanded_end,
                        current_scores,
                    )
                )
    
            results.append(spans)
    
        return results

    def save(self, output_dir: str | Path) -> None:
        """Save model and tokenizer in Hugging Face format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

def _expand_to_word_boundaries(sentence: str, start: int, end: int) -> tuple[int, int]:
    """Expand a predicted subword span to the full surface word.

    Letters, digits, underscore, and hyphen are treated as part of words.
    """
    while start > 0 and (sentence[start - 1].isalnum() or sentence[start - 1] in "_-"):
        start -= 1

    while end < len(sentence) and (sentence[end].isalnum() or sentence[end] in "_-"):
        end += 1

    return start, end

def _scored_span(start: int, end: int, scores: Sequence[float]) -> dict[str, Any]:
    """Build the public scored-span dict from merged token scores."""
    if not scores:
        average_score = 0.0
        max_score = 0.0
    else:
        average_score = sum(scores) / len(scores)
        max_score = max(scores)
    return {
        "span": (start, end),
        "score": average_score,
        "average_score": average_score,
        "max_score": max_score,
    }
