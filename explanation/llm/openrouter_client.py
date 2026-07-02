"""OpenRouter LLM client for contextual difficult-word explanations."""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from typing import Any, Dict

from explanation.llm.prompts import EXPLAIN_SYSTEM, EXPLAIN_USER

load_dotenv()

DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClientError(Exception):
    """Raised when OpenRouter returns an unexpected response."""


class OpenRouterClient:
    """Thin wrapper around the OpenAI SDK configured for OpenRouter."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file or environment."
            )
        self._client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _chat(self, system: str, user: str) -> str:
        """Send a single-turn chat and return the raw text response."""
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        choice = response.choices[0]
        content = choice.message.content

        if content is None:
            finish_reason = getattr(choice, "finish_reason", None)
            raise OpenRouterClientError(
                "OpenRouter returned empty message content "
                f"for model={self.model!r}, finish_reason={finish_reason!r}."
            )

        content = content.strip()

        if not content:
            finish_reason = getattr(choice, "finish_reason", None)
            raise OpenRouterClientError(
                "OpenRouter returned blank message content "
                f"for model={self.model!r}, finish_reason={finish_reason!r}."
            )

        return content
    
    def explain_difficult_words(
        self, sentence: str, words: list[str]
    ) -> Dict[str, Any]:
        """Return the full explanation JSON for *words* in *sentence*.

        Args:
            sentence: The original English sentence.
            words: Surface-form strings of the difficult words to explain.

        Returns:
            Parsed JSON dict matching the ExplanationResult schema.

        Raises:
            OpenRouterClientError: If the model response is not valid JSON or missing keys.
        """
        words_list = ", ".join(f'"{w}"' for w in words)
        raw = self._chat(
            system=EXPLAIN_SYSTEM,
            user=EXPLAIN_USER.format(sentence=sentence, words_list=words_list),
        )
        
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0].strip()
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise OpenRouterClientError(
                f"explain_difficult_words: model returned non-JSON: {raw!r}"
            ) from exc
        if "difficult_words" not in result:
            raise OpenRouterClientError(
                f"explain_difficult_words: missing 'difficult_words' key in response"
            )
        return result
