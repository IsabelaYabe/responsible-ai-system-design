"""
Model-agnostic LLM client.

A thin wrapper so the rest of the package never imports a vendor SDK directly;
the backend can be swapped in config without touching anything downstream.

The same client serves two roles with two different models:
  - the *answerer* (cheaper model, e.g. Haiku) generates reading-assistant replies
  - the *judge*    (stronger model, e.g. Sonnet) grades them

Keeping them distinct avoids the self-preference bias documented for
LLM-as-a-judge setups (a model tends to favour its own generations).
"""

from __future__ import annotations

import os

from . import config


class LLMClient:
    """Usage: client.complete(system_prompt, user_prompt) -> str"""

    def __init__(
        self,
        backend: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.backend = (backend or config.BACKEND).lower()
        key = api_key if api_key is not None else config.get_api_key()

        if self.backend == "anthropic":
            import anthropic

            key = key or os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=key)
            self._model = model or config.ANSWERER_MODEL
        elif self.backend == "openai":
            import openai

            key = key or os.environ.get("OPENAI_API_KEY")
            self._client = openai.OpenAI(api_key=key)
            self._model = model or config.OPENAI_MODEL
        elif self.backend == "ollama":
            import openai  # Ollama exposes an OpenAI-compatible /v1 endpoint

            self._client = openai.OpenAI(
                api_key="ollama", base_url=f"{config.OLLAMA_BASE_URL}/v1"
            )
            self._model = model or config.OLLAMA_MODEL
        else:
            raise ValueError(
                f"Unknown backend {self.backend!r}. Choose anthropic | openai | ollama"
            )

    @property
    def model(self) -> str:
        return self._model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        if self.backend == "anthropic":
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text.strip()
        # openai-compatible (openai + ollama)
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()


def make_answerer(api_key: str | None = None) -> LLMClient:
    return LLMClient(api_key=api_key, model=config.ANSWERER_MODEL)


def make_judge(api_key: str | None = None) -> LLMClient:
    """Stronger model than the answerer, to reduce self-preference bias."""
    return LLMClient(api_key=api_key, model=config.JUDGE_MODEL)
