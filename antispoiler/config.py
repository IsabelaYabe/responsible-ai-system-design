"""
Configuration + credential resolution.

Credentials resolve in this order so the same code runs in Colab and locally:
  1. google.colab.userdata  (Colab secrets)
  2. .env / os.environ      (local VS Code runs via python-dotenv)

Nothing here imports a vendor SDK or a heavy ML dependency, so it is cheap to
import from tests.
"""

from __future__ import annotations

import os

# ── BOOK ─────────────────────────────────────────────────────────────────────
BOOK_URL = "https://raw.githubusercontent.com/GITenberg/Pride-and-Prejudice_1342/master/1342.txt"
BOOK_TITLE = "Pride and Prejudice"
BOOK_AUTHOR = "Jane Austen"

# Simulated reader position: chapters 1..N are "already read".
# Pride and Prejudice has 61 chapters; 15 ~= 25% through the book.
READER_POSITION = 15

# ── LLM BACKEND — dev / prod modes ───────────────────────────────────────────
# Selected by the APP_MODE env var (default "prod"), set on the run command:
#   prod — Anthropic: Haiku generator + Sonnet validator (the characterized setup).
#   dev  — OpenRouter: one cheap model for fast/cheap iteration. NOT for evaluation:
#          it collapses the judge≠generator separation (D13) and isn't characterized.
# Example:  APP_MODE=dev uvicorn app:app --port 8000
APP_MODE = os.environ.get("APP_MODE", "prod").strip().lower()

# OpenRouter is OpenAI-compatible. The dev model is overridable so you can swap cheap
# models without editing code:  OPENROUTER_MODEL=deepseek/deepseek-chat APP_MODE=dev …
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat")

if APP_MODE == "dev":
    BACKEND = "openrouter"
    ANSWERER_MODEL = OPENROUTER_MODEL   # generator
    JUDGE_MODEL = OPENROUTER_MODEL      # eval grader
    VALIDATOR_MODEL = OPENROUTER_MODEL  # LLM 3 validator
else:
    BACKEND = "anthropic"  # "anthropic" | "openai" | "openrouter" | "ollama"
    # Answerer vs. judge are deliberately different models: using the same model to
    # answer and to grade invites self-preference bias (judges favour their own
    # generations — see report's validation section). The judge is the stronger one.
    ANSWERER_MODEL = "claude-haiku-4-5"
    JUDGE_MODEL = "claude-sonnet-4-6"
    # LLM 3 validator (validator/ package) — its own knob, separate from JUDGE_MODEL
    # (the anti-spoiler eval's grader), so the validator's model can be switched
    # independently. Sonnet today: the verdict stage is entailment/NLI where a stronger
    # model helps (D3); a different *family* from the generator would further reduce
    # self-preference bias (D13). The validator uses single-pass, temperature-default
    # calls, so switching to Opus/Fable (which reject `temperature`) is safe.
    VALIDATOR_MODEL = "claude-sonnet-4-6"

OPENAI_MODEL = "gpt-4o"
OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"

# ── RETRIEVAL ────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"  # fast 384-dim model; no API key required
TOP_K = 10  # retrieved chunks per query

# ── CHUNKING ─────────────────────────────────────────────────────────────────
MIN_CHARS = 400   # below this, merge a paragraph with the next
MAX_CHARS = 2000  # soft ceiling; we never split a paragraph to hit it

# ── EVAL ─────────────────────────────────────────────────────────────────────
N_QUESTIONS_PER_TIER = 4  # x3 tiers = 12 generated questions


def _from_colab(name: str) -> str | None:
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return None
    try:
        return userdata.get(name)
    except Exception:
        return None


# Locate .env by absolute path so it resolves regardless of the process's working
# directory (the notebook runs from notebooks/, not the repo root). We check a few
# sensible locations: repo root (canonical), the package dir, and the cwd.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
_DOTENV_CANDIDATES = [
    os.path.join(_REPO_ROOT, ".env"),   # canonical
    os.path.join(_PKG_DIR, ".env"),     # antispoiler/.env
    os.path.join(os.getcwd(), ".env"),  # wherever you launched from
]


def _from_dotenv(name: str) -> str | None:
    # Already in the environment? Use it (lets real env vars win).
    val = os.environ.get(name)
    if val:
        return val
    try:
        from dotenv import dotenv_values  # type: ignore

        for path in _DOTENV_CANDIDATES:
            if os.path.exists(path):
                val = dotenv_values(path).get(name)
                if val:
                    return val
    except Exception:
        pass
    return None


def get_secret(name: str, *aliases: str) -> str | None:
    """
    Resolve a secret by name, trying Colab userdata first, then .env/os.environ.
    Extra positional args are treated as alternative env-var names to try.
    """
    for key in (name, *aliases):
        val = _from_colab(key) or _from_dotenv(key)
        if val:
            return val
    return None


def get_api_key() -> str | None:
    """API key for the active BACKEND. Accepts API_KEY or the vendor env names."""
    if BACKEND == "anthropic":
        return get_secret("API_KEY", "ANTHROPIC_API_KEY")
    if BACKEND == "openai":
        return get_secret("API_KEY", "OPENAI_API_KEY")
    if BACKEND == "openrouter":
        return get_secret("OPENROUTER_API_KEY")
    return None  # ollama needs none


def get_hf_token() -> str | None:
    return get_secret("HF_TOKEN", "HUGGINGFACE_TOKEN")
