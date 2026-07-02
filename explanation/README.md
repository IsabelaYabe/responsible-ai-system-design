# explanation

Local difficult-word identification + contextual LLM explanation for the
reading assistant's "Define"-style feature. Given an English sentence, the
pipeline returns each difficult word's character span and a short gloss of
what it means *in that sentence*.

## Idea

Identifying "which words are hard" and "what a hard word means here" are two
different problems, so the pipeline is split into two stages that never share
a responsibility:

1. **Identification is local and supervised.** A small token classifier (the
   *Edit Predictor*, fine-tuned `distilbert-base-uncased`) labels every token
   K (keep) or M (mask/difficult). It runs offline, deterministically, and
   without any API cost.
2. **Explanation is the LLM's only job.** Once spans are fixed, OpenRouter is
   asked only to gloss the words it's given — never to decide which words are
   difficult, substitute synonyms, or simplify/rewrite the sentence. If the
   Edit Predictor finds nothing, the LLM is never called.

This split means identification is cheap, reproducible, and evaluable with
ordinary classification metrics, while the one LLM call per sentence is
narrowly scoped and easy to validate (its output can only add a gloss to
already-fixed spans, so it cannot introduce new — possibly hallucinated —
difficult words).

## Layout

| Path | Responsibility |
|---|---|
| `schemas.py` | Public result types: `DifficultWord`, `ExplanationResult` |
| `inference/edit_predictor_identifier.py` | Loads a checkpoint, calls the Edit Predictor, returns validated `(word, span, score)` items |
| `inference/pipeline.py` | `ExplanationPipeline`: identify locally on one sentence → explain via LLM → merge into `ExplanationResult` |
| `inference/selection_explainer.py` | `SelectionExplainer`: same idea, but for a reader's arbitrary selection (word / sentence / excerpt) inside a paragraph |
| `inference/_shared.py` | Span-coercion + LLM-response-to-word-mapping helpers shared by both pipelines |
| `llm/openrouter_client.py` | `OpenRouterClient` — one method, `explain_difficult_words`, OpenAI-SDK-compatible call to OpenRouter |
| `llm/prompts.py` | The explanation system/user prompt templates (JSON-only output) |
| `model/edit_predictor.py` | `EditPredictor` — the Hugging Face token-classifier wrapper (load/predict/save) |
| `model/edit_predictor_dataset.py` | Sentence-level pseudo-labels → token-level K/M labels, and the train/validation/test split |
| `model/train_edit_predictor.py` | Training loop, metrics, checkpoint selection |
| `model/manage_edit_predictor.py` | Interactive/CLI front-end over train/continue/checkpoint-eval/predict |
| `scripts/generate_complex_word_pseudo_labels.py` | Lexicon-based pseudo-label generator (spaCy lemma matching against a CEFR word list) |
| `scripts/build_edit_predictor_dataset.py` | Sentence-level pseudo-labels → tokenized `.pt` dataset artifact |
| `scripts/run_inference.py` | CLI: run the maintained pipeline on one sentence |
| `data/` | Raw CEFR vocab lists + sentence corpora, prep scripts, and `processed/` build artifacts |
| `model/checkpoints/` | Trained Edit Predictor checkpoints (Hugging Face format) |
| `tests/` | Unit tests, one file per module above |

`notebooks/explanation/train_edit_predictor.ipynb` is the interactive
training/analysis workbook: it calls into `model/train_edit_predictor.py`
(never reimplements training), then plots the loss/F1 curves, audits the
K/M token balance, and inspects individual predictions.

`notebooks/explanation/test_selection_explainer.ipynb` exercises
`SelectionExplainer` on real book text fetched through
`antispoiler.book.fetch_and_chunk` (the same book/chunking pipeline the
anti-spoiler feature uses), covering all selection shapes — including an
excerpt that crosses a paragraph boundary, to check that only the touched
paragraphs are sent as context.

## Key design decisions

**Supervised local classifier instead of an LLM for identification.**
Asking an LLM "which words in this sentence are hard?" is nondeterministic
and expensive to run per sentence, and the spans it returns are guesses. A
small fine-tuned classifier gives fixed, reproducible spans, runs offline,
and can be evaluated on a held-out test set like any classification model.

**Pseudo-labels from a CEFR lexicon, not manual annotation.** Manually
labeling difficult-word spans doesn't scale. `generate_complex_word_pseudo_labels.py`
lemmatizes each sentence with spaCy and marks a span difficult when its lemma
matches a B2/C1/C2 entry in a CEFR vocabulary lexicon. This is a weak/lexical
label source (it doesn't look at context), used only to bootstrap supervised
training data for the Edit Predictor.

**Model selection is by validation `f1_M`, never token accuracy.** K (keep)
tokens vastly outnumber M (difficult) tokens, so a model that never predicts
M can still score high accuracy. Every training run also computes and saves
an "all-K baseline" so this failure mode is visible in the metrics rather
than hidden behind an accuracy number.

**Span merging expands to full word boundaries.** The tokenizer works on
subwords, so adjacent tokens predicted M are first merged, then the merged
span is expanded to the nearest non-alphanumeric boundary — so the model
never reports a partial word (e.g. "laureate" when the surface form is
"laureates").

**The LLM client is created lazily.** `ExplanationPipeline` only constructs
an `OpenRouterClient` (which requires `OPENROUTER_API_KEY`) the first time a
sentence actually has a difficult word to explain. A sentence with nothing
difficult never touches the network and never needs the API key.

**A reader's selection is never fed to the Edit Predictor as raw text, and
the LLM only ever sees the paragraph(s) the selection actually touches.**
`SelectionExplainer` (used by the "select text in the reader UI" flow, as
opposed to `ExplanationPipeline`'s one-sentence-at-a-time API) takes text
that may span several paragraphs, and always identifies on whole sentences,
because that's the only input shape the Edit Predictor was trained on:
- a single selected **word** skips identification entirely (there's nothing
  to identify) — the LLM is asked to define that exact word directly,
  grounded on the one paragraph it sits in;
- a selection that's exactly one full **sentence** runs the Edit Predictor
  on it directly, grounded on the one paragraph containing that sentence;
- any other **excerpt** (multiple sentences, a partial one, or a span that
  crosses a paragraph break) runs the Edit Predictor on every full sentence
  the excerpt touches — using the complete sentence even where the excerpt
  doesn't cover it end to end, and pulling sentences from more than one
  paragraph if the excerpt crosses a paragraph break — then only the
  difficult words whose span falls inside the excerpt are kept and
  explained.

In every case, the context sent to the LLM is exactly the paragraph(s)
touched by the selection (one, for a word or single-sentence selection; as
many as the excerpt actually spans otherwise) — never the caller's whole
input text and never just the bare selection, since a word's meaning can
depend on nearby context outside the highlighted text.

**Checkpoints are versioned, not overwritten.** `v1`, `final`, and
`big_final` under `model/checkpoints/` are kept as reproducible runs rather
than a single mutable "latest" checkpoint (see each checkpoint's own
`training_config.json` / `metrics.json` for the exact dataset and
hyperparameters used). `big_final` was trained on the largest token dataset
build, improves validation/test `f1_M` from 0.84 (`final`) to 0.92, and is
the one used by the pipeline/demo by default.

## Running the pipeline

```bash
python -m explanation.scripts.run_inference \
  --sentence "A committee of the institute appoints the laureates for the Nobel Prize." \
  --edit-predictor-checkpoint explanation/model/checkpoints/edit_predictor_complex_words_distilbert_max256_big_final \
  --max-length 256
```

Needs `OPENROUTER_API_KEY` in the repo-root `.env` (only used if a difficult
word is actually found).

## Rebuilding the training data / retraining

```bash
# 1. sentence-level pseudo-labels from the CEFR lexicon (needs: python -m spacy download en_core_web_sm)
python -m explanation.scripts.generate_complex_word_pseudo_labels

# 2. tokenize + split into train/validation/test
python -m explanation.scripts.build_edit_predictor_dataset

# 3. train (or use notebooks/explanation/train_edit_predictor.ipynb interactively)
python -m explanation.model.manage_edit_predictor --mode train
```

## Tests

```bash
pytest explanation/tests/
```
