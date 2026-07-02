# README — Difficult-Word Identification and Explanation Pipeline Scripts

This README explains how to run the three main pipeline scripts:

```text
explanation/scripts/generate_complex_word_pseudo_labels.py
explanation/scripts/build_edit_predictor_dataset.py
explanation/scripts/run_inference.py
```

The pipeline has three main stages:

```text
1. Generate difficult-word pseudo-labels
2. Convert pseudo-labels into a token-level K/M dataset
3. Run inference with the trained Edit Predictor + LLM for explanation
```

Important: these three scripts do not train the model. Between stage 2 and stage 3, you need to have an already trained Edit Predictor checkpoint.

---

## 1. Prepare the environment

Run the commands from the repository root:

```bash
cd ~/projects/responsible-ai-system-design
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Install the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

spaCy is required because the pseudo-label script uses lemmatization to compare words from the sentences against the CEFR vocabulary.

---

## 2. Expected file structure

Before running the pipeline, check whether these files exist:

```bash
ls explanation/data/sentences.txt
ls explanation/data/cefr_vocabularies.csv
```

The `sentences.txt` file must contain one sentence per line.

Example:

```text
A committee of the institute appoints the laureates for the Nobel Prize.
The group talked for a long time before they agreed on the ultimate decision.
```

The `cefr_vocabularies.csv` file must contain the CEFR vocabulary. The script accepts columns such as:

```text
lemma,label
word,CEFR
headword,level
```

Accepted CEFR levels are:

```text
A1, A2, B1, B2, C1, C2
```

By default, the pipeline treats the following levels as difficult:

```text
B2, C1, C2
```

---

# Stage 1 — Generate difficult-word pseudo-labels

Script:

```text
explanation/scripts/generate_complex_word_pseudo_labels.py
```

This script reads sentences and identifies difficult words using the local CEFR vocabulary.

It does not call an LLM.
It does not train a model.
It only generates a `.jsonl` file with pseudo-labels.

## Input

```text
explanation/data/sentences.txt
explanation/data/cefr_vocabularies.csv
```

## Output

```text
explanation/data/processed/complex_words_pseudo_labels.jsonl
```

## Recommended command

```bash
python -m explanation.scripts.generate_complex_word_pseudo_labels \
  --input-file explanation/data/sentences.txt \
  --complex-words-csv explanation/data/cefr_vocabularies.csv \
  --output-file explanation/data/processed/complex_words_pseudo_labels.jsonl \
  --cefr-levels "B2,C1,C2" \
  --skip-mojibake
```

## Quick test command

Use `--limit` to process only a small number of sentences:

```bash
python -m explanation.scripts.generate_complex_word_pseudo_labels \
  --input-file explanation/data/sentences.txt \
  --complex-words-csv explanation/data/cefr_vocabularies.csv \
  --output-file explanation/data/processed/complex_words_pseudo_labels.jsonl \
  --cefr-levels "B2,C1,C2" \
  --skip-mojibake \
  --limit 100
```

## What each argument does

```text
--input-file
    Input file with one sentence per line.

--complex-words-csv
    CSV file with CEFR vocabulary.

--output-file
    Path where the pseudo-label JSONL file will be saved.

--cefr-levels
    CEFR levels treated as difficult.

--skip-mojibake
    Skips sentences with corrupted characters or encoding problems.

--limit
    Limits the number of processed sentences. Useful for testing.

--include-phrases
    Allows detection of multi-word expressions.

--spacy-model
    spaCy model used for lemmatization. Default: en_core_web_sm.
```

## Expected terminal output

```text
=== Complex-word lexicon pseudo-label generation complete ===
input_file                      : explanation/data/sentences.txt
complex_words_csv               : explanation/data/cefr_vocabularies.csv
output_file                     : explanation/data/processed/complex_words_pseudo_labels.jsonl
accepted_cefr_levels            : ['B2', 'C1', 'C2']
spacy_model                     : en_core_web_sm
lexicon_size                    : ...
sentences_read                  : ...
sentences_written               : ...
sentences_with_complex_words    : ...
sentences_without_complex_words : ...
total_complex_word_matches      : ...
```

## Check the generated file

```bash
head -n 3 explanation/data/processed/complex_words_pseudo_labels.jsonl
```

Expected line example:

```json
{"sentence": "A committee of the institute appoints the laureates for the Nobel Prize.", "difficult_words": ["laureates"], "spans": [[38, 47]], "source": "complex_words_csv", "model": "spacy_lemma_lexicon"}
```

---

# Stage 2 — Convert pseudo-labels into a token-level dataset

Script:

```text
explanation/scripts/build_edit_predictor_dataset.py
```

This script converts the pseudo-label JSONL file into a `.pt` dataset used to train the Edit Predictor.

It creates token-level labels:

```text
K = keep
M = mask/difficult
-100 = ignore special token or padding
```

## Input

```text
explanation/data/processed/complex_words_pseudo_labels.jsonl
```

## Output

```text
explanation/data/processed/edit_predictor_token_labels_complex_words_distilbert_max256.pt
```

## Recommended command

```bash
python -m explanation.scripts.build_edit_predictor_dataset \
  --input-file explanation/data/processed/complex_words_pseudo_labels.jsonl \
  --output-file explanation/data/processed/edit_predictor_token_labels_complex_words_distilbert_max256.pt \
  --model-name distilbert-base-uncased \
  --max-length 256 \
  --train-ratio 0.8 \
  --validation-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 13
```

## What each argument does

```text
--input-file
    JSONL file generated in the previous stage.

--output-file
    .pt file that will be used to train the Edit Predictor.

--model-name
    Hugging Face tokenizer used to tokenize the sentences.
    Default: distilbert-base-uncased.

--max-length
    Maximum tokenized sequence length.
    In the current project, use 256.

--train-ratio
    Proportion of the dataset used for training.

--validation-ratio
    Proportion used for validation.

--test-ratio
    Proportion used for testing.

--seed
    Seed used for deterministic splitting.

--keep-invalid-rows
    Keeps rows with invalid spans as all-K examples.
    By default, invalid rows are discarded.
```

## Expected terminal output

```text
=== Edit Predictor token dataset build complete ===
Rows read                         : ...
Rows written                      : ...
Rows skipped                      : ...
Rows skipped malformed            : ...
Invalid spans ignored             : ...
Out-of-bounds spans ignored       : ...
Examples with no difficult words  : ...
Tokens K / M / -100               : ...
Train / validation / test examples: ...
Output path                       : explanation/data/processed/edit_predictor_token_labels_complex_words_distilbert_max256.pt
```

## Check the generated file

```bash
ls -lh explanation/data/processed/edit_predictor_token_labels_complex_words_distilbert_max256.pt
```

---

# Intermediate stage — Train the Edit Predictor

The three scripts covered by this README do not train the model.

After generating the `.pt` file, train the Edit Predictor with the project training script:

```bash
python -m explanation.model.train_edit_predictor \
  --dataset-file explanation/data/processed/edit_predictor_token_labels_complex_words_distilbert_max256.pt \
  --output-dir explanation/model/checkpoints/edit_predictor_complex_words_distilbert_max256_big_final \
  --model-name distilbert-base-uncased \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --weight-decay 0.01 \
  --early-stopping-patience 2 \
  --early-stopping-min-delta 0.001 \
  --seed 13 \
  --use-class-weights
```

After training, the expected checkpoint will be located at:

```text
explanation/model/checkpoints/edit_predictor_complex_words_distilbert_max256_big_final
```

The folder should contain files such as:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.txt
special_tokens_map.json
metrics.json
history.csv
```

---

# Stage 3 — Run final inference

Script:

```text
explanation/scripts/run_inference.py
```

This script runs the final pipeline:

```text
input sentence
    ↓
local Edit Predictor identifies difficult words
    ↓
OpenRouter/LLM generates contextual explanations
    ↓
final JSON with word, span, and meaning in context
```

## Requirement: OpenRouter key

Create a `.env` file in the project root:

```bash
nano .env
```

Add:

```env
OPENROUTER_API_KEY=your_key_here
```

Without this key, the pipeline will fail when it needs to call the LLM to generate explanations.

## Input

An English sentence passed through the `--sentence` argument.

## Output

A JSON printed in the terminal.

## Recommended command

```bash
python -m explanation.scripts.run_inference \
  --sentence "A committee of the institute appoints the laureates for the Nobel Prize." \
  --edit-predictor-checkpoint explanation/model/checkpoints/edit_predictor_complex_words_distilbert_max256_big_final \
  --max-length 256 \
  --edit-predictor-threshold 0.5
```

## Run using GPU

```bash
python -m explanation.scripts.run_inference \
  --sentence "A committee of the institute appoints the laureates for the Nobel Prize." \
  --edit-predictor-checkpoint explanation/model/checkpoints/edit_predictor_complex_words_distilbert_max256_big_final \
  --max-length 256 \
  --edit-predictor-threshold 0.5 \
  --device cuda
```

## Run using CPU

```bash
python -m explanation.scripts.run_inference \
  --sentence "A committee of the institute appoints the laureates for the Nobel Prize." \
  --edit-predictor-checkpoint explanation/model/checkpoints/edit_predictor_complex_words_distilbert_max256_big_final \
  --max-length 256 \
  --edit-predictor-threshold 0.5 \
  --device cpu
```

## What each argument does

```text
--sentence
    English sentence to analyze.

--model
    Model used by OpenRouter to generate explanations.
    If not provided, the default model defined in the OpenRouter client is used.

--edit-predictor-checkpoint
    Path to the trained Edit Predictor checkpoint.

--edit-predictor-threshold
    Probability threshold used to classify a word as difficult.
    Default: 0.5.

--max-length
    Maximum tokenized sequence length.
    Use 256 to remain compatible with the max256 checkpoint.

--device
    Device used by the local model.
    Examples: cuda or cpu.
```

## Expected output

```json
{
  "sentence": "A committee of the institute appoints the laureates for the Nobel Prize.",
  "difficult_words": [
    {
      "word": "laureates",
      "span": [
        38,
        47
      ],
      "meaning_in_context": "people who receive an important prize or honor"
    }
  ]
}
```

---

# Complete workflow

This is the full workflow to run everything from scratch, assuming `sentences.txt` and `cefr_vocabularies.csv` already exist.

```bash
source .venv/bin/activate

python -m spacy download en_core_web_sm

python -m explanation.scripts.generate_complex_word_pseudo_labels \
  --input-file explanation/data/sentences.txt \
  --complex-words-csv explanation/data/cefr_vocabularies.csv \
  --output-file explanation/data/processed/complex_words_pseudo_labels.jsonl \
  --cefr-levels "B2,C1,C2" \
  --skip-mojibake

python -m explanation.scripts.build_edit_predictor_dataset \
  --input-file explanation/data/processed/complex_words_pseudo_labels.jsonl \
  --output-file explanation/data/processed/edit_predictor_token_labels_complex_words_distilbert_max256.pt \
  --model-name distilbert-base-uncased \
  --max-length 256 \
  --train-ratio 0.8 \
  --validation-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 13

python -m explanation.model.train_edit_predictor \
  --dataset-file explanation/data/processed/edit_predictor_token_labels_complex_words_distilbert_max256.pt \
  --output-dir explanation/model/checkpoints/edit_predictor_complex_words_distilbert_max256_big_final \
  --model-name distilbert-base-uncased \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --weight-decay 0.01 \
  --early-stopping-patience 2 \
  --early-stopping-min-delta 0.001 \
  --seed 13 \
  --use-class-weights

python -m explanation.scripts.run_inference \
  --sentence "A committee of the institute appoints the laureates for the Nobel Prize." \
  --edit-predictor-checkpoint explanation/model/checkpoints/edit_predictor_complex_words_distilbert_max256_big_final \
  --max-length 256 \
  --edit-predictor-threshold 0.5
```

---

# Common problems

## Error: `spaCy model not found`

Install the model:

```bash
python -m spacy download en_core_web_sm
```

## Error: `Sentence file not found`

Check whether the file exists:

```bash
ls explanation/data/sentences.txt
```

If it does not exist, generate the sentence file before running the pseudo-label step.

## Error: `Complex word CSV not found`

Check whether the CEFR vocabulary exists in the expected path:

```bash
ls explanation/data/cefr_vocabularies.csv
```

Or pass another path with:

```bash
--complex-words-csv path/to/your_file.csv
```

## Error: `OPENROUTER_API_KEY is not set`

Create the `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_key_here
```

## Error: `Edit Predictor checkpoint not found`

Check whether the checkpoint exists:

```bash
ls explanation/model/checkpoints/edit_predictor_complex_words_distilbert_max256_big_final
```

Or pass the correct path:

```bash
--edit-predictor-checkpoint path/to/the/checkpoint
```

---

# Generated files that should not be pushed to GitHub

The following files are local generated artifacts and should not be versioned:

```text
*.pt
*.safetensors
*.bin
explanation/model/checkpoints/
```

Recommended `.gitignore`:

```gitignore
# Model checkpoints / generated ML artifacts
explanation/model/checkpoints/
*.safetensors
*.bin
*.pt
```

This prevents GitHub push errors caused by large files.

---

# Summary of each script

| Script                                   | Purpose                                                                | Input                                     | Output                                     |
| ---------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------ |
| `generate_complex_word_pseudo_labels.py` | Generates difficult-word pseudo-labels using CEFR vocabulary and spaCy | `sentences.txt` + `cefr_vocabularies.csv` | `complex_words_pseudo_labels.jsonl`        |
| `build_edit_predictor_dataset.py`        | Converts pseudo-labels into a token-level K/M dataset                  | `complex_words_pseudo_labels.jsonl`       | `.pt` with train/validation/test splits    |
| `run_inference.py`                       | Runs the final pipeline with local checkpoint + OpenRouter             | sentence + trained checkpoint             | JSON with difficult words and explanations |
