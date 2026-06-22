# responsible-ai-system-design

Course project: an AI reading assistant that supports text comprehension with
on-demand, context-grounded help — without spoiling what the reader hasn't reached.
See `report.md` for the full write-up.

## Repository structure

```
report.md            Written report (problem, proposal, architecture, validation)
antispoiler/         Position-bounded anti-spoiler mechanism + eval harness (see its README)
notebooks/           antispoiler_eval.ipynb — thin orchestration over the package
scripts/             build_notebook.py (regenerates the notebook)
tests/               Unit tests for the deterministic logic
docs/                Persona + user journey, and reference papers
links.md             Shared links (slides, etc.)
```

## Running the anti-spoiler eval

Apple Silicon needs a native arm64 conda env (the repo's base miniconda is x86):

```bash
CONDA_SUBDIR=osx-arm64 conda create -y -n antispoiler-arm python=3.11
conda run -n antispoiler-arm conda config --env --set subdir osx-arm64
conda run -n antispoiler-arm pip install -r requirements.txt
conda run -n antispoiler-arm python tests/test_logic.py   # no API key needed
```

Put `API_KEY` and `HF_TOKEN` in a `.env` at the repo root (git-ignored). Full
details in `antispoiler/README.md`.