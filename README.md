# Sparse Autoencoders are Capable LLM Jailbreak Mitigators

This software project accompanies the research paper, [Sparse Autoencoders are Capable LLM Jailbreak Mitigators](https://arxiv.org/abs/2602.12418).

This repository contains the code for **CC-Delta**, a sparse autoencoder (SAE) based inference-time defense against jailbreak attacks on language models.

## Layout

- `ccdelta/` — main package: hooks, models, evaluations, scripts, configs
- `scripts/` — standalone scripts for data prep and SAE feature description
- `notebooks/` — analysis notebooks for paper results

## Setup

Install [uv](https://docs.astral.sh/uv/) to manage dependencies, then:

```bash
git clone <repo-url>
cd ml-ccdelta
uv sync
```

Dependency versions are pinned to those used in the paper for reproducibility; loosen them in `pyproject.toml` if exact reproduction isn't needed.

To also install the optional SAE feature interpretation dependency (for `scripts/interpret_sae_features.py`):

```bash
uv sync --extra interpret
```

## Data Setup

Data must be generated locally before running experiments.

### 1. Base prompts (required)

Downloads AdvBench and StrongREJECT, creates train/test splits:

```bash
uv run python -m ccdelta.scripts.prepare_dataset_splits --dataset all -v
```

Output: `data/forbidden_prompts/0.5/`

### 2. Jailbroken prompts (required)

Applies jailbreak attacks to the base prompts. Some attack methods (PAP variants) use the OpenAI API:

```bash
export OPENAI_API_KEY=<your-key>
uv run python -m ccdelta.scripts.apply_strongreject
```

Output: `data/jailbroken/0.5/`

### 3. Prompt defense variants (optional)

Wraps jailbroken prompts in defense templates. Only needed for prompt-defense evaluation:

```bash
uv run python -m ccdelta.scripts.apply_prompt_defense
```

Output: `data/prompt_defense/0.5/`

### 4. OR-Bench data (optional)

Downloads OR-Bench from HuggingFace for over-refusal evaluation:

```bash
uv run python scripts/prepare_or_bench.py prepare
uv run python scripts/prepare_or_bench.py partition \
    --existing-sample data/or_bench/or_bench_80k_10pct_seed42.parquet
```

To also run prompt defense on OR-Bench:

```bash
uv run python -m ccdelta.scripts.apply_prompt_defense_orbench
```

Output: `data/or_bench/`

## Running Experiments

Run all experiments (baseline, CC-Delta, CAA) with a single command:

```bash
uv run python -m ccdelta.scripts.run_experiments
```

By default this runs in **fast mode** (small subset of prompts) for quick validation. To run on the full dataset:

```bash
uv run python -m ccdelta.scripts.run_experiments --full
```

Run specific stages:

```bash
# Only baseline and CC-Delta
uv run python -m ccdelta.scripts.run_experiments --stages baseline ccdelta

# Only CAA
uv run python -m ccdelta.scripts.run_experiments --stages caa
```

Override the model:

```bash
uv run python -m ccdelta.scripts.run_experiments --model gemma-2-9b-it
```

Results are written to `./results/` by default. See `notebooks/results_analysis.ipynb` to analyze results.

**Running on Mac (MPS) or CPU:**

The default device is `cuda`. On a Mac or machine without a GPU, pass a device override (MPS or CPU):

```bash
uv run python -m ccdelta.scripts.run_experiments device=cpu --stages ccdelta
```

## Baselines

This repository also includes implementations of the following baselines:

- **CAA (Contrastive Activation Addition)** — dense contrastive steering vectors
- **LinearAcT** — dense optimal transport activation steering ([ml-act](https://github.com/apple-aiml-research/ml-act))

## Tests

```bash
uv run python -m pytest ccdelta/tests/ -x -m "not integration"
```
