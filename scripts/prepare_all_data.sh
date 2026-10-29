#!/usr/bin/env bash
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
# Generates all data splits needed to run experiments.
# Usage: bash scripts/prepare_all_data.sh
set -euo pipefail

echo "=== Step 1: Base prompts (AdvBench + StrongREJECT) ==="
uv run python -m ccdelta.scripts.prepare_dataset_splits --dataset all -v

echo ""
echo "=== Step 2: Jailbroken prompts ==="
uv run python -m ccdelta.scripts.apply_strongreject

echo ""
echo "=== Step 3: Prompt defense variants ==="
uv run python -m ccdelta.scripts.apply_prompt_defense

echo ""
echo "=== Step 4: OR-Bench ==="
uv run python scripts/prepare_or_bench.py prepare
uv run python scripts/prepare_or_bench.py partition \
    --existing-sample data/or_bench/or_bench_80k_10pct_seed42.parquet

echo ""
echo "=== Step 5: OR-Bench prompt defense ==="
uv run python -m ccdelta.scripts.apply_prompt_defense_orbench

echo ""
echo "=== Done ==="
echo "Data written to data/"
