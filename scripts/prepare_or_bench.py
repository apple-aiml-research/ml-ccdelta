# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""Download and prepare OR-Bench data for over-refusal evaluation.

Downloads or-bench-80k and or-bench-hard-1k from HuggingFace (bench-llm/or-bench),
takes a stratified sample of 80k, keeps all of hard-1k, and saves as parquet files
matching the existing pipeline column conventions.

Also supports partitioning the full 80k dataset into sequential stratified chunks
for incremental evaluation.
"""

import argparse
import math
import os
import shutil

import pandas as pd
from datasets import load_dataset


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "or_bench")


def prepare_or_bench(sample_pct: float = 10, seed: int = 42):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- or-bench-80k: stratified sample ---
    print("Loading or-bench-80k from HuggingFace...")
    ds_80k = load_dataset("bench-llm/or-bench", "or-bench-80k", split="train", revision="main")
    df_80k = ds_80k.to_pandas()
    print(f"  Total rows: {len(df_80k)}")

    frac = sample_pct / 100.0
    df_80k_sample = (
        df_80k.groupby("category", group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=seed))
        .reset_index(drop=True)
    )
    print(f"  Sampled {sample_pct}%: {len(df_80k_sample)} rows")
    print(f"  Categories: {df_80k_sample['category'].nunique()}")

    print(f"\n  or-bench-80k {sample_pct}% sample counts per category:")
    orig_counts = df_80k["category"].value_counts().sort_index()
    sample_counts = df_80k_sample["category"].value_counts().sort_index()
    for cat in orig_counts.index:
        print(f"    {cat}: {sample_counts.get(cat, 0)} / {orig_counts[cat]}")

    df_80k_out = _to_pipeline_format(df_80k_sample, dataset_name="or_bench_80k")
    out_80k = os.path.join(OUTPUT_DIR, f"or_bench_80k_{sample_pct}pct_seed{seed}.parquet")
    df_80k_out.to_parquet(out_80k, index=False)
    print(f"  Saved to {out_80k}")

    # --- or-bench-hard-1k: keep all ---
    print("Loading or-bench-hard-1k from HuggingFace...")
    ds_hard = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train", revision="main")
    df_hard = ds_hard.to_pandas()
    print(f"  Total rows: {len(df_hard)}")

    print(f"\n  or-bench-hard-1k counts per category:")
    for cat, count in df_hard["category"].value_counts().sort_index().items():
        print(f"    {cat}: {count}")

    df_hard_out = _to_pipeline_format(df_hard, dataset_name="or_bench_hard_1k")
    out_hard = os.path.join(OUTPUT_DIR, "or_bench_hard_1k.parquet")
    df_hard_out.to_parquet(out_hard, index=False)
    print(f"  Saved to {out_hard}")

    print("Done.")


def _to_pipeline_format(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Rename/add columns to match existing pipeline conventions."""
    return pd.DataFrame({
        "forbidden_prompt": df["prompt"],
        "dataset": dataset_name,
        "source": "or_bench",
        "category": df["category"],
        "split": "test",
        "jailbroken_prompt": df["prompt"],
        "jailbreak": "none",
    })


def partition_or_bench(existing_sample: str, chunk_pct: float = 10, seed: int = 42):
    """Partition OR-Bench 80k into stratified chunks for sequential evaluation.

    Takes an existing sample parquet (chunk 0) and partitions the remaining rows
    into additional stratified chunks via round-robin assignment within each category.
    """
    # Load existing sample and extract its prompts
    print(f"Loading existing sample from {existing_sample}...")
    df_existing = pd.read_parquet(existing_sample)
    existing_prompts = set(df_existing["forbidden_prompt"])
    print(f"  Existing sample: {len(df_existing)} rows, {len(existing_prompts)} unique prompts")

    # Download full dataset
    print("Loading or-bench-80k from HuggingFace...")
    ds_80k = load_dataset("bench-llm/or-bench", "or-bench-80k", split="train", revision="main")
    df_80k = ds_80k.to_pandas()
    print(f"  Total rows: {len(df_80k)}")

    # Exclude already-sampled prompts
    df_remaining = df_80k[~df_80k["prompt"].isin(existing_prompts)].copy()
    print(f"  Remaining after excluding existing sample: {len(df_remaining)} rows")

    # Determine number of chunks from remaining data
    n_chunks = math.ceil(len(df_remaining) / len(df_existing)) if len(df_existing) > 0 else 1
    print(f"  Will create {n_chunks} additional chunk(s) (plus chunk 0 = existing sample)")

    # Round-robin assignment within each category for stratified chunks
    chunk_assignments = pd.Series(index=df_remaining.index, dtype=int)
    for _cat, group in df_remaining.groupby("category"):
        shuffled = group.sample(frac=1, random_state=seed)
        assignments = [i % n_chunks for i in range(len(shuffled))]
        chunk_assignments.loc[shuffled.index] = assignments

    df_remaining["_chunk"] = chunk_assignments

    # Set up output directory
    chunk_dir = os.path.join(OUTPUT_DIR, f"chunks_seed{seed}")
    os.makedirs(chunk_dir, exist_ok=True)

    # Copy existing sample as chunk 0
    chunk_0_path = os.path.join(chunk_dir, f"orbench_{int(chunk_pct)}pct_seed{seed}_chunk_0.parquet")
    shutil.copy2(existing_sample, chunk_0_path)
    print(f"\n  Chunk 0 (existing sample): {len(df_existing)} rows -> {chunk_0_path}")

    # Save remaining chunks
    total_rows = len(df_existing)
    for chunk_idx in range(n_chunks):
        df_chunk = df_remaining[df_remaining["_chunk"] == chunk_idx].drop(columns=["_chunk"])
        df_chunk_out = _to_pipeline_format(df_chunk, dataset_name="or_bench_80k")
        chunk_path = os.path.join(
            chunk_dir, f"orbench_{int(chunk_pct)}pct_seed{seed}_chunk_{chunk_idx + 1}.parquet"
        )
        df_chunk_out.to_parquet(chunk_path, index=False)
        total_rows += len(df_chunk_out)
        print(f"  Chunk {chunk_idx + 1}: {len(df_chunk_out)} rows -> {chunk_path}")

    # Summary
    print(f"\n  Total chunks: {n_chunks + 1} (chunk 0 + {n_chunks} new)")
    print(f"  Total rows across all chunks: {total_rows}")
    print(f"  Full dataset size: {len(df_80k)}")

    # Category distribution per chunk
    print("\n  Category distribution per chunk:")
    cats = sorted(df_80k["category"].unique())
    header = f"  {'Category':<30}" + "".join(f"  chunk_{i:<4}" for i in range(n_chunks + 1))
    print(header)
    for cat in cats:
        counts = [len(df_existing[df_existing["category"] == cat])]
        for chunk_idx in range(n_chunks):
            df_chunk = df_remaining[df_remaining["_chunk"] == chunk_idx]
            counts.append(len(df_chunk[df_chunk["category"] == cat]))
        row = f"  {cat:<30}" + "".join(f"  {c:<9}" for c in counts)
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare OR-Bench data for evaluation")
    subparsers = parser.add_subparsers(dest="command")

    # --- prepare subcommand (default) ---
    prepare_parser = subparsers.add_parser("prepare", help="Download and prepare OR-Bench data")
    prepare_parser.add_argument(
        "--sample-pct",
        type=float,
        default=10,
        help="Percentage of or-bench-80k to sample (default: 10)",
    )
    prepare_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )

    # --- partition subcommand ---
    partition_parser = subparsers.add_parser(
        "partition", help="Partition OR-Bench 80k into stratified chunks"
    )
    partition_parser.add_argument(
        "--existing-sample",
        type=str,
        required=True,
        help="Path to existing sample parquet file (becomes chunk 0)",
    )
    partition_parser.add_argument(
        "--chunk-pct",
        type=float,
        default=10,
        help="Approximate percentage per chunk (default: 10)",
    )
    partition_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )

    args = parser.parse_args()

    if args.command == "partition":
        partition_or_bench(
            existing_sample=args.existing_sample,
            chunk_pct=args.chunk_pct,
            seed=args.seed,
        )
    else:
        # Default to prepare (handles both explicit "prepare" and no subcommand)
        sample_pct = getattr(args, "sample_pct", 10)
        seed = getattr(args, "seed", 42)
        prepare_or_bench(sample_pct=sample_pct, seed=seed)
