#!/usr/bin/env python3
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""
SAE Feature Explorer - Command Line Utility

Inspect and analyze SAE steering features from saved hook state dicts.

Usage:
    uv run scripts/explore_sae_features.py <state_dict_path> [options]

Examples:
    # Show summary statistics
    uv run scripts/explore_sae_features.py extras/steering_statedicts/qwen-2.5-7b-it-model.layers.15.statedict

    # List top 20 features
    uv run scripts/explore_sae_features.py extras/steering_statedicts/qwen-2.5-7b-it-model.layers.15.statedict --top 20

    # Export top 50 features to CSV
    uv run scripts/explore_sae_features.py extras/steering_statedicts/qwen-2.5-7b-it-model.layers.15.statedict --export features.csv --top 50

    # Look up specific feature
    uv run scripts/explore_sae_features.py extras/steering_statedicts/qwen-2.5-7b-it-model.layers.15.statedict --lookup 122548
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_steering_state(state_dict_path: Path) -> dict:
    """
    Load and parse SAE hook state dict.

    Args:
        state_dict_path: Path to saved state dict file

    Returns:
        dict with keys:
            - mul_indices: torch.Tensor of feature IDs
            - mul_values: torch.Tensor of multiplication factors
            - add_values: torch.Tensor of steering strengths
            - sae_config: dict with SAE configuration
    """
    if not state_dict_path.exists():
        print(f"❌ Error: State dict not found at {state_dict_path}")
        sys.exit(1)

    state = torch.load(state_dict_path, map_location="cpu", weights_only=True)

    # Parse SAE config from tensor
    sae_config = None
    if "sae_config_tensor" in state:
        config_str = "".join(chr(int(c)) for c in state["sae_config_tensor"])
        sae_config = json.loads(config_str)

    return {
        "mul_indices": state["mul_indices"],
        "mul_values": state["mul_values"],
        "add_values": state["add_values"],
        "sae_config": sae_config,
    }


def create_feature_dataframe(state: dict) -> pd.DataFrame:
    """
    Create ranked DataFrame of features.

    Features are ranked by abs(steering_strength) descending.

    Args:
        state: Output from load_steering_state()

    Returns:
        pd.DataFrame with columns:
            - rank: 0-indexed rank by abs(steering_strength)
            - feature_id: SAE feature index
            - steering_strength: add_values (signed)
            - abs_strength: abs(add_values)
            - mul_value: multiplication factor
            - direction: 'positive' or 'negative'
    """
    # Convert to float32 to handle BFloat16
    feature_ids = state["mul_indices"].to(torch.long).numpy()
    add_vals = state["add_values"].to(torch.float32).numpy()
    mul_vals = state["mul_values"].to(torch.float32).numpy()

    df = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "steering_strength": add_vals,
            "abs_strength": np.abs(add_vals),
            "mul_value": mul_vals,
        }
    )

    # Sort by absolute strength descending
    df = df.sort_values("abs_strength", ascending=False).reset_index(drop=True)
    df["rank"] = df.index
    df["direction"] = df["steering_strength"].apply(
        lambda x: "positive" if x > 0 else "negative"
    )

    return df


def parse_qwen_sae_id(sae_id: str) -> str:
    """
    Parse Qwen SAE ID to extract hook point for Neuronpedia URL.

    Example: 'resid_post_layer_15_trainer_1' -> 'resid-post-aa'
    """
    # Qwen SAEs use format: {hook_point}_layer_{layer}_trainer_{trainer_id}
    # Neuronpedia format: {hook_point}-aa (autoencoder architecture)
    if "resid_post" in sae_id:
        return "resid-post-aa"
    if "mlp_out" in sae_id:
        return "mlp-out-aa"
    if "attn_out" in sae_id:
        return "attn-out-aa"
    return "resid-post-aa"  # default


def build_neuronpedia_url(feature_id: int, sae_config: dict, embed: bool = False) -> str:
    """
    Generate Neuronpedia URL for a feature.

    Args:
        feature_id: SAE feature index
        sae_config: SAE configuration dict
        embed: If True, append ?embed=true for iframe display

    Returns:
        Neuronpedia URL string
    """
    release = sae_config.get("release", "")
    sae_id = sae_config.get("sae_id", "")
    layer = sae_config.get("layer", 0)

    # Handle Qwen
    if "qwen" in release.lower():
        hook_point = parse_qwen_sae_id(sae_id)
        model_name = "qwen2.5-7b-it"
        url = f"https://www.neuronpedia.org/{model_name}/{layer}-{hook_point}/{feature_id}"

    # Handle Gemma
    elif "gemma" in release.lower():
        width = "65k" if "65k" in sae_id else "16k"
        url = f"https://www.neuronpedia.org/gemma-2-2b/{layer}-res-{width}/{feature_id}"

    # Handle Llama
    elif "llama" in release.lower():
        url = f"https://www.neuronpedia.org/llama-{layer}/{feature_id}"

    else:
        url = f"https://www.neuronpedia.org/feature/{feature_id}"

    if embed:
        url += "?embed=true"

    return url


def print_summary(df: pd.DataFrame, sae_config: dict) -> None:
    """Print summary statistics about the features."""
    print("\n" + "=" * 80)
    print("SAE STEERING STATE SUMMARY")
    print("=" * 80)

    print(f"\nSAE Configuration:")
    print(f"  Release: {sae_config.get('release', 'N/A')}")
    print(f"  SAE ID: {sae_config.get('sae_id', 'N/A')}")
    print(f"  Model: {sae_config.get('model', 'N/A')}")
    print(f"  Layer: {sae_config.get('layer', 'N/A')}")

    print(f"\nFeature Statistics:")
    print(f"  Total features: {len(df)}")
    print(f"  Positive features: {(df['steering_strength'] > 0).sum()}")
    print(f"  Negative features: {(df['steering_strength'] < 0).sum()}")
    print(f"  Mean strength: {df['steering_strength'].mean():.6f}")
    print(f"  Median strength: {df['steering_strength'].median():.6f}")
    print(f"  Mean abs strength: {df['abs_strength'].mean():.6f}")
    print(f"  Median abs strength: {df['abs_strength'].median():.6f}")
    print(f"  Max abs strength: {df['abs_strength'].max():.6f}")
    print(f"  Min abs strength: {df['abs_strength'].min():.6f}")

    print("\n" + "=" * 80)


def print_features_table(df: pd.DataFrame, sae_config: dict, top_n: int = 20) -> None:
    """Print table of top N features."""
    subset = df.head(top_n)

    print(f"\nTop {top_n} Features by Absolute Steering Strength:")
    print("=" * 100)
    print(
        f"{'Rank':<6} {'Feature ID':<12} {'Strength':<15} {'Abs Strength':<15} {'Direction':<10} {'Neuronpedia URL'}"
    )
    print("-" * 100)

    for _, row in subset.iterrows():
        rank = int(row["rank"])
        fid = int(row["feature_id"])
        strength = row["steering_strength"]
        abs_strength = row["abs_strength"]
        direction = row["direction"]
        url = build_neuronpedia_url(fid, sae_config)

        print(
            f"{rank:<6} {fid:<12} {strength:>14.6f} {abs_strength:>14.6f} {direction:<10} {url}"
        )

    print("=" * 100)


def lookup_feature(df: pd.DataFrame, sae_config: dict, feature_id: int) -> None:
    """Look up and display a specific feature by ID."""
    row = df[df["feature_id"] == feature_id]

    if len(row) == 0:
        print(f"\n❌ Feature {feature_id} not found in state dict")
        return

    row = row.iloc[0]
    url = build_neuronpedia_url(feature_id, sae_config)

    print("\n" + "=" * 80)
    print(f"FEATURE {feature_id} DETAILS")
    print("=" * 80)
    print(f"  Rank: {int(row['rank'])}")
    print(f"  Steering Strength: {row['steering_strength']:.6f}")
    print(f"  Absolute Strength: {row['abs_strength']:.6f}")
    print(f"  Direction: {row['direction']}")
    print(f"  Neuronpedia URL: {url}")
    print("=" * 80 + "\n")


def export_features(
    df: pd.DataFrame, sae_config: dict, output_path: Path, top_n: int = None
) -> None:
    """Export feature list to CSV."""
    export_df = df.copy()

    if top_n is not None:
        export_df = export_df.head(top_n)

    # Add Neuronpedia URLs
    export_df["neuronpedia_url"] = export_df["feature_id"].apply(
        lambda fid: build_neuronpedia_url(int(fid), sae_config, embed=False)
    )

    # Reorder columns
    cols = [
        "rank",
        "feature_id",
        "steering_strength",
        "abs_strength",
        "direction",
        "mul_value",
        "neuronpedia_url",
    ]
    export_df[cols].to_csv(output_path, index=False)

    print(f"\n✓ Exported {len(export_df)} features to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Explore SAE steering features from saved hook state dicts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "state_dict_path",
        type=Path,
        help="Path to saved hook state dict file",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Number of top features to display (default: show summary only)",
    )

    parser.add_argument(
        "--lookup",
        type=int,
        default=None,
        help="Look up a specific feature by ID",
    )

    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Export features to CSV file",
    )

    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip summary statistics",
    )

    args = parser.parse_args()

    # Load state dict
    print(f"Loading state dict from: {args.state_dict_path}")
    state = load_steering_state(args.state_dict_path)

    # Create feature dataframe
    print(f"✓ Loaded {len(state['mul_indices'])} features")
    df = create_feature_dataframe(state)

    # Print summary by default
    if not args.no_summary:
        print_summary(df, state["sae_config"])

    # Look up specific feature
    if args.lookup is not None:
        lookup_feature(df, state["sae_config"], args.lookup)

    # Display top N features
    if args.top is not None:
        print_features_table(df, state["sae_config"], args.top)

    # Export to CSV
    if args.export is not None:
        export_features(df, state["sae_config"], args.export, args.top)


if __name__ == "__main__":
    main()
