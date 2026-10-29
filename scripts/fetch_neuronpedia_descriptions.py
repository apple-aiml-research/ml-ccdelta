#!/usr/bin/env python3
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""
Fetch Neuronpedia Descriptions for SAE Features

Loads top N features from a state dict and fetches their human-written
descriptions from Neuronpedia, outputting a simple CSV for bulk review.

Usage:
    uv run scripts/fetch_neuronpedia_descriptions.py \
        extras/steering_statedicts/qwen-2.5-7b-it-model.layers.15.statedict \
        --model-id qwen2.5-7b-it \
        --layer 15 \
        --hook-point resid-post-aa \
        --top 100 \
        --output descriptions.csv

Example:
    uv run scripts/fetch_neuronpedia_descriptions.py \
        extras/steering_statedicts/llama-3.1-8b-it-2k-model.layers.17.statedict \
        --model-id llama3.1-8b \
        --layer 17 \
        --hook-point llamascope-res-131k \
        --top 50
"""

import sys
import time
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

# Import SAE feature utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from notebooks.sae_feature_utils import (
    create_feature_dataframe,
    fetch_neuronpedia_description,
    load_steering_state,
)


@click.command()
@click.argument(
    "state_dict_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--model-id",
    required=True,
    help="Neuronpedia model ID (e.g., 'qwen2.5-7b-it')",
)
@click.option(
    "--layer",
    type=int,
    required=True,
    help="Layer number",
)
@click.option(
    "--hook-point",
    default="resid-post-aa",
    help="Hook point ID (default: resid-post-aa)",
)
@click.option(
    "--top",
    type=int,
    default=100,
    help="Number of top features to fetch (default: 100)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output CSV file (default: {state_dict_name}_descriptions.csv)",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Print detailed progress",
)
def main(
    state_dict_path: Path,
    model_id: str,
    layer: int,
    hook_point: str,
    top: int,
    output: Path | None,
    verbose: bool,
):
    """Fetch Neuronpedia descriptions for SAE features.

    Loads top N features from STATE_DICT_PATH and fetches their human-written
    descriptions from Neuronpedia.
    """
    # Load state dict
    click.echo(f"Loading state dict from: {state_dict_path}")
    state = load_steering_state(state_dict_path)
    click.echo(f"✓ Loaded {len(state['mul_indices'])} features")

    # Create feature dataframe (preserves order from state dict)
    click.echo(f"\nCreating feature dataframe...")
    df = create_feature_dataframe(state)
    click.echo(f"✓ Created dataframe with {len(df)} features")

    # Take top N features (preserves order)
    click.echo(f"\nSelecting top {top} features...")
    df_top = df.head(top)
    click.echo(f"✓ Selected {len(df_top)} features")

    # Fetch descriptions
    click.echo(f"\nFetching Neuronpedia descriptions...")
    click.echo(f"  Model: {model_id}")
    click.echo(f"  Layer spec: {layer}-{hook_point}")

    layer_spec = f"{layer}-{hook_point}"

    results = []
    for row in tqdm(
        df_top.itertuples(), total=len(df_top), desc="Fetching descriptions"
    ):
        result = fetch_neuronpedia_description(
            model_id, layer_spec, int(row.feature_id), verbose=verbose
        )

        # Add feature metadata
        result["rank"] = int(row.rank)
        result["steering_strength"] = float(row.steering_strength)

        results.append(result)

        if verbose and result["success"]:
            click.echo(f"\n  Rank {result['rank']}: Feature {result['feature_id']}")
            click.echo(f"    Description: {result.get('description', 'N/A')[:100]}...")
        elif verbose and not result["success"]:
            click.echo(
                f"\n  Rank {result['rank']}: Feature {result['feature_id']} - FAILED"
            )
            click.echo(f"    Error: {result['error']}")

        time.sleep(0.5)

    # Count successes/failures
    successes = sum(1 for r in results if r["success"])
    failures = len(results) - successes

    click.echo(f"\n{'='*60}")
    click.echo("FETCH RESULTS")
    click.echo(f"{'='*60}")
    click.echo(f"  Total features: {len(results)}")
    click.echo(f"  Successful: {successes}")
    click.echo(f"  Failed: {failures}")

    # Determine output path
    if output is None:
        base_name = state_dict_path.stem
        output_path = state_dict_path.parent / f"{base_name}_descriptions_top{top}.csv"
    else:
        output_path = output

    # Create DataFrame
    df = pd.DataFrame(results)
    # print(results)

    # Extract top 5 explanations into separate columns
    for i in range(5):
        df[f"explanation_{i+1}"] = df["top_explanations"].apply(
            lambda x: x[i]["description"] if isinstance(x, list) and len(x) > i else ""
        )
        df[f"explanation_{i+1}_score"] = df["top_explanations"].apply(
            lambda x: x[i]["score"] if isinstance(x, list) and len(x) > i else None
        )

    # Reorder and select columns for CSV
    csv_columns = [
        "rank",
        "feature_id",
        "steering_strength",
        "num_explanations",
        "description",  # This is explanation_1
        "explanation_2",
        "explanation_3",
        "explanation_4",
        "explanation_5",
        "url",
    ]

    # Handle failed fetches
    if "error" in df.columns:
        df["description"] = df.apply(
            lambda row: (
                row.get("description", "")
                if row["success"]
                else f"ERROR: {row.get('error', 'Unknown error')}"
            ),
            axis=1,
        )

    # Save to CSV
    df[csv_columns].to_csv(output_path, index=False)

    click.echo(f"\n✓ Saved results to: {output_path}")
    click.echo(f"{'='*60}")

    # Show sample descriptions
    if successes > 0:
        click.echo("\nSample descriptions (top 5):")
        for result in results[:5]:
            if result["success"]:
                desc = result.get("description", "N/A")
                if len(desc) > 100:
                    desc = desc[:100] + "..."
                click.echo(f"\n  Rank {result['rank']}: Feature {result['feature_id']}")
                click.echo(f"    Strength: {result['steering_strength']:.6f}")
                click.echo(f"    Description: {desc}")


if __name__ == "__main__":
    main()
