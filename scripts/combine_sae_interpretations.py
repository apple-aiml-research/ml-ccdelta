#!/usr/bin/env python3
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""
Combine SAE Feature Data from Multiple Sources

Combines Neuronpedia descriptions (CSV) and semantic regex interpretations (JSON)
into a single table with the most important information for analysis.

Usage:
    uv run scripts/combine_sae_interpretations.py \
        descriptions.csv \
        interpretations.json \
        --top 50 \
        --output combined_features.csv

Example:
    uv run scripts/combine_sae_interpretations.py \
        extras/steering_statedicts/llama-3.1-8b-it-2k-model.layers.17_descriptions_top100.csv \
        extras/steering_statedicts/llama-3.1-8b-it-2k-model.layers.17_interpretations_top100_model-gpt-5_1_tokens32.json \
        --top 25 \
        --output combined_analysis.csv
"""

import json
import sys
from pathlib import Path

import click
import pandas as pd


def load_neuronpedia_descriptions(csv_path: Path) -> pd.DataFrame:
    """Load Neuronpedia descriptions from CSV file."""
    df = pd.read_csv(csv_path)

    # Select relevant columns and rename for clarity
    df_selected = df[
        ["rank", "feature_id", "steering_strength", "description", "url"]
    ].copy()
    df_selected.rename(
        columns={"description": "neuronpedia_description", "url": "neuronpedia_url"},
        inplace=True,
    )

    return df_selected


def load_semantic_interpretations(json_path: Path) -> pd.DataFrame:
    """Load semantic regex interpretations from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract interpretations
    interpretations = data["interpretations"]

    # Create DataFrame with relevant fields
    records = []
    for item in interpretations:
        record = {
            "rank": item["rank"],
            "feature_id": item["feature_id"],
            "steering_strength": item["steering_strength"],
            "semantic_regex": (
                item.get("interpretation", "")
                if item.get("success", False)
                else "No Activations Available"
            ),
            "interpretation_success": item.get("success", False),
        }
        records.append(record)

    return pd.DataFrame(records)


def load_sae_metadata(state_dict_path: Path) -> pd.DataFrame:
    """Load SAE hook metadata and rename fields for clarity.

    Args:
        state_dict_path: Path to SAE hook state dict (.statedict file)

    Returns:
        DataFrame with columns: feature_id, harmful_mean, jailbroken_mean

    Raises:
        RuntimeError: If state dict cannot be loaded or processed
    """
    try:
        # Import here to avoid dependency issues if notebooks module not available
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from notebooks.sae_feature_utils import load_steering_state

        state, _ = load_steering_state(state_dict_path)
        feature_metadata = state.get("feature_metadata", {})

        if not feature_metadata:
            return pd.DataFrame(
                columns=["feature_id", "harmful_mean", "jailbroken_mean"]
            )

        # Convert to DataFrame and rename columns for semantic clarity
        records = []
        for feat_id_str, metadata in feature_metadata.items():
            # Round activation values to 3 decimal places for cleaner output
            harmful_val = metadata.get("mean_positive", float("nan"))
            jailbroken_val = metadata.get("mean_negative", float("nan"))

            records.append(
                {
                    "feature_id": int(feat_id_str),
                    "harmful_mean": round(harmful_val, 3) if not pd.isna(harmful_val) else harmful_val,
                    "jailbroken_mean": round(jailbroken_val, 3) if not pd.isna(jailbroken_val) else jailbroken_val,
                }
            )

        return pd.DataFrame(records)

    except ImportError as e:
        raise RuntimeError(
            f"Cannot import sae_feature_utils. Ensure notebooks directory is accessible: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load SAE metadata: {e}")


def combine_data(
    neuronpedia_df: pd.DataFrame,
    interpretations_df: pd.DataFrame,
    top_n: int,
    sae_metadata_df: pd.DataFrame = None,
    include_steering_strength: bool = True,
) -> pd.DataFrame:
    """Combine data from multiple sources into a single table.

    Args:
        neuronpedia_df: DataFrame with Neuronpedia descriptions
        interpretations_df: DataFrame with semantic regex interpretations
        top_n: Number of top features to include
        sae_metadata_df: Optional DataFrame with SAE hook metadata (harmful_mean, jailbroken_mean)
        include_steering_strength: Whether to include steering_strength column in output

    Returns:
        Combined DataFrame with all available data sources
    """

    # Merge Neuronpedia and interpretations data (existing logic)
    combined = pd.merge(
        neuronpedia_df,
        interpretations_df[
            ["rank", "feature_id", "semantic_regex"]
        ],
        on=["rank", "feature_id"],
        how="left",
    )

    # Merge SAE metadata if provided
    if sae_metadata_df is not None and len(sae_metadata_df) > 0:
        combined = pd.merge(
            combined,
            sae_metadata_df[["feature_id", "harmful_mean", "jailbroken_mean"]],
            on=["feature_id"],
            how="left",
        )

    # Sort by rank and take top N
    combined = combined.sort_values("rank").head(top_n)

    # Define column order with conditional steering_strength inclusion
    final_columns = ["rank", "feature_id"]

    if include_steering_strength:
        final_columns.append("steering_strength")

    final_columns.extend([
        "harmful_mean",  # NEW: Mean activation in target samples
        "jailbroken_mean",  # NEW: Mean activation in source samples
        "neuronpedia_description",
        "semantic_regex",
        "neuronpedia_url",
    ])

    # Handle missing values for existing columns
    combined["semantic_regex"] = combined["semantic_regex"].fillna("NOT_AVAILABLE")

    # Replace NaN values in SAE metadata columns with "Inactive"
    if "harmful_mean" in combined.columns:
        combined["harmful_mean"] = combined["harmful_mean"].fillna("Inactive")
    if "jailbroken_mean" in combined.columns:
        combined["jailbroken_mean"] = combined["jailbroken_mean"].fillna("Inactive")

    # Select only available columns (for backwards compatibility when no SAE metadata)
    available_columns = [col for col in final_columns if col in combined.columns]
    return combined[available_columns]


@click.command()
@click.argument(
    "descriptions_csv",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "interpretations_json",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--top",
    type=int,
    default=50,
    help="Number of top features to include in output (default: 50)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output CSV file (default: combined_features.csv)",
)
@click.option(
    "--state-dict",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional SAE hook state dict file (.statedict) containing feature metadata with mean activations",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Print detailed progress",
)
@click.option(
    "--include-steering-strength",
    is_flag=True,
    help="Include steering_strength column in output (default: False)",
)
def main(
    descriptions_csv: Path,
    interpretations_json: Path,
    top: int,
    output: Path | None,
    state_dict: Path | None,
    verbose: bool,
    include_steering_strength: bool,
):
    """Combine SAE feature data from Neuronpedia descriptions and semantic interpretations.

    Takes a CSV file from fetch_neuronpedia_descriptions.py and a JSON file from
    interpret_sae_features.py and creates a combined table with the most important
    information for analysis.
    """

    # Load data from both sources
    click.echo(f"Loading Neuronpedia descriptions from: {descriptions_csv}")
    neuronpedia_df = load_neuronpedia_descriptions(descriptions_csv)
    click.echo(f"✓ Loaded {len(neuronpedia_df)} features from CSV")

    click.echo(f"\nLoading semantic interpretations from: {interpretations_json}")
    interpretations_df = load_semantic_interpretations(interpretations_json)
    click.echo(f"✓ Loaded {len(interpretations_df)} interpretations from JSON")

    # Check for successful interpretations
    successful_interpretations = interpretations_df["interpretation_success"].sum()
    failed_interpretations = len(interpretations_df) - successful_interpretations
    click.echo(f"  - Successful interpretations: {successful_interpretations}")
    click.echo(f"  - Failed interpretations: {failed_interpretations}")

    # Load SAE metadata if provided
    sae_metadata_df = None
    if state_dict:
        click.echo(f"\nLoading SAE metadata from: {state_dict}")
        try:
            sae_metadata_df = load_sae_metadata(state_dict)
            click.echo(
                f"✓ Loaded metadata for {len(sae_metadata_df)} features from state dict"
            )
        except Exception as e:
            click.echo(f"❌ Error loading SAE metadata: {e}", err=True)
            sys.exit(1)

    # Combine data
    click.echo(f"\nCombining data and selecting top {top} features...")
    combined_df = combine_data(
        neuronpedia_df, interpretations_df, top, sae_metadata_df, include_steering_strength
    )
    click.echo(f"✓ Created combined table with {len(combined_df)} features")

    # Determine output path
    if output is None:
        output_path = Path("combined_features.csv")
    else:
        output_path = output

    # Save results
    combined_df.to_csv(output_path, index=False)
    click.echo(f"\n✓ Saved combined data to: {output_path}")

    # Show summary statistics
    click.echo(f"\n{'='*60}")
    click.echo("SUMMARY")
    click.echo(f"{'='*60}")
    click.echo(f"Total features in output: {len(combined_df)}")
    click.echo(
        f"Features with Neuronpedia descriptions: {combined_df['neuronpedia_description'].notna().sum()}"
    )
    click.echo(
        f"Features with semantic regex interpretations: {successful_interpretations}"
    )
    click.echo(
        f"Features with both descriptions: {min(len(combined_df[combined_df['neuronpedia_description'].notna()]), successful_interpretations)}"
    )

    # Show SAE metadata statistics if available
    if sae_metadata_df is not None:
        harmful_mean_count = (
            combined_df["harmful_mean"].notna().sum()
            if "harmful_mean" in combined_df.columns
            else 0
        )
        jailbroken_mean_count = (
            combined_df["jailbroken_mean"].notna().sum()
            if "jailbroken_mean" in combined_df.columns
            else 0
        )
        both_means_count = (
            (
                (combined_df["harmful_mean"].notna())
                & (combined_df["jailbroken_mean"].notna())
            ).sum()
            if "harmful_mean" in combined_df.columns
            and "jailbroken_mean" in combined_df.columns
            else 0
        )

        click.echo(f"Features with harmful_mean: {harmful_mean_count}")
        click.echo(f"Features with jailbroken_mean: {jailbroken_mean_count}")
        click.echo(f"Features with both mean activations: {both_means_count}")

    # Show steering strength range
    if len(combined_df) > 0 and "steering_strength" in combined_df.columns:
        min_strength = combined_df["steering_strength"].min()
        max_strength = combined_df["steering_strength"].max()
        click.echo(f"Steering strength range: {min_strength:.4f} to {max_strength:.4f}")

    # Show preview of top features if verbose
    if verbose and len(combined_df) > 0:
        click.echo("\nTop 5 features preview:")
        for _, row in combined_df.head(5).iterrows():
            click.echo(f"\n  Rank {int(row['rank'])}: Feature {int(row['feature_id'])}")
            if "steering_strength" in combined_df.columns:
                click.echo(f"    Strength: {row['steering_strength']:.4f}")

            # Show Neuronpedia description
            desc = str(row["neuronpedia_description"])
            if pd.notna(row["neuronpedia_description"]) and desc.strip():
                if len(desc) > 80:
                    desc = desc[:80] + "..."
                click.echo(f"    Neuronpedia: {desc}")
            else:
                click.echo("    Neuronpedia: [No description]")

            # Show semantic regex
            regex = str(row["semantic_regex"])
            if (
                regex != "No Activations Available"
                and regex != "NOT_AVAILABLE"
                and pd.notna(regex)
                and regex.strip()
            ):
                if len(regex) > 80:
                    regex = regex[:80] + "..."
                click.echo(f"    Semantic Regex: {regex}")
            else:
                click.echo("    Semantic Regex: [Failed/Unavailable]")

            # Show SAE metadata if available
            if (
                "harmful_mean" in combined_df.columns
                and "jailbroken_mean" in combined_df.columns
            ):
                harmful = row["harmful_mean"]
                jailbroken = row["jailbroken_mean"]

                # Format values (handle both numeric and "Inactive" string values)
                def format_activation(val):
                    if isinstance(val, str):
                        return val  # "Inactive"
                    elif pd.notna(val):
                        return f"{val:.3f}"
                    else:
                        return "Inactive"

                harmful_str = format_activation(harmful)
                jailbroken_str = format_activation(jailbroken)

                click.echo(f"    Activation Pattern: {jailbroken_str} → {harmful_str}")

    click.echo(f"\n{'='*60}")


if __name__ == "__main__":
    main()
