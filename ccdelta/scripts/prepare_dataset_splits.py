# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from strong_reject.load_datasets import load_strongreject

logger = logging.getLogger(__name__)


def prepare_dataset_splits(
    dataset: str,
    output_dir: str,
    train_ratio: float,
    random_seed: int,
    output_format: str,
    advbench_data_path: Optional[str] = None,
) -> None:
    """
    Creates deterministic train/test splits for both AdvBench and StrongReject datasets.

    Args:
        dataset: Either "advbench" or "strongreject"
        output_dir: Directory to save the split files
        train_ratio: Proportion of data for training (default: 0.75)
        random_seed: Seed for deterministic splitting
        output_format: Output format, either "parquet" or "json"
        advbench_data_path: Path to AdvBench CSV file (only used for advbench dataset)
    """
    if output_format not in ["parquet", "json"]:
        raise ValueError(
            f"output_format must be 'parquet' or 'json', got {output_format}"
        )

    output_dir = os.path.join(output_dir, str(train_ratio))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set up deterministic random generator
    rng = random.Random(random_seed)

    if dataset == "advbench":
        df = _prepare_advbench_data(
            advbench_data_path or "data/raw/advbench_behaviours/harmful_behaviors.csv"
        )
    elif dataset == "strongreject":
        df = _prepare_strongreject_data()
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. Must be 'advbench' or 'strongreject'"
        )

    # Create deterministic train/test split
    indices = list(range(len(df)))
    rng.shuffle(indices)

    train_size = int(len(indices) * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    # Add split column
    train_df["split"] = "train"
    test_df["split"] = "test"

    # Save splits based on output format
    file_extension = output_format
    train_path = Path(output_dir) / f"{dataset}_train.{file_extension}"
    test_path = Path(output_dir) / f"{dataset}_test.{file_extension}"

    if output_format == "parquet":
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
    elif output_format == "json":
        train_df.to_json(train_path, orient="records", indent=2)
        test_df.to_json(test_path, orient="records", indent=2)

    logger.info(
        f"Saved {dataset} train split ({len(train_df)} samples) to {train_path}"
    )
    logger.info(f"Saved {dataset} test split ({len(test_df)} samples) to {test_path}")


def _prepare_advbench_data(data_path: str) -> pd.DataFrame:
    """Prepare AdvBench data with unified column schema."""
    # Download the dataset if it doesn't exist
    if not Path(data_path).exists():
        logger.info(f"Downloading {data_path}...")
        subprocess.run(
            [
                "curl",
                "-o",
                str(data_path),
                "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
            ],
            check=True,
        )
    else:
        logger.info(f"Dataset already exists at {data_path}")

    df = pd.read_csv(data_path)

    # Rename goal column to forbidden_prompt for consistency
    df = df.rename(columns={"goal": "forbidden_prompt"})

    # Add unified columns
    df["dataset"] = "advbench"
    df["source"] = "AdvBench"
    df["category"] = None  # AdvBench doesn't have categories

    # Keep only the columns we need
    df = df[["forbidden_prompt", "dataset", "source", "category"]]

    return df


def _prepare_strongreject_data() -> pd.DataFrame:
    """Prepare StrongReject data with unified column schema."""
    logger.info("Loading StrongReject dataset...")
    strongreject_ds = load_strongreject()
    df = strongreject_ds.to_pandas()

    # Filter out samples that are sourced from AdvBench to avoid duplication
    logger.info(f"Original StrongReject samples: {len(df)}")
    df = df[df["source"] != "AdvBench"]
    logger.info(f"StrongReject samples after filtering AdvBench: {len(df)}")

    # Add dataset column
    df["dataset"] = "strongreject"

    # Reorder columns to match unified schema
    df = df[["forbidden_prompt", "dataset", "source", "category"]]

    return df


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["advbench", "strongreject", "all"]),
    default="all",
    help="Dataset to prepare splits for",
)
@click.option(
    "--output-dir",
    default="data/forbidden_prompts",
    help="Directory to save split files",
)
@click.option(
    "--train-ratio",
    type=float,
    default=0.5,
    help="Proportion of data for training",
)
@click.option(
    "--random-seed",
    type=int,
    default=42,
    help="Random seed for deterministic splitting",
)
@click.option(
    "--output-format",
    type=click.Choice(["parquet", "json"]),
    default="parquet",
    help="Output file format",
)
@click.option(
    "--advbench-data-path",
    help="Path to AdvBench CSV file (defaults to data/raw/advbench_behaviours/harmful_behaviors.csv)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    dataset: str,
    output_dir: str,
    train_ratio: float,
    random_seed: int,
    output_format: str,
    advbench_data_path: Optional[str],
    verbose: bool,
) -> None:
    """Prepare train/test splits for AdvBench and/or StrongReject datasets with unified schema."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    if dataset == "all":
        datasets_to_process = ["advbench", "strongreject"]
    else:
        datasets_to_process = [dataset]

    for dataset_name in datasets_to_process:
        logger.info(f"Preparing {dataset_name} dataset splits...")
        prepare_dataset_splits(
            dataset=dataset_name,
            output_dir=output_dir,
            train_ratio=train_ratio,
            random_seed=random_seed,
            output_format=output_format,
            advbench_data_path=advbench_data_path,
        )

    click.echo(f"Dataset preparation complete! Files saved to {output_dir}/")


if __name__ == "__main__":
    main()
