# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""Takes jailbroken prompts and implements a prompt based defense by wrapping them in a template."""

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ccdelta.utils.defense import sample_and_apply_defense

logger = logging.getLogger(__name__)


def load_dataset_split(
    data_dir: str,
    train_split: str,
    dataset: str,
    split: str,
    jailbreak_variant: str = "jailbroken",
) -> pd.DataFrame:
    """
    Loads a specific split from the unified dataset format.

    Args:
        data_dir: Base directory containing the dataset splits
        train_split: The train split folder (e.g., "0.5")
        dataset: Either "advbench" or "strongreject"
        split: Either "train" or "test"
        jailbreak_variant: The jailbreak variant (e.g., "jailbroken" or "jailbroken_none")
    """
    if dataset not in ["advbench", "strongreject"]:
        raise ValueError(
            f"Invalid dataset: {dataset}. Must be 'advbench' or 'strongreject'."
        )
    if split not in ["train", "test"]:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'.")

    file_path = (
        Path(data_dir) / train_split / f"{dataset}_{split}.{jailbreak_variant}.parquet"
    )
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_parquet(file_path)
    return df


def apply_prompt_defense_to_split(
    data_dir: str,
    train_split: str,
    dataset: str,
    split: str,
    output_dir: str,
    prompt_templates: list[str],
    jailbreak_variant: str = "jailbroken",
) -> None:
    """
    Applies prompt defenses to a dataset split and saves the result.

    Args:
        data_dir: Base directory containing the dataset splits
        train_split: The train split folder (e.g., "0.5")
        dataset: Either "advbench" or "strongreject"
        split: Either "train" or "test"
        output_dir: Directory to save the prompt defense dataset
        prompt_templates: Prompt templates to use
        jailbreak_variant: The jailbreak variant (e.g., "jailbroken" or "jailbroken_none")
    """
    logger.info(
        f"Loading {dataset}_{split} ({jailbreak_variant}) from train_split {train_split}"
    )
    jailbroken_prompts = load_dataset_split(
        data_dir, train_split, dataset, split, jailbreak_variant
    )

    logger.info(f"Loaded {len(jailbroken_prompts)} samples")
    logger.info(f"Applying {len(prompt_templates)} prompts to data (random sampling)")

    # Create output directory (include train_split in path)
    output_path_dir = Path(output_dir) / train_split
    output_path_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    if jailbreak_variant == "jailbroken_none":
        dataset_base_name = f"{dataset}_{split}_none"
    else:
        dataset_base_name = f"{dataset}_{split}"
    output_path = output_path_dir / f"{dataset_base_name}.prompt_defense.parquet"

    # Extract the content from the jailbroken_prompt structure
    jailbroken_prompts["jailbroken_prompt"] = jailbroken_prompts["jailbroken_prompt"].apply(
        lambda text: sample_and_apply_defense(
            text=text,
            prompt_templates=None,
            split_str=None,
            payload_last=True,
        )
    )

    jailbroken_prompts.to_parquet(output_path, index=False)
    logger.info(f"Saved prompt defense dataset to {output_path}")


def apply_prompt_defense(cfg: DictConfig) -> None:
    """
    Applies prompt defenses to unified dataset splits and saves the results.

    Args:
        cfg (DictConfig): A Hydra configuration object.
    """
    # Extract configuration
    data_dir = cfg.data_dir
    train_split_pct = str(cfg.train_split_pct)
    datasets = OmegaConf.to_container(cfg.datasets, resolve=True)
    splits = OmegaConf.to_container(cfg.splits, resolve=True)

    prompt_templates = list(OmegaConf.to_container(cfg.prompt_templates, resolve=True))

    # Ensure they are lists
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not isinstance(splits, list):
        splits = [splits]

    # Validate datasets
    valid_datasets = {"advbench", "strongreject"}
    for dataset in datasets:
        if dataset not in valid_datasets:
            raise ValueError(
                f"Invalid dataset: {dataset}. Must be one of: {valid_datasets}"
            )

    # Validate splits
    valid_splits = {"train", "test"}
    for split in splits:
        if split not in valid_splits:
            raise ValueError(f"Invalid split: {split}. Must be one of: {valid_splits}")

    output_dir = cfg.output_dir

    logger.info(f"   Train split percentage: {train_split_pct}")
    logger.info(f"   Datasets: {datasets}")
    logger.info(f"   Splits: {splits}")
    logger.info(f"   Output: {output_dir}/{train_split_pct}/")

    total_combinations = len(datasets) * len(splits)
    current = 0

    # Process both jailbreak variants
    jailbreak_variants = ["jailbroken", "jailbroken_none"]
    total_combinations = len(datasets) * len(splits) * len(jailbreak_variants)
    current = 0

    for dataset in datasets:
        for split in splits:
            for jailbreak_variant in jailbreak_variants:
                current += 1
                variant_suffix = (
                    "_none" if jailbreak_variant == "jailbroken_none" else ""
                )
                logger.info(
                    f"Processing {current}/{total_combinations}: {dataset}_{split}{variant_suffix}"
                )
                try:
                    apply_prompt_defense_to_split(
                        data_dir=data_dir,
                        train_split=train_split_pct,
                        dataset=dataset,
                        split=split,
                        output_dir=output_dir,
                        prompt_templates=prompt_templates,
                        jailbreak_variant=jailbreak_variant,
                    )
                    logger.info(f"Completed {dataset}_{split}{variant_suffix}")
                except Exception as e:
                    logger.error(
                        f"Failed to process {dataset}_{split}{variant_suffix}: {e}"
                    )
                    raise

    logger.info("Finished applying prompt defense to all datasets")


@hydra.main(
    config_path="../configs", config_name="apply_prompt_defenses", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    apply_prompt_defense(cfg)


if __name__ == "__main__":
    main()
