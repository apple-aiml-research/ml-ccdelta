# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import logging
from pathlib import Path

import hydra
import pandas as pd
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from strong_reject.jailbreaks import apply_jailbreaks_to_dataset

logger = logging.getLogger(__name__)


def load_dataset_split(
    data_dir: str, train_split: str, dataset: str, split: str
) -> Dataset:
    """
    Loads a specific split from the unified dataset format.

    Args:
        data_dir: Base directory containing the dataset splits
        train_split: The train split folder (e.g., "0.5")
        dataset: Either "advbench" or "strongreject"
        split: Either "train" or "test"
    """
    if dataset not in ["advbench", "strongreject"]:
        raise ValueError(
            f"Invalid dataset: {dataset}. Must be 'advbench' or 'strongreject'."
        )
    if split not in ["train", "test"]:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'.")

    file_path = Path(data_dir) / train_split / f"{dataset}_{split}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_parquet(file_path)
    return Dataset.from_pandas(df)


def apply_jailbreaks_to_split(
    data_dir: str,
    train_split: str,
    dataset: str,
    split: str,
    jailbreaks: list[str],
    output_dir: str,
    attack_models: list[str] = [],
) -> None:
    """
    Applies strong_reject jailbreaks to a dataset split and saves the result.

    Args:
        data_dir: Base directory containing the dataset splits
        train_split: The train split folder (e.g., "0.5")
        dataset: Either "advbench" or "strongreject"
        split: Either "train" or "test"
        jailbreaks: List of jailbreak method names
        output_dir: Directory to save the jailbroken dataset
    """
    logger.info(f"Loading {dataset}_{split} from train_split {train_split}")
    forbidden_prompt_dataset = load_dataset_split(data_dir, train_split, dataset, split)

    logger.info(f"Loaded {len(forbidden_prompt_dataset)} samples")
    logger.info(f"Applying {len(jailbreaks)} jailbreaks: {jailbreaks}")

    kwargs = {
        "model": attack_models[0] if len(attack_models) else None,
        "models": attack_models,
    }

    jailbroken_dataset = apply_jailbreaks_to_dataset(
        forbidden_prompt_dataset, jailbreaks, kwargs=kwargs
    )
    jb_dataset_list = jailbroken_dataset.to_list()

    # Create output directory (include train_split in path)
    output_path_dir = Path(output_dir) / train_split
    output_path_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    dataset_base_name = f"{dataset}_{split}"
    output_path = output_path_dir / f"{dataset_base_name}.jailbroken.parquet"

    df = pd.DataFrame(jb_dataset_list)

    # Extract the content from the jailbroken_prompt structure
    df["jailbroken_prompt"] = df["jailbroken_prompt"].apply(
        lambda x: (
            x[0]["content"] if isinstance(x, list) and x and "content" in x[0] else None
        )
    )

    df.to_parquet(output_path, index=False)
    logger.info(f"Saved jailbroken dataset to {output_path}")


def apply_jailbreaks(cfg: DictConfig) -> None:
    """
    Applies strong_reject jailbreaks to unified dataset splits and saves the results.

    Args:
        cfg (DictConfig): A Hydra configuration object.
    """
    # Extract configuration
    data_dir = cfg.data_dir
    train_split_pct = str(cfg.train_split_pct)
    datasets = OmegaConf.to_container(cfg.datasets, resolve=True)
    splits = OmegaConf.to_container(cfg.splits, resolve=True)

    attack_models = list(OmegaConf.to_container(cfg.attack_models, resolve=True)) if cfg.attack_models else []

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
    jailbreaks_to_apply = list(OmegaConf.to_container(cfg.jailbreaks, resolve=True))
    output_dir = cfg.output_dir

    logger.info(
        f"Applying jailbreaks to {len(datasets)} datasets, {len(splits)} splits"
    )
    logger.info(f"   Train split percentage: {train_split_pct}")
    logger.info(f"   Datasets: {datasets}")
    logger.info(f"   Splits: {splits}")
    logger.info(f"   Jailbreaks: {jailbreaks_to_apply}")
    logger.info(f"   Output: {output_dir}/{train_split_pct}/")

    total_combinations = len(datasets) * len(splits)
    current = 0

    for dataset in datasets:
        for split in splits:
            current += 1
            logger.info(f"Processing {current}/{total_combinations}: {dataset}_{split}")
            try:
                apply_jailbreaks_to_split(
                    data_dir=data_dir,
                    train_split=train_split_pct,
                    dataset=dataset,
                    split=split,
                    jailbreaks=jailbreaks_to_apply,
                    output_dir=output_dir,
                    attack_models=attack_models,
                )
                logger.info(f"Completed {dataset}_{split}")
            except Exception as e:
                logger.error(f"Failed to process {dataset}_{split}: {e}")
                raise

    logger.info(f"Finished applying jailbreaks to all datasets")


@hydra.main(
    config_path="../configs", config_name="apply_strongreject", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    apply_jailbreaks(cfg)


if __name__ == "__main__":
    main()
