# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

import typing as t
from pathlib import Path

import torch
import transformers

from .json_subsets_dataset import (
    get_json_subsets_dataset,
    get_json_from_tabular_dataset,
)
from .tabular_paired_dataset import get_tabular_paired_dataset
from .samplers import StratifiedSampler


def get_dataset(
    name: str,
    datasets_folder: Path,
    split: str,
    subsets: t.Set[str],
    tokenizer: t.Optional[transformers.PreTrainedTokenizer] = None,
    **kwargs,
) -> t.Tuple[torch.utils.data.Dataset, t.Callable]:
    """Loads and returns a dataset split given its name. It also returns a collator function for the dataloader

    Args:
        name (str): dataset name
        datasets_folder (Path): path where dataset is located
        split (str): train, val, test
        subsets (Set[str]): dataset subsets to load
        tokenizer (t.Optional[transformers.PreTrainedTokenizer], optional): a huggingface tokenizer in case it is a text dataset. Defaults to None.
        **kwargs: Dataset-specific parameters

    Returns:
        t.Tuple[torch.utils.data.Dataset, t.Callable]: pytorch Dataset instance and collator function
    """
    if name == "json-subsets":
        return get_json_subsets_dataset(
            datasets_folder,
            split=split,
            subsets=subsets,
            tokenizer=tokenizer,
            **kwargs,
        )
    elif name == "tabular-paired":
        return get_tabular_paired_dataset(
            datasets_folder,  # Required by interface but unused
            split=split,  # Required by interface but unused
            subsets=subsets,  # Required by interface but unused
            tokenizer=tokenizer,
            **kwargs,  # Contains file_path, source_column, target_column
        )
    elif name == "tabular_to_json":
        return get_json_from_tabular_dataset(
            datasets_folder,
            split=split,
            subsets=subsets,
            tokenizer=tokenizer,
            **kwargs,  # Contains file_path, source_column, target_column
        )
    else:
        raise ValueError(
            f"Unknown dataset type: {name}. Available: ['json-subsets', 'tabular-paired']"
        )


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    drop_last: bool,
    shuffle: bool,
    collate_fn: t.Optional[t.Callable] = None,
    balanced: bool = False,
    seed: int = 0,
    **kwargs: dict,
) -> torch.utils.data.DataLoader:
    import logging

    logger = logging.getLogger(__name__)

    # Get dataset type to determine sampling strategy
    dataset_type = getattr(dataset, "data_type", type(dataset).__name__)

    # Choose sampling strategy based on dataset type
    if dataset_type == "tabular_paired":
        # Paired datasets: always preserve pairs with random sampling
        sampler = None
        shuffle = shuffle
        if balanced:
            # Log info if balanced was requested but ignored
            logger.info(
                "Paired dataset detected: using random pair sampling "
                "(balanced sampling not applicable to preserve pairs)"
            )

    elif (
        dataset_type in ["json_subsets", "JsonSubsetsDataset", "tabular_to_json"]
        and balanced
    ):
        # Subset datasets: use stratified sampling when requested
        if hasattr(dataset, "subsets") and getattr(dataset, "subsets", None):
            sampler = StratifiedSampler(dataset.subsets, seed=seed)
            shuffle = False
            logger.info(f"Using stratified sampling for {dataset_type}")
        else:
            # Fallback if subsets not available
            sampler = None
            shuffle = shuffle
            logger.warning(
                f"Balanced sampling requested but no subsets found in {dataset_type}. "
                f"Using regular sampling."
            )

    else:
        # Default: regular sampling for all other cases
        sampler = None
        shuffle = shuffle
        if balanced:
            logger.info(
                f"Using regular sampling for {dataset_type} (balanced sampling not applicable)"
            )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        shuffle=shuffle,
        sampler=sampler,
        **kwargs,
    )


if __name__ == "__main__":
    pass
